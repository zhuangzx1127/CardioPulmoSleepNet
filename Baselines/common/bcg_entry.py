"""BCGNet training on native MAT-axis IHR/ABD with reviewed XML supervision."""
import ast
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from .data import ROOT
from .native_inputs import NativeCache,bcg_stage,bcg_events
from .native_hparams import resolve,optimizer_and_scheduler
from .models_torch import bcgnet,focal_bcg,BASE
from .protocol import make_run,write_json,verify_source,metrics
from .prediction_export import export_predictions
from .torch_entry import parser


def event_counts(y,logits,mask):
    """Original 0.5-threshold segment metric, evaluated on contiguous valid spans."""
    if not hasattr(event_counts,'scope'):
        path=BASE/'bcgnet/upstream/metrics/event_eval.py'
        scope={'np':np}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node,ast.FunctionDef) and node.name in ('cal_cls_segment','merge_intervals','match','cal_metrics'):
                exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        event_counts.scope=scope
    f=event_counts.scope;counts=np.zeros(3)
    changes=np.diff(np.pad(mask.astype(int),(1,1)))
    for start,end in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
        gt=f['cal_cls_segment'](y[start:end]);pred=f['cal_cls_segment']((logits[start:end]>0).astype(int))
        gt=f['merge_intervals'](gt) if gt else []
        pred=f['merge_intervals'](pred) if pred else []
        pred=[s for s in pred if s[1]-s[0]>=10]
        counts+=f['cal_metrics'](gt,pred)
    return counts


def normalize_event_ihr(r):
    """Per-record valid IHR z-score within the prepared 10-hour window."""
    h=np.asarray(r['heartbeat'],dtype=np.float32)
    valid=np.asarray(r['heartbeat_valid'],dtype=bool)&np.isfinite(h)
    z=np.zeros_like(h)
    if valid.any():
        values=h[valid].astype(np.float64)
        z[valid]=((values-values.mean())/max(float(values.std()),1e-8)).astype(np.float32)
    return dict(r,heartbeat=z)


class Samples(Dataset):
    def __init__(self,cache,cohorts,part,task,smoke=False,smoke_samples=2,ihr_zscore=False):
        self.ihr_zscore=ihr_zscore;self.cache=cache;self.task=task;self.smoke=smoke;self.items=[];self.last=None
        refs=cache.index(cohorts,part)
        if smoke:refs=refs[:smoke_samples]
        for rid in refs:
            m=cache.rows[rid];seconds=min(36000,int(m['source_duration_seconds']))
            if task=='stage':self.items.append((rid,0,min(1200,seconds//30)))
            else:
                for start in range(0,seconds,1800):
                    self.items.append((rid,start,min(seconds,start+1800)))
        if smoke:self.items=self.items[:smoke_samples]
        if not self.items:raise ValueError('No eligible samples: '+part)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        rid,start,end=self.items[i]
        if self.last is None or self.last[0]!=rid:self.last=(rid,self.cache.read(rid))
        r=self.last[1]
        if self.task=='stage':
            x,y=bcg_stage(r);m=y>=0;factor=120
        else:
            if self.ihr_zscore:r=normalize_event_ihr(r)
            x,y,m=bcg_events(r);factor=4
        if self.smoke:end=min(end,start+(32 if self.task=='stage' else 60))
        return {k:np.asarray(v[start*factor:end*factor],np.float32) for k,v in x.items()},np.asarray(y[start:end],np.int64),np.asarray(m[start:end],bool),rid


def collate(batch):
    lengths=[len(b[1]) for b in batch];n=max(lengths);factor=len(batch[0][0]['heartbeat'])//lengths[0]
    x={k:torch.from_numpy(np.stack([np.pad(b[0][k],(0,(n-len(b[1]))*factor)) for b in batch])) for k in ('heartbeat','breath')}
    x['length']=torch.tensor(lengths)
    y=torch.from_numpy(np.stack([np.pad(b[1],(0,n-len(b[1])),constant_values=-1) for b in batch]))
    mask=torch.from_numpy(np.stack([np.pad(b[2],(0,n-len(b[1])),constant_values=False) for b in batch]))
    return x,y,mask,[b[3] for b in batch]


def loader(cache,a,part,smoke):
    data=Samples(cache,a.cohorts,part,a.task,smoke,max(2,getattr(a,'gpus',1)),ihr_zscore=getattr(a,'ihr_zscore',False))
    gen=torch.Generator().manual_seed(a.seed)
    sampler=None
    if part=='train' and a.task=='events':
        classes=np.array([np.digitize(cache.rows[rid]['ahi'],[5,15,30]) for rid,_,_ in data.items])
        counts=np.bincount(classes,minlength=4)
        sampler=WeightedRandomSampler(torch.as_tensor(1/counts[classes],dtype=torch.double),len(classes),replacement=True,generator=gen)
    return DataLoader(data,batch_size=a.batch_size if part=='train' else a.val_batch_size,shuffle=part=='train' and sampler is None,sampler=sampler,collate_fn=collate,num_workers=a.workers,generator=gen,pin_memory=torch.cuda.is_available())


def phase(model,batches,a,device,opt=None):
    model.train(opt is not None);total=weight=correct=count=updates=skipped=0;segments=np.zeros(3)
    for x,y,m,ids in batches:
        if not m.any():skipped+=len(ids);continue
        # Normalize over the entire effective batch, including class weights.
        w=torch.tensor([.18,.22,.18,.22,.20],device=device)
        denominator=float(torch.tensor([.18,.22,.18,.22,.20])[y[m]].sum()) if a.task=='stage' else int(m.sum())
        micro=a.micro_batch_size or len(ids);batch_loss=0.
        if opt is not None:opt.zero_grad(set_to_none=True)
        for start in range(0,len(ids),micro):
            stop=min(len(ids),start+micro)
            yy=y[start:stop].to(device);mm=m[start:stop].to(device)
            if not mm.any():continue
            xx={k:v[start:stop].to(device) for k,v in x.items()}
            with torch.set_grad_enabled(opt is not None):
                pred=model(**xx)
                if a.task=='stage':
                    loss=nn.functional.cross_entropy(pred[mm],yy[mm],weight=w,reduction='sum')/denominator
                    correct+=int((pred.argmax(-1)[mm]==yy[mm]).sum())
                else:
                    pred=pred.reshape_as(yy)
                    loss=focal_bcg(pred,yy.clamp_min(0),mm)*int(mm.sum())/denominator
                    if opt is None:
                        for target,pp,valid in zip(yy.cpu().numpy(),pred.detach().cpu().numpy(),mm.cpu().numpy()):segments+=event_counts(target,pp,valid)
                if not torch.isfinite(loss):raise ValueError('Nonfinite BCGNet loss')
                if opt is not None:loss.backward()
                batch_loss+=float(loss.detach())
        if opt is not None:
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite BCGNet gradients')
            opt.step();updates+=1
        total+=batch_loss*len(ids);weight+=len(ids);count+=int(m.sum())
        print(json.dumps(dict(phase='train' if opt else 'val',samples=weight,updates=updates,loss=total/weight)),flush=True)
    if not weight:raise ValueError('No supervised samples')
    tp,fp,fn=segments;denom=2*tp+fp+fn
    return dict(loss=total/weight,samples=weight,valid_targets=count,updates=updates,skipped_no_valid_targets=skipped,accuracy=correct/count if a.task=='stage' else None,event_f1=float(2*tp/denom) if denom>0 else 0.,event_counts=segments.tolist())


def build_model(task, no_batchnorm=False):
    """Optional event-only BN ablation; leave the pinned upstream source intact."""
    if no_batchnorm and task != 'events':
        raise ValueError('--no-batchnorm is supported only for --task events')
    model = bcgnet(task)
    if no_batchnorm:
        def replace(parent):
            for name, child in list(parent.named_children()):
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    setattr(parent, name, nn.Identity())
                else:
                    replace(child)
        replace(model)
    return model


def load(run,task,cache,device):
    c=torch.load(Path(run)/'best.pt',map_location=device,weights_only=False)
    if c.get('native_index_sha256')!=cache.meta['native_index_sha256'] or c.get('task')!=task or c.get('baseline')!='bcgnet':raise ValueError('Checkpoint/native dataset mismatch')
    model=build_model(task,c.get('arguments',{}).get('no_batchnorm',False)).to(device);model.load_state_dict(c['state_dict'],strict=True);model.eval();model.ihr_zscore=c.get('arguments',{}).get('ihr_zscore',False);return model


@torch.no_grad()
def predict(model,r,task,device,stage_model=None,sleep_source='predicted'):
    out=dict(recording_id=r['rid'],participant_id=r['pid'],ahi_true=r['ahi'],stage_true=r['stage'],mask=r['mask'])
    def stage(net):
        x,_=bcg_stage(r);length=min(1200,int(r['metadata']['source_duration_seconds'])//30)
        xx={k:torch.from_numpy(v[None,:length*120]).to(device) for k,v in x.items()};xx['length']=torch.tensor([length],device=device)
        p=net(**xx)[0].softmax(-1).cpu().numpy();p=np.pad(p,((0,1200-length),(0,0)))
        return p,np.stack([p[:,0],p[:,1]+p[:,2],p[:,3],p[:,4]],-1)
    if task=='stage':out['stage_prob5'],out['stage_prob']=stage(model)
    else:
        x,y,m=bcg_events(normalize_event_ihr(r) if getattr(model,'ihr_zscore',False) else r);out['event_true']=y;out['event_mask']=m;prob=np.zeros(36000,np.float32)
        seconds=min(36000,int(r['metadata']['source_duration_seconds']))
        for start in range(0,seconds,1800):
            end=min(seconds,start+1800)
            xx={k:torch.from_numpy(v[None,start*4:end*4]).to(device) for k,v in x.items()};xx['length']=torch.tensor([end-start],device=device)
            prob[start:end]=model(**xx).sigmoid().reshape(-1).cpu().numpy()
        out['event_prob']=prob
        if stage_model is not None or sleep_source=='reference':
            if sleep_source=='reference':
                asleep=np.repeat(r['stage5']>0,30)&np.repeat(r['mask'],30)&m
                out['sleep_source']='reference_stage_labels'
            else:
                out['stage_prob5'],out['stage_prob']=stage(stage_model)
                asleep=np.repeat(out['stage_prob'].argmax(-1)!=0,30)&np.repeat(r['mask'],30)&m
                out['sleep_source']='predicted_bcgnet_stage'
            positive=(prob>=.5)&asleep
            changes=np.diff(np.pad(positive.astype(int),(1,1)));lengths=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
            hours=asleep.sum()/3600
            out['ahi_pred']=float((lengths>=10).sum()/hours) if hours else float('nan')
            out['ahi_status']='VALID' if hours else 'NO_VALID_SLEEP'
            out['predicted_event_count']=int((lengths>=10).sum());out['sleep_hours']=float(hours)
            out['ahi_method']='contiguous_10s_events_in_sleep / valid_sleep_hours'
    return out


def validate_sleep_run(cache,a):
    if a.task!='events' or a.sleep_source=='reference':return
    if not a.stage_run:raise ValueError('Event AHI export needs --stage-run STAGING_RUN; alternatively explicitly use --sleep-source reference, or --skip-test-export for training only')
    run=Path(a.stage_run)
    if not (run/'best.pt').is_file():raise ValueError('Staging best.pt missing: '+str(run))
    proto=json.loads((run/'protocol.json').read_text())
    if proto.get('native_index_sha256')!=cache.meta['native_index_sha256'] or proto.get('baseline')!='bcgnet' or proto['arguments']['task']!='stage':raise ValueError('Staging run/native dataset mismatch')


def export_test(cache,a,device,run,partition):
    validate_sleep_run(cache,a)
    if getattr(a,'gpus',1)>1:
        from .parallel_export import parallel_export
        return parallel_export('bcgnet',cache,a,run,partition)
    model=load(run,a.task,cache,device)
    stage_model=load(a.stage_run,'stage',cache,device) if a.task=='events' and a.sleep_source=='predicted' else None
    return export_predictions(cache,a.cohorts,partition,Path(getattr(a,'export_output',run)),lambda r:predict(model,r,a.task,device,stage_model,a.sleep_source),dict(baseline='bcgnet',task=a.task,sleep_source=a.sleep_source if a.task=='events' else 'predicted',reference_sleep_assisted=a.task=='events' and a.sleep_source=='reference',stage_run=str(Path(a.stage_run).resolve()) if stage_model is not None else None))


def main():
    p=parser('bcgnet');p.add_argument('--native-inputs',default=str(ROOT/'BaselineInputs/native_10h'));p.add_argument('--workers',type=int,default=0);p.add_argument('--micro-batch-size',type=int,default=0,help='Accumulate within the native effective batch; 0 disables splitting')
    next(action for action in p._actions if action.dest=='command').choices.append('export')
    p.add_argument('--gpus',type=int,default=1,help='DataParallel GPUs; batch-size remains global')
    p.add_argument('--ihr-zscore',action='store_true',help='Event-only per-night z-score on valid IHR samples before chunking; padding remains zero')
    p.add_argument('--no-batchnorm',action='store_true',help='Event-only ablation: replace BatchNorm with Identity; train from scratch')
    p.add_argument('--skip-test-export',action='store_true')
    p.add_argument('--sleep-source',choices=['predicted','reference'],default='predicted',help='Reference uses true staging labels; explicitly labelled assisted AHI evaluation')
    a=resolve('bcgnet',p.parse_args())
    if a.ihr_zscore and a.task!='events':raise ValueError('--ihr-zscore requires --task events')
    if a.no_batchnorm and a.task!='events':raise ValueError('--no-batchnorm requires --task events')
    if a.task=='ahi':raise ValueError('Use --task events --stage-run STAGING_RUN to estimate AHI')
    if int(os.environ.get('WORLD_SIZE','1'))>1:raise ValueError('Launch with python --gpus 4, not torchrun; this entry uses DataParallel')
    if min(a.epochs,a.batch_size,a.val_batch_size,a.threads)<1 or a.workers<0 or a.micro_batch_size<0:raise ValueError('Invalid training settings')
    if a.gpus<1:raise ValueError('--gpus must be positive')
    if a.gpus>1 and (not torch.cuda.is_available() or torch.cuda.device_count()<a.gpus or a.device=='cpu'):raise ValueError('Requested GPU count unavailable')
    if a.gpus>1 and (min(a.batch_size,a.val_batch_size)<a.gpus or (a.micro_batch_size and a.micro_batch_size<a.gpus)):raise ValueError('Global batch and nonzero micro-batch must cover all GPUs')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    cache=NativeCache(a.native_inputs,'bcgnet');verify_source('bcgnet')
    if a.command=='check':
        print(json.dumps(dict(baseline='bcgnet',channels=['RAW_ECG_DERIVED_IHR_4HZ','RAW_ABD_BESSEL_4HZ'],task=a.task,hyperparameters=vars(a),native_index_sha256=cache.meta['native_index_sha256'],counts={c:{p:len(cache.index([c],p)) for p in ['train','val','test']} for c in a.cohorts},excluded=cache.excluded),indent=2));return
    if a.gpus>1:device=torch.device('cuda:0')
    model=build_model(a.task,a.no_batchnorm).to(device)
    if a.gpus>1:model=nn.DataParallel(model,device_ids=list(range(a.gpus)))
    if a.command=='model-check':
        n=32 if a.task=='stage' else 60;factor=120 if a.task=='stage' else 4
        pred=model(heartbeat=torch.randn(2,n*factor,device=device),breath=torch.randn(2,n*factor,device=device),length=torch.tensor([n,n],device=device));pred.square().mean().backward()
        print('SYNTHETIC_FORWARD_BACKWARD',list(pred.shape));return
    if not a.run:raise ValueError('--run required')
    if a.command in ('evaluate','export'):
        del model
        export_test(cache,a,device,Path(a.run),a.partition);return
    if a.command=='train' and not a.skip_test_export:validate_sleep_run(cache,a)
    a.training_protocol=dict(stage_loss='weighted_CE_0.18_0.22_0.18_0.22_0.20',event_loss='sigmoid_focal_gamma2_alpha0.25',event_chunk_seconds=1800,stage_input='first_10h_1200_epochs',event_training_sampler='inverse_AHI_class_frequency_with_replacement',validation_sampler='deterministic_all_samples',monitor='val_accuracy' if a.task=='stage' else 'val_native_segment_f1',precision='float32',gpu_count=a.gpus,parallelism='DataParallel' if a.gpus>1 else 'single',global_batch_size=a.batch_size,nominal_per_gpu_batch=a.batch_size/a.gpus,micro_batch_size=a.micro_batch_size,batchnorm_statistics='per GPU/microbatch; DataParallel retains GPU0 running buffers; differs from global-batch BN',mask='unknown_stage_and_invalid_IHR_excluded',ahi_estimator='contiguous_10s_events_in_predicted_sleep / valid_predicted_sleep_hours; adaptation')
    a.training_protocol['event_ihr_normalization']='per_record_valid_10h_zscore_before_chunking' if a.ihr_zscore else 'raw_bpm'
    a.training_protocol.update(normalization_variant='no_batchnorm_ablation' if a.no_batchnorm else 'upstream_batchnorm')
    if a.no_batchnorm:a.training_protocol['batchnorm_statistics']='not applicable: all BatchNorm layers replaced by Identity'
    smoke=a.command=='smoke';train=loader(cache,a,'train',smoke);val=loader(cache,a,'val',smoke)
    run=make_run(a.run,a,cache,'bcgnet');opt,_,_=optimizer_and_scheduler(model,'bcgnet',a)
    history=[];best=-float('inf');stale=0
    for e in range(1,(1 if smoke else a.epochs)+1):
        t=phase(model,train,a,device,opt);v=phase(model,val,a,device)
        score=v['accuracy'] if a.task=='stage' else v['event_f1']
        history.append(dict(epoch=e,train=t,val=v));write_json(run/'history.json',history)
        state=dict(baseline='bcgnet',task=a.task,state_dict=(model.module if isinstance(model,nn.DataParallel) else model).state_dict(),optimizer_state=opt.state_dict(),epoch=e,arguments=vars(a),manifest_sha256=cache.meta['manifest_sha256'],native_index_sha256=cache.meta['native_index_sha256'])
        torch.save(state,run/'last.pt')
        if score>best:best=score;stale=0;torch.save(state,run/'best.pt')
        else:stale+=1
        if a.patience and stale>=a.patience:break
    del model,opt,state
    if device.type=='cuda':torch.cuda.empty_cache()
    restored=load(run,a.task,cache,device);del restored
    write_json(run/'completion.json',dict(status='REAL_DATA_SMOKE' if smoke else 'TRAINED',epochs=len(history),best_validation_score=best,checkpoint_strict_reload=True,test_evaluated=False))
    if not smoke and not a.skip_test_export:
        export_test(cache,a,device,run,'test')
        write_json(run/'completion.json',dict(status='TRAINED',epochs=len(history),best_validation_score=best,checkpoint_strict_reload=True,test_evaluated=True,test_predictions='test_predictions.json'))
    print('Completed:',run)
