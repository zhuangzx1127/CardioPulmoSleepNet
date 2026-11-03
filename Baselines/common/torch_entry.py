"""From-scratch native model training; missing event annotations fail closed."""
import argparse
import json
import os
import random
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
from torch import nn
from .data import Cache, ROOT, signal_mask, zscore, bcg_features
from .labels import EventLabels
from . import models_torch as models
from .protocol import verify_source, make_run, write_json, metrics
from .native_hparams import resolve,optimizer_and_scheduler

def parser(name):
    p=argparse.ArgumentParser(description=name+' reproducible processed-cache baseline')
    p.add_argument('command',choices=['check','model-check','smoke','train','evaluate'])
    p.add_argument('--cache',default=str(ROOT/'TrainingCache'));p.add_argument('--manifest',default=str(ROOT/'Dataset/splits.json'))
    p.add_argument('--cohorts',nargs='+',choices=['mesa','mros','shhs'],default=['mesa','mros','shhs'])
    p.add_argument('--task',choices=['stage','events','ahi'],default='stage' if name!='tsdnet' else 'events')
    p.add_argument('--event-labels',default=str(ROOT/'EventLabels'));p.add_argument('--event-run');p.add_argument('--stage-run')
    p.add_argument('--run',required=False);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int)
    p.add_argument('--lr',type=float);p.add_argument('--batch-size',type=int);p.add_argument('--val-batch-size',type=int)
    p.add_argument('--optimizer',choices=['Adam','AdamW']);p.add_argument('--weight-decay',type=float);p.add_argument('--patience',type=int)
    p.add_argument('--grad-clip',type=float);p.add_argument('--warmup-steps',type=int);p.add_argument('--tau',type=float);p.add_argument('--min-lr',type=float)
    p.add_argument('--prediction-batch-size',type=int,default=2048)
    p.add_argument('--device',default='auto');p.add_argument('--threads',type=int,default=4)
    p.add_argument('--smoke-epochs',type=int,default=32);p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--event-positive-seconds',type=int,default=1)
    return p

def tensor(x,device,dtype=torch.float32):return torch.as_tensor(np.array(x,copy=True),dtype=dtype,device=device)

def prepare_wav(r,device,epochs=1200):
    ecg=zscore(r['ecg'],signal_mask(r,1024))[:epochs*1024]
    resp=resample_poly(r['respi'],1,4).astype(np.float32)
    resp=zscore(resp,signal_mask(r,256))[:epochs*256]
    return {'ECG':tensor(ecg[None],device),'THX':tensor(resp[None],device)}

def tsd_windows(r):
    x=r['spo2'];x=np.pad(x,(120,120));out=[]
    for i in range(1200):
        v=x[i*30:i*30+270]
        # Match upstream 256 points / 270 seconds, retaining the original timeline.
        v=np.interp(np.linspace(0,269,256),np.arange(270),v).astype(np.float32)
        out.append((v-v.mean())/max(float(v.std()),1e-8))
    return np.asarray(out)[:,None,:]

def event_target(r,labels,cache,threshold):
    y,m=labels.read(r,cache)
    return (y.reshape(1200,30).sum(1)>=threshold).astype(np.int64),m.reshape(1200,30).all(1)

def make_model(name,task):
    if name=='wav2sleep':return models.wav2sleep()
    if name=='tsdnet':return models.ahinet() if task=='ahi' else models.tsdnet()
    return models.bcgnet(task)

def load_model(name,task,run,device,cache):
    checkpoint=torch.load(Path(run)/'best.pt',map_location=device,weights_only=False)
    if checkpoint['manifest_sha256']!=cache.meta['manifest_sha256'] or checkpoint['baseline']!=name or checkpoint['task']!=task:raise ValueError('Checkpoint/protocol mismatch')
    model=make_model(name,task).to(device);model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval();return model

def event_probability(model,r,device,batch):
    x=tsd_windows(r);out=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(x),batch):out.append(model(tensor(x[start:start+batch],device)).cpu().numpy())
    return np.concatenate(out)

def ahi_input(prob,mask):
    # Avoid joining gaps: mask on the original 1200-position time axis.
    prob=np.where(mask[:,None],prob,0)
    x=np.arange(1200);q=np.linspace(0,1199,1024)
    return np.stack([np.interp(q,x,prob[:,i]) for i in range(2)]).astype(np.float32)[None]

def grouped(items,size):
    group=[]
    for item in items:
        group.append(item)
        if len(group)==size:yield group;group=[]
    if group:yield group

def prepared_samples(records,name,task,cache,labels,a,device,event_model,training):
    for r in records:
        if name=='wav2sleep':yield r
        elif task=='events':
            x=tsd_windows(r);y,m=event_target(r,labels,cache,a.event_positive_seconds);idx=np.flatnonzero(m)
            if training:np.random.shuffle(idx)
            for i in idx:yield (x[i],y[i],r['rid'])
        else:
            prob=event_probability(event_model,r,device,a.prediction_batch_size)
            yield (ahi_input(prob,r['mask'])[0],r['ahi'],r['rid'])

def batch_loss(model,group,name,task,a,device,smoke):
    if name=='wav2sleep':
        length=a.smoke_epochs if smoke else 1200
        prepared=[prepare_wav(r,device,length) for r in group]
        x={k:torch.cat([r[k] for r in prepared],dim=0) for k in ('ECG','THX')}
        if model.training:
            # Native training/main.yaml flip_polarity=True, p=0.5 per signal/night.
            for key,value in x.items():x[key]=value*(2*torch.randint(0,2,(len(group),1),device=device)-1)
        y=tensor(np.stack([r['stage'][:length] for r in group]),device,torch.long)
        m=tensor(np.stack([r['mask'][:length] for r in group]),device,torch.bool)
        pred=model(x)
        if not m.any():raise ValueError('No scored epochs in batch')
        return nn.functional.cross_entropy(pred[m],y[m]),int(m.sum()),[r['rid'] for r in group]
    x=tensor(np.stack([r[0] for r in group]),device);ids=[r[2] for r in group]
    y=tensor(np.asarray([r[1] for r in group]),device,torch.long if task=='events' else torch.float32)
    pred=model(x)
    loss=models.focal_tsd(pred,y) if task=='events' else nn.functional.mse_loss(pred.squeeze(-1),y)
    return loss,len(group),ids

def epoch(model,records,name,task,cache,labels,a,device,optimizer=None,event_model=None,smoke=False,scheduler=None):
    if name=='bcgnet':bcg_features(None)
    model.train(optimizer is not None);total=count=targets=updates=0;ids=set();sizes=[];lrs=[]
    size=a.batch_size if optimizer is not None else a.val_batch_size
    samples=prepared_samples(records,name,task,cache,labels,a,device,event_model,optimizer is not None)
    for group in grouped(samples,size):
        with torch.set_grad_enabled(optimizer is not None):
            loss,n,rids=batch_loss(model,group,name,task,a,device,smoke)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradients')
                if a.grad_clip:nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip,error_if_nonfinite=True)
                lrs.append(optimizer.param_groups[0]['lr']);optimizer.step();updates+=1
                if scheduler is not None:scheduler.step()
            # Native Lightning aggregates logged losses with the night batch size.
            total+=float(loss.detach().cpu())*len(group);count+=len(group);targets+=n;ids.update(rids);sizes.append(len(group))
        print(json.dumps({'phase':'train' if optimizer else 'val','recordings':len(ids),'batch_samples':len(group),'updates':updates}),flush=True)
    if count==0 or (optimizer is not None and updates==0):raise ValueError('No usable targets/updates')
    return dict(loss=total/count,target_count=targets,recordings=len(ids),optimizer_updates=updates,batch_sizes=sizes,learning_rates=lrs)

def predict(name,task,model,r,cache,a,device,event_model=None,stage_model=None):
    out=dict(recording_id=r['rid'],participant_id=r['pid'],ahi_true=r['ahi'],stage_true=r['stage'],mask=r['mask'])
    with torch.no_grad():
        if name=='wav2sleep':out['stage_prob']=model(prepare_wav(r,device))[0].softmax(-1).cpu().numpy()
        elif name=='tsdnet':
            probability=event_probability(event_model if task=='ahi' else model,r,device,a.prediction_batch_size)
            out['event_score']=probability;out['epoch_indices']=np.arange(1200)
            if task=='ahi':out['ahi_pred']=float(model(tensor(ahi_input(probability,r['mask']),device)).cpu().squeeze())
        else:
            features,qc=bcg_features(r);out['r_peak_qc_json']=json.dumps(qc)
            if task=='stage':
                x={k:tensor(v[None],device) for k,v in features.items()};x['length']=torch.tensor([1200],device=device)
                p=model(**x)[0].softmax(-1).cpu().numpy();out['stage_prob5']=p;out['stage_prob']=np.stack([p[:,0],p[:,1]+p[:,2],p[:,3],p[:,4]],-1)
            else:
                pred=[]
                for start in range(0,36000,1800):
                    x={k:tensor(v[None,start*4:(start+1800)*4],device) for k,v in features.items()};x['length']=torch.tensor([1800],device=device)
                    pred.append(model(**x)[0].sigmoid().cpu().numpy())
                out['event_prob']=np.concatenate(pred)
                if stage_model is not None:
                    x={k:tensor(v[None],device) for k,v in features.items()};x['length']=torch.tensor([1200],device=device)
                    p=stage_model(**x)[0].softmax(-1).cpu().numpy();out['stage_prob']=np.stack([p[:,0],p[:,1]+p[:,2],p[:,3],p[:,4]],-1)
                    asleep=np.repeat(out['stage_prob'].argmax(-1)!=0,30)&np.repeat(r['mask'],30)
                    binary=(out['event_prob']>=.5)&asleep
                    changes=np.diff(np.pad(binary.astype(int),(1,1)));lengths=np.flatnonzero(changes==-1)-np.flatnonzero(changes==1)
                    sleep_hours=asleep.sum()/3600
                    out['ahi_pred']=float((lengths>=10).sum()/sleep_hours) if sleep_hours>0 else float('nan')
                    out['ahi_status']='VALID' if sleep_hours>0 else 'NO_PREDICTED_SLEEP'
        if 'ahi_pred' in out:out['ahi_pred_nonnegative']=max(out['ahi_pred'],0)
    return out

def main(name):
    a=resolve(name,parser(name).parse_args())
    if int(os.environ.get('WORLD_SIZE','1'))>1:
        if name!='wav2sleep':raise ValueError('DDP is currently supported only for wav2sleep')
        from .wav_ddp import main as distributed_main
        return distributed_main(a)
    if (name=='wav2sleep' and a.task!='stage') or (name=='tsdnet' and a.task=='stage') or (name=='bcgnet' and a.task=='ahi'):raise ValueError('Requested endpoint is not native to this baseline')
    if a.epochs<1 or a.batch_size<1 or a.threads<1 or not 1<=a.smoke_epochs<=1200 or not 1<=a.event_positive_seconds<=30:raise ValueError('Invalid training settings')
    if a.val_batch_size<1 or a.prediction_batch_size<1 or a.patience<0:raise ValueError('Invalid batch size/patience')
    torch.set_num_threads(a.threads);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    cache=Cache(a.cache,a.manifest);source=verify_source(name)
    if a.command=='check':
        report=dict(baseline=name,upstream=source,manifest_sha256=cache.meta['manifest_sha256'],hyperparameters=vars(a),event_labels='NOT_REQUIRED' if name=='wav2sleep' or a.task=='stage' else 'MISSING_OR_NOT_VALIDATED')
        for c in a.cohorts:
            r=next(cache.records([c],'val',limit=1));entry={'recording_id':r['rid'],'stage_valid_epochs':int(r['mask'].sum())}
            if name=='bcgnet':
                cache.stage5(r)
                try:entry['r_peak_qc']=bcg_features(r)[1]
                except ValueError as e:entry['r_peak_qc_error']=str(e)
            report[c]=entry
        if name!='wav2sleep' and a.task!='stage':
            try:
                labels=EventLabels(a.event_labels,cache);labels.require(cache,a.cohorts,['train','val']);report['event_labels']='READY_FOR_TRAIN_VAL'
            except ValueError as e:report['event_label_error']=str(e)
        print(json.dumps(report,indent=2));return
    if name=='bcgnet' and a.command!='model-check':
        bcg_features(None)  # Explicit user-requested input gate, before run creation.
    model=make_model(name,a.task).to(device)
    if a.command=='model-check':
        model.train()
        if name=='wav2sleep':pred=model({'ECG':torch.randn(2,32*1024,device=device),'THX':torch.randn(2,32*256,device=device)})
        elif name=='tsdnet':pred=model(torch.randn(2,2,1024,device=device) if a.task=='ahi' else torch.randn(2,1,256,device=device))
        else:
            samples=32*120 if a.task=='stage' else 60*4;tokens=32 if a.task=='stage' else 60
            pred=model(heartbeat=torch.randn(2,samples,device=device),breath=torch.randn(2,samples,device=device),length=torch.tensor([tokens,tokens],device=device))
        loss=pred.square().mean();loss.backward()
        if not torch.isfinite(pred).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Model gradient check failed')
        print(json.dumps(dict(baseline=name,task=a.task,shape=list(pred.shape),parameters=sum(p.numel() for p in model.parameters()),check='SYNTHETIC_FORWARD_BACKWARD_ONLY')));return
    if not a.run:raise ValueError('--run is required')
    labels=None;event_model=stage_model=None
    if a.task=='events' and a.command in ('train','smoke'):
        labels=EventLabels(a.event_labels,cache);labels.require(cache,a.cohorts,['train','val'])
    if name=='tsdnet' and a.task=='ahi':
        if not a.event_run:raise ValueError('--event-run is required for native AHINet input')
        event_model=load_model(name,'events',a.event_run,device,cache)
    if name=='bcgnet' and a.task=='events' and a.stage_run:stage_model=load_model(name,'stage',a.stage_run,device,cache)
    if a.command=='evaluate':
        model=load_model(name,a.task,a.run,device,cache);output=Path(a.run)/('evaluation_'+a.partition);output.mkdir(exist_ok=False)
        for cohort in a.cohorts:
            results=[]
            for r in cache.records([cohort],a.partition):
                pred=predict(name,a.task,model,r,cache,a,device,event_model,stage_model);np.savez_compressed(output/(r['rid']+'.npz'),**pred);results.append(pred)
                # Metrics accumulator keeps only lightweight arrays, never waveforms.
            write_json(output/(cohort+'_metrics.json'),metrics(results))
        return
    optimizer,scheduler,interval=optimizer_and_scheduler(model,name,a)
    run=make_run(a.run,a,cache,name)
    smoke=a.command=='smoke';history=[];best=float('inf');stale=0
    for epoch_no in range(1,(1 if smoke else a.epochs)+1):
        train=epoch(model,cache.records(a.cohorts,'train',seed=a.seed+epoch_no,limit=a.batch_size if smoke else 0),name,a.task,cache,labels,a,device,optimizer,event_model,smoke,scheduler if interval=='step' else None)
        val=epoch(model,cache.records(a.cohorts,'val',limit=a.val_batch_size if smoke else 0),name,a.task,cache,labels,a,device,None,event_model,smoke)
        if scheduler is not None and interval=='epoch':scheduler.step()
        history.append(dict(epoch=epoch_no,train=train,val=val));write_json(run/'history.json',history)
        if val['loss']<best:
            best=val['loss'];stale=0
            torch.save(dict(state_dict=model.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict() if scheduler else None,manifest_sha256=cache.meta['manifest_sha256'],baseline=name,task=a.task,epoch=epoch_no,arguments=vars(a)),run/'best.pt')
        else:stale+=1
        if a.patience and stale>=a.patience:break
    restored=load_model(name,a.task,run,device,cache)
    if smoke:
        output=run/'smoke_val_predictions';output.mkdir()
        for cohort in a.cohorts:
            r=next(cache.records([cohort],'val',limit=1));pred=predict(name,a.task,restored,r,cache,a,device,event_model,stage_model)
            np.savez_compressed(output/(r['rid']+'.npz'),**pred);write_json(output/(cohort+'_metrics.json'),metrics([pred]))
    write_json(run/'completion.json',dict(status='REAL_DATA_SMOKE' if smoke else 'TRAINED',best_val_loss=best,epochs=len(history),checkpoint_strict_reload=True,full_night_processing=not smoke or a.smoke_epochs==1200,test_evaluated=False))
    print('Completed:',run)
