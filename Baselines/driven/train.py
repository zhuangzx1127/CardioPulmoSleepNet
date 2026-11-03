"""DRIVEN adaptation: native per-channel CNNs, probability fusion, AHI calibration.

Uses native raw-MAT ABD and SpO2 with adjusted XML labels, no airflow. Original
architecture/function body is extracted from the pinned source without importing
its hard-coded training script. Events and sleep/wake remain binary endpoints.
"""
import argparse
import ast
import gc
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import Cache,ROOT,zscore,signal_mask
from common.labels import EventLabels
from common.prediction_export import export_predictions
from common.native_inputs import NativeCache,driven_signal,driven_targets
from common.protocol import verify_source,make_run,write_json,metrics

CHANNELS=('abd','spo2');TARGETS=('events','sleep')

def build(tf,lr):
    path=Path(__file__).parent/'upstream/python_code/training/train_nn.py'
    node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='get_model')
    scope={'tf':tf,'Adam':tf.keras.optimizers.Adam,'pr_metric':tf.keras.metrics.AUC(curve='PR',num_thresholds=100)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope['get_model'](64,30,lr)

def window_info(r,cache,labels=None,positive_seconds=1):
    if 'abd_64hz' in r:
        t=driven_targets(r);event=np.array([r['event'][s:s+30].sum()>=positive_seconds for s in t['starts']],dtype=np.int64)&t['sleep']
        return t['starts'],t['valid'],{'events':event.astype(np.int64),'sleep':t['sleep'].astype(np.int64)}
    starts=np.arange(0,36000-30+1,15);valid=np.repeat(r['mask'],30)
    event=None
    if labels is not None:event,coverage=labels.read(r,cache);valid&=coverage
    good=np.array([valid[s:s+30].all() for s in starts]);sleep=np.repeat(r['stage']!=0,30)
    sleep_y=np.array([sleep[s:s+30].any() for s in starts],dtype=np.int64)
    event_y=np.array([event[s:s+30].sum()>=positive_seconds for s in starts],dtype=np.int64)&sleep_y if event is not None else None
    return starts,good,{'events':event_y,'sleep':sleep_y}

def channel_signal(r,name):
    if 'abd_64hz' in r:return driven_signal(r,name)
    if name=='spo2':return np.repeat(r['spo2']/100,64).astype(np.float32)
    return resample_poly(zscore(r['respi'],signal_mask(r,1024)),15,8).astype(np.float32)

def window_batch(x,starts):return np.stack([x[s*64:(s+30)*64] for s in starts])[:,:,None,None]

def cnn_phase(model,cache,labels,a,channel,target,train=False,epoch=0,smoke=False):
    count=updates=steps=empty=0;total=0;sizes=[]
    refs=cache.index(a.cohorts,'train' if train else 'val')
    rng=np.random.default_rng(a.seed+epoch+(0 if train else 100000))
    target_steps=1 if smoke else (a.train_steps if train else a.val_steps)
    while steps<target_steps:
        r=cache.read(refs[int(rng.integers(len(refs)))])
        starts,valid,targets=window_info(r,cache,labels,a.event_positive_seconds)
        awake=np.flatnonzero(valid&(targets['sleep']==0));sleep=np.flatnonzero(valid&(targets['sleep']==1)&(targets['events']==0));event=np.flatnonzero(valid&(targets['events']==1))
        n=min(len(awake),len(sleep),len(event),64)
        if n<2:
            empty+=1
            if empty>max(100,len(refs)*2):raise ValueError('No sufficiently represented classes for native per-night balanced sampling')
            continue
        empty=0
        if target=='events':idx=np.r_[rng.choice(sleep,n,replace=False),rng.choice(event,n,replace=False)]
        else:idx=np.r_[rng.choice(awake,n,replace=False),rng.choice(sleep,n//2,replace=False),rng.choice(event,n//2,replace=False)]
        rng.shuffle(idx)
        x=channel_signal(r,channel)
        for k in range(0,len(idx),a.batch_size):
            selected=idx[k:k+a.batch_size];batch=window_batch(x,starts[selected]);y=np.eye(2,dtype=np.float32)[targets[target][selected]]
            result=model.train_on_batch(batch,y,return_dict=True) if train else model.test_on_batch(batch,y,return_dict=True)
            if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite DRIVEN loss')
            total+=float(result['loss'])*len(selected);count+=len(selected);updates+=int(train);steps+=1;sizes.append(len(selected))
            if steps>=target_steps:break
        print(json.dumps(dict(channel=channel,target=target,phase='train' if train else 'val',rid=r['rid'],windows=count)),flush=True)
    if not count:raise ValueError('No reviewed windows')
    return dict(loss=total/count,windows=count,optimizer_updates=updates,steps=steps,batch_sizes=sizes,learning_rate=float(model.optimizer.lr.numpy()))

def cnn_features(models,r,cache,a,labels=None):
    starts,valid,targets=window_info(r,cache,labels,a.event_positive_seconds);idx=np.flatnonzero(valid)
    def one(key):
        target,channel=key;signal=channel_signal(r,channel);pred=[]
        for start in range(0,len(idx),a.batch_size):
            batch=window_batch(signal,starts[idx[start:start+a.batch_size]])
            model=models[key]
            if hasattr(model,'_fusion_infer'):
                import tensorflow as tf
                with tf.device(model._fusion_device):
                    result=model._fusion_infer(tf.convert_to_tensor(batch,dtype=tf.float32))
            else:
                result=model(batch,training=False)
            pred.append(result.numpy())
        return np.concatenate(pred) if pred else np.empty((0,2),np.float32)
    keys=[(t,c) for t in TARGETS for c in CHANNELS]
    if getattr(a,'gpus',1)>1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:arrays=list(pool.map(one,keys))
    else:arrays=[one(k) for k in keys]
    return np.concatenate(arrays,axis=1),{k:v[idx] if v is not None else None for k,v in targets.items()},starts[idx]


def load_cnns(tf,run,a):
    models={}
    for i,(target,channel) in enumerate((t,c) for t in TARGETS for c in CHANNELS):
        device='/GPU:'+str(i%a.gpus) if getattr(a,'gpus',1)>1 else None
        if device:
            with tf.device(device):m=build(tf,a.lr)
        else:m=build(tf,a.lr)
        m.load_weights(str(run/(target+'_'+channel+'.weights.h5')))
        if device:
            # Device scopes are thread-local: bind actual forward ops, not just variables.
            def make_infer(model,placement):
                @tf.function(input_signature=[tf.TensorSpec([None,1920,1,1],tf.float32)])
                def infer(batch):
                    with tf.device(placement):
                        return tf.identity(model(batch,training=False))
                return infer
            infer=make_infer(m,device)
            with tf.device(device):
                actual=infer(tf.zeros((1,1920,1,1),tf.float32)).device
            if ('GPU:'+str(i%a.gpus)) not in actual:
                raise RuntimeError('Fusion inference device mismatch: '+actual+' expected '+device)
            m._fusion_device=device
            m._fusion_infer=infer
            print(json.dumps(dict(phase='fusion_device',target=target,channel=channel,device=actual)),flush=True)
        models[(target,channel)]=m
    return models

def feature_table(models,cache,labels,a,run,part):
    refs=cache.index(a.cohorts,part);capacity=len(refs)*2399
    x=np.lib.format.open_memmap(run/(part+'_features.npy'),mode='w+',dtype=np.float32,shape=(capacity,8))
    y=np.lib.format.open_memmap(run/(part+'_targets.npy'),mode='w+',dtype=np.int8,shape=(capacity,2))
    n=0;night_index=[]
    for r in cache.records(a.cohorts,part):
        f,t,starts=cnn_features(models,r,cache,a,labels);size=len(f);x[n:n+size]=f;y[n:n+size]=np.stack([t[k] for k in TARGETS],axis=1)
        night_index.append(dict(recording_id=r['rid'],participant_id=r['pid'],start=n,end=n+size,ahi=r['ahi'],window_start_seconds=starts.tolist()));n+=size
        print(json.dumps(dict(phase='fusion_features',partition=part,rid=r['rid'],windows=n)),flush=True)
    x.flush();y.flush();write_json(run/(part+'_feature_index.json'),night_index)
    return x[:n],y[:n],night_index

def fit_fusion(models,cache,labels,a,run):
    import lightgbm as lgb
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    from types import SimpleNamespace
    from sklearn.linear_model import LinearRegression
    x,y,_=feature_table(models,cache,labels,a,run,'train');v,w,nights=feature_table(models,cache,labels,a,run,'val')
    for col,target in enumerate(TARGETS):
        if len(np.unique(y[:,col]))!=2 or len(np.unique(w[:,col]))!=2:raise ValueError('Both classes required for fusion train and val: '+target)
        # Execute the native objective; adapt only removed LightGBM API kwargs.
        def compatible_train(params,data,valid_sets,early_stopping_rounds,verbose_eval):
            return lgb.train(params,data,valid_sets=valid_sets,callbacks=[lgb.early_stopping(early_stopping_rounds,verbose=False)])
        source=Path(__file__).parent/'upstream/python_code/training/train_lgbm.py'
        node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='objective')
        features=x[:,col*4:(col+1)*4];val_features=v[:,col*4:(col+1)*4]
        scope=dict(x_train=features,y_train=y[:,col],x_val=val_features,y_val=w[:,col],train_test_split=train_test_split,roc_auc_score=roc_auc_score,gc=gc,lgb=SimpleNamespace(train=compatible_train,Dataset=lgb.Dataset))
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=a.seed))
        study.optimize(scope['objective'],n_trials=100,gc_after_trial=True)
        model=lgb.LGBMClassifier(metric=None,**study.best_trial.params)
        model.fit(features,y[:,col],eval_set=[(val_features,w[:,col])],callbacks=[lgb.early_stopping(35,verbose=False)])
        model.booster_.save_model(str(run/(target+'_fusion.txt')))
        write_json(run/(target+'_optuna.json'),dict(n_trials=100,best_params=study.best_trial.params,best_value=study.best_value,sampler_seed=a.seed,source=str(source)))
    event=lgb.Booster(model_file=str(run/'events_fusion.txt')).predict(v[:,:4]);sleep=lgb.Booster(model_file=str(run/'sleep_fusion.txt')).predict(v[:,4:])
    # Fixed 0.5 thresholds; class-balanced WLS on VAL, as explicitly disclosed.
    ratios=[];targets=[];ids=[]
    for row in nights:
        begin,end=row['start'],row['end'];asleep=sleep[begin:end]>=.5
        if not asleep.any():raise ValueError('No predicted sleep during calibration: '+row['recording_id'])
        ratios.append(float(((event[begin:end]>=.5)&asleep).sum()/asleep.sum()));targets.append(row['ahi']);ids.append(row['recording_id'])
    classes=np.digitize(targets,[5,15,30]);counts=np.bincount(classes,minlength=4);weights=np.array([1/counts[c] for c in classes])
    reg=LinearRegression().fit(np.asarray(ratios)[:,None],targets,sample_weight=weights)
    write_json(run/'ahi_calibration.json',dict(coefficient=float(reg.coef_[0]),intercept=float(reg.intercept_),event_threshold=.5,sleep_threshold=.5,fit_partition='val',weighting='inverse_AHI_class_frequency',recording_ids=ids,reference_ahi=targets,validation_metrics_are_calibration_in_sample=True))

def export_test(tf,cache,a,run,partition):
    if getattr(a,'gpus',1)>1:
        from common.parallel_export import parallel_export
        return parallel_export('driven',cache,a,run,partition)
    import copy
    import lightgbm as lgb
    args=copy.copy(a);args.batch_size=a.prediction_batch_size
    if args.batch_size<1:raise ValueError('Invalid prediction batch size')
    proto=json.loads((run/'protocol.json').read_text())
    if proto['manifest_sha256']!=cache.meta['manifest_sha256']:raise ValueError('Run/cache mismatch')
    if not a.legacy_cache and proto.get('native_index_sha256')!=cache.meta['native_index_sha256']:raise ValueError('Native dataset changed since training')
    args.event_positive_seconds=proto['arguments']['event_positive_seconds']
    tf.keras.backend.clear_session();gc.collect()
    models=load_cnns(tf,run,args)
    fusion={k:lgb.Booster(model_file=str(run/(k+'_fusion.txt'))) for k in TARGETS}
    cal=json.loads((run/'ahi_calibration.json').read_text())
    def predict(r):
        f,_,starts=cnn_features(models,r,cache,args)
        ep=fusion['events'].predict(f[:,:4]);sp=fusion['sleep'].predict(f[:,4:]);asleep=sp>=cal['sleep_threshold']
        ratio=float(((ep>=cal['event_threshold'])&asleep).sum()/asleep.sum()) if asleep.any() else None
        value=ratio*cal['coefficient']+cal['intercept'] if ratio is not None else float('nan')
        return dict(recording_id=r['rid'],participant_id=r['pid'],ahi_true=r['ahi'],ahi_pred=value,ahi_pred_nonnegative=max(value,0),event_prob=ep,sleep_prob=sp,window_start_seconds=starts,ahi_status='VALID' if ratio is not None else 'NO_PREDICTED_SLEEP',sleep_source='predicted_sleep_wake',ahi_method='validation_calibrated_positive_sleep_window_ratio')
    return export_predictions(cache,a.cohorts,partition,Path(getattr(a,'export_output',run)),predict,dict(baseline='driven',sleep_source='predicted_sleep_wake',prediction_batch_size=args.batch_size))

def train_cnn(tf,cache,labels,a,run,target,channel,smoke=False):
    history=[]
    model=build(tf,a.lr);best=float('inf')
    source=Path(__file__).parent/'upstream/python_code/training/train_nn.py'
    node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='ReduceLRBacktrack')
    import logging
    scope=dict(ReduceLROnPlateau=tf.keras.callbacks.ReduceLROnPlateau,logging=logging)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    weight_path=str(run/(target+'_'+channel+'.weights.h5'))
    reduce_lr=scope['ReduceLRBacktrack'](best_path=weight_path,monitor='val_loss',factor=.5,patience=10,min_lr=1e-6)
    early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20,restore_best_weights=True)
    callbacks=[reduce_lr,early]
    for cb in callbacks:cb.set_model(model);cb.on_train_begin()
    for e in range(1,(1 if smoke else a.epochs)+1):
        t=cnn_phase(model,cache,labels,a,channel,target,True,e,smoke);v=cnn_phase(model,cache,labels,a,channel,target,False,e,smoke)
        history.append(dict(target=target,channel=channel,epoch=e,train=t,val=v));write_json(run/('history_'+target+'_'+channel+'.json'),history)
        for cb in callbacks:cb.on_epoch_end(e-1,{'val_loss':v['loss']})
        if v['loss']<best:best=v['loss'];model.save_weights(weight_path)
        if model.stop_training:break
    for cb in callbacks:cb.on_train_end()
    model.load_weights(str(run/(target+'_'+channel+'.weights.h5')));del model;tf.keras.backend.clear_session();gc.collect()
    return history

def configure_tf(a):
    # TF 2.8/2.9 GPU depthwise filter gradients lack a deterministic kernel.
    strict=getattr(a,'deterministic_ops',False)
    if not strict:
        os.environ['TF_DETERMINISTIC_OPS']='0'
        os.environ['TF_CUDNN_DETERMINISTIC']='0'
    a.op_determinism='strict' if strict else 'disabled_for_gpu_kernel_compatibility'
    if a.cpu:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    devices=tf.config.list_physical_devices('GPU')
    if a.cpu:tf.config.set_visible_devices([],'GPU')
    else:
        if a.gpus>len(devices) and a.gpus>1:raise ValueError('Requested GPU count unavailable')
        tf.config.set_visible_devices(devices[:a.gpus],'GPU')
        for device in devices[:a.gpus]:tf.config.experimental.set_memory_growth(device,True)
    tf.config.threading.set_intra_op_parallelism_threads(a.threads);tf.config.threading.set_inter_op_parallelism_threads(2);tf.keras.utils.set_random_seed(a.seed)
    if strict:tf.config.experimental.enable_op_determinism()
    return tf

def main():
    global CHANNELS
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','model-check','smoke','train','evaluate','export','resume-fusion'])
    p.add_argument('--native-inputs',default=str(ROOT/'BaselineInputs/native_10h'))
    p.add_argument('--legacy-cache',action='store_true',help='Explicitly use old RIPsum cache and separate EventLabels')
    p.add_argument('--cache',default=str(ROOT/'TrainingCache'));p.add_argument('--manifest',default=str(ROOT/'Dataset/splits.json'))
    p.add_argument('--cohorts',nargs='+',choices=['mesa','mros','shhs'],default=['mesa','mros','shhs']);p.add_argument('--event-labels',default=str(ROOT/'EventLabels'))
    p.add_argument('--run');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,help='Required for training: upstream receives a CLI value and publishes no fixed default')
    p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--threads',type=int,default=4);p.add_argument('--cpu',action='store_true');p.add_argument('--partition',choices=['val','test'],default='test')
    p.add_argument('--train-steps',type=int,default=1024);p.add_argument('--val-steps',type=int,default=256)
    p.add_argument('--event-positive-seconds',type=int,default=1)
    p.add_argument('--gpus',type=int,choices=[1,4],default=1,help='Four independent CNN jobs on four GPUs; batch is per CNN')
    p.add_argument('--deterministic-ops',action='store_true',help='Require deterministic kernels; unsupported depthwise GPU gradients may fail on older TensorFlow')
    p.add_argument('--skip-test-export',action='store_true')
    p.add_argument('--prediction-batch-size',type=int,default=128)
    a=p.parse_args();cache=Cache(a.cache,a.manifest) if a.legacy_cache else NativeCache(a.native_inputs,'driven');source=verify_source('driven')
    if a.gpus>1 and (a.cpu or a.legacy_cache):raise ValueError('Four-GPU mode requires native inputs and CUDA')
    a.parallelism='four_independent_cnns' if a.gpus==4 else 'single_gpu'
    a.batch_semantics='per_CNN_optimizer_step; not multiplied by GPU count'
    CHANNELS=('respi','spo2') if a.legacy_cache else ('abd','spo2')
    labels=None
    if a.command=='check':
        report=dict(baseline='driven',upstream=source,channels=['RIPsum_IN_PLACE_OF_ABD','SAME_PROCESSED_SPO2'] if a.legacy_cache else ['RAW_MAT_ABD_64HZ','RAW_MAT_SPO2_1HZ'],event_classes=2,sleep_classes=2,four_stage_head=False,batch_size=a.batch_size,batch_size_source='UNSPECIFIED_AUTHOR_CLI_ARGUMENT',learning_rate=a.lr,optimizer='Adam',epochs=a.epochs,train_steps=a.train_steps,val_steps=a.val_steps,early_stopping_patience=20,lr_plateau_patience=10,lr_factor=.5,min_lr=1e-6,lightgbm_optuna_trials=100)
        if not a.legacy_cache:
            report.update(event_labels='AUTHOR_CONFIRMED_ADJUSTED_XML',native_index_sha256=cache.meta['native_index_sha256'],counts={c:{p:len(cache.index([c],p)) for p in ['train','val','test']} for c in a.cohorts})
            print(json.dumps(report,indent=2));return
        try:labels=EventLabels(a.event_labels,cache);labels.require(cache,a.cohorts,['train','val']);report['event_labels']='READY_FOR_TRAIN_VAL'
        except ValueError as e:report['event_labels']=str(e)
        print(json.dumps(report,indent=2));return
    if a.command in ('train','smoke'):
        if a.batch_size is None:raise ValueError('UNRESOLVED_AUTHOR_BATCH_SIZE: DRIVEN upstream takes an explicit CLI batch_size; no published fixed value was found. Supply the verified value with --batch-size.')
        if a.legacy_cache:
            labels=EventLabels(a.event_labels,cache);labels.require(cache,a.cohorts,['train','val'])
    tf=configure_tf(a)
    if a.command=='model-check':
        model=build(tf,a.lr);x=np.random.randn(2,1920,1,1).astype(np.float32);y=np.eye(2,dtype=np.float32)
        result=model.train_on_batch(x,y,return_dict=True)
        if not all(np.isfinite(v) for v in result.values()):raise ValueError('Nonfinite model check')
        print(json.dumps(dict(parameters=model.count_params(),shape=list(model(x,training=False).shape),loss=result,check='SYNTHETIC_FORWARD_BACKWARD_ONLY')));return
    if a.command=='resume-fusion':
        if not a.run:raise ValueError('--run must name the completed CNN run')
        if a.legacy_cache:raise ValueError('resume-fusion currently requires native inputs')
        run=Path(a.run)
        proto=json.loads((run/'protocol.json').read_text())
        if proto.get('baseline')!='driven' or proto.get('native_index_sha256')!=cache.meta['native_index_sha256'] or proto.get('manifest_sha256')!=cache.meta['manifest_sha256']:
            raise ValueError('Run/native input fingerprint mismatch')
        saved=proto['arguments']
        if saved.get('command')!='train':raise ValueError('Expected a full CNN training run')
        for key in ('cohorts','seed','lr','event_positive_seconds'):
            setattr(a,key,saved[key])
        for target in TARGETS:
            for channel in CHANNELS:
                weight=run/(target+'_'+channel+'.weights.h5')
                if not weight.is_file() or not weight.stat().st_size:raise ValueError('Missing CNN weight: '+str(weight))
        # Fail before feature extraction if fusion dependencies are unavailable.
        import lightgbm, optuna, sklearn
        if a.prediction_batch_size<1:raise ValueError('prediction batch must be positive')
        a.batch_size=a.prediction_batch_size  # Forward-only inference; original training protocol is untouched.
        write_json(run/'fusion_resume_protocol.json',dict(arguments=vars(a),cnn_training_skipped=True,lightgbm=lightgbm.__version__,optuna=optuna.__version__,sklearn=sklearn.__version__))
        print('RESUME_FUSION: loading four existing CNN weights; CNN training skipped',flush=True)
        fit_fusion(load_cnns(tf,run,a),cache,None,a,run)
        write_json(run/'completion.json',dict(status='TRAINED',fusion_trained=True,ahi_calibrated=True,test_evaluated=False))
        if not a.skip_test_export:
            export_test(tf,cache,a,run,'test')
            write_json(run/'completion.json',dict(status='TRAINED',fusion_trained=True,ahi_calibrated=True,test_evaluated=True,test_predictions='test_predictions.json'))
        print('Completed:',run);return
    if a.command in ('evaluate','export') and a.batch_size is None:a.batch_size=128  # Inference throughput only.
    if not a.run or a.epochs<1 or a.batch_size<1 or min(a.train_steps,a.val_steps)<1 or not 1<=a.event_positive_seconds<=30:raise ValueError('Invalid settings / --run')
    if a.command in ('evaluate','export'):
        export_test(tf,cache,a,Path(a.run),a.partition);return
    run=make_run(a.run,a,cache,'driven');smoke=a.command=='smoke';history=[]
    if a.gpus==4:
        from driven.parallel import train_four_cnns
        history=train_four_cnns(a,run,smoke)
    else:
        for target in TARGETS:
            for channel in CHANNELS:history.extend(train_cnn(tf,cache,labels,a,run,target,channel,smoke))
    write_json(run/'history.json',history)
    if not smoke:fit_fusion(load_cnns(tf,run,a),cache,labels,a,run)
    write_json(run/'completion.json',dict(status='CNN_SMOKE_ONLY' if smoke else 'TRAINED',fusion_trained=not smoke,ahi_calibrated=not smoke,test_evaluated=False))
    if not smoke and not a.skip_test_export:
        export_test(tf,cache,a,run,'test')
        write_json(run/'completion.json',dict(status='TRAINED',fusion_trained=True,ahi_calibrated=True,test_evaluated=True,test_predictions='test_predictions.json'))
    print('Completed:',run)

if __name__=='__main__':main()
