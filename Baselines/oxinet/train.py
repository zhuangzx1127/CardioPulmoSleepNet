"""Pinned OxiNet architecture, train-only scaling, processed SpO2 input."""
import argparse
import csv
import importlib.util
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common.data import Cache,ROOT
from common.protocol import make_run,write_json,verify_source,metrics

def config(hours,seed):
    with (Path(__file__).parent/'upstream/saved_model/oxinet_config.csv').open() as f:raw=next(csv.DictReader(f))
    params={}
    for k,v in raw.items():
        if v in ('TRUE','FALSE'):params[k]=v=='TRUE'
        else:
            try:params[k]=float(v);params[k]=int(params[k]) if params[k].is_integer() else params[k]
            except ValueError:params[k]=v
    for k in ('first_conv_kernel_size','residual_convolution_kernel_size','dilated_block_kernel_size','dilation','kernel_size_lstm'):params[k]=(int(params[k]),)
    params.update(shape=(int(hours*2),1800),padding_size=int(hours*3600),seed=seed,use_metadata=False,cnn_loss_weight=1.,lstm_loss_weight=1.)
    return params

def build(params):
    spec=importlib.util.spec_from_file_location('native_oxinet',Path(__file__).parent/'upstream/architecture.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod.get_duplo(params)

def fit_scaler(cache,a,smoke):
    total=squares=count=0.
    for r in cache.records(a.cohorts,'train',limit=2 if smoke else 0):
        x=r['spo2'][:a.hours*3600];mask=np.repeat(r['mask'],30)[:len(x)];v=x[mask].astype(np.float64)
        total+=v.sum();squares+=np.square(v).sum();count+=len(v)
    if count==0:raise ValueError('No train SpO2 for scaler')
    return dict(mean=total/count,std=max(float(np.sqrt(max(squares/count-(total/count)**2,0))),1e-6),sample_count=int(count),fitted_partition='train',scope='SMOKE_SUBSET' if smoke else 'ALL_SELECTED_TRAIN')

def inputs(rows,scaler,hours):
    values=[]
    for r in rows:
        x=r['spo2'][:hours*3600];m=np.repeat(r['mask'],30)[:len(x)]
        values.append(np.where(m,(x-scaler['mean'])/scaler['std'],0).reshape(hours*2,1800))
    return [np.stack(values).astype(np.float32),np.zeros((len(rows),176),np.float32)]

def batches(iterator,size):
    rows=[]
    for r in iterator:
        # Batch 256 only needs oxygen/targets; release decoded ECG/respiration.
        rows.append({k:r[k] for k in ('spo2','mask','ahi','rid','pid')})
        if len(rows)==size:yield rows;rows=[]
    if rows:yield rows

def phase(model,cache,a,scaler,train=False,smoke=False,epoch=0):
    iterator=cache.records(a.cohorts,'train' if train else 'val',seed=a.seed+epoch if train else None,limit=2 if smoke else 0)
    losses=0.;count=updates=0;sizes=[]
    for rows in batches(iterator,a.batch_size):
        y=np.array([r['ahi'] for r in rows],np.float32)[:,None];x=inputs(rows,scaler,a.hours)
        loss=model.train_on_batch(x,[y,y,y],return_dict=True) if train else model.test_on_batch(x,[y,y,y],return_dict=True)
        if not all(np.isfinite(v) for v in loss.values()):raise ValueError('Nonfinite OxiNet loss')
        losses+=float(loss['loss'])*len(rows);count+=len(rows);updates+=int(train);sizes.append(len(rows))
        print(json.dumps(dict(phase='train' if train else 'val',recordings=count,updates=updates,loss=loss)),flush=True)
    if not count:raise ValueError('No records')
    return dict(loss=losses/count,recordings=count,optimizer_updates=updates,configured_batch_size=a.batch_size,batch_sizes=sizes,small_subset_debug=smoke)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['check','model-check','smoke','train','evaluate'])
    p.add_argument('--cache',default=str(ROOT/'TrainingCache'));p.add_argument('--manifest',default=str(ROOT/'Dataset/splits.json'))
    p.add_argument('--cohorts',nargs='+',choices=['mesa','mros','shhs'],default=['mesa','mros','shhs']);p.add_argument('--run')
    p.add_argument('--hours',type=int,choices=[6,7,10],default=6);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--seed',type=int,default=32)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--partition',choices=['val','test'],default='test');p.add_argument('--cpu',action='store_true')
    a=p.parse_args();a.hyperparameter_source='upstream/saved_model/oxinet_config.csv and architecture.py; epoch cap from paper';a.unresolved_hyperparameters=['Full author fit loop / auxiliary-loss schedule not published'];a.hyperparameter_overrides={k:dict(reference=v,actual=getattr(a,k)) for k,v in dict(hours=6,batch_size=256,seed=32,epochs=100).items() if getattr(a,k)!=v};cache=Cache(a.cache,a.manifest);source=verify_source('oxinet')
    if a.command=='check':
        report=dict(baseline='oxinet',upstream=source,channel='SAME_PROCESSED_SPO2_AS_MANUSCRIPT',event_labels='NOT_REQUIRED',metadata='NOT_REQUIRED',hours=a.hours,batch_size=a.batch_size,learning_rate=config(a.hours,a.seed)['learning_rate'],optimizer='Adam',epochs=a.epochs,epoch_cap_source='paper; training loop not published',schedule_source='not specified in public training config')
        for c in a.cohorts:
            r=next(cache.records([c],'val',limit=1));report[c]={'recording_id':r['rid'],'ahi':r['ahi']}
        print(json.dumps(report,indent=2));return
    if a.cpu:os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tensorflow as tf
    if a.cpu:tf.config.set_visible_devices([],'GPU')
    else:tf.config.set_visible_devices(tf.config.list_physical_devices('GPU')[:1],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(a.seed);params=config(a.hours,a.seed)
    tf.config.experimental.enable_op_determinism()
    if a.command=='evaluate':
        run=Path(a.run);protocol=json.loads((run/'protocol.json').read_text())
        if protocol['manifest_sha256']!=cache.meta['manifest_sha256']:raise ValueError('Checkpoint manifest mismatch')
        a.hours=protocol['arguments']['hours'];params=json.loads((run/'model_config.json').read_text())
        for k in ('first_conv_kernel_size','residual_convolution_kernel_size','dilated_block_kernel_size','dilation','kernel_size_lstm','shape'):params[k]=tuple(params[k])
        model=build(params);model.load_weights(str(run/'best.weights.h5'));scaler=json.loads((run/'scaler.json').read_text())
        output=run/('evaluation_'+a.partition);output.mkdir(exist_ok=False)
        for c in a.cohorts:
            results=[]
            for rows in batches(cache.records([c],a.partition),a.batch_size):
                pred=model(inputs(rows,scaler,a.hours),training=False)[2].numpy().ravel()
                for r,value in zip(rows,pred):
                    out=dict(recording_id=r['rid'],participant_id=r['pid'],ahi_true=r['ahi'],ahi_pred=float(value),ahi_pred_nonnegative=max(float(value),0))
                    np.savez_compressed(output/(r['rid']+'.npz'),**out);results.append(out)
            write_json(output/(c+'_metrics.json'),metrics(results))
        return
    model=build(params)
    if a.command=='model-check':
        x=[np.random.randn(2,a.hours*2,1800).astype(np.float32),np.zeros((2,176),np.float32)];y=np.array([[5.],[15.]],np.float32)
        loss=model.train_on_batch(x,[y,y,y],return_dict=True)
        if not all(np.isfinite(v) for v in loss.values()):raise ValueError('Model check failed')
        print(json.dumps(dict(parameters=model.count_params(),loss=loss,check='SYNTHETIC_FORWARD_BACKWARD_ONLY')));return
    if not a.run or a.epochs<1 or a.batch_size<1:raise ValueError('Valid --run / epochs / batch size required')
    run=make_run(a.run,a,cache,'oxinet');write_json(run/'model_config.json',params)
    smoke=a.command=='smoke';scaler=fit_scaler(cache,a,smoke);write_json(run/'scaler.json',scaler)
    history=[];best=float('inf')
    for e in range(1,(1 if smoke else a.epochs)+1):
        train=phase(model,cache,a,scaler,True,smoke,e);val=phase(model,cache,a,scaler,False,smoke,e)
        history.append(dict(epoch=e,train=train,val=val));write_json(run/'history.json',history)
        if val['loss']<best:best=val['loss'];model.save_weights(str(run/'best.weights.h5'))
    model.load_weights(str(run/'best.weights.h5'))
    if smoke:
        output=run/'smoke_val_predictions';output.mkdir()
        for cohort in a.cohorts:
            probe=next(cache.records([cohort],'val',limit=1));pred=model(inputs([probe],scaler,a.hours),training=False)[2].numpy().ravel()
            result=dict(recording_id=probe['rid'],participant_id=probe['pid'],ahi_true=probe['ahi'],ahi_pred=float(pred[0]),ahi_pred_nonnegative=max(float(pred[0]),0))
            np.savez_compressed(output/(probe['rid']+'.npz'),**result);write_json(output/(cohort+'_metrics.json'),metrics([result]))
    write_json(run/'completion.json',dict(status='REAL_DATA_SMOKE' if smoke else 'TRAINED',epochs=len(history),best_val_loss=best,checkpoint_strict_reload=True,test_evaluated=False))
    print('Completed:',run)

if __name__=='__main__':main()
