"""Export a trained OxiNet snapshot without fitting or changing its weights."""
import argparse, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--root',required=True);p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
run=Path(a.run);root=Path(a.root)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False));tmp.replace(p)
protocol=json.loads((run/'protocol.json').read_text())
for rel,digest in protocol['source_sha256'].items():
 if rel.endswith('.py') and not Path(rel).name.startswith('._'):
  assert sha(run/'source'/rel)==digest,rel
sys.path.insert(0,str(run/'source'))
spec=importlib.util.spec_from_file_location('snapshot_oxinet',run/'source/oxinet/train.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
import tensorflow as tf
tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(4);tf.config.threading.set_inter_op_parallelism_threads(2)
tf.keras.utils.set_random_seed(protocol['arguments']['seed'])
cache=mod.Cache(root/'TrainingCache',root/'Dataset/splits.json')
assert protocol['manifest_sha256']==cache.meta['manifest_sha256']
assert sha(run/'cache.json')==sha(root/'TrainingCache/cache.json')
params=json.loads((run/'model_config.json').read_text());hours=protocol['arguments']['hours']
for k in ('first_conv_kernel_size','residual_convolution_kernel_size','dilated_block_kernel_size','dilation','kernel_size_lstm','shape'):params[k]=tuple(params[k])
model=mod.build(params);weight_hash=sha(run/'best.weights.h5');model.load_weights(str(run/'best.weights.h5'))
scaler=json.loads((run/'scaler.json').read_text());assert scaler['fitted_partition']=='train' and scaler['scope']=='ALL_SELECTED_TRAIN'
expected={rid for rid,v in cache.rows.items() if v['partition']=='test' and v['cohort'] in protocol['arguments']['cohorts']}
output=run/'test_predictions.json';assert not output.exists(),output
results={};start=time.time()
for cohort in protocol['arguments']['cohorts']:
 for rows in mod.batches(cache.records([cohort],'test'),a.batch_size):
  pred=model(mod.inputs(rows,scaler,hours),training=False)[2].numpy().ravel();assert len(pred)==len(rows) and np.isfinite(pred).all()
  for r,value in zip(rows,pred):
   assert r['rid'] not in results
   results[r['rid']]=dict(dataset=cohort,recording_id=r['rid'],participant_id=r['pid'],ahi_true=r['ahi'],ahi_pred=float(value),ahi_pred_nonnegative=max(0.,float(value)))
  print(json.dumps(dict(cohort=cohort,exported=len(results),total=len(expected),elapsed_seconds=round(time.time()-start,1))),flush=True)
assert set(results)==expected and sha(run/'best.weights.h5')==weight_hash
save(output,results)
history=json.loads((run/'history.json').read_text());best=min(history,key=lambda x:x['val']['loss'])
save(run/'test_export_protocol.json',dict(status='PASS',partition='test',records=len(results),manifest_sha256=cache.meta['manifest_sha256'],checkpoint_sha256=weight_hash,predictions_sha256=sha(output),hours=hours,best_epoch=best['epoch'],completed_epochs=len(history),tensorflow=tf.__version__,device='CPU',inference_batch_size=a.batch_size,source='verified training snapshot',scaler='saved train-only scaler',ahi_policy='raw final_output; nonnegative sensitivity stored separately',test_used_for_fitting=False))
print('DONE',output,flush=True)
