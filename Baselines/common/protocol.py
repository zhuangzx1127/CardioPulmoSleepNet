import json
import platform
import shutil
import subprocess
from pathlib import Path
import numpy as np
from .data import sha

BASE=Path(__file__).resolve().parents[1]

def verify_source(name):
    d=json.loads((BASE/name/'upstream_inventory.json').read_text())
    for f in d['files']:
        if sha(BASE/name/'upstream'/f['path'])!=f['sha256']:raise ValueError('Upstream source modified: '+f['path'])
    return {k:d[k] for k in ('repo','commit')}

def copy_source_tree(src, dst):
    """Copy snapshot bytes without chmod/utime (WSL mounted-drive safe)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=False)
    for entry in src.iterdir():
        if entry.name == '__pycache__' or entry.name == '.DS_Store' or entry.name.startswith('._'):
            continue
        target = dst / entry.name
        if entry.is_dir():
            copy_source_tree(entry, target)
        else:
            shutil.copyfile(entry, target)

def make_run(path,args,cache,name):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    source=path/'source';source.mkdir()
    copy_source_tree(BASE/'common',source/'common')
    copy_source_tree(BASE/name,source/name)
    for filename in ('README.md','HYPERPARAMETERS.md','LABEL_REQUIREMENTS.md','requirements-torch.txt','requirements-tf-linux.txt','event_review.template.json'):
        if (BASE/filename).is_file():shutil.copyfile(BASE/filename,source/filename)
    shutil.copyfile(cache.manifest,path/'splits.json');shutil.copyfile(cache.root/('index.json' if 'native_index_sha256' in cache.meta else 'cache.json'),path/('native_index.json' if 'native_index_sha256' in cache.meta else 'cache.json'))
    protocol=dict(baseline=name,arguments=vars(args),manifest_sha256=cache.meta['manifest_sha256'],upstream=verify_source(name),python=platform.python_version(),platform=platform.platform(),initialization='FROM_SCRATCH',test_used_for_fitting=False)
    if 'native_index_sha256' in cache.meta:
        protocol.update(native_index_sha256=cache.meta['native_index_sha256'],input_format=cache.meta['format'],time_axis=cache.meta['time_axis'],excluded_records=cache.excluded)
    protocol['source_sha256']={str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write_json(path/'protocol.json',protocol)
    freeze=subprocess.run([__import__('sys').executable,'-m','pip','freeze'],capture_output=True,text=True)
    (path/'environment.txt').write_text(freeze.stdout)
    return path

def write_json(path,data):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2,allow_nan=False));temp.replace(p)

def metrics(predictions):
    cm=np.zeros((4,4),dtype=np.int64); errors=[]
    for r in predictions:
        if 'stage_prob' in r:
            m=r['mask'];y=r['stage_true'][m];p=r['stage_prob'][m].argmax(-1)
            np.add.at(cm,(y,p),1)
        if 'ahi_pred' in r and np.isfinite(r['ahi_pred']):errors.append(float(r['ahi_pred'])-float(r['ahi_true']))
    out={'n_recordings':len(predictions)}
    if cm.sum():
        denom=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),denom,out=np.zeros(4,dtype=float),where=denom>0)
        acc=float(np.trace(cm)/cm.sum());chance=float(np.dot(cm.sum(0),cm.sum(1))/cm.sum()**2)
        out.update(stage_confusion=cm.tolist(),stage_accuracy=acc,stage_macro_f1=float(f1.mean()),stage_kappa=(acc-chance)/(1-chance) if chance<1 else None)
    if errors:out.update(ahi_mae=float(np.mean(np.abs(errors))),ahi_rmse=float(np.sqrt(np.mean(np.square(errors)))),ahi_n=len(errors))
    return out
