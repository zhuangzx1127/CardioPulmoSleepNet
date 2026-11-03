"""One inference process per GPU, disjoint nights, atomic final JSON merge."""
import copy
import json
import multiprocessing as mp
import os
import time
from pathlib import Path
import numpy as np
from .native_inputs import NativeCache
from .prediction_export import export_predictions


def _worker(backend,args,run,partition,gpu,rank,world):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    a=copy.copy(args);a.gpus=1
    cache=NativeCache(a.native_inputs,backend)
    ids=[rid for cohort in a.cohorts for rid in cache.index([cohort],partition)]
    selected=set(ids[rank::world]);cache.rows={rid:v for rid,v in cache.rows.items() if rid in selected}
    out=Path(run)/('export_shards_'+partition)/('rank_'+str(rank));out.mkdir(parents=True,exist_ok=True)
    a.export_output=str(out)
    if backend=='bcgnet':
        import torch
        from .bcg_entry import export_test
        torch.set_num_threads(a.threads)
        export_test(cache,a,torch.device('cuda:0'),Path(run),partition)
    else:
        from driven.train import configure_tf,export_test
        tf=configure_tf(a)
        export_test(tf,cache,a,Path(run),partition)


def run_processes(processes):
    try:
        for p in processes:p.start()
        while any(p.is_alive() for p in processes):
            failed=[p for p in processes if p.exitcode not in (None,0)]
            if failed:raise RuntimeError('GPU worker failed: '+str([(p.pid,p.exitcode) for p in failed]))
            time.sleep(.25)
        if any(p.exitcode for p in processes):raise RuntimeError('GPU worker failed: '+str([p.exitcode for p in processes]))
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:
            if p.pid is not None:p.join()


def gpu_ids(count):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    ids=visible.split(',') if visible else [str(i) for i in range(count)]
    if len(ids)<count or any(not x.strip() or x.strip()=='-1' for x in ids[:count]):raise ValueError('Not enough GPUs in CUDA_VISIBLE_DEVICES')
    return [x.strip() for x in ids[:count]]


def merge_shards(cache,cohorts,run,partition,world,metadata):
    run=Path(run);expected={rid for c in cohorts for rid in cache.index([c],partition)};paths={}
    for rank in range(world):
        folder=run/('export_shards_'+partition)/('rank_'+str(rank))
        report=json.loads((folder/(partition+'_predictions.meta.json')).read_text())
        if report['status']!='COMPLETE' or report.get('native_index_sha256')!=cache.meta['native_index_sha256']:raise ValueError('Invalid shard metadata')
        rows=json.loads((folder/(partition+'_predictions.json')).read_text())
        for rid in rows:
            if rid in paths:raise ValueError('Duplicate inference record '+rid)
            paths[rid]=folder/('evaluation_'+partition)/(rid+'.npz')
    if set(paths)!=expected:raise ValueError('Shard coverage differs from selected partition')
    class View:
        meta=cache.meta
        def records(self,selected,part):
            for c in selected:
                for rid in cache.index([c],part):yield {'rid':rid}
    def prediction(r):
        with np.load(paths[r['rid']],allow_pickle=False) as f:
            return {k:f[k].item() if f[k].ndim==0 else f[k] for k in f.files}
    return export_predictions(View(),cohorts,partition,run,prediction,metadata)


def parallel_export(backend,cache,a,run,partition):
    count=a.gpus;ids=gpu_ids(count)
    ctx=mp.get_context('spawn')
    workers=[ctx.Process(target=_worker,args=(backend,a,str(run),partition,gpu,rank,count)) for rank,gpu in enumerate(ids)]
    run_processes(workers)
    return merge_shards(cache,a.cohorts,run,partition,count,dict(baseline=backend,task=getattr(a,'task','ahi'),stage_run=getattr(a,'stage_run',None),gpu_processes=count,parallelism='disjoint_recordings',sleep_source=getattr(a,'sleep_source','predicted_sleep_wake'),reference_sleep_assisted=getattr(a,'sleep_source','predicted')=='reference'))
