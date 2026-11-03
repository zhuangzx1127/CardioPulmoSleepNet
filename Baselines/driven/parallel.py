"""Four independent native DRIVEN CNNs, one CUDA process per target/channel."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
from common.native_inputs import NativeCache
from common.parallel_export import run_processes,gpu_ids


def _train_worker(args,run,target,channel,gpu,smoke):
    os.environ['CUDA_VISIBLE_DEVICES']=gpu
    from driven.train import configure_tf,train_cnn
    a=copy.copy(args);a.gpus=1
    cache=NativeCache(a.native_inputs,'driven');tf=configure_tf(a)
    train_cnn(tf,cache,None,a,Path(run),target,channel,smoke)


def train_four_cnns(a,run,smoke):
    jobs=[(t,c) for t in ('events','sleep') for c in ('abd','spo2')]
    ctx=mp.get_context('spawn')
    processes=[ctx.Process(target=_train_worker,args=(a,str(run),t,c,gpu,smoke)) for (t,c),gpu in zip(jobs,gpu_ids(4))]
    run_processes(processes)
    rows=[]
    for target,channel in jobs:rows.extend(json.loads((Path(run)/('history_'+target+'_'+channel+'.json')).read_text()))
    return rows
