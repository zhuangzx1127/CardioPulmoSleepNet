"""Exact global batches for torchrun; no duplicated targets in uneven tails."""
import json, os, random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def local_refs(batch,rank,world):
    return batch[rank::world]

def epoch(model,cache,a,device,number,optimizer=None,scheduler=None):
    from .torch_entry import batch_loss
    train=optimizer is not None;model.train(train)
    rank=dist.get_rank();world=dist.get_world_size();size=a.batch_size if train else a.val_batch_size
    refs=list(cache.index(a.cohorts,'train' if train else 'val'))
    if train:np.random.default_rng(a.seed+number).shuffle(refs)
    smoke=a.command=='smoke'
    if smoke:refs=refs[:size]
    total=count=targets=updates=0;sizes=[];lrs=[]
    for start in range(0,len(refs),size):
        batch=refs[start:start+size];mine=local_refs(batch,rank,world)
        # All ranks participate in DDP backward. Empty ranks forward a real
        # record with zero objective weight; it is never counted as a sample.
        rows=[cache.read(r) for r in (mine or batch[:1])]
        with torch.set_grad_enabled(train):
            loss,n,_=batch_loss(model,rows,'wav2sleep','stage',a,device,smoke)
            n=n if mine else 0
            packed=torch.tensor([float(loss.detach())*n,n,int(not torch.isfinite(loss))],dtype=torch.float64,device=device)
            dist.all_reduce(packed)
            if packed[2] or packed[1]<=0:raise ValueError('Invalid distributed loss/targets')
            global_loss=float(packed[0]/packed[1])
            if train:
                optimizer.zero_grad(set_to_none=True)
                # DDP averages ranks; correct for differing valid epoch counts.
                (loss*(world*n/float(packed[1]))).backward()
                if a.grad_clip:torch.nn.utils.clip_grad_norm_(model.parameters(),a.grad_clip,error_if_nonfinite=True)
                elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Nonfinite gradients')
                lrs.append(optimizer.param_groups[0]['lr']);optimizer.step();updates+=1
                if scheduler is not None:scheduler.step()
        total+=global_loss*len(batch);count+=len(batch);targets+=int(packed[1]);sizes.append(len(batch))
        if rank==0:print(json.dumps(dict(phase='train' if train else 'val',recordings=count,global_batch_samples=len(batch),updates=updates,loss=global_loss)),flush=True)
    if not count:raise ValueError('No records')
    return dict(loss=total/count,target_count=targets,recordings=count,optimizer_updates=updates,batch_sizes=sizes,learning_rates=lrs)

def main(a):
    from .torch_entry import Cache,verify_source,make_run,write_json,make_model,load_model,optimizer_and_scheduler
    if a.command not in ('train','smoke'):raise ValueError('torchrun supports train/smoke; use single-process check/evaluate')
    if a.task!='stage' or not a.run:raise ValueError('wav2sleep requires stage task and --run')
    if min(a.batch_size,a.val_batch_size,a.epochs,a.threads)<1 or not 1<=a.smoke_epochs<=1200:raise ValueError('Invalid training settings')
    local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    cpu=a.device=='cpu'
    if not cpu:
        if a.device not in ('auto','cuda'):raise ValueError('DDP uses --device cuda or auto; select GPUs with CUDA_VISIBLE_DEVICES')
        torch.cuda.set_device(local)
    device=torch.device('cpu' if cpu else 'cuda',None if cpu else local)
    dist.init_process_group('gloo' if cpu else 'nccl')
    try:
        rank=dist.get_rank();torch.set_num_threads(a.threads)
        random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
        cache=Cache(a.cache,a.manifest);verify_source('wav2sleep')
        model=make_model('wav2sleep','stage').to(device)
        model=DDP(model,device_ids=None if cpu else [local],broadcast_buffers=False,find_unused_parameters=True)
        # Separate reproducible augmentation/dropout streams after identical init.
        random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
        optimizer,scheduler,interval=optimizer_and_scheduler(model,'wav2sleep',a)
        a.distributed=dict(strategy='DDP',world_size=world,backend=dist.get_backend(),global_train_batch=a.batch_size,global_val_batch=a.val_batch_size,tail_policy='no dropped/repeated targets; zero-weight forward on empty ranks',loss_reduction='global valid-epoch mean; epoch logging weighted by global night count')
        run=Path(a.run)
        if rank==0:make_run(run,a,cache,'wav2sleep')
        dist.barrier();history=[];best=float('inf');stale=0
        for number in range(1,(1 if a.command=='smoke' else a.epochs)+1):
            train=epoch(model,cache,a,device,number,optimizer,scheduler)
            val=epoch(model,cache,a,device,number)
            history.append(dict(epoch=number,train=train,val=val))
            if rank==0:write_json(run/'history.json',history)
            if val['loss']<best:
                best=val['loss'];stale=0
                if rank==0:torch.save(dict(state_dict=model.module.state_dict(),optimizer_state=optimizer.state_dict(),scheduler_state=scheduler.state_dict(),manifest_sha256=cache.meta['manifest_sha256'],baseline='wav2sleep',task='stage',epoch=number,arguments=vars(a)),run/'best.pt')
            else:stale+=1
            dist.barrier()
            if a.patience and stale>=a.patience:break
        if rank==0:
            load_model('wav2sleep','stage',run,device,cache)
            write_json(run/'completion.json',dict(status='REAL_DATA_SMOKE' if a.command=='smoke' else 'TRAINED',best_val_loss=best,epochs=len(history),checkpoint_strict_reload=True,full_night_processing=a.command!='smoke' or a.smoke_epochs==1200,test_evaluated=False,distributed=a.distributed))
            print('Completed:',run,flush=True)
        dist.barrier()
    finally:dist.destroy_process_group()
