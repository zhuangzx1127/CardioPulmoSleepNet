"""Source-qualified defaults. Unknown author settings are never guessed."""

def resolve(name,a):
    if name=='wav2sleep':
        defaults=dict(batch_size=16,val_batch_size=32,epochs=50,lr=1e-3,optimizer='AdamW',weight_decay=1e-4,patience=5,grad_clip=1.,warmup_steps=2000,tau=10000.,min_lr=0.)
        source='upstream/scripts/config/{main,training/main,training/optimizer/adamw,training/scheduler/expdecay,training/callbacks/early_stopping}.yaml'
    elif name=='tsdnet':
        defaults=dict(batch_size=512,val_batch_size=512,epochs=100,lr=1e-3 if a.task=='ahi' else 1e-4,optimizer='AdamW',weight_decay=None if a.task=='ahi' else .05,patience=15,grad_clip=0.,warmup_steps=0,tau=0.,min_lr=1e-5 if a.task=='events' else 0.)
        source='paper DOI 10.1038/s41467-025-62864-x, Model training; public repo has no training loop'
    else:
        defaults=dict(batch_size=32 if a.task=='stage' else 48,val_batch_size=32 if a.task=='stage' else 48,epochs=100,lr=1e-4,optimizer='Adam',weight_decay=0.,patience=10,grad_clip=0.,warmup_steps=0,tau=0.,min_lr=0.)
        source='upstream/configs/sleep_staging/{psg5,base}.yaml or configs/ahi/{psg_ahi,wxbase}.yaml'
    overrides={}
    for key,value in defaults.items():
        actual=getattr(a,key,None)
        if actual is None:setattr(a,key,value)
        elif actual!=value:overrides[key]=dict(reference=value,actual=actual)
    a.hyperparameter_source=source;a.hyperparameter_overrides=overrides
    a.unresolved_hyperparameters=['AHINet weight_decay not stated in paper or training code'] if name=='tsdnet' and a.task=='ahi' and a.weight_decay is None else []
    return a

def optimizer_and_scheduler(model,name,a):
    import torch
    from .models_torch import BASE,module
    if a.unresolved_hyperparameters:raise ValueError('UNRESOLVED_AUTHOR_HYPERPARAMETERS: '+', '.join(a.unresolved_hyperparameters)+'. An explicit --weight-decay is recorded as an unverified override, not an author setting.')
    cls=getattr(torch.optim,a.optimizer)
    opt=cls(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scheduler=None;interval=None
    if name=='wav2sleep':
        native=module('native_wav_scheduler',BASE/'wav2sleep/upstream/src/wav2sleep/trainer/scheduler.py')
        scheduler=native.ExpWarmUpScheduler(opt,lr_max=a.lr,warmup_steps=a.warmup_steps,tau=a.tau);interval='step'
    elif name=='tsdnet' and a.task=='events':
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.min_lr);interval='epoch'
    return opt,scheduler,interval
