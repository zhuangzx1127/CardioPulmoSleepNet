"""Instantiate pinned upstream models without upstream data/CLI dependencies."""
import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path
import torch
from torch import nn

BASE=Path(__file__).resolve().parents[1]

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def package(name,path):
    # Native __init__ imports the full inference API; models themselves need no API.
    if name not in sys.modules:
        p=types.ModuleType(name);p.__path__=[str(path)];sys.modules[name]=p

def wav2sleep():
    root=BASE/'wav2sleep/upstream/src/wav2sleep';package('wav2sleep',root);package('wav2sleep.models',root/'models')
    m=importlib.import_module('wav2sleep.models.wav2sleep')
    return m.Wav2Sleep(
        m.SignalEncoders({'ECG':'ECG','THX':'THX'},128,'gelu',norm='instance',causal=False,chunk_causal=False,initial_channels=16,max_channels=128,output_norm=False,use_residual=True),
        m.MultiModalAttentionEmbedder(128,layers=2,dropout=.1,dim_ff=512,nhead=8),
        m.SequenceCNN(128,dropout=.1,num_layers=2,activation='gelu',norm='layer',causal=False,num_dilations=6,kernel_size=7),4)

def tsdnet(): return module('native_tsd',BASE/'tsdnet/upstream/TSDNet.py').TSDNet_B()
def ahinet(): return module('native_ahi',BASE/'tsdnet/upstream/AHI_regressionNet.py').AHINet()

def bcgnet(task):
    root=BASE/'bcgnet/upstream'
    # Extract unmodified native class bodies. Remove only unused Lightning,
    # pretrained-loading and CLI imports; no copied architecture is rewritten.
    from transformers import RoFormerConfig,RoFormerModel
    import typing
    from sympy.ntheory import factorint
    rb=module('native_resblock',root/'wuji_dl/ops/res_block1d.py').ResBlock1d
    def length_mask(max_len,length): return torch.arange(max_len,device=length.device)[None,:]<length[:,None]
    scope=dict(nn=nn,torch=torch,t=typing,RoFormerConfig=RoFormerConfig,RoFormerModel=RoFormerModel,factorint=factorint,ResBlock1d=rb,length_mask=length_mask)
    source=(root/'models/sleep_net_zero.py').read_text();tree=ast.parse(source)
    for node in tree.body:
        if isinstance(node,ast.ClassDef) and node.name in ('ResNetFeatureExtractor','SleepNetZeroHead','SleepNetZeroBackbone'):
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(root/'models/sleep_net_zero.py'),'exec'),scope)
    scope['SleepNetZeroFacade']=scope['SleepNetZeroBackbone']
    node=next(n for n in ast.parse((root/'models/sleep_staging.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='SleepStagingNet')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'native_sleep_staging','exec'),scope)
    kwargs=dict(features=[dict(head=k,token_len=120 if task=='stage' else 4,input_channels=1) for k in ['heartbeat','breath']],hidden_size=512,num_hidden_layers=6,num_attention_heads=8,intermediate_size=2048,max_position_embeddings=1536 if task=='stage' else 1801)
    if task=='stage':return scope['SleepStagingNet'](num_classes=5,**kwargs)
    head=module('native_heads',root/'models/heads.py').seqProj('ahi_event',512,num_classes=1,token_len=1)
    class EventNet(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=scope['SleepNetZeroBackbone'](**kwargs);self.head=head
        def forward(self,**x):return self.head(self.backbone(**x))['ahi_event']
    return EventNet()

def focal_tsd(pred,y):
    # Same unusual upstream evaluation objective: softmax over sigmoid outputs.
    p=torch.softmax(pred,dim=1).gather(1,y[:,None]).squeeze(1).clamp_min(1e-7)
    return (-.5*(1-p).square()*p.log()).mean()

def focal_bcg(pred,y,mask):
    p=torch.sigmoid(pred);pt=torch.where(y.bool(),p,1-p);alpha=torch.where(y.bool(),.25,.75)
    loss=alpha*(1-pt).square()*torch.nn.functional.binary_cross_entropy_with_logits(pred,y.float(),reduction='none')
    return loss[mask].mean()
