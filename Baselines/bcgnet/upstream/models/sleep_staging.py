import torch.nn as nn
from typing import List, Tuple, Dict
import torch
from .sleep_net_zero import SleepNetZeroFacade
import pdb

class SleepStagingNet(nn.Module):
    def __init__(self, num_classes: int = 5, **kwargs) -> None:
        super().__init__()

        self.backbone = SleepNetZeroFacade(**kwargs)
        #pdb.set_trace()
        self.fc = nn.Linear(self.backbone.hidden_size, num_classes)

    def forward(self, **kwargs):
        #pdb.set_trace()
        x = self.backbone(**kwargs)
        if isinstance(x, torch.Tensor):
            x = x[:, 1:, :]
            if x.shape[-1] == self.backbone.hidden_size:
                x = self.fc(x)
            
        return x

class SleepStagingMultiheadNet(nn.Module):
    def __init__(self, num_classes: Dict , **kwargs) -> None:
        super().__init__()

        self.backbone = SleepNetZeroFacade(**kwargs)
        self.num_classes = num_classes
        for head_name in num_classes.keys():
            setattr(self, f"{head_name}", nn.Linear(self.backbone.hidden_size, num_classes[head_name]))



    def forward(self, **kwargs):
        #pdb.set_trace()
        x = self.backbone(**kwargs)[:, 1:, :]
        #x = self.fc(x)
        result = {}
        for head_name in self.num_classes.keys():
            result[head_name] = getattr(self, f"{head_name}")(x)
        return result



