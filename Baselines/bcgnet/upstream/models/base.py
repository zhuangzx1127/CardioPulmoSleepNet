from wuji_dl.factory import create_instance
from torch import nn
import torch


class base_model(nn.Module):
    def __init__(self,  backbone: dict, heads: list[dict]):
        super().__init__()
        self.backbone = create_instance(nn.Module, backbone)
        self.heads = nn.ModuleList([create_instance(nn.Module, head) for head in heads])
    def forward(self, **kwargs):
        x = self.backbone(**kwargs)
        result = {}# {"x": x}
        for head in self.heads:
            result.update(head(x, **kwargs))
        return result
        