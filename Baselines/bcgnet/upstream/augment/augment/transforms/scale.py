import torch
import torch.nn as nn


class RandomScale(nn.Module):
    def __init__(self, scale_min: float = 1.0, scale_max: float = 1.0):
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: torch.Tensor):
        factor = torch.rand(1) * (self.scale_max - self.scale_min) + self.scale_min
        return factor * x
