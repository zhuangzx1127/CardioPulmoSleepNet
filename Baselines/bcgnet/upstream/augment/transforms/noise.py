import torch
import torch.nn as nn


class GaussianRandomNoise(nn.Module):
    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor):
        return x + torch.randn_like(x) * self.std
