import torch
import torch.nn as nn


class Transpose(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2)
