from typing import Optional, Union

import torch
import torch.nn as nn

from sklearn.preprocessing import robust_scale


class ZScore(nn.Module):
    def __init__(self, mode: Union[bool, str] = True, eps: float = 1e-5):
        super().__init__()
        self.mode = mode
        assert self.mode in [True, False, "robust", "masked"]
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if not self.mode:
            return x
        if self.mode == "robust":
            return torch.from_numpy(robust_scale(x.cpu().numpy())).to(x)
        if self.mode == "masked" and mask is not None:
            return x.masked_scatter(mask, self._forward_impl(x[mask]))
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor):
        if x.numel() <= 1:
            return x
        return (x - x.mean()) / torch.sqrt(x.var() + self.eps)


# kept for backward compatibility
class RobustZScore(nn.Module):
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        y = robust_scale(x.numpy())
        return torch.from_numpy(y)
