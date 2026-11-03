import torch

def resample_stairs(x: torch.Tensor, target_len: int) -> torch.Tensor:
    assert x.ndim == 1
    i = torch.arange(target_len) + 0.5
    i = i / target_len * len(x)
    i = i.floor().type(torch.int)
    return x[i]
