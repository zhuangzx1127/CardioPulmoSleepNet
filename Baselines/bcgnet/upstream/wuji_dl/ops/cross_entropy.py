import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int, **kwargs) -> None:
        super().__init__()
        if kwargs.get("weight") is not None:
            kwargs["weight"] = torch.tensor(kwargs["weight"])
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target != self.ignore_index
        return self.criterion(logits[mask], target[mask])
