import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, pos_weight=None, **kwargs) -> None:
        if pos_weight is not None:
            kwargs["pos_weight"] = torch.tensor(pos_weight)
        super().__init__(**kwargs)
