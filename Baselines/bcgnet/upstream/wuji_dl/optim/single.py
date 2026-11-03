import torch
import torch.nn as nn
import typing as t

from wuji_dl.factory import create_instance


class SingleOptimizerFactory:
    def __init__(
        self, optimizer: dict[str, t.Any], lr_scheduler: t.Union[dict[str, t.Any], None]
    ) -> None:
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def __call__(self, parameters: t.Iterator[nn.Parameter]):
        optimizer = create_instance(torch.optim.Optimizer, self.optimizer, parameters)
        if self.lr_scheduler is None:
            return optimizer
        lr_scheduler = self.lr_scheduler.copy()
        lr_scheduler["scheduler"] = create_instance(
            torch.optim.lr_scheduler.LRScheduler, lr_scheduler["scheduler"], optimizer
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
