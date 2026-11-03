import torch
import torch.nn as nn
import typing as t


class ResBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        downsample: t.Union[nn.Module, None] = None,
        norm_layer: t.Callable[[int], nn.Module] = nn.BatchNorm1d,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, 7, stride=stride, padding=3, bias=False
            ),
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1, bias=False),
            norm_layer(out_channels),
        )
        self.act = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        return self.act(self.stem(x) + identity)
