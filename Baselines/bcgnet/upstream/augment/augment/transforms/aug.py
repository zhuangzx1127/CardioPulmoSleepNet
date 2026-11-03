import torch
import torch.nn as nn
import numpy as np
import neurokit2 as nk
import typing as t
from numpy.typing import NDArray
from scipy import stats

class GaussianRandomNoise(nn.Module):
    def __init__(self, std: float):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor):
        noise = torch.randn_like(x) * self.std
        return x + noise

class TimeShifting(nn.Module):
    def __init__(self, shift_max: int):
        super().__init__()
        self.shift_max = shift_max

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        shift = torch.randint(-self.shift_max, self.shift_max, (1,))
        return torch.roll(x, shifts=shift.item(), dims=1), label

class Slicing(nn.Module):
    def __init__(self, slice_len: int, token_len: int):
        super().__init__()
        self.token_len = token_len
        self.slice_size = slice_len * token_len

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        if x.size(1) <= self.slice_size:
            return x, label
        start = torch.randint(0, x.size(1) - self.slice_size, (1,))
        label_start = start // self.token_len
        start = label_start * self.token_len
        return x[:, start:start + self.slice_size], label[label_start:label_start + self.slice_size//self.token_len]

class BandFilter(nn.Module):
    def __init__(
        self,
        fs=4,
        lower_bound: tuple[float, float] = (0.05, 0.15),
        upper_bound: tuple[float, float] = (0.3, 0.4),
        is_val_test=False,
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.fs = fs
        self.is_val_test = is_val_test

    def forward(self, resp: t.Union[torch.Tensor, np.ndarray]):
        dtype = resp.dtype
        if not self.is_val_test:
            lowcut = (np.random.uniform(*self.lower_bound),)
            highcut = (np.random.uniform(*self.upper_bound),)
        else:
            lowcut = (self.lower_bound[0] + self.lower_bound[1]) / 2
            highcut = (self.upper_bound[0] + self.upper_bound[1]) / 2
        resp = nk.signal_filter(
            resp,
            lowcut=lowcut,  # 可以看一下
            highcut=highcut,
            method="bessel",
            sampling_rate=self.fs,
            order=3,
        )
        resp = t.cast(
            NDArray,
            resp,
        )
        return torch.tensor(resp.copy(), dtype=dtype)
    
class Mirroring(nn.Module):
    def forward(self, x: torch.Tensor, label: torch.Tensor):
        return torch.flip(x, [-1]), torch.flip(label, [-1])

class Rotation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        # Assuming x is a multi-dimensional time series data (batch_size, channels, length)
        # Generate a random rotation matrix
        angle = torch.rand(1) * 2 * np.pi
        rotation_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)], 
                                        [torch.sin(angle),  torch.cos(angle)]])
        # Apply rotation (this is a simple example for 2D rotation)
        return torch.matmul(rotation_matrix, x)

class Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.length_sampler = stats.gamma(
            2.118015930559434 * 2.5, 1.6073826305607342, 0.5189391605678708
        )
        self.freq_smplaer = stats.norm(3.5371236491378086 * 1.3, 1.1905298824126544)


    def forward(self, x: torch.Tensor):
        seq_length = x.shape[-1]
        mask_counts = np.exp(self.freq_smplaer.rvs(round(seq_length / (3600 * 4))))
        mask_counts[mask_counts > 400] = np.random.randint(
            0, 401, (sum(mask_counts > 400),)
        )  # 400改300
        mask_count = int(np.sum(mask_counts).item())
        if mask_count == 0:
            return x
        mask_lengths = torch.tensor(
            np.exp(self.length_sampler.rvs(mask_count)), dtype=torch.int
        )
        mask_lengths[mask_lengths > 360] = torch.randint(
            7, 361, (torch.sum(mask_lengths > 360),), dtype=torch.int
        )  # 360改240
        mask_lengths[mask_lengths < 7] = 7
        """
        # uniform 版本
        mask_count = int(torch.sum(torch.randint(0, 281, (round(seq_length / (3600 * 4)), ), dtype=torch.int)).item())
        mask_lengths = torch.randint(7, 161, (mask_count, ), dtype=torch.int)
        """
        mask = torch.zeros(seq_length, dtype=torch.bool)
        start_positions = torch.randint(0, seq_length, (mask_count,))

        # 生成掩码
        mask_indices = (
            start_positions[:, None] + torch.arange(mask_lengths.max())[None, :]
        )
        end_positions = (start_positions + mask_lengths - 1)[:, None]
        mask_indices = torch.masked_select(
            mask_indices,
            torch.logical_and(mask_indices < end_positions, mask_indices < seq_length),
        )

        # print(mask)
        # 应用掩码
        mask[mask_indices] = True
        if x.ndim > 1:
            avg = torch.mean(x, 1, keepdim=True, dtype=x.dtype)
            x[:, mask] = avg
        else:
            avg = torch.mean(x)
            x[mask] = avg
        return x

class Resampling(nn.Module):
    def __init__(self, new_size: int):
        super().__init__()
        self.new_size = new_size

    def forward(self, x: torch.Tensor):
        return torch.nn.functional.interpolate(x.unsqueeze(0), size=self.new_size, mode='linear', align_corners=False).squeeze(0)

class ChannelShuffling(nn.Module):
    def forward(self, x: torch.Tensor):
        # Assuming x is multi-channel data (batch_size, channels, length)
        channels = x.size(1)
        shuffle_idx = torch.randperm(channels)
        return x[:, shuffle_idx, :]

class ContrastAdjustment(nn.Module):
    def __init__(self, factor: float):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        return (x - mean) * self.factor + mean
    
class NormalizationPerturbation(nn.Module):
    def __init__(self, perturb_factor: float):
        super().__init__()
        self.perturb_factor = perturb_factor

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) * (1 + self.perturb_factor)
        return (x - mean) / std

# You can add more classes for other techniques like Rotation, Resampling, etc.
