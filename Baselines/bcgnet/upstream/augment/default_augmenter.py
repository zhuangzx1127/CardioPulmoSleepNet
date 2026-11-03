from augment.utils import resample_stairs
from augment.transforms import *

import torch
import torch.nn as nn

val_test_aug = {"BandFilter"}

class DefaultAugmenter(nn.Module):
    def __init__(self, label, noise_std=0.1, speed_factors=[0.75, 0.76, 0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85,
                      0.86, 0.87, 0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96,
                      0.97, 0.98, 0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07,
                      1.08, 1.09, 1.1 , 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18,
                      1.19, 1.2 , 1.21, 1.22, 1.23, 1.24, 1.25], amplitude_range=[0.9,1.1], augcoposition=None, token_len = 120, heads = ["breath", "heartbeat"], max_tokens=1535, fs=4, resp="breath", uniform_ratio=0.15, is_val_test=True,) -> None:
        super().__init__()
        assert resp in heads
        self.separate_augmentation = nn.ModuleDict({head: nn.Sequential() for head in heads})
        if is_val_test:
            augcoposition = list(val_test_aug.intersection(augcoposition))
        if augcoposition is not None:
            for augtype in augcoposition:
                if augtype == 'TimeShifting':
                    self.TimeShifting = TimeShifting(shift_max=4)
                elif augtype == 'TimeShifting2':
                    self.TimeShifting = TimeShifting(shift_max=2)
                elif augtype == 'Slicing':
                    self.slicing = Slicing(slice_len=1000, token_len=token_len)
                elif augtype == 'RandomScale':
                    for aug in self.separate_augmentation.values():
                        aug.add_module('RandomScale', RandomScale(*amplitude_range))
                elif augtype == 'noise':
                    for aug in self.separate_augmentation.values():
                        aug.add_module('GaussianRandomNoise', GaussianRandomNoise(noise_std))
                elif augtype == 'Mirroring':
                    self.mirroring = Mirroring()
                elif augtype == 'Rotation':
                    self.rotation = Rotation()
                elif augtype == 'ContrastAdjustment':
                    for aug in self.separate_augmentation.values():
                        aug.add_module('ContrastAdjustment', ContrastAdjustment(0.1))
                elif augtype == "BandFilter":
                    self.bandfilter = BandFilter(fs=fs, is_val_test=is_val_test)
                elif augtype == "Mask":
                    self.mask = Mask()
                self.speed_pert = SpeedPerturbation(orig_freq=16000, factors=speed_factors)
        if augcoposition is None:
            augcoposition = ['None']
        self.augcoposition = augcoposition
        self.label = label
        self.token_len = token_len
        self.max_tokens = max_tokens
        self.slice_to_limit = Slicing(slice_len=max_tokens, token_len=token_len)
        self.fs = fs
        self.resp = resp

    def forward(self, data: dict) -> dict:
        if 'BandFilter' in self.augcoposition:
            data[self.resp] = self.bandfilter(data[self.resp])

        for k, v in self.separate_augmentation.items():
            data[k] = v(data[k])
        
        
        x = torch.stack([data[k] for k in self.separate_augmentation.keys()], dim=0)
        label = data[self.label]
            

        if 'TimeShifting' in self.augcoposition:
            x, label = self.TimeShifting(x, label)
        if 'TimeShifting2' in self.augcoposition:
            x, label = self.TimeShifting(x, label)
        if 'Mirroring' in self.augcoposition:
            x, label = self.mirroring(x, label)
        if 'Rotation' in self.augcoposition:
            x = self.rotation(x)
        if 'Slicing' in self.augcoposition:
            x, label = self.slicing(x, label)
        if 'speed' in self.augcoposition:
            x = self.speed_pert(x.to(dtype=torch.float32))[0].to(dtype=x.dtype)
        if "Mask" in self.augcoposition:
            x = self.mask(x)
        x, label = self.slice_to_limit(x, label)
        length = x.size(1) // self.token_len
        label = resample_stairs(label, length)
        x = x[:, : length * self.token_len]
        for i,k in enumerate(self.separate_augmentation.keys()):
            data[k] = x[i]
        data[self.label] = label
        return data, length
