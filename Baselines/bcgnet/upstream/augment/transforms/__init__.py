from torchaudio.transforms import SpeedPerturbation

from .noise import GaussianRandomNoise
from .scale import RandomScale
from .zscore import ZScore, RobustZScore
from .aug import TimeShifting, Slicing, Mirroring, Rotation, Resampling, ChannelShuffling, GaussianRandomNoise, NormalizationPerturbation, ContrastAdjustment, BandFilter, Mask
