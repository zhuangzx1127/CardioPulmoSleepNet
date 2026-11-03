from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import typing as t
from wuji_dl.data import BaseDataset
import wuji_dl.ops.functional as wf

from augment.default_augmenter import DefaultAugmenter
import pdb
import numpy as np


@dataclass  # 这个装饰器用于自动把下列内容包装为一个类
class SampleIndex:
    id: t.Union[int, str]
    path: str
    start: int
    end: int
    payload: dict = field(default_factory=lambda: {})


@dataclass
class Sample:
    id: t.Union[int, str]
    length: int
    payload: dict


def default_extractor(name: str, frames_per_token: int, input_channels, dtype):
    if input_channels == 1:

        def extract(npz, start: int, end: int, value):
            if name in npz:
                # if name == 'stage':
                #     pdb.set_trace()
                return torch.from_numpy(
                    npz[name].flatten()[start * frames_per_token : end * frames_per_token]
                ).to(dtype=getattr(torch, dtype))
            if not 'stage' in name:
                print(f"{name}并不在{npz}中")
            zs = torch.zeros(
                size=((end - start) * frames_per_token,), dtype=getattr(torch, dtype)
            )
            return zs if not 'stage' in name else zs - 1

    else:

        def extract(npz, start: int, end: int, value):
            if name in npz:
                return torch.from_numpy(
                    npz[name][:, start * frames_per_token : end * frames_per_token].T
                ).to(dtype=getattr(torch, dtype))
            print(f"{name}并不在{npz}中")
            return torch.zeros(
                ((end - start) * frames_per_token, input_channels),
                dtype=getattr(torch, dtype),
            )

    return extract


def pad_collator(name: str, **kwargs):
    def collate(samples: list[Sample]):
        return wf.pad_batch(
            [s.payload[name] for s in samples], **kwargs
        )

    return collate


class DefaultDataset(BaseDataset):
    """
    General dataset behavior comes here
    """

    def __init__(
        self,
        data: t.Sequence[SampleIndex],
        data_memmap: t.Optional[t.Dict[str, np.memmap]],
        extractors: t.Mapping[str, t.Callable],
        collators: t.Mapping[str, tuple[bool, t.Callable]],
        dataloader_config: t.Mapping[str, t.Any],
        augmenter: t.Optional[DefaultAugmenter] = None,
        weighted_sampling: t.Union[str, None] = None,
    ) -> None:
        """
        Args:
            data: list of SampleIndex
            extractors: mapping, key : ((NpzFile, start, end) -> fetched value)
            collators: mapping, key : (is_input, (list[Sample]) -> batched value)
            dataloader_config: DataLoader kwargs except dataset and collate_fn
        """
        self.data = data
        self.data_memmap = data_memmap
        self.extractors = extractors
        self.collators = collators
        self.dataloader_config = dataloader_config
        self.augmenter = augmenter
        self.weighted_sampling = weighted_sampling

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Sample:
        src = self.data[idx]
        if self.data_memmap is not None:
            payload = {
                key: fn(self.data_memmap, src.start, src.end, src.payload[key] if key in src.payload else None)
                for key, fn in self.extractors.items()
            }
        else:
            with np.load(src.path) as npz:
                payload = {
                    key: fn(npz, src.start, src.end, src.payload[key] if key in src.payload else None)
                    for key, fn in self.extractors.items()
                }
        length = src.end - src.start

        # if self.augmenter:
        #     payload, length = self.augmenter(payload)

        src.payload.update(payload)  # update时扔进去另一本字典
        return Sample(id=src.id, length=length, payload=src.payload)

    def dataloader(self) -> DataLoader:
        def collate_fn(samples: list[Sample]):
            batch: dict = {"id": [s.id for s in samples]}
            net_input = {
                "length": torch.tensor([s.length for s in samples]),
            }
            for key, (is_input, fn) in self.collators.items():
                if is_input:
                    net_input[key] = fn(samples)
                else:
                    batch[key] = fn(samples)
            batch["net_input"] = net_input
            return batch

        if self.weighted_sampling is not None:
            labels = [s.payload[self.weighted_sampling] for s in self.data]
            class_sample_count = np.array(
                [len(np.where(np.array(labels) == t)[0]) for t in np.unique(labels)]
            )
            print(class_sample_count)
            weight = 1.0 / class_sample_count
            samples_weight = [weight[t].item() for t in labels]
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            return DataLoader(
                self, **self.dataloader_config, collate_fn=collate_fn, sampler=sampler, persistent_workers=False
            )
        else:
            return DataLoader(self, **self.dataloader_config, collate_fn=collate_fn, persistent_workers=False)
