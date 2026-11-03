import pandas as pd
import torch
import typing as t
from wuji_dl import slicing

from data.default_dataset import *
from wuji_dl.factory import create_instance
from augment.default_augmenter import DefaultAugmenter
from tempfile import mkdtemp
import os.path as path
from concurrent.futures import ThreadPoolExecutor
import shutil
import torch.types
from data.extractors import BaseExtractor
from datetime import datetime
from tqdm import tqdm

@dataclass
class Feature:
    dst_name: str|None = None
    src_name: str|None = None
    input_channels: int = 1
    token_len: int = 1
    dtype: str = "float"  # 仅在使用memmap时有意义
    is_input: bool = True
    pad_value: torch.types.Number | str = 0
    is_attribute: bool = False  # 是否作为attribute输入到后续。
    from_csv: bool = False  # 原始格式是不是属性。
    att_preprocesser: None | dict | BaseExtractor = None
    seq_extractor: None | dict | BaseExtractor = None
    
    def __post_init__(self):
        assert not (self.dst_name is None and self.src_name is None)
        if self.dst_name is None:
            self.dst_name = self.src_name
        elif self.src_name is None:
            self.src_name = self.dst_name
        if isinstance(self.seq_extractor, dict):
            self.seq_extractor = create_instance(BaseExtractor, self.seq_extractor)
        if isinstance(self.att_preprocesser, dict):
            self.att_preprocesser = create_instance(BaseExtractor, self.att_preprocesser)
        if self.is_attribute:
            assert self.from_csv

class SleepStagingNetInput(t.TypedDict):
    """
    task interface
    """

    length: torch.Tensor
    heartbeat: t.Optional[torch.Tensor]
    breath: t.Optional[torch.Tensor]
    body_movement: t.Optional[torch.Tensor]
    eeg: t.Optional[torch.Tensor]


class SleepStagingBatch(t.TypedDict):
    """
    task interface
    """

    id: list[t.Union[int, str]]
    net_input: SleepStagingNetInput
    stage5: torch.Tensor


PAD_STAGE = -1


class SleepStagingDataset(DefaultDataset):
    def __init__(
        self,
        index: str,
        split: str,
        max_tokens: int,
        token_sec: int,
        features: list[dict] = [],
        stride_tokens: int = 0,  # 0 for truncation
        r_margin: float = 0,
        id: str | None = "path",
        memmap: bool = False,
        memmap_path: str = ".",
        aug: t.Optional[t.Dict] = None,
        weighted_sampling: t.Union[str, None] = None,
        n: int | None = None,
        time_cut: bool = False,
        abandon_wake: bool = False,
        **kwargs: t.Any,
    ) -> None:
        csv = pd.read_csv(index)
        csv = csv[(csv["split"] == split) & (csv["duration"] >= token_sec)]
        if n is not None:
            csv = csv.sample(n)
        data: list[SampleIndex] = []
        rs = []  # 记录右侧读到多远
        total_n = 0
        self.features = [Feature(**args) for args in features]
        #loading = [(feat.src_name, feat.dst_name) for feat in self.features if feat.from_csv]
        for i, (_, row) in enumerate(csv.iterrows()):
            n = int((row["duration"] - r_margin) // token_sec)
            if "shaoyifu" in row["path"] and stride_tokens == 0:  # shaoyifu
                r = n
                l = max(0, n - max_tokens)
                windows = [[l, r]]
            elif time_cut:
                windows = slicing.window_every_hour(n, max_tokens, stride_tokens, row['time'])
            else:
                windows = slicing.window(n, max_tokens, stride_tokens)
            if abandon_wake:
                npz = np.load(row['path'])
                stage = np.repeat(npz['stage5'] > 0, 30)[:n] if 'stage5' in npz else np.zeros((n, ))
            for l, r in windows:
                if abandon_wake and sum(stage[int(l): int(r)]) == 0:
                    print(f"{row['path']} 的 {l}: {r} 被删除了！")
                    continue
                payload={}
                for f in self.features:
                    if f.from_csv:
                        payload[f.dst_name] = row[f.src_name] if f.att_preprocesser is None else f.att_preprocesser(None, l, r, row[f.src_name])
                data.append(
                    SampleIndex(
                        id=i if id is None else row[id],
                        path=row["path"],
                        start=l + total_n if memmap else l,
                        end=r + total_n if memmap else r,
                        payload=payload
                    )
                )
            rs.append(r)
            total_n += r

        self.memmap = memmap
        if self.memmap:
            temp_dir = mkdtemp(dir=memmap_path)
            data_memmap = {
                feat.src_name: np.memmap(
                    filename=path.join(temp_dir, feat.src_name),
                    mode="w+",
                    shape=(
                        (feat.input_channels, total_n * feat.token_len)
                        if feat.input_channels != 1
                        else (total_n * feat.token_len)
                    ),
                    dtype=np.dtype(feat.dtype),
                )
                for feat in self.features
            }

            self.path = temp_dir

            def load_data(i, path):
                with np.load(path) as npz:
                    for feat in self.features:
                        if feat.input_channels == 1 and feat.src_name in npz:
                            data_memmap[feat.src_name][
                                sum(rs[:i])
                                * feat.token_len : sum(rs[: i + 1])
                                * feat.token_len
                            ] = npz[feat.src_name][0 : rs[i] * feat.token_len].astype(np.dtype(feat.dtype))
                        elif feat.src_name in npz:
                            data_memmap[feat.src_name][
                                :,
                                sum(rs[:i])
                                * feat.token_len : sum(rs[: i + 1])
                                * feat.token_len,
                            ] = npz[feat.src_name][:, 0 : rs[i] * feat.token_len].astype(np.dtype(feat.dtype))
                for d in data_memmap.values():
                    d.flush()

            with ThreadPoolExecutor(max_workers=kwargs["num_workers"]) as executor:
                for i, (_, row) in enumerate(csv.iterrows()):
                    executor.submit(load_data, i, row["path"])
        else:
            data_memmap = None

        if aug is not None:
            augmenter = DefaultAugmenter(
                label="stage",
                max_tokens=max_tokens,
                is_val_test= split in ['val', 'test'],
                **aug,
                )
        else:
            augmenter = None

        extractors = {}

        for feat in self.features:
            if not feat.is_attribute:
                if feat.seq_extractor is None:
                    extractors[feat.dst_name] = default_extractor(
                        feat.src_name,
                        feat.token_len,
                        feat.input_channels,
                        feat.dtype,  # rename(
                    )
                else:
                    extractors[feat.dst_name] = feat.seq_extractor

        collators = {
            feat.dst_name: (
                feat.is_input,
                (
                    pad_collator(
                        feat.dst_name,
                        pad_value=(
                            torch.nan if feat.pad_value == "nan" else feat.pad_value
                        ),
                    )
                    if not feat.is_attribute
                    else lambda samples, head=feat.dst_name: torch.tensor(
                        [s.payload[head] for s in samples]
                    )
                ),
            )
            for feat in self.features
        }

        super().__init__(
            data,
            data_memmap=data_memmap,
            extractors=extractors,
            collators=collators,
            dataloader_config=kwargs,
            augmenter=augmenter,
            weighted_sampling=weighted_sampling if split == "train" else None,
        )

    def __del__(self):
        if self.memmap:
            shutil.rmtree(self.path)
