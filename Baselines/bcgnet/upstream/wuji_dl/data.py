import lightning as L
from torch.utils.data import DataLoader, Dataset
import typing as t

from wuji_dl.factory import Factory, create_instance


class BaseDataset(Dataset, Factory):
    def dataloader() -> DataLoader:
        raise NotImplementedError


class DefaultDataModule(L.LightningDataModule):
    def __init__(
        self,
        common: t.Dict[str, t.Any] = {},
        train: t.Dict[str, t.Any] = {},
        val: t.Dict[str, t.Any] = {},
        test: t.Dict[str, t.Any] = {},
        predict: t.Dict[str, t.Any] = {},
    ) -> None:
        super().__init__()
        self._configs = {
            "train": common | train,
            "val": common | val,
            "test": common | test,
            "predict": common | predict,
        }

    def setup(self, stage: str) -> None:
        self._datasets: dict[str, BaseDataset] = {}

    def dataloader(self, split: str) -> DataLoader:
        if split not in self._datasets:
            self._datasets[split] = create_instance(BaseDataset, self._configs[split])
        return self._datasets[split].dataloader()

    def train_dataloader(self) -> DataLoader:
        return self.dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self.dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.dataloader("test")

    def predict_dataloader(self) -> DataLoader:
        return self.dataloader("predict")
