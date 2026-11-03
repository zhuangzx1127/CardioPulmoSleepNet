from collections import OrderedDict
from numpy.typing import NDArray
import pandas as pd
import torch
import torchmetrics as tm
import typing as t


class ClassificationMetrics(tm.MetricCollection):
    SCALAR_KEYS = {"acc", "macro_f1", "weighted_f1", "cohen_kappa"}

    def __init__(
        self, classes: t.List[str], ignore_index: t.Optional[int] = None
    ) -> None:
        self.classes = classes
        common = {
            "task": "multiclass",
            "num_classes": len(classes),
            "ignore_index": ignore_index,
        }
        super().__init__(
            {
                "acc": tm.Accuracy(**common),
                "macro_f1": tm.F1Score(**common, average="macro"),
                "weighted_f1": tm.F1Score(**common, average="weighted"),
                "cohen_kappa": tm.CohenKappa(**common),
                "confusion_matrix": tm.ConfusionMatrix(**common, normalize="all"),
                "precision": tm.Precision(**common, average="none"),
                "recall": tm.Recall(**common, average="none"),
            }
        )

    def loggable_items(self) -> OrderedDict[str, tm.Metric]:
        self._compute_groups_create_state_ref(False)
        od: OrderedDict[str, tm.Metric] = OrderedDict()
        for k, v in self._modules.items():
            if k in self.SCALAR_KEYS:
                od[self._set_name(k)] = v
        return od

    def _yield(self, key: str) -> tuple[str, NDArray]:
        self._compute_groups_create_state_ref(False)
        value: torch.Tensor = self[key].compute()
        self[key].reset()
        return self._set_name(key), value.detach().cpu().numpy()

    def yield_precision(self) -> tuple[str, pd.Series]:
        key, value = self._yield("precision")
        return key, pd.Series(value, index=self.classes, name="precision")

    def yield_recall(self) -> tuple[str, pd.Series]:
        key, value = self._yield("recall")
        return key, pd.Series(value, index=self.classes, name="recall")

    def yield_confusion_matrix(self) -> tuple[str, pd.DataFrame]:
        key, value = self._yield("confusion_matrix")
        return key, pd.DataFrame(value, index=self.classes, columns=self.classes)
