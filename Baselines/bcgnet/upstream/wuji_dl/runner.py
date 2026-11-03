import lightning as L
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import time
import torch.distributed
import typing as t

from wuji_dl.factory import create_instance, get_class


def _parse_hierarchy(path: str) -> DictConfig:
    config = t.cast(DictConfig, OmegaConf.load(path))
    base_paths = config.pop("_base", [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]
    bases = [_parse_hierarchy(base) for base in base_paths]
    return t.cast(DictConfig, OmegaConf.merge(*bases, config))


def _resolve_dict_config(path: str, **kwargs) -> dict[str, t.Any]:
    file_config = _parse_hierarchy(path)
    kwargs_config = OmegaConf.create(kwargs)
    config = OmegaConf.merge(file_config, kwargs_config)
    resolved_config = t.cast(
        dict[str, t.Any], OmegaConf.to_container(config, resolve=True)
    )
    return resolved_config


class Runner:
    def __init__(
        self,
        devices: t.Union[int, tuple[int, ...], str] = "auto",
        seed: int = 42,
        fast_dev_run: t.Union[bool, int] = False,
        project: t.Union[str, None] = None,
        group: t.Union[str, None] = None,
        job_type: t.Union[str, None] = None,
        **kwargs,
    ):
        self.devices = list(devices) if isinstance(devices, tuple) else devices
        self.config: dict[str, t.Any] = {}
        L.seed_everything(seed, workers=True)
        self.fast_dev_run = fast_dev_run
        self.version = f"{seed}-" + time.strftime("%m%d-%H%M%S")
        self.project = project
        self.group = group
        self.job_type = job_type
        self.kwargs = kwargs

    def _set_config(self, config: str, stage: str) -> None:
        self.config = _resolve_dict_config(config, **self.kwargs)
        if self.project is None:
            self.project = self.config.get("project")
        if self.group is None:
            self.group = self.config.get("group", Path(config).stem)
        if self.job_type is None:
            self.job_type = self.config.get("job_type", stage)
        self.version = f"{self.group}-{self.job_type}-{self.version}"

    def _logger(self) -> WandbLogger:
        save_dir = Path("log")
        save_dir.mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            save_dir=save_dir,
            project=self.project,
            group=self.group,
            job_type=self.job_type,
            version=self.version,
        )

    def _callbacks(self) -> list[L.Callback]:
        callbacks = self.config.get("callbacks", [])
        if isinstance(callbacks, dict):
            callbacks = callbacks.values()
        return [create_instance(L.Callback, item) for item in callbacks]

    def fit(
        self,
        config: str,
        *,
        find_unused_parameters: bool = False,
        ckpt: t.Union[str, None] = None,
    ) -> t.Union[str, None]:
        self._set_config(config, "fit")
        task = create_instance(L.LightningModule, self.config["task"])
        data = create_instance(L.LightningDataModule, self.config["data"])
        trainer_config = self.config.get("trainer", {})
        if find_unused_parameters:
            trainer_config["strategy"] = "ddp_find_unused_parameters_true"
        trainer = L.Trainer(
            devices=self.devices,
            logger=self._logger(),
            callbacks=self._callbacks(),
            fast_dev_run=self.fast_dev_run,
            accelerator="gpu",
            **trainer_config,
        )
        trainer.fit(task, data, ckpt_path=ckpt)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if not trainer.is_global_zero:
            return
        return f"run {self.version} finished"

    def validate(self, config: str, ckpt: str) -> str:
        return self._eval("validate", config, ckpt)

    def test(self, config: str, ckpt: str) -> str:
        return self._eval("test", config, ckpt)

    def predict(self, config: str, ckpt: str) -> str:
        return self._eval("predict", config, ckpt)

    def _eval(self, stage: str, config: str, ckpt: str) -> str:
        self._set_config(config, stage)
        task_config = self.config["task"]
        task_class = task_config.pop("_name")
        task = get_class(L.LightningModule, task_class).load_from_checkpoint(
            ckpt, map_location="cpu", **task_config
        )
        data = create_instance(L.LightningDataModule, self.config["data"])
        trainer = L.Trainer(
            devices=self.devices,
            logger=self._logger(),
            callbacks=self._callbacks(),
            fast_dev_run=self.fast_dev_run,
            **self.config.get("trainer", {}),
        )
        getattr(trainer, stage)(task, data)
        return f"run {self.version} finished"

    def export_onnx(self, config: str, ckpt: str, save_path: str) -> str:
        self._set_config(config, "export_onnx")
        task_config = self.config["task"]
        task_class = task_config.pop("_name")
        task = get_class(L.LightningModule, task_class).load_from_checkpoint(
            ckpt, map_location="cpu", **task_config
        )
        task.eval()
        input_sample_config = self.config.get("input_sample", [])
        input_names = list(input_sample_config.keys())
        input_sample = {}
        for key, item in input_sample_config.items():
            input_sample[key] = eval(item)
        input_sample = {"net_input": input_sample}
        output_names = self.config.get("output_names", [])
        dynamic_axes = {}
        for name in input_names + output_names:
            dynamic_axes[name] = {0: "batch_size"}

        task.to_onnx(
            file_path=save_path,
            input_sample=input_sample,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
        return f"exporting onnx to {save_path}"


def fit_runner(
    config: str,
    *,
    find_unused_parameters: bool = False,
    ckpt: t.Union[str, None] = None,
    **kwargs,
) -> t.Union[str, None]:
    return Runner(**kwargs).fit(
        config, find_unused_parameters=find_unused_parameters, ckpt=ckpt
    )


def test_runner(config: str, ckpt: str, **kwargs) -> t.Union[str, None]:
    return Runner(**kwargs).test(config, ckpt)


def export_onnx_runner(config: str, ckpt: str, save_path: str, **kwargs) -> str:
    return Runner(**kwargs).export_onnx(config, ckpt, save_path)
