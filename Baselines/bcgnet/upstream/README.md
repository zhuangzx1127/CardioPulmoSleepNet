# BCGNet: Pretrained on Nearly 600K Hours of PSG Enables Contactless BCG-based Sleep Monitoring

## Installation

```bash
pip install .
````

## Training

```bash
wuji-fit CONFIG [--find-unused-parameters] [--ckpt CKPT]
```

Train using the configuration file at `CONFIG`.

* `--find-unused-parameters`: for multi-GPU training
* `--ckpt CKPT`: resume training from the checkpoint at `CKPT`

## Testing

```bash
wuji-test CONFIG CKPT
```

Test using the configuration file at `CONFIG`; load the model from the checkpoint at `CKPT`.

## Common CLI Arguments

```
--devices DEVICES       Number or list of GPU IDs (e.g. “0,” for a single GPU). Defaults to all available GPUs.
--seed SEED             Random seed (default: 42).
--fast-dev-run          Debug mode: disable logging and checkpointing, run only 1 batch.
--fast-dev-run N        Debug mode: disable logging and checkpointing, run only N batches.
--project PROJECT       Manually set the wandb project name.  
                        If not provided, it’s read from config, then from env var; falls back to `lightning_logs`.
--group GROUP           Manually set the wandb group name.  
                        If not provided, read from config; otherwise defaults to the config file’s basename.
--job-type JOB_TYPE     Manually set the wandb job type.  
                        If not provided, read from config; defaults to `fit` for training, `test` for testing.
```

## Creating Instances from Config Files

Use:

```python
wuji_dl.factory.create_instance(BaseClass, config, *args, **kwargs)
```

Config file YAML syntax:

```yaml
_name: package.Class
argument1: value1
argument2: value2
...
```

* `BaseClass` restricts `package.Class` to be a subclass of `BaseClass`.
* `*args`: positional arguments
* `**kwargs`: keyword arguments


## Config File Inheritance

Configs support inheritance via a `_base` list. Entries in `_base` are merged with the current file; later keys override earlier ones.

## Basic Structure of Config Files

### `task`

A dict used to create a `LightningModule` instance.

### `data`

A dict used to create a `LightningDataModule` instance.

* `wuji_dl.data.DefaultDataModule` will create four datasets (`train`/`val`/`test`/`predict`) using `wuji_dl.data.BaseDataset`.
* `BaseDataset` is an abstract class extending `torch.utils.data.Dataset` and must implement a `dataloader()` method.
* DefaultDataModule parameters (all dicts, default empty):

  * `train`/`val`/`test`/`predict`: dataset settings for each split
  * `common`: shared settings; per-split keys override `common`

### `callbacks`

Can be a list of dicts or a dict of dicts.
A dict of dicts allows inheritance per key without losing others; it’s flattened into a list of callback configurations.
Each entry creates a `lightning.Callback` instance.

### `trainer`

Parameters passed directly to `lightning.Trainer`.

### Other Modules

* `wuji_dl.metrics`: custom metrics (usage like torchmetrics)
* `wuji_dl.ops`: shared model layers
* `wuji_dl.ops.functional`: shared functions
* `wuji_dl.optim`: optimizer and scheduler configs
* `wuji_dl.display`: logging display helpers
* `wuji_dl.slicing`: data-slicing utilities

## Config File Examples

### AHI

* `configs/ahi/psg_ahi.yaml` – PSG-based
* `configs/ahi/pure_bcg.yaml` – BCG-based (w/o fine-tuning)
* `configs/ahi/fine_tuning_bcg.yaml` – BCGNet (proposed)

### Sleep Staging

* `configs/sleep_staging/psg5.yaml` – PSG-based
* `configs/sleep_staging/bcg5.yaml` – BCG-based (w/o fine-tuning)
* `configs/sleep_staging/fine_tuning_bcg5.yaml` – BCGNet (proposed)

## Join us

We're Five Seasons Medical, building the full stack of AI for human health —
contactless sensors, foundation models for physiology, and LLM agents
that ship to real users every day. BCGNet is one piece of it.

The team comes from Tsinghua, Peking University, and top industry labs.
Small, focused, and shipping.

Hiring across ML research, signal processing, LLM agents, and clinical
science. Reach real users, not just benchmarks — chenxuesong@wuji-inc.com
