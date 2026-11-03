# Reproduced baselines and public participant splits

Training and evaluation code for ResSleepNet, OxiNet, wav2sleep, BCGNet and
DRIVEN, with participant-disjoint splits for MESA, MrOS and SHHS.

| Method | Adapter inputs | Outputs | Input format |
|---|---|---|---|
| ResSleepNet | RIPsum | Four-class staging, AHI | TrainingCache |
| OxiNet | SpO2 | AHI | TrainingCache; specify `--hours 10` |
| wav2sleep | ECG + RIPsum in its THX input position | Four-class staging | TrainingCache |
| BCGNet | ECG-derived IHR + ABD, both 4 Hz | Five-class staging, binary respiratory events, derived AHI | native_10h |
| DRIVEN | ABD + SpO2 | Binary sleep/wake, binary events, calibrated AHI | native_10h |

BCGNet probabilities merge N1+N2 for four-class evaluation. Its AHI evaluation
uses predicted sleep from the staging run by default. DRIVEN also uses predicted
sleep/wake to estimate AHI.

## Data placement

Keep the following paths relative to the repository root:

```text
Baselines/                         # source code
Dataset/splits.json                # public participant split
BaselineInputs/native_10h/splits.json  # common native-input subset
TrainingCache/cache.json           # prepared cache metadata
TrainingCache/<cohort>/*.tfrecord   # prepared signals and labels
BaselineInputs/native_10h/index.json
BaselineInputs/native_10h/data/<cohort>/<partition>/*.npz
Runs/                              # created by training
```

Obtain cohort data through NSRR and prepare the inputs described in
[DATA_FORMATS.md](DATA_FORMATS.md). Preserve the supplied split files: their
checksums are used to validate data membership. See
[splits_summary.json](splits_summary.json) for cohort counts and participant IDs.

| Split | Main cache nights | BCGNet/DRIVEN nights |
|---|---:|---:|
| Train | 9,905 | 9,902 |
| Validation | 1,394 | 1,394 |
| Test | 2,870 | 2,870 |

The native subset excludes only three training records: one lacking ABD and two
with unusable IHR. Both native methods use the same exclusions. Main and native
test participants are identical. Multiple visits from one participant stay in
the same partition.

## Dependencies

Install dependencies in separate PyTorch and TensorFlow environments:

```bash
python -m pip install -r Baselines/requirements-torch.txt
# In a separate environment:
python -m pip install -r Baselines/requirements-tf-linux.txt
```

## Run from `Baselines/`

The following examples select seed 42 explicitly. Options and realized settings
are saved in each run directory. Use a new run path for a separate experiment.

```bash
# TensorFlow: ResSleepNet
CUDA_VISIBLE_DEVICES=0,1,2,3 bash resleepnet/run.sh train 42 \
  --run_dir ../Runs/baselines/resleepnet/seed_42

# TensorFlow: OxiNet, 10-hour reproduction
python -u oxinet/train.py train --hours 10 --batch-size 256 --epochs 100 \
  --seed 42 --run ../Runs/baselines/oxinet/seed_42_10h
python oxinet/train.py evaluate --hours 10 --seed 42 \
  --run ../Runs/baselines/oxinet/seed_42_10h --partition test
python oxinet/export_test.py --run ../Runs/baselines/oxinet/seed_42_10h \
  --root ..

# PyTorch: wav2sleep, four DDP processes; batch is GLOBAL
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --standalone --nproc_per_node=4 wav2sleep/train.py train \
  --device cuda --seed 42 --run ../Runs/baselines/wav2sleep/seed_42
python wav2sleep/train.py evaluate --device cuda --seed 42 \
  --run ../Runs/baselines/wav2sleep/seed_42 --partition test

# PyTorch: BCGNet, stage before events (DataParallel; no torchrun)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u bcgnet/train.py train \
  --task stage --gpus 4 --workers 4 --seed 42 \
  --run ../Runs/baselines/bcgnet/stage_seed_42
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u bcgnet/train.py train \
  --task events --gpus 4 --workers 4 --micro-batch-size 32 --ihr-zscore \
  --stage-run ../Runs/baselines/bcgnet/stage_seed_42 --seed 42 \
  --run ../Runs/baselines/bcgnet/events_seed_42_ihrZ

# TensorFlow: DRIVEN; batch is explicit, 128 is the reproduction choice
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u driven/train.py train \
  --gpus 4 --batch-size 128 --seed 42 \
  --run ../Runs/baselines/driven/seed_42
```

BCGNet and DRIVEN export test JSON after successful training; their `export`
commands also regenerate predictions without training. wav2sleep `evaluate`
writes per-record NPZ and cohort metrics. ResSleepNet exports its test predictions at the end
unless `--skip_test` is used. See the per-method READMEs for export commands.

The BCGNet options `--ihr-zscore`, `--micro-batch-size` and `--no-batchnorm` are
explicit variants recorded in the run protocol. BatchNorm is retained unless
the last flag is supplied. DRIVEN's four GPUs train its four CNNs independently;
its LightGBM fusion/calibration runs on CPU. `resume-fusion` continues after the
CNN stage, without starting those four CNNs again.

## Source checks

From the repository root, without importing a deep-learning framework:

```bash
python Baselines/verify_release.py
shasum -a 256 -c BASELINES_SHA256SUMS
```

Upstream repositories, pinned versions and licenses are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
