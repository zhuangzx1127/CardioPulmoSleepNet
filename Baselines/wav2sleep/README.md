# wav2sleep

The ECG and RIPsum adapter runs with DDP via `python -m torch.distributed.run`. Batch sizes are global. The original THX model position receives resampled RIPsum. `evaluate --run RUN --partition test` loads best.pt and writes cohort NPZ/metrics.

See [baseline README](../README.md), [input formats](../DATA_FORMATS.md) and [source notices](../THIRD_PARTY_NOTICES.md).
