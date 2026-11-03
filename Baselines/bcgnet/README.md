# bcgnet

Train `--task stage` before `--task events --stage-run STAGE_RUN`. Use `python`, not torchrun; `--gpus 4` enables DataParallel. Events train exports test JSON by default. `export --task events --run EVENT_RUN --stage-run STAGE_RUN` regenerates results. Explicit `--ihr-zscore` and microbatch flags are saved in protocol.

See [baseline README](../README.md), [input formats](../DATA_FORMATS.md) and [source notices](../THIRD_PARTY_NOTICES.md).
