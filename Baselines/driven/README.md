# driven

Train with an explicit batch, e.g. `train --batch-size 128 --gpus 4 --seed 42 --run RUN`. Four CNNs precede LightGBM fusion and validation calibration. `resume-fusion --run RUN --gpus 4` continues fusion without retraining CNNs. `export --run RUN --gpus 4` exports test JSON after fitting.

See [baseline README](../README.md), [input formats](../DATA_FORMATS.md) and [source notices](../THIRD_PARTY_NOTICES.md).
