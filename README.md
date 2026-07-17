# CardioPulmoSleepNet

This repository contains the public inference example associated with
*From cardiopulmonary signals to sleep phenotypes: a personalized low-burden
framework for longitudinal sleep-health monitoring*.

The example loads a pretrained CardioPulmoSleepNet checkpoint and predicts
four-class sleep staging and a night-level apnea-hypopnea index (AHI) from a
de-identified single-night sample. Training code, training configurations, and
clinical datasets are intentionally not included.

## Repository contents

```text
.
├── checkpoints/best_model/  # TensorFlow checkpoint used by the demo
├── data/demo_sample.npz     # De-identified single-night example
├── demo.py                  # Inference entry point
├── model.py                 # Model definition required for inference
├── verify_demo.py           # Reference-result regression check
├── requirements.txt         # Reproduced Python dependencies
├── environment.yml          # Optional Conda environment
└── SHA256SUMS               # Integrity hashes for model/data artifacts
```

## Environment

The archived result was reproduced with Python 3.8.19, TensorFlow/Keras 2.9.0,
and NumPy 1.24.3. On Apple Silicon, the requirements install
`tensorflow-macos`; other supported platforms install `tensorflow`.

Using Conda:

```bash
conda env create -f environment.yml
conda activate cardiopulmosleepnet-demo
```

Alternatively, create a Python 3.8 virtual environment and run:

```bash
python -m pip install -r requirements.txt
```

## Run the demo

The paths are resolved relative to `demo.py`, so this command can be launched
from any working directory:

```bash
python demo.py
```

Expected result:

```text
AHI prediction: approximately 7.237
AHI ground truth: 8.2421875
Sleep output shape: (1, 1200, 4)
Sleep staging accuracy: 0.8312428734321551
```

TensorFlow kernels and mixed-precision execution can produce a small
device-dependent difference in the last AHI digits. Run the regression check to
validate the result with an absolute AHI tolerance of `1e-3`:

```bash
python verify_demo.py
```

The sleep output shape must match exactly; accuracy uses a numerical tolerance
of `1e-12`.

To verify that the downloaded model and sample files are intact:

```bash
shasum -a 256 -c SHA256SUMS
```

## Input format

`demo.py` expects an unpickled NumPy `.npz` archive with these arrays:

| Key | Shape | Description |
| --- | ---: | --- |
| `respi` | `(1228800,)` | Respiratory signal, 1,200 epochs × 1,024 samples |
| `cardi` | `(1228800,)` | Cardiac signal, 1,200 epochs × 1,024 samples |
| `oxi` | `(36000,)` | Oximetry signal, 1,200 epochs × 30 samples |
| `ahi` | scalar | Reference night-level AHI |
| `sleep` | `(1200, 4)` | One-hot four-class sleep-stage labels |
| `is_data` | `(1200,)` | Valid-epoch mask |

For a different sample or checkpoint, use `python demo.py --help`.

## Scope

This release is an inference-only research artifact. It is not a medical device
and must not be used for clinical diagnosis or treatment decisions. The bundled
example contains no explicit identifier field; authorization to redistribute
physiological recordings must still be confirmed by the data owner before a
public release.
