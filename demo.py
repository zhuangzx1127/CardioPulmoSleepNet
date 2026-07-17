"""Run CardioPulmoSleepNet inference on the bundled single-night example."""

import argparse
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEMO_FILE = PROJECT_ROOT / "data" / "demo_sample.npz"
DEFAULT_MODEL_FILE = PROJECT_ROOT / "model.py"
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "checkpoints" / "best_model" / "best_model"

EXPECTED_SHAPES = {
    "respi": (1200 * 1024,),
    "cardi": (1200 * 1024,),
    "oxi": (36000,),
    "ahi": (),
    "sleep": (1200, 4),
    "is_data": (1200,),
}


def load_model_class(model_path):
    """Load CardioPulmoSleepNet without requiring this directory as a package."""
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model definition not found: {model_path}")

    spec = importlib.util.spec_from_file_location("cardiopulmosleepnet_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load model definition: {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CardioPulmoSleepNet


def load_demo_data(demo_file):
    """Load and validate the arrays required by the public inference example."""
    demo_file = Path(demo_file).expanduser().resolve()
    if not demo_file.is_file():
        raise FileNotFoundError(f"Demo sample not found: {demo_file}")

    with np.load(demo_file, allow_pickle=False) as archive:
        missing = sorted(set(EXPECTED_SHAPES) - set(archive.files))
        if missing:
            raise ValueError(f"Demo sample is missing arrays: {', '.join(missing)}")
        arrays = {name: np.asarray(archive[name]) for name in EXPECTED_SHAPES}

    for name, expected_shape in EXPECTED_SHAPES.items():
        if arrays[name].shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: {arrays[name].shape}; "
                f"expected {expected_shape}"
            )
    return arrays


def run_demo(
    demo_file=DEFAULT_DEMO_FILE,
    model_file=DEFAULT_MODEL_FILE,
    weights_path=DEFAULT_WEIGHTS_PATH,
    precision="mixed_float16",
):
    """Run inference and return the four values used by the verifier."""
    tf.keras.mixed_precision.set_global_policy(precision)
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for index, gpu in enumerate(gpus):
            print(f"GPU {index}: {gpu}")
    else:
        print("No GPU detected by TensorFlow")

    arrays = load_demo_data(demo_file)
    print("====================")
    print("Input")
    print("====================")
    print("respi:", arrays["respi"].shape)
    print("cardi:", arrays["cardi"].shape)
    print("oxi:", arrays["oxi"].shape)
    print("ahi label:", float(arrays["ahi"]))
    print("sleep label:", arrays["sleep"].shape)
    print("is_data:", arrays["is_data"].shape)

    respi = arrays["respi"][None, ...]
    cardi = arrays["cardi"][None, ...]
    oxi = arrays["oxi"][None, ...]

    model_class = load_model_class(model_file)
    model = model_class()
    model(respi, cardi, oxi, training=False)

    weights_path = Path(weights_path).expanduser().resolve()
    if not weights_path.with_suffix(".index").is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    load_status = model.load_weights(str(weights_path))
    if hasattr(load_status, "assert_existing_objects_matched"):
        load_status.assert_existing_objects_matched()
    print("Model loaded")

    sleep_pred, ahi_pred, _ = model(respi, cardi, oxi, training=False)
    sleep_pred = sleep_pred.numpy()
    ahi_prediction = float(ahi_pred.numpy()[0])
    ahi_ground_truth = float(arrays["ahi"])

    sleep_pred_class = np.argmax(sleep_pred, axis=-1)[0]
    sleep_true_class = np.argmax(arrays["sleep"], axis=-1)
    valid_idx = arrays["is_data"].astype(bool)
    if not np.any(valid_idx):
        raise ValueError("The demo sample contains no valid sleep-stage epochs.")
    sleep_accuracy = float(
        np.mean(sleep_pred_class[valid_idx] == sleep_true_class[valid_idx])
    )

    result = {
        "ahi_prediction": ahi_prediction,
        "ahi_ground_truth": ahi_ground_truth,
        "sleep_output_shape": tuple(sleep_pred.shape),
        "sleep_staging_accuracy": sleep_accuracy,
    }

    print("====================")
    print("Result")
    print("====================")
    print("AHI prediction:", result["ahi_prediction"])
    print("AHI ground truth:", result["ahi_ground_truth"])
    print("Sleep output shape:", result["sleep_output_shape"])
    print("Sleep staging accuracy:", result["sleep_staging_accuracy"])
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="CardioPulmoSleepNet single-night inference demo"
    )
    parser.add_argument("--demo-file", type=Path, default=DEFAULT_DEMO_FILE)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--precision",
        choices=("mixed_float16", "float32"),
        default="mixed_float16",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_demo(
        demo_file=args.demo_file,
        model_file=args.model_file,
        weights_path=args.weights_path,
        precision=args.precision,
    )


if __name__ == "__main__":
    main()
