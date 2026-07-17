"""Verify the bundled demo against the archived reference result."""

import argparse
from pathlib import Path

import numpy as np

from demo import (
    DEFAULT_DEMO_FILE,
    DEFAULT_MODEL_FILE,
    DEFAULT_WEIGHTS_PATH,
    run_demo,
)


REFERENCE = {
    "ahi_prediction": 7.236681938171387,
    "ahi_ground_truth": 8.2421875,
    "sleep_output_shape": (1, 1200, 4),
    "sleep_staging_accuracy": 0.8312428734321551,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify the public inference demo")
    parser.add_argument("--demo-file", type=Path, default=DEFAULT_DEMO_FILE)
    parser.add_argument("--model-file", type=Path, default=DEFAULT_MODEL_FILE)
    parser.add_argument("--weights-path", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument(
        "--precision",
        choices=("mixed_float16", "float32"),
        default="mixed_float16",
    )
    parser.add_argument(
        "--ahi-atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for small device-dependent AHI differences.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    actual = run_demo(
        demo_file=args.demo_file,
        model_file=args.model_file,
        weights_path=args.weights_path,
        precision=args.precision,
    )

    checks = {
        "AHI prediction": np.isclose(
            actual["ahi_prediction"],
            REFERENCE["ahi_prediction"],
            rtol=0.0,
            atol=args.ahi_atol,
        ),
        "AHI ground truth": actual["ahi_ground_truth"]
        == REFERENCE["ahi_ground_truth"],
        "sleep output shape": actual["sleep_output_shape"]
        == REFERENCE["sleep_output_shape"],
        "sleep staging accuracy": np.isclose(
            actual["sleep_staging_accuracy"],
            REFERENCE["sleep_staging_accuracy"],
            rtol=0.0,
            atol=1e-12,
        ),
    }

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Verification failed: " + ", ".join(failed))

    difference = abs(actual["ahi_prediction"] - REFERENCE["ahi_prediction"])
    print("====================")
    print("Verification: PASS")
    print(f"AHI absolute difference: {difference:.10f} (tolerance {args.ahi_atol:g})")


if __name__ == "__main__":
    main()
