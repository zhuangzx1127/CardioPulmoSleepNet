#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${INFOFU_DATA_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-}"; if [ "$#" -gt 0 ]; then shift; fi
SEED=42
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then SEED="$1"; shift; fi
RUN_DIR="$DATA_ROOT/Runs/baselines/resleepnet/seed_$SEED"
case "$MODE" in
  verify) OPTIONS=(--verify_cache) ;;
  check) OPTIONS=(--check_data) ;;
  smoke) RUN_DIR="${RUN_DIR}_smoke"; OPTIONS=(--n_epoch 1 --smoke_steps 2) ;;
  train) OPTIONS=() ;;
  evaluate) OPTIONS=(--evaluate) ;;
  *) printf 'Usage: bash run.sh [verify|check|smoke|train|evaluate] [seed=42] [arguments]\n' >&2; exit 2 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
exec "$PYTHON_BIN" "$SCRIPT_DIR/train.py" --cache_dir "$DATA_ROOT/TrainingCache" \
  --manifest "$DATA_ROOT/Dataset/splits.json" --run_dir "$RUN_DIR" \
  --seed "$SEED" "${OPTIONS[@]}" "$@"
