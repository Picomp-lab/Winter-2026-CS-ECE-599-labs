#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_lab1_root
ensure_cuda

mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA_DIR="${DATA_DIR:-$LAB1_ROOT/data}"
MODEL_PATH="${MODEL_PATH:-$OUT_DIR/mnist_model_baseline.pth}"

cd "$LAB1_ROOT"
python -m src.train_mnist \
  --device cuda \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --data-dir "$DATA_DIR" \
  --model-path "$MODEL_PATH"
