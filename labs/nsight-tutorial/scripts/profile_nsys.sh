#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_lab1_root
ensure_cuda
ensure_tool nsys

mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DATA_DIR="${DATA_DIR:-$LAB1_ROOT/data}"
REPORT_BASE="${REPORT_BASE:-$OUT_DIR/mnist_lab1_nsys}"

cd "$LAB1_ROOT"
nsys profile \
  --output "$REPORT_BASE" \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample none \
  --stats true \
  python -m src.train_mnist \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --data-dir "$DATA_DIR" \
    --model-path "$OUT_DIR/mnist_model_nsys.pth"

echo "Generated: ${REPORT_BASE}.nsys-rep"
echo "Quick summary: nsys stats ${REPORT_BASE}.nsys-rep"
