#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_cuda
ensure_tool nsys

mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-80}"
REPORT_BASE="${REPORT_BASE:-$OUT_DIR/mnist_nvtx_nsys}"

python "$REPO_ROOT/examples/mnist_nvtx_profile.py" --help >/dev/null

nsys profile \
  --output "$REPORT_BASE" \
  --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample none \
  --stats true \
  python "$REPO_ROOT/examples/mnist_nvtx_profile.py" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --profile-steps "$PROFILE_STEPS" \
    --enable-cuda-profiler 0

echo "Generated: ${REPORT_BASE}.nsys-rep"
