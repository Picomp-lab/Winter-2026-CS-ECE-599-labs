#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

ensure_cuda
ensure_tool ncu

mkdir -p "$OUT_DIR"

EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WARMUP_STEPS="${WARMUP_STEPS:-20}"
PROFILE_STEPS="${PROFILE_STEPS:-60}"
NCU_SET="${NCU_SET:-launchStats}"
REPORT_BASE="${REPORT_BASE:-$OUT_DIR/mnist_nvtx_ncu}"

python "$REPO_ROOT/examples/mnist_nvtx_profile.py" --help >/dev/null

ncu \
  --set "$NCU_SET" \
  --target-processes all \
  --profile-from-start off \
  --force-overwrite \
  --output "$REPORT_BASE" \
  python "$REPO_ROOT/examples/mnist_nvtx_profile.py" \
    --device cuda \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --warmup-steps "$WARMUP_STEPS" \
    --profile-steps "$PROFILE_STEPS" \
    --enable-cuda-profiler 1

echo "Generated: ${REPORT_BASE}.ncu-rep"
