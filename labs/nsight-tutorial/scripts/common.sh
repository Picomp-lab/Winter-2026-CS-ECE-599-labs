#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_LAB1_ROOT="$(cd "$REPO_ROOT/../lab1" 2>/dev/null && pwd || true)"

LAB1_ROOT="${LAB1_ROOT:-$DEFAULT_LAB1_ROOT}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/reports}"

ensure_lab1_root() {
  if [[ -z "${LAB1_ROOT}" || ! -f "$LAB1_ROOT/src/train_mnist.py" ]]; then
    echo "ERROR: LAB1_ROOT is not set correctly."
    echo "Set LAB1_ROOT to your lab1 path, e.g.:"
    echo "  export LAB1_ROOT=/path/to/Winter-2026-CS-ECE-599-labs/labs/lab1"
    exit 1
  fi
}

ensure_cuda() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. Run this on a GPU node."
    exit 1
  fi
}

ensure_tool() {
  local tool="$1"
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: $tool not found in PATH."
    exit 1
  fi
}
