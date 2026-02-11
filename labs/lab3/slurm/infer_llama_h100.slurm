#!/bin/bash
#SBATCH -A eecs
#SBATCH -J lab3-llama-h100
#SBATCH -p gpu,dgxh
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -t 0-01:00:00
#SBATCH -o slurm/%x-%j.out

set -euo pipefail

# Run this script from repo root:
# sbatch labs/lab3/slurm/infer_llama_h100.slurm
cd "${SLURM_SUBMIT_DIR}/labs/lab3"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csece599

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

python -m src.llama_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype bfloat16 \
  --max-new-tokens 128 \
  --prompt "Summarize data parallelism vs model parallelism in 6 short bullets."
