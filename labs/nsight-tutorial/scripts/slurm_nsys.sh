#!/usr/bin/env bash
#SBATCH -A eecs
#SBATCH -p gpu,dgx2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=nsys-mnist
#SBATCH --output=nsys-%j.out

set -euo pipefail

# Example activation. Update if your shell init differs.
source ~/.bashrc
conda activate csece599

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
bash scripts/profile_nsys_nvtx.sh
