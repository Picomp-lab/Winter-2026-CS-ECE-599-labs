#!/bin/bash
#SBATCH --job-name=csece599
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "Running on $(hostname)"
nvidia-smi || true

# Example:
# python labs/lab0/handout/main.py
