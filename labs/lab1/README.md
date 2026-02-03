# Lab 1 — MNIST MLP Starter

This lab provides a minimal MNIST training + evaluation script you can run on CPU
or GPU. The goal is to verify your PyTorch setup and get comfortable with the
training loop before moving on.

## Files
- `src/train_mnist.py`: baseline MLP model with train/test routines.

## Quick start
From the repo root:

```bash
cd labs/lab1
python -m src.train_mnist --epochs 1
```

## Recap: reserve a GPU node (srun)
On the submit node, request an interactive GPU allocation:

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:1 --mem=64G --pty bash
```

Then verify the GPU and note the hostname:

```bash
nvidia-smi
hostname -f
```

When finished, exit the shell to release the allocation:

```bash
exit
```

## Activate the course conda environment
From the repo root (students can create it themselves the first time):

```bash
cd ~/hpc-share/Winter-2026-CS-ECE-599-labs
conda env create -f labs/lab1/env/environment.yml
conda activate csece599
```

## Useful options
- `--device cpu` or `--device cuda`
- `--batch-size 128`
- `--learning-rate 1e-3`
- `--data-dir ./data`
- `--model-path mnist_model.pth`

## Expected output
You should see loss every ~100 steps and a final test accuracy printout. Accuracy
will vary but should be well above random guessing after 1 epoch.
