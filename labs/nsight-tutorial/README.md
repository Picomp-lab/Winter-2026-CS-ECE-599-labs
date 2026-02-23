# CSECE 599 Nsight Tutorial (Lab1 DNN Example)

This repo is a ready-to-teach tutorial for profiling the **Lab1 MNIST MLP** workload with:
- **Nsight Systems (`nsys`)**: timeline and CUDA API/kernel traces
- **Nsight Compute (`ncu`)**: kernel-level metrics

## 1) What this repo uses from lab1

Original lab workload:
- `labs/lab1/src/train_mnist.py`

This repo also includes:
- `examples/mnist_nvtx_profile.py`: same MNIST MLP structure as lab1, with NVTX ranges (`h2d`, `forward`, `backward`, `optimizer_step`) to make timeline demos cleaner.

## 2) Prereqs

Run on a GPU node and activate your class env:

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:1 --mem=64G --pty bash
source ~/.bashrc
conda activate csece599
```

Check tools:

```bash
which nsys
which ncu
nvidia-smi
```

## 3) Baseline run (no profiler)

```bash
cd /nfs/hpc/share/dongwenq/Winter-2026-CS-ECE-599-labs/labs/nsight-tutorial
bash scripts/run_baseline.sh
```

## 4) Nsight Systems examples

### A) Profile original lab1 script

```bash
bash scripts/profile_nsys.sh
```

Outputs:
- `reports/mnist_lab1_nsys.nsys-rep`

Quick terminal summary:

```bash
nsys stats reports/mnist_lab1_nsys.nsys-rep
```

### B) Profile NVTX-instrumented script (better for teaching)

```bash
bash scripts/profile_nsys_nvtx.sh
```

Outputs:
- `reports/mnist_nvtx_nsys.nsys-rep`

## 5) Nsight Compute example

```bash
bash scripts/profile_ncu.sh
```

Outputs:
- `reports/mnist_nvtx_ncu.ncu-rep`

Default metric set is lightweight (`launchStats`).
For deeper analysis:

```bash
NCU_SET=full bash scripts/profile_ncu.sh
```

## 6) Open reports

If GUI is available:

```bash
nsys-ui reports/mnist_nvtx_nsys.nsys-rep
ncu-ui reports/mnist_nvtx_ncu.ncu-rep
```

If GUI is not available, use CLI summaries:

```bash
nsys stats reports/mnist_nvtx_nsys.nsys-rep
ncu --import reports/mnist_nvtx_ncu.ncu-rep --page details
```

## 7) Useful knobs for class demos

Environment overrides:
- `EPOCHS` (default `1`)
- `BATCH_SIZE` (default `256`)
- `WARMUP_STEPS` (default `20`)
- `PROFILE_STEPS` (default `60`)
- `OUT_DIR` (default `./reports`)
- `LAB1_ROOT` (auto-detected, override if needed)

Examples:

```bash
EPOCHS=1 BATCH_SIZE=512 bash scripts/profile_nsys.sh
WARMUP_STEPS=10 PROFILE_STEPS=40 NCU_SET=full bash scripts/profile_ncu.sh
```

## 8) Slurm batch examples

```bash
sbatch scripts/slurm_nsys.sh
sbatch scripts/slurm_ncu.sh
```
