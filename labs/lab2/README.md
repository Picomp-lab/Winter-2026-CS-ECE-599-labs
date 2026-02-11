# Lab 2 — Tensor and Pipeline Parallelism

## 1) Overview
This lab introduces two model-parallel training strategies:
- Tensor parallelism:
  - Native PyTorch tensor-parallel APIs (`src/tensor_parallel_native.py`)
  - Manual sharding implementation (`src/tensor_parallel_manual.py`)
- Pipeline parallelism with GPipe schedule (`src/pipeline.py`)

By the end of this lab, you should be able to launch multi-process GPU jobs
with `torchrun`, understand how model layers are split across ranks, and debug
common distributed failures.

Files you will use:
- `src/tensor_parallel_native.py`
- `src/tensor_parallel_manual.py`
- `src/pipeline.py`

## 2) Setup and GPU allocation
From your repo root:

```bash
cd labs/lab2
```

Use the same environment from Lab 1 (`csece599`). If needed:

```bash
conda activate csece599
```

Reserve an interactive GPU allocation (example: 2 GPUs):

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:2 --mem=64G --pty bash
```

Sanity-check your node:

```bash
nvidia-smi
hostname -f
```

## 3) Tensor parallelism (native API)
This script extends the Lab 1 MNIST MLP and parallelizes `fc1`/`fc2` with
`torch.distributed.tensor.parallel`.

Run:

```bash
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

Important notes:
- Always launch with `torchrun` (not plain `python`) so
  `RANK/LOCAL_RANK/WORLD_SIZE` are configured.
- `--nproc_per_node` must match the GPUs you intend to use.
- This script requires CUDA and NCCL.
- World size must divide hidden size `512` in the current model.
- First run may download MNIST on rank 0.

Debug sequence:

1. CUDA visibility:
```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY
```

2. Tensor-parallel import:
```bash
python - <<'PY'
try:
    from torch.distributed.tensor.parallel import ColwiseParallel
    print("tensor.parallel: OK")
except Exception as e:
    print("tensor.parallel import failed:", e)
PY
```

3. Single process smoke test:
```bash
torchrun --standalone --nproc_per_node 1 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

4. Two-process run with explicit device visibility:
```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

5. Verbose distributed logs:
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

## 4) Tensor parallelism (manual sharding)
This educational version implements:
- Column-parallel first linear layer
- Row-parallel second linear layer
- `all_reduce` to aggregate partial outputs

Run:

```bash
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_manual \
  --batch-size 64 --input-dim 1024 --hidden-dim 2048 --num-classes 10 --steps 5
```

Important notes:
- Uses synthetic data (no dataset download).
- `--hidden-dim` must be divisible by world size.
- Use this script to understand communication patterns before using larger models.

## 5) Pipeline parallelism (GPipe)
This script uses `torch.distributed.pipelining` with `ScheduleGPipe`.

Run:

```bash
torchrun --standalone --nproc_per_node 2 -m src.pipeline
```

Important notes tied to current `src/pipeline.py`:
- `num_stages = 2` is hardcoded; run with exactly 2 processes unless you edit it.
- `n_microbatches`, `batch_size`, and `num_iterations` are constants in code.
- Stage partitioning is currently split half-and-half across transformer layers.
- Rank 0 feeds token IDs; the second stage expects hidden-state tensors.

## 6) Common failure modes
- Hang at startup:
  - Usually one rank crashed early or did not reach a collective call.
  - Re-run with `TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO`.
- NCCL/init error:
  - Confirm you are on GPU nodes and CUDA devices are visible.
- Port collision:
  - Set a different master port, e.g.
    `MASTER_PORT=29511 torchrun --standalone ...`
- Shape mismatch in pipeline:
  - Ensure stage partition and `example_input_microbatch` agree with stage inputs.
- Import error for tensor/pipeline APIs:
  - Use a PyTorch build that includes `torch.distributed.tensor.parallel` and
    `torch.distributed.pipelining`.

## 7) After you finish
Post your progress or questions in the course discussions:
https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs/discussions
