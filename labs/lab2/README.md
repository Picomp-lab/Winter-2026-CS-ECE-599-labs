# Lab 2 — Tensor & Pipeline Parallelism

This lab introduces two distributed training strategies:
- **Tensor parallelism** (native API and manual sharding)
- **Pipeline parallelism** (multi-stage execution with microbatches)

## 1) Reserve GPUs (Slurm)
On the submit node, request an interactive GPU allocation (example: 2 GPUs):

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:2 --mem=64G --pty bash
```

Then verify the GPUs:

```bash
nvidia-smi
hostname -f
```

## 2) Tensor Parallel — Native API
This example builds on the Lab 1 MNIST training loop, but shards the MLP layers
with PyTorch’s tensor-parallel API.

```bash
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

Notes:
- `--nproc_per_node` should match the number of GPUs you allocated.
- Requires a PyTorch build that includes `torch.distributed.tensor.parallel`.
- World size must divide the hidden size used in the model (512).
- Use `torchrun` (not plain `python`) so each GPU gets its own process and
  `RANK/LOCAL_RANK/WORLD_SIZE` are set automatically.

### Debugging steps (tensor parallel native)
Use these steps to isolate issues incrementally.

Step 1: Confirm you are on a GPU node and PyTorch can see CUDA.
```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY
```

Step 2: Sanity check the import for tensor-parallel API.
```bash
python - <<'PY'
try:
    from torch.distributed.tensor.parallel import ColwiseParallel
    print("tensor.parallel: OK")
except Exception as e:
    print("tensor.parallel import failed:", e)
PY
```

Step 3: Run with a single process (still using `torchrun`).
```bash
torchrun --standalone --nproc_per_node 1 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

Step 4: Run with two GPUs and explicit CUDA visibility.
```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

Step 5: If it hangs, enable debug logs.
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO \
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data
```

## 3) Tensor Parallel — Manual Sharding (Educational)
This version shows the mechanics explicitly: column-parallel first layer and
row-parallel second layer (with an `all_reduce` on outputs).

```bash
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_manual \
  --batch-size 64 --input-dim 1024 --hidden-dim 2048 --num-classes 10 --steps 5
```

Notes:
- This script intentionally uses synthetic data for clarity and speed.
- `--hidden-dim` must be divisible by the number of GPUs.

## 4) Pipeline Parallel (GPipe schedule)
This example uses `torch.distributed.pipelining` and the `ScheduleGPipe`
pipeline execution strategy.

```bash
torchrun --standalone --nproc_per_node 2 -m src.pipeline
```

Notes:
- The number of GPUs should match the `num_stages` in `src/pipeline.py`.
- You can edit `num_stages`, `n_microbatches`, and other constants in
  `src/pipeline.py` to experiment.

## 5) Troubleshooting
- Hangs usually mean one rank never reached a barrier or a mismatch in args.
- If a port is in use, set a different `MASTER_PORT`.
- If imports fail for `pipelining` or `tensor.parallel`, update PyTorch.
