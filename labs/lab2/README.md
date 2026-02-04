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
This example uses PyTorch’s tensor-parallel API to shard the MLP layers.

```bash
torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --batch-size 64 --input-dim 1024 --hidden-dim 2048 --num-classes 10 --steps 5
```

Notes:
- `--nproc_per_node` should match the number of GPUs you allocated.
- Requires a PyTorch build that includes `torch.distributed.tensor.parallel`.

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
