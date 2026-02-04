import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class TPMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def setup_dist():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this example.")
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return device, rank, world_size


def parse_args():
    parser = argparse.ArgumentParser(description="Native tensor parallel MLP example")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device, rank, world_size = setup_dist()

    try:
        from torch.distributed.tensor.parallel import (
            ColwiseParallel,
            RowwiseParallel,
            parallelize_module,
        )
        from torch.distributed._tensor import DeviceMesh
    except Exception as exc:
        raise SystemExit(
            "This example requires torch.distributed.tensor.parallel "
            "(PyTorch 2.1+)."
        ) from exc

    if args.hidden_dim % world_size != 0:
        raise SystemExit("--hidden-dim must be divisible by world size.")

    torch.manual_seed(args.seed)
    model = TPMLP(args.input_dim, args.hidden_dim, args.num_classes).to(device)

    mesh = DeviceMesh("cuda", torch.arange(world_size))
    tp_plan = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    model = parallelize_module(model, mesh, tp_plan)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for step in range(1, args.steps + 1):
        if rank == 0:
            x = torch.randn(args.batch_size, args.input_dim, device=device)
            y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)
        else:
            x = torch.empty(args.batch_size, args.input_dim, device=device)
            y = torch.empty(args.batch_size, dtype=torch.long, device=device)
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)

        optimizer.zero_grad()
        logits = model(x)
        if hasattr(logits, "to_local"):
            logits = logits.to_local()
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
