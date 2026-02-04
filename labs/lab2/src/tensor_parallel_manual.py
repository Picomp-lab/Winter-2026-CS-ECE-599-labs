import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed=42):
        super().__init__()
        if out_features % world_size != 0:
            raise ValueError("out_features must be divisible by world_size")
        self.out_per_rank = out_features // world_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        self.bias = nn.Parameter(torch.empty(self.out_per_rank))
        self._init_from_global(in_features, out_features, world_size, rank, seed)

    def _init_from_global(self, in_features, out_features, world_size, rank, seed):
        if rank == 0:
            torch.manual_seed(seed)
            weight_full = torch.randn(out_features, in_features, device=self.weight.device) * 0.02
            bias_full = torch.zeros(out_features, device=self.weight.device)
        else:
            weight_full = torch.empty(out_features, in_features, device=self.weight.device)
            bias_full = torch.empty(out_features, device=self.weight.device)
        dist.broadcast(weight_full, src=0)
        dist.broadcast(bias_full, src=0)
        self.weight.data.copy_(torch.chunk(weight_full, world_size, dim=0)[rank])
        self.bias.data.copy_(torch.chunk(bias_full, world_size, dim=0)[rank])

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank, seed=42):
        super().__init__()
        if in_features % world_size != 0:
            raise ValueError("in_features must be divisible by world_size")
        self.in_per_rank = in_features // world_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        self.bias = nn.Parameter(torch.empty(out_features))
        self._init_from_global(in_features, out_features, world_size, rank, seed)

    def _init_from_global(self, in_features, out_features, world_size, rank, seed):
        if rank == 0:
            torch.manual_seed(seed + 1)
            weight_full = torch.randn(out_features, in_features, device=self.weight.device) * 0.02
            bias_full = torch.zeros(out_features, device=self.weight.device)
        else:
            weight_full = torch.empty(out_features, in_features, device=self.weight.device)
            bias_full = torch.empty(out_features, device=self.weight.device)
        dist.broadcast(weight_full, src=0)
        dist.broadcast(bias_full, src=0)
        self.weight.data.copy_(torch.chunk(weight_full, world_size, dim=1)[rank])
        self.bias.data.copy_(bias_full)

    def forward(self, x_shard):
        # Each rank computes a partial output; sum across ranks.
        partial = F.linear(x_shard, self.weight, None)
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial + self.bias


class ManualTPMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, world_size, rank, seed=42):
        super().__init__()
        self.fc1 = ColumnParallelLinear(in_dim, hidden_dim, world_size, rank, seed)
        self.act = nn.ReLU()
        self.fc2 = RowParallelLinear(hidden_dim, out_dim, world_size, rank, seed)

    def forward(self, x):
        h_shard = self.fc1(x)
        h_shard = self.act(h_shard)
        return self.fc2(h_shard)


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
    parser = argparse.ArgumentParser(description="Manual tensor parallel MLP example")
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

    if args.hidden_dim % world_size != 0:
        raise SystemExit("--hidden-dim must be divisible by world size.")

    model = ManualTPMLP(
        args.input_dim, args.hidden_dim, args.num_classes, world_size, rank, args.seed
    ).to(device)
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
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"Step {step}/{args.steps} | Loss: {loss.item():.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
