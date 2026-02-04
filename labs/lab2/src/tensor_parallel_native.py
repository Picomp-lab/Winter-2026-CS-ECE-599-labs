import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.flatten(x)
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


def build_dataloader(data_dir, batch_size, rank):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    if rank == 0:
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
        dist.barrier()
    else:
        dist.barrier()
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=False, transform=transform
        )

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Native tensor parallel MNIST MLP (builds on Lab 1 training)"
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
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

    if 512 % world_size != 0:
        raise SystemExit("World size must divide 512 for this MLP.")

    torch.manual_seed(args.seed)
    train_loader = build_dataloader(args.data_dir, args.batch_size, rank)

    model = MLP().to(device)
    mesh = DeviceMesh("cuda", torch.arange(world_size))
    tp_plan = {
        "fc1": ColwiseParallel(),
        "fc2": RowwiseParallel(),
    }
    model = parallelize_module(model, mesh, tp_plan)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    steps_per_epoch = len(train_loader) if rank == 0 else 0
    steps_tensor = torch.tensor([steps_per_epoch], device=device)
    dist.broadcast(steps_tensor, src=0)
    steps_per_epoch = int(steps_tensor.item())

    for epoch in range(args.epochs):
        running_loss = 0.0
        if rank == 0:
            loader_iter = iter(train_loader)
        for step in range(1, steps_per_epoch + 1):
            if rank == 0:
                images, labels = next(loader_iter)
                images = images.to(device)
                labels = labels.to(device)
            else:
                images = torch.empty(
                    args.batch_size, 1, 28, 28, device=device, dtype=torch.float
                )
                labels = torch.empty(
                    args.batch_size, device=device, dtype=torch.long
                )
            dist.broadcast(images, src=0)
            dist.broadcast(labels, src=0)

            optimizer.zero_grad()
            logits = model(images)
            if hasattr(logits, "to_local"):
                logits = logits.to_local()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0 and step % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}], "
                    f"Step [{step}/{steps_per_epoch}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
