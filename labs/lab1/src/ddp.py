import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)


def build_dataloader(data_dir, batch_size, rank, world_size):
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

    sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False
    )
    return train_loader


def train(model, train_loader, device, epochs, learning_rate, rank):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_loader.sampler.set_epoch(epoch)
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if rank == 0 and step % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"[GPU{rank}] Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{step}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0


def ddp_setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.master_addr, args.master_port)
    torch.manual_seed(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device(f"cuda:{rank}")

    train_loader = build_dataloader(args.data_dir, args.batch_size, rank, world_size)
    model = MLP().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    train(ddp_model, train_loader, device, args.epochs, args.learning_rate, rank)

    if rank == 0:
        torch.save(ddp_model.module.state_dict(), args.model_path)
        print(f"Saved model to {args.model_path}")

    destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training on MNIST (MLP)")
    parser.add_argument(
        "--epochs",
        "--total-epochs",
        "--total_epochs",
        dest="epochs",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--learning-rate",
        "--learning_rate",
        dest="learning_rate",
        default=1e-3,
        type=float,
    )
    parser.add_argument("--data-dir", default="./data", type=str)
    parser.add_argument(
        "--model-path",
        "--checkpoint-path",
        dest="model_path",
        default="mnist_ddp.pth",
        type=str,
    )
    parser.add_argument("--master-addr", default="localhost", type=str)
    parser.add_argument("--master-port", default=12355, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--gpus",
        default=0,
        type=int,
        help="Number of GPUs to use (0 = all available).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA devices found. Request a GPU node or run a GPU-equipped machine."
        )

    available_gpus = torch.cuda.device_count()
    if args.gpus < 0:
        raise SystemExit("--gpus must be >= 0.")
    if args.gpus == 0:
        world_size = available_gpus
    else:
        if args.gpus > available_gpus:
            raise SystemExit(
                f"Requested {args.gpus} GPUs but only {available_gpus} are available."
            )
        world_size = args.gpus

    mp.spawn(main, args=(world_size, args), nprocs=world_size)
