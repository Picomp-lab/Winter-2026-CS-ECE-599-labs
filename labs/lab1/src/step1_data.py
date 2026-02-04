import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def build_dataloaders(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Step 1: MNIST data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    train_loader, test_loader = build_dataloaders(args.data_dir, args.batch_size)

    images, labels = next(iter(train_loader))
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    print(f"Batch images: {images.shape} | Batch labels: {labels.shape}")
    print(f"Label sample: {labels[:10].tolist()}")


if __name__ == "__main__":
    main()
