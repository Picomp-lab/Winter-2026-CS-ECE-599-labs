import argparse
import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: MNIST model + forward")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    device = torch.device(args.device)
    train_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = MLP().to(device)
    images, _ = next(iter(train_loader))
    images = images.to(device)
    outputs = model(images)

    print(f"Images: {images.shape}")
    print(f"Outputs (logits): {outputs.shape}")


if __name__ == "__main__":
    main()
