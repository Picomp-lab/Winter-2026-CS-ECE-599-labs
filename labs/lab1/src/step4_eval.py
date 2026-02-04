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

    test_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test set: {accuracy:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Step 4: MNIST evaluation")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="mnist_model.pth")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.model_path):
        raise SystemExit(
            f"Model not found at {args.model_path}. "
            "Run step3_train.py first to create it."
        )

    device = torch.device(args.device)
    test_loader = build_dataloaders(args.data_dir, args.batch_size)

    model = MLP().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    test(model, test_loader, device)


if __name__ == "__main__":
    main()
