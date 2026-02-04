import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
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


def train(model, train_loader, device, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for step, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if step % 100 == 0:
                avg_loss = running_loss / 100
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{step}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}"
                )
                running_loss = 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Step 3: MNIST training loop")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-path", type=str, default="mnist_model.pth")
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
    train(model, train_loader, device, args.epochs, args.learning_rate)

    torch.save(model.state_dict(), args.model_path)
    print(f"Saved model to {args.model_path}")


if __name__ == "__main__":
    main()
