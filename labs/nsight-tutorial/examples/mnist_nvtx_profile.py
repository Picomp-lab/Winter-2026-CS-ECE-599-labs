import argparse
import contextlib
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MLP(nn.Module):
    """Same MLP architecture used in lab1/src/train_mnist.py."""

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


@contextlib.contextmanager
def nvtx_range(name, enabled):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def build_train_loader(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def maybe_start_cuda_profiler(step, start_step, enabled, already_started):
    if enabled and (not already_started) and step >= start_step:
        torch.cuda.cudart().cudaProfilerStart()
        return True
    return already_started


def train(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but no CUDA device is available.")

    device = torch.device(args.device)
    use_nvtx = args.device == "cuda"

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = build_train_loader(args.data_dir, args.batch_size)

    model.train()
    global_step = 0
    prof_started = False
    profile_start_step = args.warmup_steps + 1
    profile_end_step = args.warmup_steps + args.profile_steps

    t0 = time.time()
    for epoch in range(args.epochs):
        for images, labels in train_loader:
            global_step += 1

            prof_started = maybe_start_cuda_profiler(
                global_step,
                profile_start_step,
                args.enable_cuda_profiler == 1 and args.device == "cuda",
                prof_started,
            )

            with nvtx_range("h2d", use_nvtx):
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with nvtx_range("forward", use_nvtx):
                outputs = model(images)
                loss = criterion(outputs, labels)

            with nvtx_range("backward", use_nvtx):
                optimizer.zero_grad(set_to_none=True)
                loss.backward()

            with nvtx_range("optimizer_step", use_nvtx):
                optimizer.step()

            if global_step % 20 == 0:
                print(f"step={global_step} loss={loss.item():.5f}")

            if global_step >= profile_end_step:
                break

        if global_step >= profile_end_step:
            break

    if args.device == "cuda":
        torch.cuda.synchronize()

    if prof_started and args.device == "cuda":
        torch.cuda.cudart().cudaProfilerStop()

    elapsed = time.time() - t0
    print(
        f"done steps={global_step} warmup_steps={args.warmup_steps} "
        f"profile_steps={args.profile_steps} elapsed_sec={elapsed:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="NVTX-instrumented MNIST training for Nsight demos"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--profile-steps", type=int, default=60)
    parser.add_argument(
        "--enable-cuda-profiler",
        type=int,
        default=1,
        choices=[0, 1],
        help="If 1, call cudaProfilerStart/Stop (useful for ncu --profile-from-start off).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
