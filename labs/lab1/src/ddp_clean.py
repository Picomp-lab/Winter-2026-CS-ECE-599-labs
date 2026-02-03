import argparse
import os
import time

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.models as models
from datasets import load_dataset
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


class PokemonDataset(Dataset):
    def __init__(self, split="train", config="full"):
        self.dataset = load_dataset("keremberke/pokemon-classification", config)[split]
        self.num_classes = len(self.dataset.features["labels"].names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        image_tensor = transform(image)
        label = torch.tensor(sample["labels"], dtype=torch.long)
        return image_tensor, label


class Trainer:
    def __init__(self, model, train_data, optimizer, gpu_id, save_every, test_every):
        self.gpu_id = gpu_id
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id])
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.test_every = test_every
        self.step_every = 10
        self.accuracy = 0.0

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        batch_size = len(next(iter(self.train_data))[0])
        self.train_data.sampler.set_epoch(epoch)
        for step, (source, targets) in enumerate(self.train_data, start=1):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            if self.gpu_id == 0 and step % self.step_every == 0:
                print(
                    f"[GPU{self.gpu_id}] Epoch {epoch} | "
                    f"Batchsize: {batch_size} | "
                    f"Steps:{step}/{len(self.train_data)} | "
                    f"Loss: {loss:.4f} | Accuracy: {self.accuracy:.4f}"
                )

    def _save_checkpoint(self, epoch, path):
        torch.save(self.model.module.state_dict(), path)
        print(f"Epoch {epoch} | Training checkpoint saved at {path}")

    def _measure_accuracy(self, eval_data):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in eval_data:
                images = images.to(self.gpu_id)
                labels = labels.to(self.gpu_id)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def train(self, max_epochs, checkpoint_path):
        for epoch in range(max_epochs):
            start_time = time.time()
            self._run_epoch(epoch)
            end_time = time.time()
            if self.gpu_id == 0:
                print(f"Epoch {epoch} took {end_time - start_time:.2f} seconds")
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch, checkpoint_path)
            if self.gpu_id == 0 and epoch % self.test_every == 0:
                self.accuracy = self._measure_accuracy(self.train_data)


def ddp_setup(rank, world_size, master_addr, master_port):
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def load_train_objs():
    train_set = PokemonDataset("train", "full")
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, train_set.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    return train_set, model, optimizer


def prepare_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset),
    )


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.master_addr, args.master_port)
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, args.batch_size)
    trainer = Trainer(model, train_data, optimizer, rank, args.save_every, args.test_every)
    trainer.train(args.total_epochs, args.checkpoint_path)
    destroy_process_group()


def parse_args():
    parser = argparse.ArgumentParser(description="DDP training on Pokemon dataset")
    parser.add_argument("--total-epochs", default=10, type=int)
    parser.add_argument("--save-every", default=100, type=int)
    parser.add_argument("--test-every", default=2, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--checkpoint-path", default="checkpoint.pt", type=str)
    parser.add_argument("--master-addr", default="localhost", type=str)
    parser.add_argument("--master-port", default=12355, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
