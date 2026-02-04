# Lab 1 — MNIST + DDP Starter

## 1) Overview
This lab verifies your PyTorch setup and introduces both single-GPU training
(MNIST) and multi-GPU distributed data parallel (DDP) training. You will:
- Run a basic MNIST MLP training loop.
- Reserve a GPU node via Slurm.
- Create and use a course conda environment.
- Run a DDP training example on the Pokemon dataset.

Files you will use:
- `src/train_mnist.py`: baseline MNIST MLP training and evaluation.
- `src/ddp.py`: DDP example (as-is from prior lab2).
- `src/ddp.py`: cleaned DDP example with a tidy CLI.
- `env/environment.yml`: conda env definition.
- `env/requirements.txt`: pip packages installed into the env.

## 2) GPU reserve (recap), conda env setup and vs code debugging
### 2.1 Reserve a GPU node (srun)
On the submit node, request an interactive GPU allocation:

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:1 --mem=64G --pty bash
```

Then verify the GPU and note the hostname:

```bash
nvidia-smi
hostname -f
```

When finished, exit the shell to release the allocation:

```bash
exit
```

### 2.2 Create and activate the conda env
From your github pull:

```bash
cd ./Winter-2026-CS-ECE-599-labs/labs/lab1
conda env create -f env/environment.yml
conda activate csece599
pip install -r env/requirements.txt
```

What the env files do:
- `env/environment.yml` defines the base conda env (Python version + pip).
- `env/requirements.txt` lists pip packages (e.g., `torch`, `torchvision`, `datasets`).

### 2.3 VS Code debugging tutorial

https://code.visualstudio.com/docs/debugtest/debugging#_data-inspection

## 3) MNIST steps + tutorial
### 3.1 Run MNIST training
From your github pull:

```bash
python -m src.train_mnist --epochs 100
```

Useful options:
- `--device cpu` or `--device cuda`
- `--batch-size 128`
- `--learning-rate 1e-3`
- `--data-dir ./data`
- `--model-path mnist_model.pth`

Expected output:
You should see loss every ~100 steps and a final test accuracy printout. Accuracy
will vary but should be well above random guessing after 10 epoch.

### 3.2 Step-by-step MNIST walkthrough (optional) - if you are new to pytorch
These scripts split `train_mnist.py` into small, instructional steps. Run them
in order:

```bash
python -m src.step1_data
python -m src.step2_model
python -m src.step3_train --epochs 10 --model-path mnist_model.pth
python -m src.step4_eval --model-path mnist_model.pth
```

Notes:
- `step3_train` saves the model weights; `step4_eval` loads and evaluates them.
- You can pass `--device cpu` or `--device cuda` to steps 2–4 if needed.

## 4) DDP steps + tutorial
### 4.1 Run DDP training (Pokemon dataset)
This section adds a multi-GPU example using the
`keremberke/pokemon-classification` dataset from Hugging Face.


```bash
python -m src.ddp --total_epochs 2 --batch_size 64
```

Run the cleaned version:

```bash
python -m src.ddp --total-epochs 2 --batch-size 64
```

Notes:
- Requires multiple GPUs; `torch.cuda.device_count()` controls DDP world size.
- The dataset will download on first run.

### 4.2 DDP tutorial guidance (PyTorch)
Recommended reading and key points to keep in mind:
- DDP uses one process per GPU; each process has its own model replica and synchronizes gradients during backward.
- DDP is generally faster than `DataParallel` and supports multi-machine training.
- DDP requires setting up a process group (e.g., `MASTER_ADDR`, `MASTER_PORT`) and using `mp.spawn` to launch processes.
- All ranks must hit synchronization points in the same order; skewed workers can cause timeouts.

Tutorial link:
```text
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
```
