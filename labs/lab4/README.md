# Lab 4 - PyTorch Profiler and TensorBoard

## 1) Overview
This lab uses two notebooks:
- `pytorch_profiler.ipynb` (from `shwgao/CS-ECE-599-labs/lab4`)
- `tensorboard.ipynb` (from `shwgao/CS-ECE-599-labs/lab5`)

Goal:
- Profile PyTorch training with `torch.profiler`.
- Visualize traces, model graphs, images, embeddings, and training curves in TensorBoard.

## 2) Environment Setup
From repo root, reserve a GPU node (interactive):

```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgxh --gres=gpu:1 --mem=32G --pty bash
```

Then:

```bash
cd labs/lab4
conda activate csece599
pip install -U jupyter tensorboard torch_tb_profiler matplotlib
```

Optional quick check:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
PY
```

## 3) Run the Notebooks
Open in VS Code/Jupyter and run cells in order:
- `pytorch_profiler.ipynb`
- `tensorboard.ipynb`

If you prefer launching Jupyter manually:

```bash
jupyter lab --no-browser --port 8888
```

## 4) Notebook A - PyTorch Profiler
In `pytorch_profiler.ipynb`:
- CIFAR-10 is downloaded to `./data`.
- A `resnet18` model is profiled.
- Trace logs are written to `./log/resnet18`.

Visualize profiler traces:

```bash
tensorboard --logdir ./log --port 6006
```

Open the TensorBoard URL and check the **PyTorch Profiler** tab.

## 5) Notebook B - TensorBoard
In `tensorboard.ipynb`:
- CIFAR-10 is downloaded under `../lab4/data`.
- TensorBoard logs are written to `runs/cifar10_experiment_1`.
- The notebook logs images, model graph, embeddings, and training metrics.

Visualize TensorBoard outputs:

```bash
tensorboard --logdir ./runs --port 6006
```

## 6) Suggested Deliverables
- Screenshot of profiler timeline/operator view from `./log/resnet18`.
- Screenshot(s) from TensorBoard (`Images`, `Graphs`, `Projector`, `Scalars`).
- Short comparison note for model performance tuning:
  - Increase `batch_size`.
  - Increase `num_workers`.
  - Optionally test another model (for example `vgg16`) and compare with `resnet18`.

## 7) Troubleshooting
- `ModuleNotFoundError: torch_tb_profiler`:
  - Install with `pip install torch_tb_profiler`.
- TensorBoard shows no data:
  - Confirm `--logdir` matches `./log` or `./runs`.
  - Re-run notebook cells that write logs.
- CUDA unavailable:
  - Re-check Slurm GPU request and `torch.cuda.is_available()`.
