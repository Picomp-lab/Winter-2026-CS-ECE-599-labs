# Winter-2026-CS-ECE-599-labs

Labs for **CS/ECE 599 (Winter 2026)** — Advanced Topics in Deep Learning & High-performance Computing.

## Repo Structure
- `labs/lab0`: HPC + VS Code onboarding and setup docs
- `labs/lab1`: MNIST walkthrough and DistributedDataParallel baseline
- `labs/lab2`: tensor parallelism and pipeline parallelism tutorials
- `labs/lab3`: 2-GPU LLM inference (Hugging Face + vLLM tracks)
- `labs/lab4`: PyTorch Profiler + TensorBoard workflow
- `labs/lab5`: follow-up TensorBoard lab materials

## Getting Started
1. Read setup docs in `labs/lab0/doc`.
2. Complete `labs/lab0/README.md`.
3. Run through `labs/lab1/README.md` to validate your environment.
4. Continue with Lab 2, Lab 3, and Lab 4 from the sections below.

## Lab 0 Setup: HPC + VS Code Remote SSH
Lab 0 is the cluster onboarding lab. It covers:
- COE HPC account enablement and SSH login
- Scratch workspace usage at `/nfs/hpc/share/<ONID>`
- VS Code Remote-SSH setup
- Interactive GPU allocation with `srun` and `nvidia-smi` verification

Primary Lab 0 guide:
- `labs/lab0/README.md`
- `labs/lab0/doc`

Quick commands:
```bash
ln -s /nfs/hpc/share/<ONID> hpc-share
cd ~/hpc-share
git clone https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs.git
```

## Lab 1 Update: MNIST + DDP Starter
Lab 1 validates your PyTorch environment and introduces:
- Single-GPU MNIST training (`labs/lab1/src/train_mnist.py`)
- Multi-GPU DDP training (`labs/lab1/src/ddp.py`)
- Course environment setup from `env/environment.yml` + `env/requirements.txt`

Primary Lab 1 guide:
- `labs/lab1/README.md`

Quick run examples:
```bash
cd labs/lab1
conda env create -f env/environment.yml
conda activate csece599
pip install -r env/requirements.txt

python -m src.train_mnist --epochs 10
python -m src.ddp --epochs 2 --batch-size 128
```

## Lab 2 Update: Tensor and Pipeline Parallelism
Lab 2 now covers three distributed model-parallel workflows:
- Tensor parallelism via native PyTorch API: `labs/lab2/src/tensor_parallel_native.py`
- Tensor parallelism via manual sharding: `labs/lab2/src/tensor_parallel_manual.py`
- Pipeline parallelism with GPipe: `labs/lab2/src/pipeline.py`

Primary Lab 2 guide:
- `labs/lab2/README.md`

Quick run examples:
```bash
cd labs/lab2
conda activate csece599

torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_native \
  --epochs 1 --batch-size 128 --learning-rate 1e-3 --data-dir ./data

torchrun --standalone --nproc_per_node 2 -m src.tensor_parallel_manual \
  --batch-size 64 --input-dim 1024 --hidden-dim 2048 --num-classes 10 --steps 5

torchrun --standalone --nproc_per_node 2 -m src.pipeline
```

## Lab 3 Update: LLM Inference on 2 GPUs
Lab 3 is split into two tracks for Llama-family model inference:
- Track A (Hugging Face Transformers): `labs/lab3/src/llama_inference_2gpu.py`
- Track B (vLLM): `labs/lab3/src/vllm_inference_2gpu.py`

Primary Lab 3 guide:
- `labs/lab3/README.md`

Quick run examples:
```bash
cd labs/lab3
conda activate csece599

python -m src.llama_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype auto \
  --max-new-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."

python -m src.vllm_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."
```

Batch launch scripts are available at:
- `labs/lab3/slurm/infer_llama_v100.slurm`
- `labs/lab3/slurm/infer_llama_h100.slurm`

## Lab 4 Update: PyTorch Profiler + TensorBoard
Lab 4 uses two notebooks:
- `labs/lab4/pytorch_profiler.ipynb`
- `labs/lab4/tensorboard.ipynb`

Primary Lab 4 guide:
- `labs/lab4/README.md`

Quick run setup:
```bash
cd labs/lab4
conda activate csece599
pip install -U "setuptools==80.10.2" jupyter tensorboard torch_tb_profiler matplotlib
```

Run order:
1. `pytorch_profiler.ipynb`
2. `tensorboard.ipynb`

Useful TensorBoard commands:
```bash
tensorboard --logdir ./log --port 6006
tensorboard --logdir ./runs --port 6006
```

Compatibility note:
- If TensorBoard fails with `No module named 'pkg_resources'`, pin `setuptools` to `<81`.

## Discussions
Course Q&A and updates:
- https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs/discussions
