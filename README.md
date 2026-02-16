# Winter-2026-CS-ECE-599-labs

Labs for **CS/ECE 599 (Winter 2026)** — Advanced Topics in Deep Learning & High-performance Computing.

## Repo Structure
- `labs/lab0`: HPC + VS Code onboarding and setup docs
- `labs/lab1`: MNIST walkthrough and DistributedDataParallel baseline
- `labs/lab2`: tensor parallelism and pipeline parallelism tutorials
- `labs/lab3`: 2-GPU LLM inference (Hugging Face + vLLM tracks)
- `labs/lab4` and `labs/lab5`: later course labs

## Getting Started
1. Read setup docs in `labs/lab0/doc`.
2. Complete `labs/lab0/README.md`.
3. Run through `labs/lab1/README.md` to validate your environment.
4. Continue with Lab 2 and Lab 3 from the sections below.

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

## Discussions
Course Q&A and updates:
- https://github.com/Picomp-lab/Winter-2026-CS-ECE-599-labs/discussions
