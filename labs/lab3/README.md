# Lab 3 — LLM Inference on 2 GPUs (Llama Family)

## 1) Overview
In this lab, you will run a small Llama-family model for text generation across
two GPUs (H100 or V100) using model parallel inference.

You will:
- Load a Llama-family checkpoint from Hugging Face.
- Split transformer layers across 2 GPUs.
- Run prompt-based generation and measure throughput/memory.

Files used in this lab:
- `src/llama_inference_2gpu.py`: single-process 2-GPU inference script.
- `slurm/infer_llama_v100.slurm`: batch example for V100 nodes.
- `slurm/infer_llama_h100.slurm`: batch example for H100 nodes.

Default model:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small, ungated, good for class labs).

## 2) Environment setup
From repo root:

```bash
cd labs/lab3
conda activate csece599
```

Install inference dependencies (if not already installed):

```bash
pip install -U transformers accelerate sentencepiece safetensors
```

Optional for gated Meta Llama models:
- Set `HF_TOKEN` in your shell.
- Accept model terms on Hugging Face first.

## 3) Reserve 2 GPUs (interactive)
### V100 example (DGX2)
```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:2 --mem=64G --pty bash
```

### H100 example (DGXH)
```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgxh --gres=gpu:2 --mem=64G --pty bash
```

Sanity check:
```bash
nvidia-smi
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("gpu_count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

## 4) Run 2-GPU Llama inference
From `labs/lab3`:

```bash
python -m src.llama_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype auto \
  --max-new-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."
```

What this script does:
- Requires two visible GPUs.
- Splits Llama layers across `cuda:0` and `cuda:1` (first half/second half).
- Prints the exact GPU ids used by the HF device map.
- Prints runtime throughput and peak memory for both GPUs.

Expected key output:
- `Tensor/model parallel GPU ids: 0, 1`
- `HF device map GPUs in use: ['cuda:0', 'cuda:1']`
- Peak allocated memory reported for both GPUs.

## 5) Optional: run a Meta Llama checkpoint
If your account has access:

```bash
export HF_TOKEN=...your_token...
python -m src.llama_inference_2gpu \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --max-new-tokens 128 \
  --prompt "Give a concise explanation of model parallel inference."
```

## 6) Slurm batch launch
From repo root:

```bash
sbatch labs/lab3/slurm/infer_llama_v100.slurm
```

or:

```bash
sbatch labs/lab3/slurm/infer_llama_h100.slurm
```

You can edit model id, prompt, and generation settings inside each script.

## 7) Suggested exercises
1. Compare `--greedy` vs sampling (`temperature/top-p`) output quality.
2. Compare V100 (`float16`) vs H100 (`bfloat16`) throughput.
3. Try longer outputs (`--max-new-tokens 256`) and measure memory growth.
4. Try another Llama-family model and record latency differences.

## 8) Troubleshooting
- Error: not enough GPUs visible:
  - Confirm your Slurm job requested `--gres=gpu:2`.
  - Check `CUDA_VISIBLE_DEVICES`.
- OOM on V100:
  - Use smaller model (TinyLlama 1.1B).
  - Reduce `--max-new-tokens`.
  - Use `--dtype float16`.
- Hugging Face auth/gated model failure:
  - Set `HF_TOKEN`.
  - Ensure model access is approved on Hugging Face.
- Only one GPU appears active:
  - Verify output line `HF device map GPUs in use: ['cuda:0', 'cuda:1']`.
  - Do not run with `--gpu0` and `--gpu1` set to the same id.
