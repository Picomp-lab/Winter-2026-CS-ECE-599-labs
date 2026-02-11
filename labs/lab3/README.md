# Lab 3 — LLM Inference on 2 GPUs (Llama Family)

## 1) Overview
This lab is split into two separate tutorials:
- Track A: Hugging Face Transformers (`src/llama_inference_2gpu.py`)
- Track B: vLLM (`src/vllm_inference_2gpu.py`)

Both tracks run on 2 GPUs (V100 or H100) and support changing `--model-id`.

## 2) Common cluster setup
From repo root:

```bash
cd labs/lab3
conda activate csece599
```

Reserve 2 GPUs (interactive).

V100 (DGX2):
```bash
srun -A eecs --time=0-01:00:00 -p gpu,dgx2 --gres=gpu:2 --mem=64G --pty bash
```

H100 (DGXH):
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

Optional for gated models:
```bash
export HF_TOKEN=...your_token...
```

## 3) Track A — Hugging Face Transformers tutorial
### 3.1 Install deps (HF track)
```bash
pip install -U transformers accelerate sentencepiece safetensors
```

### 3.2 Run (HF track)
```bash
python -m src.llama_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype auto \
  --max-new-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."
```

### 3.3 Confirm you are on HF backend
Expected logs include:
- `Backend: Hugging Face Transformers (this script does not use vLLM).`
- `HF device map GPUs in use: ['cuda:0', 'cuda:1']`

Notes:
- This script manually shards Llama-family layers across 2 GPUs.
- Best fit for Llama-family architecture checkpoints.

## 4) Track B — vLLM tutorial
### 4.1 Install deps (vLLM track)
```bash
pip install -U vllm
```

### 4.2 Run (vLLM track)
```bash
python -m src.vllm_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."
```

### 4.3 Confirm you are on vLLM backend
Expected logs include:
- `Backend: vLLM`
- `tensor_parallel_size: 2`

Notes:
- vLLM is often easier for testing multiple architectures quickly.
- `CUDA_VISIBLE_DEVICES` is set to the two selected GPUs in the script.

## 5) Can different models be used?
Yes. In either track, pass a different `--model-id`.

Examples:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `meta-llama/Llama-3.2-1B-Instruct` (if your HF account has access)

Practical guidance:
- HF script is architecture-aware for Llama-style layer names.
- vLLM track is generally better for broader model compatibility.

## 6) Model sizing reference (GPU-aware)
Use this quick estimate before picking model size.

Weight memory:
- `weights_bytes ~= params * bytes_per_param`
- FP16/BF16: `bytes_per_param = 2`
- FP32: `bytes_per_param = 4`

Approx per-GPU weight memory for 2-way split:
- `per_gpu_weight_gb ~= (params * bytes_per_param) / (2 * 1024^3)`

KV cache (rough):
- `kv_bytes ~= 2 * layers * hidden_size * seq_len * batch_size * bytes_per_param`

Headroom:
- Leave 20% to 30% free GPU memory for runtime overhead.

Rule-of-thumb on 2 GPUs:
- 2x V100 32GB: usually 1B to 7B comfortable; 13B may fit with tighter settings.
- 2x H100 80GB: 7B to 13B easy; 30B can be feasible with careful seq/batch settings.

Always verify with runtime metrics:
- `nvidia-smi`
- script-reported peak memory

References:
- HF model loading/device mapping:
  https://huggingface.co/docs/transformers/main_classes/model
- vLLM docs:
  https://docs.vllm.ai/

## 7) Batch launch (Slurm)
From repo root:

```bash
sbatch labs/lab3/slurm/infer_llama_v100.slurm
```

or:

```bash
sbatch labs/lab3/slurm/infer_llama_h100.slurm
```

If you want vLLM in batch mode, replace the python command inside the `.slurm`
script with `python -m src.vllm_inference_2gpu ...`.

## 8) Troubleshooting
Common:
- Not enough GPUs visible:
  - Confirm `--gres=gpu:2` in Slurm request.
  - Check `CUDA_VISIBLE_DEVICES`.
- OOM:
  - Use a smaller model.
  - Reduce output length (`--max-new-tokens` or `--max-tokens`).
  - Use FP16/BF16.

HF track specific:
- If output looks like prompt echo:
  - Check `Generated Completion Only` section in script output.
- If model mapping fails:
  - Use a Llama-family checkpoint with `llama_inference_2gpu.py`.

vLLM track specific:
- `ModuleNotFoundError: vllm`:
  - Install `vllm` in the active environment.
- Startup OOM in engine init:
  - Reduce `--max-model-len` or `--gpu-memory-utilization`.
