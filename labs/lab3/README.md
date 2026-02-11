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
- `src/vllm_inference_2gpu.py`: optional vLLM-based 2-GPU inference script.
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

Optional vLLM path:

```bash
pip install -U vllm
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
- Uses Hugging Face `transformers` backend (not vLLM).

Expected key output:
- `Tensor/model parallel GPU ids: 0, 1`
- `HF device map GPUs in use: ['cuda:0', 'cuda:1']`
- Peak allocated memory reported for both GPUs.

Can different LLM models be used?
- Yes, pass a different `--model-id`.
- This script is best for Llama-family checkpoints because it manually maps
  `model.layers.*` across two GPUs.
- For broader model compatibility and higher-throughput serving, use vLLM
  (`src/vllm_inference_2gpu.py` below).

## 5) Optional vLLM (2 GPUs)
From `labs/lab3`:

```bash
python -m src.vllm_inference_2gpu \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-tokens 128 \
  --prompt "Explain tensor parallelism in 5 bullet points."
```

Notes:
- This path uses `tensor_parallel_size=2`.
- It is often easier for trying different model architectures.
- For gated models, set `HF_TOKEN`.

## 6) Model sizing calculations (GPU-aware reference)
Use these rough calculations when selecting a model for 2 GPUs.

1. Weight memory (dense model):
- `weights_bytes ~= params * bytes_per_param`
- `bytes_per_param`: FP16/BF16 = 2, FP32 = 4

2. Approx per-GPU weight memory (2-way split):
- `per_gpu_weight_gb ~= (params * bytes_per_param) / (2 * 1024^3)`

3. KV-cache memory (rough):
- `kv_bytes ~= 2 * layers * hidden_size * seq_len * batch_size * bytes_per_param`
- For 2-way layer split, divide per-GPU KV approximately by 2.

4. Practical headroom:
- Reserve about 20% to 30% GPU memory for runtime overhead, allocator
  fragmentation, activations, and buffers.

Quick rule-of-thumb for 2 GPUs:
- 2x V100 32GB:
  - 1B to 7B FP16/BF16 usually comfortable.
  - 13B can fit in many cases with careful settings, but headroom is tighter.
- 2x H100 80GB:
  - 7B to 13B easy.
  - 30B is often feasible with careful context/batch settings.

Always validate with runtime metrics:
- `nvidia-smi`
- script-reported peak allocated memory

Reference docs:
- Hugging Face model loading/device mapping:
  https://huggingface.co/docs/transformers/main_classes/model
- vLLM docs:
  https://docs.vllm.ai/

## 7) Optional: run a Meta Llama checkpoint
If your account has access:

```bash
export HF_TOKEN=...your_token...
python -m src.llama_inference_2gpu \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --dtype auto \
  --max-new-tokens 128 \
  --prompt "Give a concise explanation of model parallel inference."
```

## 8) Slurm batch launch
From repo root:

```bash
sbatch labs/lab3/slurm/infer_llama_v100.slurm
```

or:

```bash
sbatch labs/lab3/slurm/infer_llama_h100.slurm
```

You can edit model id, prompt, and generation settings inside each script.

## 9) Suggested exercises
1. Compare `--greedy` vs sampling (`temperature/top-p`) output quality.
2. Compare V100 (`float16`) vs H100 (`bfloat16`) throughput.
3. Try longer outputs (`--max-new-tokens 256`) and measure memory growth.
4. Try another Llama-family model and record latency differences.
5. Compare Transformers backend (`llama_inference_2gpu.py`) vs
   vLLM backend (`vllm_inference_2gpu.py`) on the same prompt/model.

## 10) Troubleshooting
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
- vLLM import error:
  - Install `vllm` in the active environment.
