import argparse
import os
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Two-GPU Llama-family inference (single-process model parallel)."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id (Llama-family recommended).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain tensor parallelism in 5 bullet points.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful teaching assistant for a deep learning class.",
        help="Used when chat template formatting is enabled.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Compute dtype for model weights.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HF token for gated model access. Defaults to env HF_TOKEN.",
    )
    parser.add_argument("--gpu0", type=int, default=0, help="First GPU id to use.")
    parser.add_argument("--gpu1", type=int, default=1, help="Second GPU id to use.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding (disables temperature/top-p sampling).",
    )
    parser.add_argument(
        "--disable-chat-template",
        action="store_true",
        help="Tokenize the raw prompt directly instead of chat-formatting it.",
    )
    return parser.parse_args()


def choose_dtype(dtype_arg, gpu_id):
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if dtype_arg == "float32":
        return torch.float32

    major, _ = torch.cuda.get_device_capability(gpu_id)
    if major >= 8:
        return torch.bfloat16
    return torch.float16


def build_llama_2gpu_device_map(num_hidden_layers, gpu0, gpu1):
    split = num_hidden_layers // 2
    device_map = {
        "model.embed_tokens": f"cuda:{gpu0}",
        "model.norm": f"cuda:{gpu1}",
        "lm_head": f"cuda:{gpu1}",
    }
    for layer_idx in range(num_hidden_layers):
        if layer_idx < split:
            device_map[f"model.layers.{layer_idx}"] = f"cuda:{gpu0}"
        else:
            device_map[f"model.layers.{layer_idx}"] = f"cuda:{gpu1}"
    return device_map, split


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this script.")
    if args.gpu0 == args.gpu1:
        raise SystemExit("--gpu0 and --gpu1 must be different.")
    if torch.cuda.device_count() <= max(args.gpu0, args.gpu1):
        raise SystemExit(
            f"Requested GPUs {args.gpu0},{args.gpu1}, but only "
            f"{torch.cuda.device_count()} visible."
        )

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    torch_dtype = choose_dtype(args.dtype, args.gpu0)

    config = AutoConfig.from_pretrained(args.model_id, token=hf_token)
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise SystemExit(
            "Model config missing num_hidden_layers. "
            "Use a Llama-family CausalLM checkpoint."
        )

    device_map, split = build_llama_2gpu_device_map(
        num_hidden_layers, args.gpu0, args.gpu1
    )
    print("Backend: Hugging Face Transformers (this script does not use vLLM).")
    print(f"Model: {args.model_id}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")
    print(f"Tensor/model parallel GPU ids: {args.gpu0}, {args.gpu1}")
    print(
        f"Layer split: [0..{split - 1}] -> cuda:{args.gpu0}, "
        f"[{split}..{num_hidden_layers - 1}] -> cuda:{args.gpu1}"
    )
    print(f"torch_dtype: {torch_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.eval()

    used_cuda_devices = sorted(
        {
            module_device
            for module_device in model.hf_device_map.values()
            if str(module_device).startswith("cuda:")
        }
    )
    print(f"HF device map GPUs in use: {used_cuda_devices}")
    if len(used_cuda_devices) < 2:
        raise SystemExit(
            "Model is not sharded across two GPUs. Check model architecture/device map."
        )

    use_chat_template = (
        (not args.disable_chat_template)
        and hasattr(tokenizer, "apply_chat_template")
        and callable(tokenizer.apply_chat_template)
        and getattr(tokenizer, "chat_template", None)
    )
    if use_chat_template:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        rendered_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        rendered_prompt = args.prompt

    prompt_inputs = tokenizer(rendered_prompt, return_tensors="pt").to(
        f"cuda:{args.gpu0}"
    )

    for gpu_id in (args.gpu0, args.gpu1):
        torch.cuda.reset_peak_memory_stats(gpu_id)

    do_sample = not args.greedy
    generate_kwargs = {
        "min_new_tokens": args.min_new_tokens,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = args.temperature
        generate_kwargs["top_p"] = args.top_p

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**prompt_inputs, **generate_kwargs)
    elapsed = time.perf_counter() - start

    prompt_len = prompt_inputs["input_ids"].shape[1]
    total_len = output_ids.shape[1]
    new_tokens = total_len - prompt_len
    toks_per_sec = new_tokens / max(elapsed, 1e-6)
    full_output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n=== Prompt (Rendered) ===")
    print(rendered_prompt)
    print("\n=== Generated Completion Only ===")
    print(generated_text)
    if generated_text.strip() == "":
        raw_generated = tokenizer.decode(generated_ids, skip_special_tokens=False)
        print("Warning: decoded completion is empty after removing special tokens.")
        print(f"Raw generated segment: {raw_generated!r}")
    print("\n=== Full Sequence (Prompt + Completion) ===")
    print(full_output_text)
    print("\n=== Runtime Stats ===")
    print(f"Prompt tokens: {prompt_len}")
    print(f"New tokens: {new_tokens}")
    print(f"Elapsed: {elapsed:.2f}s")
    print(f"Throughput: {toks_per_sec:.2f} tokens/s")
    for gpu_id in (args.gpu0, args.gpu1):
        peak_gb = torch.cuda.max_memory_allocated(gpu_id) / (1024**3)
        print(f"cuda:{gpu_id} peak allocated: {peak_gb:.2f} GiB")


if __name__ == "__main__":
    main()
