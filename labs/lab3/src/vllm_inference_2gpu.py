import argparse
import os
import time


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM inference on 2 GPUs.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Hugging Face model id.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain pipeline parallelism in 4 bullet points.",
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--gpu0", type=int, default=0)
    parser.add_argument("--gpu1", type=int, default=1)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum context length reserved by vLLM.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Fraction of each GPU memory that vLLM may use.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional token for gated model access. Defaults to env HF_TOKEN.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu0 == args.gpu1:
        raise SystemExit("--gpu0 and --gpu1 must be different.")

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu0},{args.gpu1}"
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Import after setting CUDA_VISIBLE_DEVICES so vLLM sees only 2 target GPUs.
    from vllm import LLM, SamplingParams

    print("Backend: vLLM")
    print(f"Model: {args.model_id}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print("tensor_parallel_size: 2")

    llm = LLM(
        model=args.model_id,
        tensor_parallel_size=2,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    start = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling)
    elapsed = time.perf_counter() - start

    completion = outputs[0].outputs[0].text
    print("\n=== Prompt ===")
    print(args.prompt)
    print("\n=== Generated Completion ===")
    print(completion)
    print("\n=== Runtime Stats ===")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
