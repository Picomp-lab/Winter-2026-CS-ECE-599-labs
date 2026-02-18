# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import os
import sys

from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM basic smoke test.")
    parser.add_argument("--model-id", type=str, default="facebook/opt-125m")
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value (for example: '0' or '0,1').",
    )
    return parser.parse_args()


def validate_gpu():
    try:
        import torch
    except Exception:
        raise SystemExit(f"PyTorch is not available in this env: {exc}") from exc

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit(
            "No GPU detected by PyTorch. Start an interactive GPU session first, "
            "then run this script again."
        )


def main():
    args = parse_args()
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    # In some cluster/notebook setups, vLLM V1 multiprocessing can fail to spawn.
    # Keep this smoke test in single-process mode unless user overrides externally.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    validate_gpu()

    # Create an LLM.
    try:
        llm = LLM(model=args.model_id)
    except Exception as exc:
        print("\n[vLLM init failure] Engine failed to initialize.", file=sys.stderr)
        print(
            "Hints: verify you are in a GPU allocation, vLLM env is active, and "
            "try setting CUDA_VISIBLE_DEVICES explicitly (for example, '0').",
            file=sys.stderr,
        )
        print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", file=sys.stderr)
        print(
            f"VLLM_ENABLE_V1_MULTIPROCESSING={os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '<unset>')}",
            file=sys.stderr,
        )
        raise

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
