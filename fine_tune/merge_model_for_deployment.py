"""Merge a base Qwen3-VL model with a QLoRA adapter for vLLM deployment."""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "MAI-UI-2B")


def main():
    parser = argparse.ArgumentParser(description="Merge base model + LoRA adapter into a single checkpoint.")
    parser.add_argument("--base_model", default=DEFAULT_MODEL_PATH, help="Base model path or HF id")
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter dir from train_lora.py")
    parser.add_argument("--output_dir", required=True, help="Where to write the merged model")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Weight dtype for the merged model (default: bf16)",
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading base model from {args.base_model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    print(f"Loading adapter from {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)

    processor_path = args.adapter_path if os.path.isfile(os.path.join(args.adapter_path, "preprocessor_config.json")) else args.base_model
    processor = AutoProcessor.from_pretrained(processor_path)
    processor.save_pretrained(args.output_dir)

    print("Done. Serve with vLLM using --model", args.output_dir)


if __name__ == "__main__":
    main()
