import argparse
import json
import os
import random

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForVisionLanguageModeling

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "MAI-UI-2B")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_training_row(row):
    prompt, completion, images = [], [], []
    for message in row["messages"]:
        if message["role"] == "assistant":
            completion.append(message)
            continue
        content = []
        for part in message["content"]:
            if part.get("type") == "image":
                images.append(part["image"])
                content.append({"type": "image"})
            else:
                content.append(part)
        prompt.append({"role": message["role"], "content": content})
    return {"images": images, "prompt": prompt, "completion": completion}


def make_collator(processor, max_length):
    base = DataCollatorForVisionLanguageModeling(
        processor=processor,
        max_length=max_length,
        completion_only_loss=True,
    )

    def collate(examples):
        batch = []
        for ex in examples:
            item = dict(ex)
            item["images"] = [
                Image.open(p).convert("RGB") if isinstance(p, str) else p for p in item["images"]
            ]
            batch.append(item)
        return base(batch)

    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="training_data.jsonl from prepare_data.py")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default="./mai-ui-qlora")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--save_steps", type=int, default=100)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)
    device_map = {"": local_rank} if local_rank >= 0 else ("auto" if torch.cuda.is_available() else "cpu")

    rows = [to_training_row(r) for r in load_jsonl(args.data)]
    random.seed(args.seed)
    random.shuffle(rows)
    print(f"Training on {len(rows)} samples")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    processor = AutoProcessor.from_pretrained(args.model_path)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = 256 * 28 * 28
        processor.image_processor.max_pixels = 1280 * 28 * 28

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            weight_decay=0.01,
            seed=args.seed,
            bf16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            gradient_checkpointing=True,
            max_length=args.max_length,
            completion_only_loss=True,
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        ),
        train_dataset=Dataset.from_list(rows),
        processing_class=processor,
        peft_config=LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        data_collator=make_collator(processor, args.max_length),
    )

    trainer.train()
    if local_rank <= 0:
        trainer.save_model(args.output_dir)
        processor.save_pretrained(args.output_dir)
        print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
