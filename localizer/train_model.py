#!/usr/bin/env python3
"""Fine-tune SmolVLM2 with LoRA for graph node localization."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

LOCALIZER_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ID = os.path.join(
    LOCALIZER_DIR, "models", "SmolVLM2-500M-Video-Instruct"
)
HF_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEFAULT_DATA_DIR = os.path.join(LOCALIZER_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(LOCALIZER_DIR, "checkpoints")
NODE_ID_MAP_FILENAME = "node_id_map.json"


def resolve_model_path(model_id: str) -> tuple[str, bool]:
    """Return (path_or_repo, local_files_only). Prefer an existing local directory."""
    if os.path.isdir(model_id) and os.path.isfile(os.path.join(model_id, "config.json")):
        return model_id, True
    if os.path.isdir(DEFAULT_MODEL_ID) and os.path.isfile(
        os.path.join(DEFAULT_MODEL_ID, "config.json")
    ):
        # Caller passed HF repo id but local copy exists — use local
        if model_id in (HF_REPO_ID, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
            return DEFAULT_MODEL_ID, True
    return model_id, False


IMAGE_MAX_SIDE = 512  # SmolVLM tile size; keeps sequences under max_length


def build_prompt_text(app_name: str, action1: str, action2: str) -> str:
    # Keep instructions, but compact — long prompts + image splitting blew past max_length.
    return (
        f"You are a state-localization model for the application {app_name}.\n"
        "Identify the graph node for Screenshot 3 using Screenshots 1–2 and Actions 1–2 only as context.\n"
        "Return ON_GRAPH JSON only when Screenshot 3 matches a known graph node.\n"
        "Return OFF_GRAPH when Screenshot 3 is another app, an unknown in-app page, an ad/dialog/"
        "permission/login/system/loading failure, or visually ambiguous.\n"
        "Do not pick a successor only from the previous action; the current screenshot must support the node.\n"
        "Return exactly one JSON object:\n"
        '{"status":"ON_GRAPH","node_id":"N1"} or {"status":"OFF_GRAPH","node_id":null}\n'
        "Use short node ids from training (N1, N2, ...), never the raw graph hashes.\n"
        f"Action 1: {action1}\n"
        f"Action 2: {action2}\n"
        "Screenshots 1, 2, then current Screenshot 3 follow in order."
    )


def build_node_id_map(episode_dirs: list[str]) -> dict[str, dict[str, str]]:
    """Map full graph node_ids -> N1, N2, ... (stable sorted assignment)."""
    full_ids: set[str] = set()
    for ep_dir in episode_dirs:
        info_path = os.path.join(ep_dir, "info.json")
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)
        if info.get("status") == "ON_GRAPH" and info.get("node_id"):
            full_ids.add(str(info["node_id"]))
    sorted_ids = sorted(full_ids)
    alias_to_node_id = {f"N{i}": nid for i, nid in enumerate(sorted_ids, start=1)}
    node_id_to_alias = {nid: alias for alias, nid in alias_to_node_id.items()}
    return {
        "alias_to_node_id": alias_to_node_id,
        "node_id_to_alias": node_id_to_alias,
    }


def save_node_id_map(mapping: dict[str, dict[str, str]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)
    print(f"Saved node id map ({len(mapping.get('alias_to_node_id', {}))} nodes) -> {path}")


def load_node_id_map(path: str) -> dict[str, dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        mapping = json.load(f)
    if "alias_to_node_id" not in mapping or "node_id_to_alias" not in mapping:
        raise ValueError(f"Invalid node id map format: {path}")
    return mapping


def alias_for_node(node_id: str | None, mapping: dict[str, dict[str, str]] | None) -> str | None:
    if node_id is None:
        return None
    if not mapping:
        return node_id
    return mapping["node_id_to_alias"].get(str(node_id), str(node_id))


def full_node_for_alias(alias: str | None, mapping: dict[str, dict[str, str]] | None) -> str | None:
    if alias is None:
        return None
    if not mapping:
        return alias
    return mapping["alias_to_node_id"].get(str(alias), str(alias))


def target_json(status: str, node_id: str | None) -> str:
    if status == "ON_GRAPH" and node_id:
        return json.dumps({"status": "ON_GRAPH", "node_id": node_id}, separators=(",", ":"))
    return json.dumps({"status": "OFF_GRAPH", "node_id": None}, separators=(",", ":"))


def parse_prediction(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    m = re.search(r"\{[^{}]*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            status = obj.get("status")
            node_id = obj.get("node_id")
            if status == "OFF_GRAPH":
                return {"status": "OFF_GRAPH", "node_id": None}
            if status == "ON_GRAPH":
                return {"status": "ON_GRAPH", "node_id": node_id}
            return {"status": status, "node_id": node_id}
    # Fallback for incomplete generations seen during early training
    if re.search(r"\bOFF_GRAPH\b", text):
        return {"status": "OFF_GRAPH", "node_id": None}
    if re.search(r"\bON_GRAPH\b", text):
        return {"status": "ON_GRAPH", "node_id": None}
    return {"status": None, "node_id": None}


def load_rgb_image(path: str, max_side: int = IMAGE_MAX_SIDE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)


def configure_processor_for_localizer(processor: Any) -> Any:
    """Avoid image splitting — 3 split screenshots were >2048 tokens and hurt training."""
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        if hasattr(ip, "do_image_splitting"):
            ip.do_image_splitting = False
        if hasattr(ip, "size"):
            ip.size = {"longest_edge": IMAGE_MAX_SIDE}
        if hasattr(ip, "max_image_size"):
            ip.max_image_size = {"longest_edge": IMAGE_MAX_SIDE}
    return processor


def list_episodes(data_dir: str) -> list[str]:
    eps = []
    for name in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, name)
        if os.path.isdir(path) and name.startswith("episode_") and os.path.isfile(
            os.path.join(path, "info.json")
        ):
            eps.append(path)
    return eps


def load_episode(
    ep_dir: str, id_map: dict[str, dict[str, str]] | None = None
) -> dict[str, Any]:
    with open(os.path.join(ep_dir, "info.json"), encoding="utf-8") as f:
        info = json.load(f)
    actions = info.get("actions") or ["none", "none"]
    while len(actions) < 2:
        actions.insert(0, "none")
    actions = actions[-2:]
    images = [
        os.path.join(ep_dir, "history_0.jpg"),
        os.path.join(ep_dir, "history_1.jpg"),
        os.path.join(ep_dir, "current_screenshot.jpg"),
    ]
    for p in images:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)
    status = info.get("status") or "OFF_GRAPH"
    full_node = info.get("node_id")
    short_node = alias_for_node(full_node, id_map) if status == "ON_GRAPH" else None
    return {
        "info": info,
        "node_id_full": full_node,
        "node_id_alias": short_node,
        "images": images,
        "actions": actions[:2],
        "prompt_text": build_prompt_text(
            info.get("app_name") or "the application", actions[0], actions[1]
        ),
        "target": target_json(status, short_node if status == "ON_GRAPH" else None),
    }


def build_messages(prompt_text: str, images: list[str], target: str | None) -> list[dict]:
    user_content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "image"},
        {"type": "image"},
        {"type": "image"},
    ]
    messages = [{"role": "user", "content": user_content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


class LocalizerDataset(TorchDataset):
    def __init__(
        self,
        episode_dirs: list[str],
        id_map: dict[str, dict[str, str]] | None = None,
    ):
        self.episode_dirs = episode_dirs
        self.id_map = id_map

    def __len__(self) -> int:
        return len(self.episode_dirs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return load_episode(self.episode_dirs[idx], id_map=self.id_map)


@dataclass
class LocalizerCollator:
    processor: Any
    max_length: int = 2048

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts = []
        images_batch = []
        for ex in features:
            imgs = [load_rgb_image(p) for p in ex["images"]]
            images_batch.append(imgs)
            messages = build_messages(ex["prompt_text"], ex["images"], ex["target"])
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            texts.append(text)

        batch = self.processor(
            text=texts,
            images=images_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Completion-only: supervise assistant answer (locate target JSON near the end).
        for i, ex in enumerate(features):
            seq = batch["input_ids"][i]
            target_ids = self.processor.tokenizer(
                ex["target"], add_special_tokens=False
            )["input_ids"]
            labels[i, :] = -100
            if pad_id is not None:
                nonzero = (seq != pad_id).nonzero(as_tuple=True)[0]
                if len(nonzero) == 0:
                    continue
                end = int(nonzero[-1].item()) + 1
            else:
                end = int(seq.shape[0])
            seq_list = seq[:end].tolist()
            # Find exact target token span; fall back to trailing window.
            start = None
            tlen = len(target_ids)
            for j in range(end - tlen, -1, -1):
                if seq_list[j : j + tlen] == target_ids:
                    start = j
                    break
            if start is None:
                start = max(0, end - tlen - 8)
            labels[i, start:end] = batch["input_ids"][i, start:end]

        batch["labels"] = labels
        return batch


class SaveNodeIdMapCallback(TrainerCallback):
    """Copy node_id_map.json into each Trainer checkpoint-* folder."""

    def __init__(self, mapping: dict[str, dict[str, str]]):
        self.mapping = mapping

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            save_node_id_map(self.mapping, os.path.join(ckpt_dir, NODE_ID_MAP_FILENAME))


class AccuracyCallback(TrainerCallback):
    def __init__(
        self,
        eval_episodes: list[str],
        processor: Any,
        eval_steps: int,
        id_map: dict[str, dict[str, str]] | None = None,
        max_new_tokens: int = 32,
        max_eval_samples: int = 32,
    ):
        self.eval_episodes = eval_episodes
        self.processor = processor
        self.eval_steps = eval_steps
        self.id_map = id_map
        self.max_new_tokens = max_new_tokens
        self.max_eval_samples = max_eval_samples

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        if state.global_step % self.eval_steps != 0:
            return
        self._run_eval(model, state.global_step)

    @torch.no_grad()
    def _run_eval(self, model, step: int) -> None:
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        eps = self.eval_episodes[: self.max_eval_samples]
        if not eps:
            return

        total = 0
        correct = 0
        on_total = 0
        on_correct = 0
        off_total = 0
        off_correct = 0
        off_pred = 0

        samples_shown = 0
        for ep_dir in eps:
            ex = load_episode(ep_dir, id_map=self.id_map)
            imgs = [load_rgb_image(p) for p in ex["images"]]
            messages = build_messages(ex["prompt_text"], ex["images"], target=None)
            prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=prompt,
                images=[imgs],
                return_tensors="pt",
            )
            inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            generated = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            in_len = inputs["input_ids"].shape[1]
            out_text = self.processor.tokenizer.decode(
                generated[0, in_len:], skip_special_tokens=True
            )
            pred = parse_prediction(out_text)
            gold_status = ex["info"].get("status")
            gold_alias = ex.get("node_id_alias")

            if samples_shown < 3:
                print(
                    f"  sample[{samples_shown}] gold={gold_status}/{gold_alias} "
                    f"(full={ex.get('node_id_full')}) pred={pred} raw={out_text!r}"
                )
                samples_shown += 1

            total += 1
            if gold_status == "OFF_GRAPH":
                off_total += 1
                if pred.get("status") == "OFF_GRAPH":
                    off_correct += 1
                    correct += 1
            else:
                on_total += 1
                if (
                    pred.get("status") == "ON_GRAPH"
                    and pred.get("node_id") == gold_alias
                ):
                    on_correct += 1
                    correct += 1
            if pred.get("status") == "OFF_GRAPH":
                off_pred += 1

        overall = correct / total if total else 0.0
        on_acc = on_correct / on_total if on_total else 0.0
        off_rec = off_correct / off_total if off_total else 0.0
        off_prec = off_correct / off_pred if off_pred else 0.0
        print(
            f"[eval step={step}] overall_acc={overall:.3f} "
            f"on_graph_node_acc={on_acc:.3f} ({on_correct}/{on_total}) "
            f"off_graph_recall={off_rec:.3f} ({off_correct}/{off_total}) "
            f"off_graph_precision={off_prec:.3f}"
        )
        if was_training:
            model.train()


def resolve_lora_targets(model) -> list[str]:
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    candidates = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "out_proj",
        "fc1",
        "fc2",
    ]
    found = [c for c in candidates if c in names]
    if found:
        return found
    # Fallback: any linear named *proj*
    proj = sorted(
        {
            n.split(".")[-1]
            for n, m in model.named_modules()
            if "proj" in n.split(".")[-1].lower() and hasattr(m, "weight")
        }
    )
    return proj or ["q_proj", "v_proj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=8.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_eval_samples", type=int, default=32)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    episodes = list_episodes(args.data_dir)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.data_dir}. Run prepare_data.py first.")

    id_map = build_node_id_map(episodes)
    print(f"Node id aliases: {len(id_map['alias_to_node_id'])} unique ON_GRAPH nodes")

    random.shuffle(episodes)
    n_eval = max(1, int(len(episodes) * args.eval_ratio))
    eval_eps = episodes[:n_eval]
    train_eps = episodes[n_eval:] or episodes
    print(f"Train episodes: {len(train_eps)} | Eval episodes: {len(eval_eps)}")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    model_kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    model_path, local_only = resolve_model_path(args.model_id)
    print(f"Loading model {model_path} (local_files_only={local_only}) ...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=local_only
    )
    processor = configure_processor_for_localizer(processor)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, local_files_only=local_only, **model_kwargs
    )
    model.config.use_cache = False

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    targets = resolve_lora_targets(model)
    print(f"LoRA target modules: {targets}")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_ds = LocalizerDataset(train_eps, id_map=id_map)
    collator = LocalizerCollator(processor=processor, max_length=args.max_length)

    os.makedirs(args.output_dir, exist_ok=True)
    # Persist mapping with the LoRA checkpoint for inference remapping.
    map_path = os.path.join(args.output_dir, NODE_ID_MAP_FILENAME)
    save_node_id_map(id_map, map_path)
    # Also keep a copy next to the episode data for inspection.
    save_node_id_map(id_map, os.path.join(args.data_dir, NODE_ID_MAP_FILENAME))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and dtype == torch.bfloat16,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        gradient_checkpointing=True,
        dataloader_num_workers=0,
    )

    callbacks = [
        SaveNodeIdMapCallback(id_map),
        AccuracyCallback(
            eval_episodes=eval_eps,
            processor=processor,
            eval_steps=args.eval_steps,
            id_map=id_map,
            max_eval_samples=args.max_eval_samples,
            max_new_tokens=32,
        ),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    save_node_id_map(id_map, map_path)
    print(f"Saved LoRA adapter + {NODE_ID_MAP_FILENAME} to {args.output_dir}")


if __name__ == "__main__":
    main()
