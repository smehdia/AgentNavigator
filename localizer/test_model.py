#!/usr/bin/env python3
"""Live ADB evaluation of a trained SmolVLM2 node localizer."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from typing import Any

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LOCALIZER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LOCALIZER_DIR)
INFERENCE_DIR = os.path.join(REPO_ROOT, "inference")
MAX_SIDE = 1152

if LOCALIZER_DIR not in sys.path:
    sys.path.insert(0, LOCALIZER_DIR)

from train_model import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_OUTPUT_DIR,
    IMAGE_MAX_SIDE,
    NODE_ID_MAP_FILENAME,
    build_messages,
    build_prompt_text,
    configure_processor_for_localizer,
    full_node_for_alias,
    load_node_id_map,
    load_rgb_image,
    parse_prediction,
)


def resize_longest(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == max_side:
        return img
    scale = max_side / float(longest)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def placeholder_bgr(ref: np.ndarray | None = None) -> np.ndarray:
    if ref is None:
        return np.zeros((640, 360, 3), dtype=np.uint8)
    return np.zeros_like(ref)


def load_id_map_from_checkpoint(checkpoint: str) -> dict[str, dict[str, str]] | None:
    candidates = [
        os.path.join(checkpoint, NODE_ID_MAP_FILENAME),
        os.path.join(os.path.dirname(checkpoint), NODE_ID_MAP_FILENAME),
    ]
    for path in candidates:
        if os.path.isfile(path):
            mapping = load_node_id_map(path)
            print(f"Loaded node id map from {path} ({len(mapping['alias_to_node_id'])} nodes)")
            return mapping
    print(f"Warning: {NODE_ID_MAP_FILENAME} not found under {checkpoint}")
    return None


def load_localizer(model_id: str, checkpoint: str, load_in_4bit: bool = False):
    from train_model import resolve_model_path

    model_path, local_only = resolve_model_path(model_id)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    try:
        processor = AutoProcessor.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, local_files_only=local_only
        )
    processor = configure_processor_for_localizer(processor)

    print(f"Loading base model from {model_path} (local_files_only={local_only})")
    base = AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)
    model = PeftModel.from_pretrained(base, checkpoint)
    model.eval()
    id_map = load_id_map_from_checkpoint(checkpoint)
    return model, processor, id_map


@torch.no_grad()
def predict(
    model,
    processor,
    app_name: str,
    history_imgs: list[np.ndarray],
    current: np.ndarray,
    actions: list[str],
    id_map: dict[str, dict[str, str]] | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    acts = list(actions)
    while len(acts) < 2:
        acts.insert(0, "none")
    acts = acts[-2:]

    hist = list(history_imgs[-2:])
    while len(hist) < 2:
        hist.insert(0, placeholder_bgr(current))

    prompt_text = build_prompt_text(app_name, acts[0], acts[1])

    def _prep(bgr: np.ndarray) -> Image.Image:
        pil = bgr_to_pil(bgr)
        w, h = pil.size
        longest = max(w, h)
        if longest > IMAGE_MAX_SIDE:
            scale = IMAGE_MAX_SIDE / float(longest)
            pil = pil.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC
            )
        return pil

    pil_images = [_prep(hist[0]), _prep(hist[1]), _prep(current)]
    messages = build_messages(prompt_text, ["h0", "h1", "cur"], target=None)
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(text=text, images=[pil_images], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    in_len = inputs["input_ids"].shape[1]
    out_text = processor.tokenizer.decode(generated[0, in_len:], skip_special_tokens=True)
    pred = parse_prediction(out_text)
    alias = pred.get("node_id")
    pred["node_id_alias"] = alias
    if pred.get("status") == "ON_GRAPH" and alias:
        pred["node_id"] = full_node_for_alias(alias, id_map)
    pred["raw"] = out_text
    return pred


def annotate_node(img: np.ndarray, node_id: str, alias: str | None = None) -> np.ndarray:
    out = img.copy()
    label = f"node: {alias or node_id}"
    if alias and alias != node_id:
        label = f"{alias} | {node_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    pad = 8
    cv2.rectangle(out, (10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, label, (10 + pad, 10 + th + pad - 2), font, scale, (0, 255, 0), thickness)
    return out


def side_by_side(live: np.ndarray, ref: np.ndarray | None) -> np.ndarray:
    h = live.shape[0]
    if ref is None:
        return live
    rh, rw = ref.shape[:2]
    scale = h / float(rh)
    ref_r = cv2.resize(ref, (int(rw * scale), h), interpolation=cv2.INTER_AREA)
    return np.concatenate([live, ref_r], axis=1)


def build_driver(device_id: str | None, app_package: str | None):
    if INFERENCE_DIR not in sys.path:
        sys.path.insert(0, INFERENCE_DIR)
    from Driver.factory import build_driver as _build_driver

    settings = {
        "os_name": "android",
        "device_id": device_id,
        "appPackage": app_package or "",
        "appActivity": "",
    }
    return _build_driver(settings=settings, agent=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Online localizer test via ADB")
    parser.add_argument("--app_name", required=True)
    parser.add_argument(
        "--app_dir",
        default=None,
        help="Explored app dir (for optional reference screenshot overlay)",
    )
    parser.add_argument("--checkpoint", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device_id", default=None)
    parser.add_argument("--app_package", default=None, help="Optional package to launch")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    parser.add_argument("--max_steps", type=int, default=0, help="0 = run forever")
    parser.add_argument("--action1", default="none", help="Action after history screenshot 1")
    parser.add_argument("--action2", default="none", help="Action after history screenshot 2")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(LOCALIZER_DIR, "live_preds"),
    )
    parser.add_argument("--no_display", action="store_true", help="Skip cv2.imshow")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model, processor, id_map = load_localizer(
        args.model_id, args.checkpoint, args.load_in_4bit
    )
    driver = build_driver(args.device_id, args.app_package)
    if not driver.check_device():
        raise SystemExit("No Android device connected (adb).")
    if args.app_package:
        try:
            driver.run_application()
            time.sleep(1.5)
        except Exception as e:
            print(f"Warning: failed to launch app: {e}")

    history: deque[np.ndarray] = deque(maxlen=2)
    # Manual action strings from CLI; history screenshots fill over time
    actions = [args.action1, args.action2]
    step = 0
    print("Starting live localization loop (Ctrl+C to stop)...")

    try:
        while True:
            step += 1
            shot = driver.take_screenshot()
            current = resize_longest(shot)

            hist_list = list(history)
            # Pad with placeholders if needed
            while len(hist_list) < 2:
                hist_list.insert(0, placeholder_bgr(current))

            pred = predict(
                model,
                processor,
                args.app_name,
                hist_list[-2:],
                current,
                list(actions),
                id_map=id_map,
                max_new_tokens=32,
            )
            status = pred.get("status")
            node_id = pred.get("node_id")
            alias = pred.get("node_id_alias")
            print(f"[step {step}] status={status} alias={alias} node_id={node_id}")
            if pred.get("raw"):
                print(f"  raw: {pred['raw'][:200]}")

            if status == "ON_GRAPH" and node_id:
                annotated = annotate_node(current, str(node_id), alias=alias)
                ref = None
                if args.app_dir:
                    ref_path = os.path.join(args.app_dir, "screenshots", f"{node_id}.jpg")
                    if os.path.isfile(ref_path):
                        ref = resize_longest(cv2.imread(ref_path))
                display = side_by_side(annotated, ref)
                safe_alias = alias or "node"
                out_path = os.path.join(args.out_dir, f"step_{step:05d}_{safe_alias}.jpg")
                cv2.imwrite(out_path, display)
                if not args.no_display:
                    cv2.imshow("localizer ON_GRAPH", display)
                    cv2.waitKey(1)
            else:
                print("OFF_GRAPH")
                if not args.no_display:
                    cv2.imshow("localizer OFF_GRAPH", current)
                    cv2.waitKey(1)

            history.append(current.copy())
            if args.max_steps and step >= args.max_steps:
                break
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if not args.no_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


if __name__ == "__main__":
    main()
