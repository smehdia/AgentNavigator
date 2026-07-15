#!/usr/bin/env python3
"""Prepare SmolVLM2 node-localizer training episodes from explored_apps graphs."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter
from typing import Any

import cv2
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

DEFAULT_EXPLORED_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "exploration",
    "explored_apps",
)
MAX_SIDE = 1152
# Overall batch composition target (off_graph is generated separately).
AUG_MIX = {
    "clean": 0.35,
    "mild": 0.25,
    "action_protected": 0.25,
    "off_graph": 0.10,
    "counterfactual": 0.05,
}
# ON_GRAPH-only mix renormalized from AUG_MIX without off_graph.
_ON_TOTAL = AUG_MIX["clean"] + AUG_MIX["mild"] + AUG_MIX["action_protected"] + AUG_MIX["counterfactual"]
ON_GRAPH_MIX = {
    "clean": AUG_MIX["clean"] / _ON_TOTAL,
    "mild": AUG_MIX["mild"] / _ON_TOTAL,
    "action_protected": AUG_MIX["action_protected"] / _ON_TOTAL,
    "counterfactual": AUG_MIX["counterfactual"] / _ON_TOTAL,
}


def load_graph(app_dir: str) -> nx.MultiDiGraph:
    path = os.path.join(app_dir, "graph.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    try:
        return json_graph.node_link_graph(data, edges="links")
    except TypeError:
        return json_graph.node_link_graph(data)


def screenshot_path(app_dir: str, node_id: str) -> str:
    return os.path.join(app_dir, "screenshots", f"{node_id}.jpg")


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def resize_longest(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest == max_side:
        return img
    scale = max_side / float(longest)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (nw, nh), interpolation=interp)


def scale_bboxes(
    bboxes: list[list[float]], src_shape: tuple[int, int], dst_shape: tuple[int, int]
) -> list[list[int]]:
    sh, sw = src_shape[:2]
    dh, dw = dst_shape[:2]
    sx, sy = dw / float(sw), dh / float(sh)
    out = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb
        out.append(
            [
                int(round(x1 * sx)),
                int(round(y1 * sy)),
                int(round(x2 * sx)),
                int(round(y2 * sy)),
            ]
        )
    return out


def action_bboxes_for_node(graph: nx.MultiDiGraph, node_id: str) -> list[list[float]]:
    attrs = graph.nodes[node_id]
    boxes: list[list[float]] = []
    for el in attrs.get("ui_elements") or []:
        bb = el.get("boundingBox") or el.get("bbox")
        if bb and len(bb) == 4:
            boxes.append([float(v) for v in bb])
    if boxes:
        return boxes
    for _, _, edata in graph.out_edges(node_id, data=True):
        bb = edata.get("boundingBox") or edata.get("bbox")
        if bb and len(bb) == 4:
            boxes.append([float(v) for v in bb])
    return boxes


def format_action(edata: dict[str, Any]) -> str:
    typ = edata.get("type") or "action"
    desc = (edata.get("description") or "").strip()
    if desc:
        return f"{typ}: {desc}"
    return str(typ)


def sample_history_ending_at(
    graph: nx.MultiDiGraph, node_id: str, k: int, rng: random.Random
) -> tuple[list[str], list[str]]:
    """Walk k incoming edges backwards to build chronological history ending at node_id."""
    if k <= 0:
        return [], []
    nodes_rev = [node_id]
    actions_rev: list[str] = []
    cur = node_id
    for _ in range(k):
        preds = list(graph.in_edges(cur, data=True))
        if not preds:
            break
        src, _, edata = rng.choice(preds)
        actions_rev.append(format_action(edata))
        nodes_rev.append(src)
        cur = src
    hist_nodes = list(reversed(nodes_rev[1:]))
    hist_actions = list(reversed(actions_rev))
    return hist_nodes, hist_actions


def sample_unrelated_history(
    graph: nx.MultiDiGraph, exclude_node: str, k: int, rng: random.Random, app_dir: str
) -> tuple[list[str], list[str]]:
    candidates = [
        n
        for n in graph.nodes
        if n != exclude_node and os.path.isfile(screenshot_path(app_dir, n))
    ]
    if not candidates or k <= 0:
        return [], []
    start = rng.choice(candidates)
    return sample_history_ending_at(graph, start, k, rng)


def collect_offgraph_screenshots(explored_root: str, app_dir: str) -> list[str]:
    app_dir = os.path.abspath(app_dir)
    paths: list[str] = []
    if not os.path.isdir(explored_root):
        return paths
    for name in os.listdir(explored_root):
        other = os.path.join(explored_root, name)
        if not os.path.isdir(other) or os.path.abspath(other) == app_dir:
            continue
        shot_dir = os.path.join(other, "screenshots")
        if not os.path.isdir(shot_dir):
            continue
        for fn in os.listdir(shot_dir):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(shot_dir, fn))
    return paths


def placeholder_image(shape: tuple[int, int, int] | None = None) -> np.ndarray:
    if shape is None:
        return np.zeros((640, 360, 3), dtype=np.uint8)
    h, w = shape[:2]
    return np.zeros((h, w, 3), dtype=np.uint8)


def adjust_brightness_contrast(
    img: np.ndarray, brightness: float, contrast: float
) -> np.ndarray:
    out = img.astype(np.float32)
    out = (out - 127.5) * contrast + 127.5
    out = out * brightness
    return np.clip(out, 0, 255).astype(np.uint8)


def adjust_saturation(img: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def jpeg_compress(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def mild_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def scale_and_pad(img: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = img.shape[:2]
    scale = rng.uniform(0.92, 1.0)
    nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # mild safe-area / letterbox variation
    top = rng.randint(0, max(0, h - nh))
    left = rng.randint(0, max(0, w - nw))
    canvas[top : top + nh, left : left + nw] = resized
    pad_val = rng.randint(0, 40)
    if pad_val > 0:
        canvas[:pad_val, :] = pad_val
        canvas[-pad_val:, :] = pad_val
        canvas[:, :pad_val] = pad_val
        canvas[:, -pad_val:] = pad_val
        canvas[top : top + nh, left : left + nw] = resized
    return canvas


def mild_full_augment(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.copy()
    out = adjust_brightness_contrast(
        out, rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)
    )
    out = adjust_saturation(out, rng.uniform(0.9, 1.1))
    if rng.random() < 0.7:
        out = jpeg_compress(out, rng.randint(55, 90))
    if rng.random() < 0.6:
        out = add_gaussian_noise(out, rng.uniform(2.0, 8.0))
    if rng.random() < 0.5:
        out = mild_blur(out, rng.choice([3, 5]))
    if rng.random() < 0.7:
        out = scale_and_pad(out, rng)
    return out


def build_action_mask(
    shape: tuple[int, int], bboxes: list[list[int]], dilate: int = 4
) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        x1 = max(0, min(w - 1, x1 - dilate))
        y1 = max(0, min(h - 1, y1 - dilate))
        x2 = max(0, min(w, x2 + dilate))
        y2 = max(0, min(h, y2 + dilate))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def apply_outside_mask(original: np.ndarray, augmented: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = augmented.copy()
    m = mask > 0
    out[m] = original[m]
    return out


def cutmix_outside_actions(
    img: np.ndarray,
    donor: np.ndarray,
    mask: np.ndarray,
    rng: random.Random,
    n_patches: int = 2,
) -> np.ndarray:
    h, w = img.shape[:2]
    if donor.shape[:2] != (h, w):
        donor = cv2.resize(donor, (w, h), interpolation=cv2.INTER_LINEAR)
    out = img.copy()
    free = mask == 0
    if free.sum() < 100:
        return out
    for _ in range(n_patches):
        ph = rng.randint(max(8, h // 12), max(9, h // 4))
        pw = rng.randint(max(8, w // 12), max(9, w // 4))
        y0 = rng.randint(0, max(0, h - ph))
        x0 = rng.randint(0, max(0, w - pw))
        region = free[y0 : y0 + ph, x0 : x0 + pw]
        if region.mean() < 0.6:
            continue
        patch = donor[y0 : y0 + ph, x0 : x0 + pw]
        dest = out[y0 : y0 + ph, x0 : x0 + pw]
        dest[region] = patch[region]
        out[y0 : y0 + ph, x0 : x0 + pw] = dest
    return out


def action_protected_augment(
    img: np.ndarray,
    bboxes: list[list[int]],
    donor: np.ndarray | None,
    rng: random.Random,
) -> np.ndarray:
    mask = build_action_mask(img.shape[:2], bboxes)
    aug = mild_full_augment(img, rng)
    aug = apply_outside_mask(img, aug, mask)
    if donor is not None and rng.random() < 0.85:
        aug = cutmix_outside_actions(aug, donor, mask, rng, n_patches=rng.randint(1, 3))
        aug = apply_outside_mask(img, aug, mask)
    return aug


def pick_from_mix(mix: dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    for name, p in mix.items():
        cum += p
        if r <= cum:
            return name
    return next(iter(mix))


def pick_aug_type(rng: random.Random) -> str:
    return pick_from_mix(AUG_MIX, rng)


def pick_on_graph_aug(rng: random.Random) -> str:
    return pick_from_mix(ON_GRAPH_MIX, rng)


def build_on_graph_episode(
    graph: nx.MultiDiGraph,
    app_dir: str,
    app_name: str,
    node_id: str,
    node_ids: list[str],
    aug_type: str,
    rng: random.Random,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    k = rng.choice([0, 1, 2])
    raw = load_image(screenshot_path(app_dir, node_id))
    src_shape = raw.shape
    current = resize_longest(raw)
    bboxes = scale_bboxes(
        action_bboxes_for_node(graph, node_id), src_shape, current.shape
    )

    if aug_type == "counterfactual":
        hist_nodes, hist_actions = sample_unrelated_history(
            graph, node_id, max(k, 1), rng, app_dir
        )
        if not hist_nodes:
            other = rng.choice([n for n in node_ids if n != node_id] or node_ids)
            hist_nodes = [other]
            hist_actions = ["action: counterfactual unmatched transition"]
        current_img = current.copy()
        hist_imgs = prepare_history_images(app_dir, hist_nodes, current.shape, rng)
        actions = pad_actions(hist_actions)
        info = {
            "status": "ON_GRAPH",
            "node_id": node_id,
            "app_name": app_name,
            "actions": actions,
            "num_history": len(hist_nodes),
            "aug_type": "counterfactual",
        }
        return hist_imgs, current_img, info

    hist_nodes, hist_actions = sample_history_ending_at(graph, node_id, k, rng)
    if aug_type == "mild":
        current_img = mild_full_augment(current, rng)
    elif aug_type == "action_protected":
        donor = None
        donors = [n for n in node_ids if n != node_id]
        if donors:
            donor = resize_longest(load_image(screenshot_path(app_dir, rng.choice(donors))))
        current_img = action_protected_augment(current, bboxes, donor, rng)
    else:
        current_img = current.copy()
        aug_type = "clean"

    hist_imgs = prepare_history_images(
        app_dir, hist_nodes, current_img.shape, rng, mild=(aug_type == "mild")
    )
    actions = pad_actions(hist_actions)
    info = {
        "status": "ON_GRAPH",
        "node_id": node_id,
        "app_name": app_name,
        "actions": actions,
        "num_history": len(hist_nodes),
        "aug_type": aug_type,
    }
    return hist_imgs, current_img, info


def build_off_graph_episode(
    graph: nx.MultiDiGraph,
    app_dir: str,
    app_name: str,
    node_ids: list[str],
    offgraph_pool: list[str],
    rng: random.Random,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    k = rng.choice([0, 1, 2])
    current = resize_longest(load_image(rng.choice(offgraph_pool)))
    if rng.random() < 0.5:
        current = mild_full_augment(current, rng)
    if k > 0 and node_ids:
        hist_nodes, hist_actions = sample_history_ending_at(
            graph, rng.choice(node_ids), k, rng
        )
    else:
        hist_nodes, hist_actions = [], []
    hist_imgs = prepare_history_images(app_dir, hist_nodes, current.shape, rng)
    actions = pad_actions(hist_actions)
    info = {
        "status": "OFF_GRAPH",
        "node_id": None,
        "app_name": app_name,
        "actions": actions,
        "num_history": len(hist_nodes),
        "aug_type": "off_graph",
    }
    return hist_imgs, current, info


def save_episode(
    out_dir: str,
    episode_idx: int,
    history_imgs: list[np.ndarray],
    current: np.ndarray,
    info: dict[str, Any],
) -> None:
    ep_dir = os.path.join(out_dir, f"episode_{episode_idx}")
    os.makedirs(ep_dir, exist_ok=True)
    # Left-pad so chronological order is oldest → newest across history_0, history_1
    slots = list(history_imgs[-2:])
    while len(slots) < 2:
        slots.insert(0, placeholder_image(current.shape))
    for i, himg in enumerate(slots):
        cv2.imwrite(
            os.path.join(ep_dir, f"history_{i}.jpg"),
            himg,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )
    cv2.imwrite(
        os.path.join(ep_dir, "current_screenshot.jpg"),
        current,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )
    with open(os.path.join(ep_dir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def pad_actions(actions: list[str], n: int = 2) -> list[str]:
    """Left-pad with 'none' so actions align with left-padded history screenshots."""
    out = list(actions)
    while len(out) < n:
        out.insert(0, "none")
    return out[:n]


def prepare_history_images(
    app_dir: str,
    hist_nodes: list[str],
    current_shape: tuple[int, ...],
    rng: random.Random,
    mild: bool = False,
) -> list[np.ndarray]:
    imgs: list[np.ndarray] = []
    for nid in hist_nodes:
        path = screenshot_path(app_dir, nid)
        if not os.path.isfile(path):
            imgs.append(placeholder_image(current_shape))
            continue
        himg = resize_longest(load_image(path))
        if mild and rng.random() < 0.4:
            himg = mild_full_augment(himg, rng)
        imgs.append(himg)
    return imgs


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare localizer training episodes")
    parser.add_argument(
        "--app_dir",
        required=True,
        help="Path to explored app dir, e.g. exploration/explored_apps/broccoli",
    )
    parser.add_argument("--app_name", required=True, help="Application display name for prompts")
    parser.add_argument(
        "--out_dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    )
    parser.add_argument("--explored_root", default=DEFAULT_EXPLORED_ROOT)
    parser.add_argument(
        "--samples_per_node",
        type=int,
        default=15,
        help="ON_GRAPH episodes to write per graph node (balanced oversampling)",
    )
    parser.add_argument(
        "--off_graph_ratio",
        type=float,
        default=0.10,
        help="Fraction of total episodes that should be OFF_GRAPH",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=0,
        help="Legacy total-episode budget (0 = use samples_per_node balancing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true", help="Clear out_dir before writing")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    app_dir = os.path.abspath(args.app_dir)
    graph = load_graph(app_dir)
    node_ids = [
        n
        for n in graph.nodes
        if os.path.isfile(screenshot_path(app_dir, n))
    ]
    if not node_ids:
        raise SystemExit(f"No nodes with screenshots found in {app_dir}")
    node_ids = sorted(node_ids)

    offgraph_pool = collect_offgraph_screenshots(args.explored_root, app_dir)

    # Build a balanced worklist: each node gets the same # of ON_GRAPH samples.
    if args.num_episodes and args.num_episodes > 0:
        # Legacy mode: approximate balance under a fixed budget.
        on_budget = int(round(args.num_episodes * (1.0 - args.off_graph_ratio)))
        samples_per_node = max(1, on_budget // len(node_ids))
        print(
            f"Legacy --num_episodes={args.num_episodes}: using samples_per_node={samples_per_node}"
        )
    else:
        samples_per_node = max(1, args.samples_per_node)

    on_jobs: list[tuple[str, str]] = []
    for node_id in node_ids:
        for _ in range(samples_per_node):
            on_jobs.append((node_id, pick_on_graph_aug(rng)))
    rng.shuffle(on_jobs)

    n_on = len(on_jobs)
    if offgraph_pool and args.off_graph_ratio > 0:
        # off / (on + off) ≈ off_graph_ratio  =>  off = on * r / (1-r)
        n_off = int(round(n_on * args.off_graph_ratio / max(1e-6, 1.0 - args.off_graph_ratio)))
    else:
        n_off = 0

    print(
        f"App nodes with screenshots: {len(node_ids)}; OFF_GRAPH pool: {len(offgraph_pool)}\n"
        f"Balanced ON_GRAPH: {n_on} ({samples_per_node}/node) + OFF_GRAPH: {n_off} "
        f"(ratio≈{n_off / max(1, n_on + n_off):.2f})"
    )

    if args.clear and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    episode_idx = 0
    aug_counts: Counter[str] = Counter()
    node_counts: Counter[str] = Counter()

    for node_id, aug_type in on_jobs:
        hist_imgs, current_img, info = build_on_graph_episode(
            graph, app_dir, args.app_name, node_id, node_ids, aug_type, rng
        )
        save_episode(args.out_dir, episode_idx, hist_imgs, current_img, info)
        aug_counts[info["aug_type"]] += 1
        node_counts[node_id] += 1
        if episode_idx % 100 == 0:
            print(f"  wrote episode_{episode_idx} ({info['aug_type']}) node=...{node_id[-12:]}")
        episode_idx += 1

    for _ in range(n_off):
        hist_imgs, current_img, info = build_off_graph_episode(
            graph, app_dir, args.app_name, node_ids, offgraph_pool, rng
        )
        save_episode(args.out_dir, episode_idx, hist_imgs, current_img, info)
        aug_counts[info["aug_type"]] += 1
        if episode_idx % 100 == 0:
            print(f"  wrote episode_{episode_idx} (off_graph)")
        episode_idx += 1

    per_node_vals = list(node_counts.values())
    print(
        f"Done. Wrote {episode_idx} episodes under {args.out_dir}\n"
        f"  aug_counts={dict(aug_counts)}\n"
        f"  per-node ON_GRAPH: min={min(per_node_vals)} max={max(per_node_vals)} "
        f"(nodes={len(per_node_vals)})"
    )


if __name__ == "__main__":
    main()
