#!/usr/bin/env python3
"""Standalone OOD training-data prep (SigLIP screenshots only)."""
import os
import random
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from typing import Any

import cv2
import numpy as np
import torch


SIGLIP_ID = "google/siglip-base-patch16-224"
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


# Broken/unreachable proxies make Hub HEAD calls hang or raise Errno 101/104.
for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY",
            "all_proxy", "ALL_PROXY", "socks5_proxy", "SOCKS5_PROXY"]:
    os.environ.pop(var, None)

# transformers still HEADs optional files (e.g. processor_config.json) unless
# offline mode is on — even with local_files_only=True in some versions.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def from_pretrained(loader, model_id, **kwargs):
    """Load from local HF cache (offline). Unset HF_HUB_OFFLINE to allow Hub."""
    kwargs.setdefault("local_files_only", True)
    return loader(model_id, **kwargs)


OOD_FEATURE_NAMES = [
    "max_cos", "top2_cos", "margin_12", "mean_top5", "mean_all", "std_all",
    "entropy_top20", "max_over_mean", "mean5_minus_mean", "max_minus_mean5",
    "pct_above_0_90", "pct_above_0_85", "pct_above_0_80",
    "pct_above_0_75", "pct_above_0_70", "pct_above_0_65",
]


def cosine_profile(sims: torch.Tensor) -> np.ndarray:
    """16-d cosine-similarity profile used by the OOD SVM."""
    s = sims.float().clone()
    s = s[s > -0.5]
    if s.numel() == 0:
        return np.zeros(len(OOD_FEATURE_NAMES), dtype=np.float64)
    vals = torch.topk(s, k=min(20, s.numel())).values
    max_cos = float(vals[0])
    top2 = float(vals[1]) if vals.numel() > 1 else max_cos
    mean5 = float(vals[: min(5, vals.numel())].mean())
    mean_all = float(s.mean())
    std_all = float(s.std(unbiased=False))
    p = torch.softmax(vals[: min(20, vals.numel())] / 0.05, dim=0)
    ent = float(-(p * p.clamp_min(1e-12).log()).sum())
    dens = [float((s >= t).float().mean()) for t in (0.90, 0.85, 0.80, 0.75, 0.70, 0.65)]
    return np.array(
        [
            max_cos, top2, max_cos - top2, mean5, mean_all, std_all, ent,
            max_cos / max(mean_all, 1e-6), mean5 - mean_all, max_cos - mean5,
            *dens,
        ],
        dtype=np.float64,
    )


def ood_features(z: np.ndarray, gallery_z: np.ndarray) -> np.ndarray:
    """Profile features for one SigLIP embedding against the in-app gallery."""
    G = F.normalize(torch.from_numpy(np.asarray(gallery_z, dtype=np.float32)).float(), dim=-1)
    z_t = F.normalize(torch.from_numpy(np.asarray(z, dtype=np.float32)).float().view(-1), dim=-1)
    return cosine_profile(G @ z_t)


def prepare_data_for_ood(
    app_dir: str,
    root_dir: str,
    device: str = "cuda",
    per_app: int = 12,
    max_apps: int = 40,
    seed: int = 0,
):
    """
    Build sklearn-ready on/off-graph features from screenshots.

    Parameters
    ----------
    app_dir : str
        In-graph app folder (…/explored_apps/<app>) with screenshots/.
    root_dir : str
        Explored-apps root; *all* sibling apps are OOD candidates (no
        domain-specific shortlist). If there are more than ``max_apps``,
        a random subset is used.
    per_app : int
        Max screenshots sampled per sibling app.
    max_apps : int
        Cap on number of sibling apps (random sample if more exist).
    """
    app_dir = os.path.abspath(app_dir)
    root_dir = os.path.abspath(root_dir)
    app = os.path.basename(app_dir.rstrip(os.sep))

    # ---- SigLIP ----
    from transformers import AutoModel, AutoProcessor
    from PIL import Image

    processor = from_pretrained(AutoProcessor.from_pretrained, SIGLIP_ID)
    model = from_pretrained(
        AutoModel.from_pretrained, SIGLIP_ID, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    def encode(bgr):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pixel_values = processor(images=pil, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            pooled = model.vision_model(pixel_values=pixel_values).pooler_output
            z = model.visual_projection(pooled) if hasattr(model, "visual_projection") else pooled
            z = z[0].float().cpu().numpy().reshape(-1)
        z = z / (np.linalg.norm(z) + 1e-8)
        return z.astype(np.float32)

    def letterbox(bgr, hw):
        th, tw = hw
        h, w = bgr.shape[:2]
        s = min(tw / w, th / h)
        nh, nw = int(h * s), int(w * s)
        resized = cv2.resize(bgr, (nw, nh))
        out = np.zeros((th, tw, 3), dtype=np.uint8)
        y0, x0 = (th - nh) // 2, (tw - nw) // 2
        out[y0 : y0 + nh, x0 : x0 + nw] = resized
        return out

    def list_shots(folder):
        d = os.path.join(folder, "screenshots")
        if not os.path.isdir(d):
            return []
        return sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

    # ---- in-app gallery ----
    in_paths = list_shots(app_dir)
    if not in_paths:
        raise RuntimeError(f"no screenshots in {app_dir}/screenshots")

    # majority letterbox size
    counts = {}
    for p in in_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        hw = (im.shape[0], im.shape[1])
        counts[hw] = counts.get(hw, 0) + 1
    target_hw = max(counts, key=counts.get)

    node_ids, gallery = [], []
    for p in in_paths:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        z = encode(letterbox(im, target_hw))
        # skip near-black
        if float(im.mean()) < 5.0:
            continue
        gallery.append(z)
        node_ids.append(os.path.splitext(os.path.basename(p))[0])
    gallery_z = np.stack(gallery).astype(np.float32)
    G = torch.from_numpy(gallery_z)
    G = F.normalize(G.float(), dim=-1)

    # LOO positives
    sims = G @ G.T
    X_in = []
    for i in range(sims.shape[0]):
        row = sims[i].clone()
        row[i] = -1.0
        X_in.append(cosine_profile(row))
    X_in = np.stack(X_in)
    # keep well-connected nodes
    loo_max = X_in[:, 0]
    keep = loo_max >= 0.75
    if keep.sum() < 20:
        keep = loo_max >= np.percentile(loo_max, 20)
    X_in = X_in[keep]

    # ---- OOD negatives: all sibling apps under root_dir (equal treatment) ----
    others = []
    for name in sorted(os.listdir(root_dir)):
        if name == app:
            continue
        d = os.path.join(root_dir, name)
        shots = list_shots(d)
        if len(shots) >= 5 and os.path.isfile(os.path.join(d, "graph.json")):
            others.append(name)

    rng = random.Random(seed)
    # optional cap: random subset of apps (uniform — no domain bias)
    if len(others) > max_apps:
        others = rng.sample(others, max_apps)
        others.sort()

    X_ood, ood_apps = [], []
    for name in others:
        shots = list_shots(os.path.join(root_dir, name))
        rng.shuffle(shots)
        for p in shots[:per_app]:
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            if im is None:
                continue
            X_ood.append(ood_features(encode(letterbox(im, target_hw)), gallery_z))
            ood_apps.append(name)
    if not X_ood:
        raise RuntimeError(f"no OOD screenshots under {root_dir}")
    X_ood = np.stack(X_ood)

    X = np.concatenate([X_in, X_ood], axis=0)
    y = np.concatenate(
        [np.ones(len(X_in), dtype=np.int64), np.zeros(len(X_ood), dtype=np.int64)]
    )

    return {
        "X": X,
        "y": y,
        "X_in": X_in,
        "X_ood": X_ood,
        "ood_apps": np.asarray(ood_apps),
        "feature_names": OOD_FEATURE_NAMES,
        "gallery_z": gallery_z,
        "node_ids": node_ids,
        "app": app,
        "target_hw": target_hw,
    }

def train_ood_classifier(
    data: dict, seed: int = 0, n_splits: int = 5, out_path: str | None = None
):
    """
    Train on/off-graph classifier on ``prepare_data_for_ood`` output with K-Fold validation for balanced accuracy.

    Model: StandardScaler + SVM-RBF (C=20), class_weight=balanced.
    Threshold: maximize balanced accuracy on the train set (and compute validation acc via cross-validation).

    Returns a dict with ``model`` (Pipeline), ``threshold``, ``train_balanced_acc``, ``val_balanced_acc``.
    If ``out_path`` is set, dumps that dict via joblib.
    """
    import numpy as np
    import joblib
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold

    X, y = data["X"], data["y"]

    # ----- K-Fold cross-validation -----
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    val_bals = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        C=20.0,
                        gamma="scale",
                        class_weight="balanced",
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(X_train, y_train)
        proba_val = pipe.predict_proba(X_val)[:, 1]
        # Find the threshold maximizing val balanced accuracy using only val split
        best_t, best_bal = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 37):
            pred = (proba_val >= t).astype(np.int64)
            tp = ((pred == 1) & (y_val == 1)).sum()
            tn = ((pred == 0) & (y_val == 0)).sum()
            p = max(int((y_val == 1).sum()), 1)
            n = max(int((y_val == 0).sum()), 1)
            bal = 0.5 * (tp / p + tn / n)
            if bal > best_bal:
                best_bal, best_t = float(bal), float(t)
        val_bals.append(best_bal)
    val_balanced_acc = float(np.mean(val_bals))

    # ----- Train final model on all data for output -----
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=20.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)[:, 1]
    best_t, best_bal = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 37):
        pred = (proba >= t).astype(np.int64)
        tp = ((pred == 1) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum()
        p = max(int((y == 1).sum()), 1)
        n = max(int((y == 0).sum()), 1)
        bal = 0.5 * (tp / p + tn / n)
        if bal > best_bal:
            best_bal, best_t = float(bal), float(t)

    result = {
        "model": pipe,
        "threshold": best_t,
        "train_balanced_acc": best_bal,
        "val_balanced_acc": val_balanced_acc,
        "feature_names": data["feature_names"],
        "app": data["app"],
    }
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        joblib.dump(result, out_path)
        print(f"saved OOD model → {out_path}")
    return result


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-8)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def _letterbox(bgr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(hw[0]), int(hw[1])
    h, w = bgr.shape[:2]
    if h == th and w == tw:
        return bgr
    s = min(tw / max(w, 1), th / max(h, 1))
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), dtype=bgr.dtype)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _is_blank(bgr: np.ndarray, mean_thresh: float = 15.0, dark_frac: float = 0.92) -> bool:
    if bgr is None or bgr.size == 0:
        return True
    crop = bgr[: max(1, int(bgr.shape[0] * 0.92))]
    return bool((crop.mean(axis=2) < mean_thresh).mean() > dark_frac)


def _majority_hw(shots: list[np.ndarray]) -> tuple[int, int]:
    counts: dict[tuple[int, int], int] = {}
    for im in shots:
        hw = (int(im.shape[0]), int(im.shape[1]))
        counts[hw] = counts.get(hw, 0) + 1
    return max(counts, key=counts.get)


def _load_app_screenshots(app_dir: str) -> tuple[list[str], list[np.ndarray]]:
    shot_dir = os.path.join(app_dir, "screenshots")
    node_ids: list[str] = []
    images: list[np.ndarray] = []
    for name in sorted(os.listdir(shot_dir)):
        if not name.endswith(".jpg"):
            continue
        nid = name[: -len(".jpg")]
        path = os.path.join(shot_dir, name)
        img = cv2.imread(path)
        if img is None or _is_blank(img):
            continue
        node_ids.append(nid)
        images.append(img)
    if not node_ids:
        raise RuntimeError(f"No usable screenshots under {shot_dir}")
    return node_ids, images


class _Encoders:
    """Lazy SigLIP + SmolVLM vision encoders."""

    def __init__(self, device: str = "cuda"):
        from PIL import Image
        from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText

        self.device = device
        self._Image = Image

        self.siglip_proc = from_pretrained(AutoProcessor.from_pretrained, SIGLIP_ID)
        self.siglip = from_pretrained(
            AutoModel.from_pretrained, SIGLIP_ID, torch_dtype=torch.float16
        ).to(device)
        self.siglip.eval()

        self.smol_proc = from_pretrained(AutoProcessor.from_pretrained, SMOLVLM_ID)
        self.smol = from_pretrained(
            AutoModelForImageTextToText.from_pretrained,
            SMOLVLM_ID,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        ).to(device)
        self.smol.eval()
        self.image_token_id = getattr(self.smol.config, "image_token_id", None)

    def _pil(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._Image.fromarray(rgb)

    @torch.inference_mode()
    def siglip_feat(self, bgr: np.ndarray) -> np.ndarray:
        pil = self._pil(bgr)
        pixel_values = self.siglip_proc(images=pil, return_tensors="pt")["pixel_values"].to(
            self.device
        )
        pooled = self.siglip.vision_model(pixel_values=pixel_values).pooler_output
        z = self.siglip.visual_projection(pooled) if hasattr(self.siglip, "visual_projection") else pooled
        z = z[0].float().cpu().numpy().reshape(-1)
        return _l2(z)

    @torch.inference_mode()
    def smol_vision_feat(self, bgr: np.ndarray) -> np.ndarray:
        """Mean-pool SmolVLM last-layer hidden states over image tokens only."""
        pil = self._pil(bgr)
        pil.thumbnail((512, 512))
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Screenshot."},
                ],
            }
        ]
        prompt = self.smol_proc.apply_chat_template(msgs, add_generation_prompt=False)
        inputs = self.smol_proc(text=prompt, images=[pil], return_tensors="pt")
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }
        out = self.smol(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1][0]
        input_ids = inputs["input_ids"][0]
        if self.image_token_id is not None:
            mask = input_ids == self.image_token_id
            if bool(mask.any()):
                return _l2(hs[mask].mean(0).float().cpu().numpy().reshape(-1))
        am = inputs["attention_mask"][0].bool()
        return _l2(hs[am].mean(0).float().cpu().numpy().reshape(-1))

    def concat_feat(self, bgr: np.ndarray) -> np.ndarray:
        sig = np.asarray(self.siglip_feat(bgr), dtype=np.float32).reshape(-1)
        smol = np.asarray(self.smol_vision_feat(bgr), dtype=np.float32).reshape(-1)
        return _l2(np.concatenate([sig, smol], axis=0))


def save_screenshot_features(
    app_dir: str,
    out_path: str,
    *,
    device: str = "cuda",
    target_hw: tuple[int, int] | None = None,
    encoders: _Encoders | None = None,
) -> dict[str, Any]:
    """
    Encode all non-blank screenshots under ``app_dir/screenshots/`` and save.

    Each gallery vector is
    ``l2(concat(l2(SigLIP(img)), l2(SmolVLM-vision(img))))``.

    Parameters
    ----------
    app_dir : str
        Explored-app folder containing ``screenshots/<node_id>.jpg``.
    out_path : str
        Destination ``.pt`` path.
    device, target_hw, encoders : optional
        Device, letterbox size (default = majority screenshot HW), reused encoders.

    Returns
    -------
    dict
        Saved payload (also written to ``out_path``).
    """
    node_ids, images = _load_app_screenshots(app_dir)
    if target_hw is None:
        target_hw = _majority_hw(images)

    enc = encoders or _Encoders(device)
    feats: list[np.ndarray] = []
    for i, (nid, img) in enumerate(zip(node_ids, images)):
        shot = _letterbox(img, target_hw)
        feats.append(enc.concat_feat(shot))
        if (i + 1) % 25 == 0 or i + 1 == len(node_ids):
            print(f"  features {i + 1}/{len(node_ids)}")

    gallery_z = np.stack(feats).astype(np.float32)
    payload = {
        "method": "siglip+smolvlm256_vision_concat",
        "node_ids": node_ids,
        "target_hw": (int(target_hw[0]), int(target_hw[1])),
        "gallery_z": gallery_z,
        "siglip_id": SIGLIP_ID,
        "smolvlm_id": SMOLVLM_ID,
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    torch.save(payload, out_path)
    print(f"saved {gallery_z.shape[0]} nodes → {out_path}  dim={gallery_z.shape[1]}")
    return payload



if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--app_dir", type=str, default="explore_apps/amazon")
    parser.add_argument("--root_dir", type=str, default="explore_apps")

    args = parser.parse_args()
    app_dir = args.app_dir
    root_dir = args.root_dir

    data_ood = prepare_data_for_ood(app_dir, root_dir)
    print("Training OOD classifier...")
    result = train_ood_classifier(
        data_ood, out_path=os.path.join(app_dir, "ood_classifier.joblib")
    )
    print("Saving screenshot features...")
    payload = save_screenshot_features(
        app_dir, os.path.join(app_dir, "siglip_smolvlm_features.pt")
    )

