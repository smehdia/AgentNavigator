"""Minimal training loop for UIGraphEmbedder."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F

from losses import (
    batch_siamese_text_similarity_loss,
    batch_soft_geodesic_loss,
    direct_text_alignment_loss,
    get_symmetric_graph_distance,
    interactive_element_loss,
    node_prototype_loss,
    on_graph_detection_loss,
    soft_set_alignment_loss,
    transition_embedding_loss,
)
from model import (
    NONE,
    TAP,
    UIGraphEmbedder,
    FrozenMAIUIVisualExtractor,
    count_trainable_parameters,
)
from prepare_data import DEFAULT_HISTORY_MODE_PROBS, get_data, load_all_graphs

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForImageTextToText, AutoProcessor


for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)


@dataclass
class LossWeights:
    node_localization: float = 1.0
    soft_geodesic: float = 1.0
    transition: float = 1.0
    interactive_element: float = 1.0
    description_relational: float = 1.0
    tab_relational: float = 1.0
    subtab_relational: float = 1.0
    waypoint_direct: float = 1.0
    waypoint_siamese: float = 1.0
    transition_hint_direct: float = 1.0
    transition_hint_siamese: float = 1.0
    intent_set: float = 1.0
    intent_siamese: float = 1.0
    on_graph: float = 1.0
    canonical_layout: float = 1.0


def grid_hw_from_positions(positions: torch.Tensor) -> tuple[int, int]:
    xs = positions[:, 0].unique().numel()
    ys = positions[:, 1].unique().numel()
    return int(ys), int(xs)


def encode_texts(text_encoder, texts: list[str], device) -> torch.Tensor:
    """
    text_encoder(texts) -> np.ndarray | torch.Tensor of shape [B, D].
    Empty strings become zero vectors.
    """
    if not texts:
        raise ValueError("texts must be non-empty.")

    raw = text_encoder(texts)
    if isinstance(raw, np.ndarray):
        emb = torch.from_numpy(raw).float()
    else:
        emb = raw.float()

    emb = emb.to(device)
    for i, text in enumerate(texts):
        if not str(text).strip():
            emb[i].zero_()
    return F.normalize(emb, dim=-1)


def encode_intent_sets(
    text_encoder,
    intent_lists: list[list[str]],
    device,
    max_intents: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        set_embeddings: [B, K, D]
        set_mask: [B, K]
        set_pooled: [B, D] mean over available intents
    """
    batch_size = len(intent_lists)
    flat_texts = []
    owners = []
    for batch_index, intents in enumerate(intent_lists):
        kept = [str(x).strip() for x in intents if str(x).strip()][:max_intents]
        if not kept:
            kept = [""]
        for text in kept:
            flat_texts.append(text)
            owners.append(batch_index)

    flat = encode_texts(text_encoder, flat_texts, device)
    dim = flat.shape[-1]
    set_embeddings = torch.zeros(batch_size, max_intents, dim, device=device)
    set_mask = torch.zeros(batch_size, max_intents, dtype=torch.bool, device=device)

    fill_count = [0] * batch_size
    for row, batch_index in enumerate(owners):
        slot = fill_count[batch_index]
        if slot >= max_intents:
            continue
        if not flat_texts[row].strip():
            continue
        set_embeddings[batch_index, slot] = flat[row]
        set_mask[batch_index, slot] = True
        fill_count[batch_index] += 1

    counts = set_mask.sum(dim=1).clamp_min(1).unsqueeze(-1).float()
    set_pooled = set_embeddings.sum(dim=1) / counts
    set_pooled = F.normalize(set_pooled, dim=-1)
    return set_embeddings, set_mask, set_pooled


def sample_batch(
    dict_graph_info,
    root_dir,
    selected_app,
    batch_size: int,
    off_graph_ratio: float = 0.25,
    history_mode_probs: dict | None = None,
) -> list[dict]:
    samples = []
    for _ in range(batch_size):
        off_graph = random.random() < off_graph_ratio
        samples.append(
            get_data(
                dict_graph_info,
                root_dir,
                selected_app,
                off_app_sample=off_graph,
                history_mode_probs=history_mode_probs,
            )
        )
    return samples


@torch.no_grad()
def extract_visual_batch(visual_extractor, screenshots, device, dtype):
    tokens, positions, image_hws, grids = [], [], [], []
    for screenshot in screenshots:
        visual = visual_extractor(screenshot)
        tokens.append(visual.features.to(device=device, dtype=dtype))
        positions.append(visual.positions.to(device=device, dtype=dtype))
        image_hws.append(visual.image_hw)
        grids.append(grid_hw_from_positions(visual.positions))
    return (
        torch.stack(tokens, dim=0),
        torch.stack(positions, dim=0),
        image_hws,
        grids,
    )


def build_action_tensors(samples, previous_image_hws, device, dtype):
    batch_size = len(samples)
    action_type = torch.full((batch_size,), NONE, device=device, dtype=torch.long)
    action_xy = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    history_available = torch.zeros(batch_size, dtype=torch.bool, device=device)
    supervise_transition = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for i, sample in enumerate(samples):
        mode = sample.get("history_mode", "none")
        has_prev = sample["screenshot_prev"] is not None
        # Corrupted/valid keep history_available=True so the model sees history.
        # "none" forces history_available=False.
        history_available[i] = has_prev and mode != "none"
        supervise_transition[i] = bool(sample.get("supervise_transition", False))
        if not history_available[i]:
            continue

        action_type[i] = int(sample["action_type"])
        if action_type[i].item() == TAP:
            xy = sample["action_coordinate_center"]
            if xy is None:
                raise ValueError(f"Tap sample {sample['node_id']} missing coordinates.")
            image_h, image_w = previous_image_hws[i]
            action_xy[i, 0] = xy[0] / max(image_w - 1, 1)
            action_xy[i, 1] = xy[1] / max(image_h - 1, 1)

    return (
        action_type,
        action_xy.clamp(0, 1),
        history_available,
        supervise_transition,
    )


def build_graph_distance_matrix(samples, dict_graph_info, device):
    batch_size = len(samples)
    distances = torch.zeros(batch_size, batch_size, device=device)
    valid = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)

    for i, sample_i in enumerate(samples):
        if not sample_i["on_graph"]:
            continue
        distance_dict = dict_graph_info[sample_i["app_name"]]["distance_dict"]
        for j, sample_j in enumerate(samples):
            if i == j or not sample_j["on_graph"]:
                continue
            if sample_i["app_name"] != sample_j["app_name"]:
                continue
            distance = get_symmetric_graph_distance(
                distance_dict,
                sample_i["node_id"],
                sample_j["node_id"],
            )
            if distance is None:
                continue
            distances[i, j] = distance
            valid[i, j] = True

    return distances, valid


def compute_losses(
    out,
    samples,
    dict_graph_info,
    node_id_to_index: dict[str, int],
    node_prototypes: nn.Parameter,
    text_encoder,
    image_hws: list[tuple[int, int]],
    grids: list[tuple[int, int]],
    supervise_transition: torch.Tensor,
    device,
    weights: LossWeights,
    max_intents: int = 8,
) -> dict[str, torch.Tensor]:
    on_graph = torch.tensor(
        [bool(s["on_graph"]) for s in samples],
        device=device,
        dtype=torch.bool,
    )
    off_graph = ~on_graph
    zero = out.embedding.new_zeros(())

    losses: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Always active
    # ------------------------------------------------------------------ #
    losses["on_graph"] = on_graph_detection_loss(
        out.on_graph_logit,
        on_graph.float(),
    )

    element_terms = []
    for i, sample in enumerate(samples):
        element_terms.append(
            interactive_element_loss(
                out.current_element_logits[i : i + 1],
                [sample["ui_bboxes"]],
                image_hw=image_hws[i],
                grid_hw=grids[i],
            )
        )
    losses["interactive_element"] = torch.stack(element_terms).mean()

    layout_text = [s["canonical_page_layout"] or "" for s in samples]
    layout_emb = encode_texts(text_encoder, layout_text, device)
    layout_available = off_graph & torch.tensor(
        [bool(str(t).strip()) for t in layout_text],
        device=device,
        dtype=torch.bool,
    )
    losses["canonical_layout"] = (
        direct_text_alignment_loss(
            out.predicted_description_embedding,
            layout_emb,
            available_mask=layout_available,
        )
        if layout_available.any()
        else zero
    )

    # ------------------------------------------------------------------ #
    # On-graph only
    # ------------------------------------------------------------------ #
    if on_graph.any():
        idx = on_graph.nonzero(as_tuple=False).squeeze(-1)

        node_indices = []
        for sample in (samples[i] for i in idx.tolist()):
            node_indices.append(node_id_to_index[sample["node_id"]])
        node_indices = torch.tensor(node_indices, device=device, dtype=torch.long)

        losses["node_localization"] = node_prototype_loss(
            out.embedding[idx],
            node_indices,
            node_prototypes,
        )

        distance_matrix, valid_pairs = build_graph_distance_matrix(
            samples,
            dict_graph_info,
            device,
        )
        # Restrict geodesic pairs to on-graph rows/cols.
        on_pair = on_graph[:, None] & on_graph[None, :]
        losses["soft_geodesic"] = batch_soft_geodesic_loss(
            out.embedding,
            distance_matrix,
            valid_pair_mask=valid_pairs & on_pair,
        )

        losses["transition"] = transition_embedding_loss(
            out.predicted_current_screen,
            out.current_screen,
            history_available=supervise_transition & on_graph,
        )

        desc_text = [s["page_description"] or "" for s in samples]
        tab_text = [s["active_tab"] or "" for s in samples]
        subtab_text = [s["active_subtab"] or "" for s in samples]
        waypoint_text = [s["waypoint_text"] or "" for s in samples]
        hint_text = [s["transition_hint_text"] or "" for s in samples]

        desc_emb = encode_texts(text_encoder, desc_text, device)
        tab_emb = encode_texts(text_encoder, tab_text, device)
        subtab_emb = encode_texts(text_encoder, subtab_text, device)
        waypoint_emb = encode_texts(text_encoder, waypoint_text, device)
        hint_emb = encode_texts(text_encoder, hint_text, device)

        desc_avail = on_graph & torch.tensor(
            [bool(t.strip()) for t in desc_text], device=device, dtype=torch.bool
        )
        tab_avail = on_graph & torch.tensor(
            [bool(t.strip()) for t in tab_text], device=device, dtype=torch.bool
        )
        subtab_avail = on_graph & torch.tensor(
            [bool(t.strip()) for t in subtab_text], device=device, dtype=torch.bool
        )
        waypoint_avail = on_graph & torch.tensor(
            [bool(t.strip()) for t in waypoint_text], device=device, dtype=torch.bool
        )
        hint_avail = on_graph & torch.tensor(
            [bool(t.strip()) for t in hint_text], device=device, dtype=torch.bool
        )

        losses["description_relational"] = batch_siamese_text_similarity_loss(
            out.predicted_description_embedding,
            desc_emb,
            available_mask=desc_avail,
        )
        losses["tab_relational"] = batch_siamese_text_similarity_loss(
            out.predicted_tab_embedding,
            tab_emb,
            available_mask=tab_avail,
        )
        losses["subtab_relational"] = batch_siamese_text_similarity_loss(
            out.predicted_subtab_embedding,
            subtab_emb,
            available_mask=subtab_avail,
        )

        losses["waypoint_direct"] = direct_text_alignment_loss(
            out.predicted_waypoint_embedding,
            waypoint_emb,
            available_mask=waypoint_avail,
        )
        losses["waypoint_siamese"] = batch_siamese_text_similarity_loss(
            out.predicted_waypoint_embedding,
            waypoint_emb,
            available_mask=waypoint_avail,
        )

        losses["transition_hint_direct"] = direct_text_alignment_loss(
            out.predicted_transition_hint_embedding,
            hint_emb,
            available_mask=hint_avail,
        )
        losses["transition_hint_siamese"] = batch_siamese_text_similarity_loss(
            out.predicted_transition_hint_embedding,
            hint_emb,
            available_mask=hint_avail,
        )

        intent_lists = [s["intent_list"] for s in samples]
        set_emb, set_mask, set_pooled = encode_intent_sets(
            text_encoder,
            intent_lists,
            device,
            max_intents=max_intents,
        )
        # Zero out off-graph rows in the set mask.
        set_mask = set_mask & on_graph[:, None]
        losses["intent_set"] = soft_set_alignment_loss(
            out.predicted_intent_embedding,
            set_emb,
            set_mask=set_mask,
            available_mask=on_graph,
        )
        losses["intent_siamese"] = batch_siamese_text_similarity_loss(
            out.predicted_intent_embedding,
            set_pooled,
            available_mask=on_graph & set_mask.any(dim=-1),
        )
    else:
        for key in (
            "node_localization",
            "soft_geodesic",
            "transition",
            "description_relational",
            "tab_relational",
            "subtab_relational",
            "waypoint_direct",
            "waypoint_siamese",
            "transition_hint_direct",
            "transition_hint_siamese",
            "intent_set",
            "intent_siamese",
        ):
            losses[key] = zero

    total = (
        weights.on_graph * losses["on_graph"]
        + weights.interactive_element * losses["interactive_element"]
        + weights.canonical_layout * losses["canonical_layout"]
        + weights.node_localization * losses["node_localization"]
        + weights.soft_geodesic * losses["soft_geodesic"]
        + weights.transition * losses["transition"]
        + weights.description_relational * losses["description_relational"]
        + weights.tab_relational * losses["tab_relational"]
        + weights.subtab_relational * losses["subtab_relational"]
        + weights.waypoint_direct * losses["waypoint_direct"]
        + weights.waypoint_siamese * losses["waypoint_siamese"]
        + weights.transition_hint_direct * losses["transition_hint_direct"]
        + weights.transition_hint_siamese * losses["transition_hint_siamese"]
        + weights.intent_set * losses["intent_set"]
        + weights.intent_siamese * losses["intent_siamese"]
    )
    losses["loss"] = total
    return losses


def localization_accuracy(
    embeddings: torch.Tensor,
    samples: list[dict],
    node_id_to_index: dict[str, int],
    node_prototypes: torch.Tensor,
) -> float | None:
    """Top-1 prototype match accuracy over on-graph samples in the batch."""
    on_graph_indices = [
        i for i, sample in enumerate(samples) if sample.get("on_graph")
    ]
    if not on_graph_indices:
        return None

    idx = torch.tensor(on_graph_indices, device=embeddings.device)
    emb = F.normalize(embeddings[idx], dim=-1)
    prototypes = F.normalize(node_prototypes, dim=-1)
    pred = (emb @ prototypes.T).argmax(dim=-1)

    targets = torch.tensor(
        [node_id_to_index[samples[i]["node_id"]] for i in on_graph_indices],
        device=embeddings.device,
        dtype=torch.long,
    )
    return float((pred == targets).float().mean().item())


def format_sample_summary(sample: dict) -> str:
    def clip(text, n=80):
        text = str(text or "").replace("\n", " ").strip()
        return text if len(text) <= n else text[: n - 3] + "..."

    return (
        f"app={sample.get('app_name')}  node={sample.get('node_id')}  "
        f"on_graph={sample.get('on_graph')}  history_mode={sample.get('history_mode')}  "
        f"prev={sample.get('previous_node_id')}  action={sample.get('action_type')}  "
        f"supervise_transition={sample.get('supervise_transition')}\n"
        f"  tab={clip(sample.get('active_tab'))!r}\n"
        f"  desc={clip(sample.get('page_description'))!r}\n"
        f"  waypoint={clip(sample.get('waypoint_text'))!r}\n"
        f"  intents={clip(sample.get('intent_list'))!r}"
    )


def save_embedding_checkpoint(
    path: str,
    model: UIGraphEmbedder,
    node_prototypes: nn.Parameter,
    node_ids: list[str],
    selected_app: str,
    step: int,
    metrics: dict | None = None,
):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "step": step,
        "selected_app": selected_app,
        "node_ids": list(node_ids),
        "config": {
            "token_dim": model.token_dim,
            "pooled_dim": model.pooled_dim,
            "embedding_dim": model.embedding_dim,
            "text_embedding_dim": model.text_embedding_dim,
        },
        # Inference backbone: screenshot + prev screenshot + prev action → embedding
        "embedding_state_dict": model.embedding_state_dict(),
        "node_prototypes": node_prototypes.detach().cpu(),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def train_step(
    model: UIGraphEmbedder,
    visual_extractor,
    text_encoder,
    samples: list[dict],
    dict_graph_info,
    node_id_to_index: dict[str, int],
    node_prototypes: nn.Parameter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    weights: LossWeights | None = None,
):
    weights = weights or LossWeights()
    model.train()
    dtype = next(model.parameters()).dtype

    current_tokens, current_positions, image_hws, grids = extract_visual_batch(
        visual_extractor,
        [s["screenshot"] for s in samples],
        device,
        dtype,
    )

    has_prev = [s["screenshot_prev"] is not None for s in samples]
    if any(has_prev):
        # Pad missing previous screenshots with a blank frame matching current size.
        prev_shots = []
        for sample, current in zip(samples, [s["screenshot"] for s in samples]):
            if sample["screenshot_prev"] is not None:
                prev_shots.append(sample["screenshot_prev"])
            else:
                prev_shots.append(np.zeros_like(current))
        previous_tokens, previous_positions, prev_hws, _ = extract_visual_batch(
            visual_extractor,
            prev_shots,
            device,
            dtype,
        )
    else:
        previous_tokens = None
        previous_positions = None
        prev_hws = image_hws

    action_type, action_xy, history_available, supervise_transition = (
        build_action_tensors(
            samples,
            prev_hws,
            device,
            dtype,
        )
    )

    out = model(
        current_tokens=current_tokens,
        current_positions=current_positions,
        previous_tokens=previous_tokens if history_available.any() else None,
        previous_positions=previous_positions if history_available.any() else None,
        action_type=action_type,
        action_xy=action_xy,
        history_available=history_available,
    )

    losses = compute_losses(
        out=out,
        samples=samples,
        dict_graph_info=dict_graph_info,
        node_id_to_index=node_id_to_index,
        node_prototypes=node_prototypes,
        text_encoder=text_encoder,
        image_hws=image_hws,
        grids=grids,
        supervise_transition=supervise_transition,
        device=device,
        weights=weights,
    )

    loc_acc = localization_accuracy(
        out.embedding.detach(),
        samples,
        node_id_to_index,
        node_prototypes.detach(),
    )

    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()

    logs = {k: float(v.detach()) for k, v in losses.items()}
    logs["localization_accuracy"] = loc_acc
    return logs


def train(
    selected_app: str = "amazon",
    root_dir: str = "/home/mehdi/Desktop/github_mehdi/AgentNavigator/exploration/explored_apps",
    steps: int = 1000,
    batch_size: int = 4,
    lr: float = 1e-4,
    off_graph_ratio: float = 0.25,
    history_mode_probs: dict | None = None,
    device: str | None = None,
    visual_extractor=None,
    text_encoder=None,
    text_embedding_dim: int = 1024,
    checkpoint_dir: str = "checkpoints",
    log_every: int = 100,
):
    if visual_extractor is None or text_encoder is None:
        raise ValueError(
            "Pass visual_extractor and text_encoder callables/objects. "
            "text_encoder(list[str]) -> [B, D] numpy/torch embeddings."
        )

    history_mode_probs = history_mode_probs or DEFAULT_HISTORY_MODE_PROBS

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dict_graph_info = load_all_graphs(root_dir)
    if selected_app not in dict_graph_info:
        raise KeyError(f"App {selected_app!r} not found in {root_dir}.")

    node_ids = dict_graph_info[selected_app]["node_ids"]
    node_id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    embedding_dim = 256
    model = UIGraphEmbedder(
        embedding_dim=embedding_dim,
        text_embedding_dim=text_embedding_dim,
    ).to(device)
    node_prototypes = nn.Parameter(
        F.normalize(
            torch.randn(len(node_ids), embedding_dim, device=device),
            dim=-1,
        )
    )

    embedding_params = model.embedding_parameter_count()
    full_params = count_trainable_parameters(model)
    print(
        f"Trainable params — embedding path (inference): {embedding_params:,}  |  "
        f"full model: {full_params:,}"
    )
    print(
        f"batch_size={batch_size}  steps={steps}  log_every={log_every}  "
        f"app={selected_app}  nodes={len(node_ids)}"
    )

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + [node_prototypes],
        lr=lr,
    )
    weights = LossWeights()
    os.makedirs(checkpoint_dir, exist_ok=True)

    for step in range(1, steps + 1):
        samples = sample_batch(
            dict_graph_info,
            root_dir,
            selected_app,
            batch_size=batch_size,
            off_graph_ratio=off_graph_ratio,
            history_mode_probs=history_mode_probs,
        )
        logs = train_step(
            model=model,
            visual_extractor=visual_extractor,
            text_encoder=text_encoder,
            samples=samples,
            dict_graph_info=dict_graph_info,
            node_id_to_index=node_id_to_index,
            node_prototypes=node_prototypes,
            optimizer=optimizer,
            device=device,
            weights=weights,
        )

        if step == 1 or step % log_every == 0 or step == steps:
            loc = logs.get("localization_accuracy")
            loc_str = "n/a" if loc is None else f"{100.0 * loc:.1f}%"
            print("=" * 72)
            print(
                f"step {step:05d}/{steps}  "
                f"localization_acc={loc_str}  "
                f"loss={logs['loss']:.4f}"
            )
            print("losses:")
            for key, value in logs.items():
                if key in ("loss", "localization_accuracy"):
                    continue
                print(f"  {key:28s} {value:.4f}")
            print("sample:")
            print(" ", format_sample_summary(samples[0]))

            ckpt_path = os.path.join(
                checkpoint_dir,
                f"{selected_app}_embedding_step{step:05d}.pt",
            )
            save_embedding_checkpoint(
                path=ckpt_path,
                model=model,
                node_prototypes=node_prototypes,
                node_ids=node_ids,
                selected_app=selected_app,
                step=step,
                metrics={
                    "loss": logs["loss"],
                    "localization_accuracy": loc,
                },
            )
            print(f"saved embedding checkpoint → {ckpt_path}")

    # Always keep a "latest" pointer for inference.
    latest_path = os.path.join(checkpoint_dir, f"{selected_app}_embedding_latest.pt")
    save_embedding_checkpoint(
        path=latest_path,
        model=model,
        node_prototypes=node_prototypes,
        node_ids=node_ids,
        selected_app=selected_app,
        step=steps,
        metrics={"loss": logs["loss"], "localization_accuracy": logs.get("localization_accuracy")},
    )
    print(f"saved latest embedding checkpoint → {latest_path}")

    return model, node_prototypes


if __name__ == "__main__":

    mai_ui_path = "/home/mehdi/Desktop/MAI-UI-2B/"
    text_embedding_dim = 1024  # BGE-M3 dense size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(mai_ui_path, local_files_only=True)
    mai_model = AutoModelForImageTextToText.from_pretrained(
        mai_ui_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    mai_model.eval()

    visual_extractor = FrozenMAIUIVisualExtractor(
        processor=processor,
        model=mai_model,
    )

    bge_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=(device == "cuda"))

    def text_encoder(texts: list[str]):
        return bge_model.encode(texts, max_length=1024)["dense_vecs"]

    train(
        visual_extractor=visual_extractor,
        text_encoder=text_encoder,
        text_embedding_dim=text_embedding_dim,
        device=device,
    )
