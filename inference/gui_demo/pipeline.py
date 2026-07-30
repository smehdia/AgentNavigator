"""Inference pipeline helpers for the GUI demo."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import networkx as nx
import numpy as np
from networkx.readwrite import json_graph


def format_candidate_label(candidate: dict) -> str:
    """Render page_tag + page_purpose for UI labels."""
    tag = str(candidate.get("page_tag") or "").strip()
    purpose = str(candidate.get("page_purpose") or "").strip()
    if tag and purpose and tag != purpose:
        return f"{tag}: {purpose}"
    return tag or purpose


def format_page_purpose_display(page_purpose) -> str:
    """Render enhanced_page_summary (or legacy string) for UI labels."""
    if isinstance(page_purpose, dict):
        return format_candidate_label(
            {
                "page_tag": page_purpose.get("tag"),
                "page_purpose": page_purpose.get("page_purpose"),
            }
        )
    return str(page_purpose or "").strip()


def build_retrieval_query(user_goal: str) -> str:
    return str(user_goal or "").strip()


def normalize_ui_navigation_memory(nav_entry) -> list:
    """
    node_navigation_plans.json stores each node as a list of plans:

    {
        "node_id": [
            {
                "relevant_waypoint_sequence": [...],
                "transition_hints": [...]
            }
        ]
    }
    """
    if nav_entry is None:
        return []
    if isinstance(nav_entry, list):
        return nav_entry
    if isinstance(nav_entry, dict):
        memory = nav_entry.get("ui_navigation_memory")
        if isinstance(memory, list):
            return memory
    return []


def build_rerank_candidates(
    task_prompt,
    results,
    node_information,
    node_intents,
    node_navigation_plans,
):
    candidates = []

    for result in results:
        node_id = result["node_id"]

        node_info = node_information.get(node_id, {}) or {}
        intent_info = node_intents.get(node_id, {}) or {}
        nav_info = node_navigation_plans.get(node_id)
        ui_navigation_memory = normalize_ui_navigation_memory(nav_info)

        user_intents = (
            result.get("user_intents")
            or intent_info.get("user_intents")
            or []
        )

        if isinstance(user_intents, dict):
            user_intents = user_intents.get("user_intents", [])

        candidates.append({
            "node_id": node_id,
            "retrieval_score": result.get("score", 0.0),
            "page_description": {
                "high_level": node_info.get("high_level", ""),
                "medium_level": node_info.get("medium_level", ""),
                "low_level": node_info.get("low_level", ""),
            },
            "user_intents": user_intents,
            "ui_navigation_memory": ui_navigation_memory,
        })

    return {
        "task_prompt": task_prompt,
        "candidates": candidates,
    }


def get_node_depths(G: nx.Graph) -> dict:
    roots = [n for n, data in G.nodes(data=True) if data.get("is_root", False)]
    if not roots:
        raise ValueError("No root node found with is_root=True")

    depths = {}
    for root in roots:
        lengths = nx.single_source_shortest_path_length(G, root)
        for node_id, depth in lengths.items():
            if node_id not in depths or depth < depths[node_id]:
                depths[node_id] = depth
    return depths


def format_navigation_plan(final_goal, transition_hints) -> str:
    """
    Build the agent navigation prompt.

    ``transition_hints``: up to 3 dicts with ``high_level`` / ``low_level`` for the
    next hop. If empty, only the final goal is passed (no "no memory" message).
    """
    if not isinstance(final_goal, str):
        raise TypeError("final_goal must be a string")
    if not isinstance(transition_hints, list):
        raise TypeError("transition_hints must be a list")

    goal = final_goal.strip() if final_goal.strip() else "(none)"
    if goal != "(none)" and not goal.lower().endswith("in the current application"):
        goal = f"{goal} in the current application"
    lines = [
        "[Navigation Instruction]",
        f"Final goal: {goal}",
    ]

    if transition_hints:
        lines += [
            "",
            "[Next-step transition hints]",
            "Up to 3 candidate next transitions are listed below, each with a high-level "
            "(intent) and low-level (visual) description. "
            "Compare them to the current screenshot and follow the best-matching one, "
            "unless there is a clear disagreement with what you see — then ignore the "
            "hints and act from the screenshot and final goal. "
            "Prefer low-level to locate the control; use high-level when the low-level "
            "target is not visible.",
            "",
        ]
        for i, hint in enumerate(transition_hints[:3], 1):
            if isinstance(hint, dict):
                low = str(hint.get("low_level", "")).strip()
                high = str(hint.get("high_level", "")).strip()
            else:
                low, high = "", str(hint).strip()
            lines.append(f"[Hint {i}]")
            if high:
                lines.append(f"High: {high}")
            if low:
                lines.append(f"Low: {low}")
            if not high and not low:
                lines.append("(empty)")
            lines.append("")

    lines += [
        "[Global usage instruction]",
        "Then take the action that most likely advances to the next waypoint in that selected plan. "
        "Do not mix steps from different plans unless the screenshot clearly supports doing so. "
        "If the screenshot clearly disagrees with all plans, ignore the plans and follow the screenshot. "
        "Once the current screen already satisfies the final goal, do not perform more navigation actions. "
        "Return the finish/done action immediately. "
        "Don't leave the current application.",
    ]

    return "\n".join(lines).strip()


def _edge_hint(edge_level_information: dict, src: str, dst: str | None) -> dict:
    if not dst:
        return {"low_level": "", "high_level": ""}
    edge_info = edge_level_information.get(f"{src}|{dst}", [])
    if isinstance(edge_info, list):
        edge_info = edge_info[0] if edge_info else {}
    if not isinstance(edge_info, dict):
        edge_info = {}
    return {
        "low_level": edge_info.get("low_level_action_description", ""),
        "high_level": edge_info.get("high_level_action_description", ""),
    }


def localize_screenshot(
    screenshot: np.ndarray,
    *,
    localizer_embedder,
    ood_classifier,
    gallery_z: np.ndarray,
    node_features: np.ndarray,
    feat_node_ids: list,
    target_hw: tuple,
    graph,
    selected_node_id: str | None,
    edge_level_information: dict,
    letterbox_fn,
    ood_features_fn,
    concat_dim: int,
    gallery_norm: np.ndarray | None = None,
) -> dict:
    """OOD check + top-3 cosine match + next-hop transition hints (inference.py parity)."""
    shot = letterbox_fn(screenshot, target_hw)
    # OOD only needs SigLIP; skip SmolVLM when off-graph.
    z = localizer_embedder.siglip_feat(shot)
    ood_feat = ood_features_fn(z, gallery_z)
    ood_label = int(ood_classifier.predict(ood_feat.reshape(1, -1))[0])

    out: dict[str, Any] = {
        "on_graph": ood_label == 1,
        "ood_label": ood_label,
        "top3": [],
        "next_hops": [],
        "transition_hints": [],
        "at_target": False,
    }
    if ood_label != 1:
        return out

    smol = localizer_embedder.smol_vision_feat(shot)
    q = np.concatenate([z, smol], axis=0).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    G = gallery_norm
    if G is None:
        G = np.asarray(node_features, dtype=np.float32)
        G = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-8)
    if q.size != G.shape[1]:
        raise RuntimeError(
            f"feature dim mismatch: query={q.size}, gallery={G.shape[1]} "
            f"(expected {concat_dim}). Re-run save_screenshot_features for this app."
        )
    sims = G @ q
    top = np.argsort(-sims)[:3]
    top3 = [(feat_node_ids[i], float(sims[i])) for i in top]
    out["top3"] = [{"node_id": nid, "score": score} for nid, score in top3]

    top_ids = {nid for nid, _ in top3}
    if selected_node_id and selected_node_id in top_ids:
        out["at_target"] = True
        return out

    next_hops = []
    hints = []
    for nid, score in top3:
        nxt = None
        if selected_node_id:
            try:
                path = nx.shortest_path(graph, nid, selected_node_id)
                nxt = path[1] if len(path) > 1 else nid
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                nxt = None
        next_hops.append({"node_id": nid, "score": score, "next_node_id": nxt})
        hints.append(_edge_hint(edge_level_information, nid, nxt))
    out["next_hops"] = next_hops
    out["transition_hints"] = hints
    return out


def retrieve_nodes_from_user_intents_embeddings(
    embeddings: np.ndarray,
    node_ids: list,
    data: dict,
    model,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    outputs = model.encode(
        [query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query_embedding = np.asarray(outputs["dense_vecs"], dtype=np.float32)
    query_embedding = query_embedding / np.maximum(
        np.linalg.norm(query_embedding, axis=1, keepdims=True),
        1e-12,
    )

    scores = embeddings @ query_embedding[0]
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        node_id = node_ids[int(idx)]
        item = data[node_id]
        results.append(
            {
                "rank": rank,
                "score": float(scores[idx]),
                "node_id": node_id,
                "user_intents": item.get("user_intents", []),
                "embedding_text": item.get("embedding_text", ""),
            }
        )
    return results


def _load_json_if_exists(logs_root: str, filename: str) -> dict:
    path = os.path.join(logs_root, filename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _load_user_intents(logs_root: str) -> dict:
    for filename in ("user_intents.json", "node_intents.json"):
        data = _load_json_if_exists(logs_root, filename)
        if data:
            return data
    return {}


def _scroll_swipe_arrow_points(
    img: np.ndarray,
    coords,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return swipe start/end pixels for a directional scroll (matches BaseDriver)."""
    h, w = img.shape[:2]
    if coords and len(coords) >= 2:
        cx, cy = int(coords[0]), int(coords[1])
    else:
        cx, cy = w // 2, h // 2

    direction = str(direction or "down").strip().lower()
    if direction not in ("up", "down", "left", "right"):
        direction = "down"

    dx, dy = int(w * 0.35), int(h * 0.35)
    if direction == "down":
        return (cx, cy), (cx, max(1, cy - dy))
    if direction == "up":
        return (cx, cy), (cx, min(h - 1, cy + dy))
    if direction == "left":
        return (cx, cy), (max(1, cx - dx), cy)
    return (cx, cy), (min(w - 1, cx + dx), cy)


def visualize_actions_on_screenshots(screenshots, actions):
    if not screenshots or not actions:
        return screenshots

    num_screenshots = len(screenshots)
    num_actions = len(actions)
    annotated = [img.copy() for img in screenshots]

    action_indices = [
        i
        for i in range(min(num_actions, num_screenshots))
        if str(actions[i].get("type", "")).strip().lower() not in {"finished", "finish"}
    ]
    for i in action_indices:
        img = annotated[i]
        act = actions[i]
        color_click = (40, 210, 40)
        color_arrow = (40, 210, 40)
        action_type = str(act["type"]).lower()
        coords = act.get("coordinate", None)

        if action_type in ("click", "tap"):
            if coords:
                x, y = int(coords[0]), int(coords[1])
                cv2.circle(img, (x, y), 40, color_click, thickness=10)
        elif action_type in ("scroll", "swipe"):
            p1, p2 = _scroll_swipe_arrow_points(img, coords, act.get("direction", "down"))
            cv2.arrowedLine(img, p1, p2, color_arrow, thickness=12, tipLength=0.2)
        elif action_type == "drag":
            start = act.get("start_coordinate") or coords
            end = act.get("end_coordinate")
            if start and end:
                p1 = (int(start[0]), int(start[1]))
                p2 = (int(end[0]), int(end[1]))
                cv2.arrowedLine(img, p1, p2, color_arrow, thickness=12, tipLength=0.2)

    return annotated


def build_trajectory_frames(
    screenshots: list,
    actions: list,
    *,
    final_screenshot=None,
    query: Optional[str] = None,
    candidates: Optional[list] = None,
    selected_node_id: Optional[str] = None,
    logs_root: Optional[str] = None,
) -> list[np.ndarray]:
    from .frame_renderer import build_gui_trajectory_frames

    return build_gui_trajectory_frames(
        screenshots,
        actions,
        final_screenshot=final_screenshot,
        query=query,
        candidates=candidates,
        selected_node_id=selected_node_id,
        logs_root=logs_root,
        annotate_screenshots=visualize_actions_on_screenshots,
    )


def trajectory_mp4_filename(query: str, config_id: str = "run") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^\w\-]+", "_", str(query or "").strip()[:40]).strip("_") or "run"
    safe_config = re.sub(r"[^\w\-]+", "_", str(config_id or "run").strip()) or "run"
    return f"{ts}_{safe_config}_{slug}.mp4"


def save_trajectory_mp4(
    screenshots: list,
    actions: list,
    output_path: os.PathLike | str,
    *,
    final_screenshot=None,
    fps: float = 1.0,
    query: Optional[str] = None,
    candidates: Optional[list] = None,
    selected_node_id: Optional[str] = None,
    logs_root: Optional[str] = None,
) -> str:
    """Write trajectory frames (GUI-style cards with action overlays) to an MP4 file."""
    frames = build_trajectory_frames(
        screenshots,
        actions,
        final_screenshot=final_screenshot,
        query=query,
        candidates=candidates,
        selected_node_id=selected_node_id,
        logs_root=logs_root,
    )
    if not frames:
        raise ValueError("No trajectory frames to write")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    return str(output_path.resolve())


def get_agent_last_prompt(agent) -> str:
    if agent is None:
        return ""
    getter = getattr(agent, "get_last_prompt", None)
    if callable(getter):
        return str(getter() or "").strip()
    return ""


def parse_action_from_step(step_result) -> dict:
    parsed = step_result[0] if isinstance(step_result, tuple) else step_result
    action_type = getattr(parsed, "action_type", None) or ""
    coords = None
    if hasattr(parsed, "sent_coords") and parsed.sent_coords and "point" in parsed.sent_coords:
        coords = parsed.sent_coords["point"]
    elif hasattr(parsed, "orig_coords") and parsed.orig_coords and "point" in parsed.orig_coords:
        coords = parsed.orig_coords["point"]
    thought = getattr(parsed, "thought", None) or ""
    params = getattr(parsed, "params", None) or {}
    direction = params.get("direction")

    start_coordinate = None
    end_coordinate = None
    orig_coords = getattr(parsed, "orig_coords", None) or {}
    if "start_point" in orig_coords:
        start_coordinate = orig_coords["start_point"]
    if "end_point" in orig_coords:
        end_coordinate = orig_coords["end_point"]

    return {
        "parsed": parsed,
        "type": action_type,
        "coordinate": coords,
        "direction": direction,
        "start_coordinate": start_coordinate,
        "end_coordinate": end_coordinate,
        "thought": thought,
    }


@dataclass
class RetrievalContext:
    graph: Any
    depths: dict
    node_intents: dict
    node_level_information: dict
    node_navigation_plans: dict
    node_ids: list
    embeddings: np.ndarray
    raw_intent_data: dict = field(default_factory=dict)
    logs_root: str = ""


def load_node_navigation_plans(logs_root: str) -> dict:
    """Load per-node navigation plans from node_navigation_plans.json."""
    return _load_json_if_exists(logs_root, "node_navigation_plans.json")


def get_ui_navigation_memory_for_node(ctx: RetrievalContext, node_id: str) -> list:
    return normalize_ui_navigation_memory(ctx.node_navigation_plans.get(node_id))


def load_artifacts(logs_root: str) -> RetrievalContext:
    logs_root = os.path.abspath(logs_root)
    with open(os.path.join(logs_root, "graph.json"), "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    graph = json_graph.node_link_graph(graph_data, edges="links")
    depths = get_node_depths(graph)

    node_intents = _load_user_intents(logs_root)
    node_level_information = _load_json_if_exists(logs_root, "node_level_information.json")
    node_navigation_plans = load_node_navigation_plans(logs_root)

    node_ids, embedding_rows = zip(
        *[(node_id, item["embedding"]) for node_id, item in node_intents.items() if "embedding" in item]
    ) if node_intents else ([], [])

    embeddings = np.asarray(embedding_rows, dtype=np.float32) if embedding_rows else np.empty((0, 0), dtype=np.float32)

    return RetrievalContext(
        graph=graph,
        depths=depths,
        node_intents=node_intents,
        node_level_information=node_level_information,
        node_navigation_plans=node_navigation_plans,
        node_ids=list(node_ids),
        embeddings=embeddings,
        raw_intent_data=node_intents,
        logs_root=logs_root,
    )


def run_retrieval(
    query: str,
    ctx: RetrievalContext,
    configs,
    embedding_model,
    vlm_client,
) -> tuple[list[dict], dict]:
    if not getattr(configs, "use_memory_for_navigation", False):
        return [], {}

    results = retrieve_nodes_from_user_intents_embeddings(
        embeddings=ctx.embeddings,
        node_ids=ctx.node_ids,
        data=ctx.raw_intent_data,
        model=embedding_model,
        query=build_retrieval_query(query),
        top_k=configs.top_k_in_first_stage_retrieval,
    )

    rerank_payload = build_rerank_candidates(
        query,
        results,
        ctx.node_level_information,
        ctx.node_intents,
        ctx.node_navigation_plans,
    )
    rerank_candidates = rerank_payload["candidates"]

    top_k_node_ids, rerank_result = vlm_client.rerank_candidates(
        query,
        rerank_candidates,
        top_k=configs.top_k_retrieval_in_stage_2,
    )
    reasoning_map = (
        rerank_result.get("reasoning", {})
        if isinstance(rerank_result, dict)
        else {}
    )

    ordered = {nid: i for i, nid in enumerate(top_k_node_ids)}
    candidates = [
        {
            **c,
            "score": c.get("retrieval_score", c.get("score", 0.0)),
            "depth": ctx.depths.get(c["node_id"], 0),
            "page_tag": "",
            "page_purpose": (
                c.get("page_description", {}).get("high_level", "")
                or c.get("page_description", {}).get("medium_level", "")
            ),
        }
        for c in rerank_candidates
        if c["node_id"] in top_k_node_ids
    ]
    candidates = sorted(candidates, key=lambda c: ordered[c["node_id"]])
    return candidates, reasoning_map


def get_navigation_memory_for_node(ctx: RetrievalContext, node_id: str, query: str) -> str:
    # Static pre-execute preview: goal only. Live steps build hints via localization.
    return format_navigation_plan(query, [])


def run_navigation_loop(
    final_goal: str,
    agent,
    driver,
    max_steps: int = 10,
    on_step: Optional[Callable[[dict], None]] = None,
    reset_first: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
    *,
    selected_node_id: Optional[str] = None,
    localizer=None,
) -> tuple[list, list, bool, bool]:
    if reset_first:
        driver.reset_to_start_page()
    agent.clear_history()

    screenshots = []
    actions = []
    finish_flag = False
    stopped = False

    gallery_norm = None
    if localizer is not None:
        nf = np.asarray(localizer["node_features"], dtype=np.float32)
        gallery_norm = nf / (np.linalg.norm(nf, axis=1, keepdims=True) + 1e-8)

    for step_idx in range(max_steps):
        if should_stop and should_stop():
            stopped = True
            break

        step_start = time.time()
        screenshot = driver.take_screenshot()
        driver_screenshot_s = time.time() - step_start

        loc_start = time.time()
        localization = {
            "on_graph": False,
            "ood_label": 0,
            "top3": [],
            "next_hops": [],
            "transition_hints": [],
            "at_target": False,
        }
        if localizer is not None:
            localization = localize_screenshot(
                screenshot,
                localizer_embedder=localizer["embedder"],
                ood_classifier=localizer["ood_classifier"],
                gallery_z=localizer["gallery_z"],
                node_features=localizer["node_features"],
                feat_node_ids=localizer["feat_node_ids"],
                target_hw=localizer["target_hw"],
                graph=localizer["graph"],
                selected_node_id=selected_node_id,
                edge_level_information=localizer.get("edge_level_information") or {},
                letterbox_fn=localizer["letterbox_fn"],
                ood_features_fn=localizer["ood_features_fn"],
                concat_dim=localizer["concat_dim"],
                gallery_norm=gallery_norm,
            )
        localization_s = time.time() - loc_start

        if localization.get("at_target"):
            action_record = {
                "type": "finished",
                "coordinate": None,
                "direction": None,
                "start_coordinate": None,
                "end_coordinate": None,
                "thought": "Selected node is already among the top-3 localized matches.",
                "prompt": "",
                "timing": {
                    "driver_screenshot_s": driver_screenshot_s,
                    "localization_s": localization_s,
                    "model_prediction_s": 0.0,
                    "action_s": 0.0,
                    "other_processing_s": 0.0,
                },
            }
            actions.append(action_record)
            screenshots.append(screenshot)
            if on_step:
                on_step(
                    {
                        "step": step_idx + 1,
                        "action": action_record,
                        "prompt": "",
                        "elapsed_s": time.time() - step_start,
                        "timing": action_record["timing"],
                        "screenshot": screenshot,
                        "parsed": None,
                        "localization": localization,
                        "navigation_plan": format_navigation_plan(final_goal, []),
                    }
                )
            print(
                f"[step {step_idx + 1}] at_target  "
                f"shot={driver_screenshot_s:.2f}s loc={localization_s:.2f}s "
                f"total={time.time() - step_start:.2f}s"
            )
            finish_flag = True
            break

        hints = localization.get("transition_hints") or []
        navigation_plan = format_navigation_plan(final_goal, hints)

        model_start = time.time()
        step_result, _ = agent.step(navigation_plan, screenshot)
        model_prediction_s = time.time() - model_start
        prompt = get_agent_last_prompt(agent)

        other_start = time.time()
        parsed_action = parse_action_from_step(step_result)
        other_processing_s = time.time() - other_start

        if should_stop and should_stop():
            stopped = True
            break

        action_type = str(parsed_action["type"]).strip().lower()
        action_s = 0.0
        if action_type not in ("finished", "finish"):
            act_start = time.time()
            driver.execute_action(parsed_action["parsed"])
            action_s = time.time() - act_start

        timing = {
            "driver_screenshot_s": driver_screenshot_s,
            "localization_s": localization_s,
            "model_prediction_s": model_prediction_s,
            "action_s": action_s,
            "other_processing_s": other_processing_s,
        }

        action_record = {
            "type": parsed_action["type"],
            "coordinate": parsed_action["coordinate"],
            "direction": parsed_action.get("direction"),
            "start_coordinate": parsed_action.get("start_coordinate"),
            "end_coordinate": parsed_action.get("end_coordinate"),
            "thought": parsed_action["thought"],
            "prompt": prompt,
            "timing": timing,
        }
        actions.append(action_record)
        screenshots.append(screenshot)

        elapsed = time.time() - step_start
        print(
            f"[step {step_idx + 1}]  "
            f"shot={driver_screenshot_s:.2f}s loc={localization_s:.2f}s "
            f"agent={model_prediction_s:.2f}s act={action_s:.2f}s total={elapsed:.2f}s"
        )

        step_payload = {
            "step": step_idx + 1,
            "action": action_record,
            "prompt": prompt,
            "elapsed_s": elapsed,
            "timing": timing,
            "screenshot": screenshot,
            "parsed": parsed_action["parsed"],
            "localization": localization,
            "navigation_plan": navigation_plan,
        }
        if on_step:
            on_step(step_payload)

        if action_type in ("finished", "finish"):
            finish_flag = True
            break

        if should_stop and should_stop():
            stopped = True
            break

    return screenshots, actions, finish_flag, stopped


def encode_image_jpeg_b64(img: np.ndarray, quality: int = 75) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
