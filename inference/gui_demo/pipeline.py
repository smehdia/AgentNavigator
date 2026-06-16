"""Inference pipeline helpers for the GUI demo."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
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
    user_goal = user_goal.strip()

    return f"""
PAGE_TAG:
{user_goal}

PAGE_PURPOSE:
{user_goal}

USER_INTENTS:
{user_goal}
""".strip()


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


def format_navigation_plan(navigation_memory_plans, final_goal) -> str:
    if not isinstance(navigation_memory_plans, list):
        raise TypeError("navigation_memory_plans must be a list")
    if not isinstance(final_goal, str):
        raise TypeError("final_goal must be a string")

    if not navigation_memory_plans:
        return (
            "[Navigation Instruction]\n"
            f"Final goal: {final_goal.strip() if final_goal.strip() else '(none)'}\n\n"
            "No navigation memory is available for this task. "
            "Use only the current screenshot and the user goal to decide the next action. "
            "If the current screen already satisfies the final goal, do not perform more navigation actions. "
            "Return the finish/done action immediately."
        )

    output_lines = []
    output_lines.append("[Navigation Memory]")
    output_lines.append(
        "There are at most 3 semantically distinct navigation plans to reach the target screen. "
        "Before acting, compare the current screenshot with the waypoint sequences and transition hints, "
        "then follow the plan that best matches the current screen. "
        "Do not use root ids; match by visible UI state only. "
        "If none of the plans match the current screenshot, rely on the screenshot instead of forcing a plan. "
        "All plans assume you start on the reset landing screen unless the screenshot clearly shows a blocker "
        "(incognito, dialog, wrong tab). Ignore leading waypoints that don't match if you are already on the "
        "shared landing screen."
    )

    for idx, plan in enumerate(navigation_memory_plans, 1):
        output_lines.append("")
        output_lines.append(f"[Plan {idx}]")

        goal = plan.get("final_goal") if plan.get("final_goal") is not None else final_goal
        goal = goal.strip() if isinstance(goal, str) else ""
        output_lines.append(f"Final goal: {goal if goal else '(none)'}")

        waypoints = plan.get("relevant_waypoint_sequence", [])
        output_lines.append("\nRelevant waypoint sequence:")
        if isinstance(waypoints, list) and waypoints:
            wp_str = " -> ".join(str(w).strip() for w in waypoints if str(w).strip())
            output_lines.append(wp_str if wp_str else "(none)")
        else:
            output_lines.append("(none)")

        hints = plan.get("transition_hints", [])
        output_lines.append("\nTransition hints:")
        if isinstance(hints, list) and hints:
            added_hint = False
            for hint in hints:
                hint_str = str(hint).strip()
                if hint_str:
                    output_lines.append(f"- {hint_str}")
                    added_hint = True
            if not added_hint:
                output_lines.append("(none)")
        else:
            output_lines.append("(none)")

        usage_instruction = plan.get("usage_instruction")
        if usage_instruction:
            output_lines.append("\nPlan-specific usage instruction:")
            output_lines.append(str(usage_instruction).strip())

    output_lines.append("")
    output_lines.append("[Global usage instruction]")
    output_lines.append(
        "First identify whether the current screenshot matches any plan. "
        "Select the plan whose current or next waypoint best corresponds to the visible screen. "
        "Then take the action that most likely advances to the next waypoint in that selected plan. "
        "Do not mix steps from different plans unless the screenshot clearly supports doing so. "
        "If the screenshot clearly disagrees with all plans, ignore the plans and follow the screenshot. "
        "Once the current screen already satisfies the final goal, do not perform more navigation actions. "
        "Return the finish/done action immediately."
    )
    return "\n".join(output_lines).strip()


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
                "ui_navigation_memory": item.get("ui_navigation_memory", []),
                "embedding_text": item.get("embedding_text", ""),
            }
        )
    return results


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
    prev_coords = None
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
            if prev_coords and coords:
                p1 = (int(prev_coords[0]), int(prev_coords[1]))
                p2 = (int(coords[0]), int(coords[1]))
                cv2.arrowedLine(img, p1, p2, color_arrow, thickness=12, tipLength=0.2)
            elif coords:
                x, y = int(coords[0]), int(coords[1])
                cv2.circle(img, (x, y), 30, color_arrow, thickness=8)
        prev_coords = coords

    return annotated


def parse_action_from_step(step_result) -> dict:
    parsed = step_result[0] if isinstance(step_result, tuple) else step_result
    action_type = getattr(parsed, "action_type", None) or ""
    coords = None
    if hasattr(parsed, "sent_coords") and parsed.sent_coords and "point" in parsed.sent_coords:
        coords = parsed.sent_coords["point"]
    elif hasattr(parsed, "orig_coords") and parsed.orig_coords and "point" in parsed.orig_coords:
        coords = parsed.orig_coords["point"]
    thought = getattr(parsed, "thought", None) or ""
    return {
        "parsed": parsed,
        "type": action_type,
        "coordinate": coords,
        "thought": thought,
    }


@dataclass
class RetrievalContext:
    graph: Any
    depths: dict
    node_intents: dict
    node_navigation_plans: dict
    node_ids: list
    embeddings: np.ndarray
    raw_intent_data: dict = field(default_factory=dict)
    logs_root: str = ""


def load_node_navigation_plans(logs_root: str, node_intents: dict | None = None) -> dict:
    """Load clustered navigation plans; fall back to node_intents if the file is missing."""
    path = os.path.join(logs_root, "node_navigation_plans.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data

    node_intents = node_intents or {}
    return {
        node_id: {"ui_navigation_memory": item.get("ui_navigation_memory", [])}
        for node_id, item in node_intents.items()
        if isinstance(item, dict) and item.get("ui_navigation_memory")
    }


def get_ui_navigation_memory_for_node(ctx: RetrievalContext, node_id: str) -> list:
    nav_entry = ctx.node_navigation_plans.get(node_id, {})
    if isinstance(nav_entry, dict):
        memory = nav_entry.get("ui_navigation_memory")
        if memory:
            return memory

    intent_entry = ctx.node_intents.get(node_id, {})
    if isinstance(intent_entry, dict):
        return intent_entry.get("ui_navigation_memory", [])
    return []


def load_artifacts(logs_root: str) -> RetrievalContext:
    logs_root = os.path.abspath(logs_root)
    with open(os.path.join(logs_root, "graph.json"), "r", encoding="utf-8") as f:
        graph_data = json.load(f)
    graph = json_graph.node_link_graph(graph_data, edges="links")
    depths = get_node_depths(graph)

    with open(os.path.join(logs_root, "node_intents.json"), "r", encoding="utf-8") as f:
        node_intents = json.load(f)

    node_navigation_plans = load_node_navigation_plans(logs_root, node_intents)

    node_ids, embedding_rows = zip(
        *[(node_id, item["embedding"]) for node_id, item in node_intents.items() if "embedding" in item]
    ) if node_intents else ([], [])

    embeddings = np.asarray(embedding_rows, dtype=np.float32) if embedding_rows else np.empty((0, 0), dtype=np.float32)

    return RetrievalContext(
        graph=graph,
        depths=depths,
        node_intents=node_intents,
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

    candidates = []
    for result in results:
        node_id = result["node_id"]
        node_entry = ctx.node_intents.get(node_id, {})
        summary = node_entry.get("enhanced_page_summary", {}) or {}

        candidates.append(
            {
                "node_id": node_id,
                "page_tag": summary.get("tag", ""),
                "page_purpose": summary.get("page_purpose", ""),
                "screen_type": summary.get("screen_type", ""),
                "selected_navigation": summary.get("selected_navigation", ""),
                "depth": ctx.depths.get(node_id),
                "score": result.get("score"),
                "user_intents": result.get("user_intents") or node_entry.get("user_intents", []),
            }
        )

    top_k_node_ids, vlm_reasoning = vlm_client.rerank_candidates(
        query, candidates, top_k=configs.top_k_retrieval_in_stage_2
    )
    reasoning_map = (
        vlm_reasoning.get("reasoning", {})
        if isinstance(vlm_reasoning, dict)
        else {}
    )
    ordered = {nid: i for i, nid in enumerate(top_k_node_ids)}
    candidates = sorted(
        [c for c in candidates if c["node_id"] in top_k_node_ids],
        key=lambda c: ordered[c["node_id"]],
    )
    return candidates, reasoning_map


def get_navigation_memory_for_node(ctx: RetrievalContext, node_id: str, query: str) -> str:
    raw = get_ui_navigation_memory_for_node(ctx, node_id)
    return format_navigation_plan(raw, query)


def run_navigation_loop(
    navigation_memory: str,
    agent,
    driver,
    max_steps: int = 10,
    on_step: Optional[Callable[[dict], None]] = None,
    reset_first: bool = True,
    should_stop: Optional[Callable[[], bool]] = None,
) -> tuple[list, list, bool, bool]:
    if reset_first:
        driver.reset_to_start_page()
    agent.clear_history()

    screenshots = [driver.take_screenshot()]
    actions = []
    finish_flag = False
    stopped = False

    for step_idx in range(max_steps):
        if should_stop and should_stop():
            stopped = True
            break

        start = time.time()
        screenshot = driver.take_screenshot()
        step_result, _ = agent.step(navigation_memory, screenshot)
        parsed_action = parse_action_from_step(step_result)

        if should_stop and should_stop():
            stopped = True
            break

        action_record = {
            "type": parsed_action["type"],
            "coordinate": parsed_action["coordinate"],
            "thought": parsed_action["thought"],
        }
        actions.append(action_record)

        step_payload = {
            "step": step_idx + 1,
            "action": action_record,
            "elapsed_s": time.time() - start,
            "screenshot": screenshot,
            "parsed": parsed_action["parsed"],
        }
        if on_step:
            on_step(step_payload)

        action_type = str(parsed_action["type"]).strip().lower()
        if action_type in ("finished", "finish"):
            finish_flag = True
            screenshots.append(driver.take_screenshot())
            break

        if should_stop and should_stop():
            stopped = True
            break

        driver.execute_action(parsed_action["parsed"])
        driver.wait()
        screenshots.append(driver.take_screenshot())

    return screenshots, actions, finish_flag, stopped


def encode_image_jpeg_b64(img: np.ndarray, quality: int = 75) -> str:
    import base64

    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")
