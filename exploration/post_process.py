import cv2
from dynaconf import Dynaconf
from PIL import Image

from Debugger import Debugger

from ImageUtils import ImageUtils
from VLM import VLM
from VLM_Yibu import VLM_Yibu

import json
import numpy as np
from networkx.readwrite import json_graph
import networkx as nx
import random
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import traceback
import dashscope
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

from networkx.readwrite import json_graph
from FlagEmbedding import BGEM3FlagModel

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.llm.openai import openai_complete_if_cache



for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)

def visualize_edge_on_screenshot(screenshot, edge_data, thickness=4):
    """Draw all parallel edges from ``get_edge_data(u, v)`` on a BGR screenshot."""
    if screenshot is None:
        return None
    vis = screenshot.copy()
    if not edge_data:
        return vis

    # MultiDiGraph: {key: attrs}; single-edge attrs: {type, description, boundingBox, ...}
    if any(k in edge_data for k in ("type", "description", "boundingBox", "bbox")):
        edge_attrs_list = [edge_data]
    else:
        edge_attrs_list = [v for v in edge_data.values() if isinstance(v, dict)]

    h, w = vis.shape[:2]
    palette = [
        (0, 255, 0),
        (0, 165, 255),
        (255, 0, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 255, 0),
    ]

    for idx, attrs in enumerate(edge_attrs_list):
        color = palette[idx % len(palette)]
        el_type = str(attrs.get("type") or "").strip().lower()
        desc = str(attrs.get("description") or attrs.get("type") or "action").strip()
        if len(edge_attrs_list) > 1:
            desc = f"[{idx + 1}] {desc}"
        bbox = attrs.get("boundingBox") or attrs.get("bbox")

        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            pad = max(8, min(x2 - x1, y2 - y1) // 6)
            if el_type == "scroll":
                cv2.arrowedLine(
                    vis, (cx, y1 + pad), (cx, y2 - pad), color, thickness, tipLength=0.3
                )
            elif el_type == "swipe":
                cv2.arrowedLine(
                    vis, (x1 + pad, cy), (x2 - pad, cy), color, thickness, tipLength=0.3
                )

            continue

        if "scroll" in el_type:
            cx = w // 2
            cv2.arrowedLine(
                vis,
                (cx, int(h * 0.22)),
                (cx, int(h * 0.78)),
                color,
                max(thickness + 2, 6),
                tipLength=0.08,
            )
            label = desc or "SCROLL"
            ty = 28 + idx * 28
            cv2.putText(
                vis,
                label[:64],
                (10, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            continue

    return vis

def simplify_edge_actions(edge_data):
    if not edge_data:
        return []

    simplified = []
    for key, attrs in edge_data.items():
        simplified.append({
            "edge_key": key,
            "type": attrs.get("type", ""),
            "description": attrs.get("description", "")
        })

    return simplified


def resolve_enhanced_page_summary(path_node_id, node_intents, nx_graph):
    """Return enhanced_page_summary dict from node_intents, or a graph-metadata fallback."""
    node_intents = node_intents or {}
    enhanced = node_intents.get(path_node_id, {}).get("enhanced_page_summary")
    if isinstance(enhanced, dict) and enhanced:
        return enhanced

    node_data = nx_graph.nodes[path_node_id]
    compact = str(node_data.get("page_summary", "") or "").strip()
    page_purpose = str(node_data.get("page_purpose", "") or "").strip() or compact
    return {
        "tag": compact[:80] if compact else path_node_id,
        "active_tab": node_data.get("active_tab"),
        "active_subtab": node_data.get("active_subtab"),
        "page_purpose": page_purpose,
        "screen_type": "",
        "selected_navigation": "",
        "visual_landmarks": [],
        "regions": {"top_bar": "", "main_area": "", "bottom_nav": ""},
        "salient_controls": [],
    }


def build_paths_for_node(nx_graph, node_id, node_intents, logs_root):
    root_node_ids = [node_id for node_id, node in nx_graph.nodes(data=True) if node.get("is_root", False)]
    paths = dict()
    for root_node_id in root_node_ids:
        try:
            node_path = nx.shortest_path(nx_graph, source=root_node_id, target=node_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        screenshots = []
        enhanced_page_summaries = []
        actions = []
        for i, node in enumerate(node_path):
            screenshot = cv2.imread(os.path.join(logs_root, "screenshots", node + ".jpg"))
            screenshots.append(screenshot)
            enhanced_page_summaries.append(
                resolve_enhanced_page_summary(node, node_intents, nx_graph)
            )
            if i < len(node_path) - 1:
                screenshots[i] = visualize_edge_on_screenshot(
                    screenshots[i],
                    nx_graph.get_edge_data(node_path[i], node_path[i + 1]),
                )
                actions.append(
                    simplify_edge_actions(
                        nx_graph.get_edge_data(node_path[i], node_path[i + 1])
                    )
                )

        paths[root_node_id] = {
            "screenshots": screenshots,
            "enhanced_page_summaries": enhanced_page_summaries,
            "actions": actions,
        }

    return paths


def normalize_clustered_navigation_plans(navigation_plans, input_root_ids=None, max_plans=3):
    """Normalize clustered VLM navigation output into ui_navigation_memory entries."""
    if not isinstance(navigation_plans, list):
        return []

    normalized = []
    for item in navigation_plans:
        if not isinstance(item, dict):
            continue

        root_ids = item.get("root_ids")
        if not root_ids and item.get("root_id"):
            root_ids = [item.get("root_id")]

        memory = item.get("ui_navigation_memory")
        if not isinstance(memory, dict):
            memory = item

        normalized.append(
            {
                "root_ids": [str(r) for r in (root_ids or []) if str(r).strip()],
                "relevant_waypoint_sequence": memory.get("relevant_waypoint_sequence", []),
                "transition_hints": memory.get("transition_hints", []),
            }
        )

    if max_plans and len(normalized) > max_plans:
        normalized = normalized[:max_plans]

    if input_root_ids:
        covered = {rid for plan in normalized for rid in plan.get("root_ids", [])}
        missing = [str(rid) for rid in input_root_ids if str(rid) not in covered]
        if missing and normalized:
            normalized[-1]["root_ids"] = list(
                dict.fromkeys(normalized[-1].get("root_ids", []) + missing)
            )

    return normalized


def _process_node_intents(node_id, nx_graph, vlm_client, logs_root):
    paths = build_paths_for_node(nx_graph, node_id, {}, logs_root)
    if not paths:
        return None

    out_edges_all = [
        attrs["description"]
        for _, _, attrs in nx_graph.out_edges(node_id, data=True)
    ]

    out_edges = [
        d for d in out_edges_all
        if not (isinstance(d, str) and ("scroll" in d.lower() or "swipe" in d.lower()))
    ] or out_edges_all

    primary_path = min(
        paths.values(),
        key=lambda v: len(v.get("actions", []))
    )

    intent_output = vlm_client.get_node_user_intents(primary_path, out_edges)

    return node_id, {
        "user_intents": list(dict.fromkeys(intent_output.get("user_intents", []))),
        "enhanced_page_summary": intent_output.get("enhanced_page_summary", {}),
    }


def save_node_intents(nx_graph, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    node_intents_path = os.path.join(logs_root, "node_intents.json")

    if os.path.exists(node_intents_path):
        with open(node_intents_path, "r", encoding="utf-8") as f:
            try:
                node_intents = json.load(f)
            except json.JSONDecodeError:
                node_intents = {}
    else:
        node_intents = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 1) or 1)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_node_intents,
                node_id,
                nx_graph,
                vlm_client,
                logs_root,
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                node_intents[node_id] = entry
                with open(node_intents_path, "w", encoding="utf-8") as f:
                    json.dump(node_intents, f, ensure_ascii=False, indent=4)

    return node_intents

def _process_node_navigation_plans(node_id, node_intents, nx_graph, vlm_client, logs_root):

    paths = build_paths_for_node(nx_graph, node_id, node_intents, logs_root)
    if not paths:
        return None

    paths = {
        root_id: bundle
        for root_id, bundle in sorted(
            paths.items(),
            key=lambda item: len(item[1].get("actions") or []),
        )[:3]
    }

    navigation_plans = []
    for root_id, bundle in paths.items():
        plans = vlm_client.get_node_navigation_plans({root_id: bundle})
        navigation_plans.extend(plans)


    ui_navigation_memory = normalize_clustered_navigation_plans(
        navigation_plans,
        input_root_ids=list(paths.keys()),
        max_plans=3,
    )

    return node_id, {"ui_navigation_memory": ui_navigation_memory}





def save_node_navigation_plans(nx_graph, node_intents, vlm_client, configs, dbg):
    logs_root = configs.logs.root
    node_navigation_plans_path = os.path.join(logs_root, "node_navigation_plans.json")

    if os.path.exists(node_navigation_plans_path):
        with open(node_navigation_plans_path, "r", encoding="utf-8") as f:
            try:
                node_navigation_plans = json.load(f)
            except json.JSONDecodeError:
                node_navigation_plans = {}
    else:
        node_navigation_plans = {}

    max_workers = int(getattr(configs.post_process, "max_workers", 1) or 1)
    write_lock = threading.Lock()
    node_ids = list(nx_graph.nodes())

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _process_node_navigation_plans,
                node_id,
                node_intents,
                nx_graph,
                vlm_client,
                logs_root,
            ): node_id
            for node_id in node_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="post-process nodes"):
            node_id = futures[future]
            try:
                result = future.result()
            except Exception:
                dbg.log(f"Failed post-process for node {node_id}", color="red")
                traceback.print_exc()
                continue

            if result is None:
                continue

            _, entry = result
            with write_lock:
                node_navigation_plans[node_id] = entry
                with open(node_navigation_plans_path, "w", encoding="utf-8") as f:
                    json.dump(node_navigation_plans, f, ensure_ascii=False, indent=4)

    return node_navigation_plans


def add_node_embeddings_to_user_intents(
    root_path,
    graph_filename="graph.json",
    user_intents_filename="user_intents.json",
    output_filename="user_intents_with_embeddings.json",
    bge_model_name="BAAI/bge-m3",
    batch_size=4,
    max_length=8192,
):
    """
    root_path must contain:
        - graph.json
        - user_intents.json

    Creates:
        - user_intents_with_embeddings.json

    Each node gets:
        - embedding_text
        - embedding
    """

    graph_path = os.path.join(root_path, graph_filename)
    user_intents_path = os.path.join(root_path, user_intents_filename)
    output_path = os.path.join(root_path, output_filename)

    with open(graph_path, "r", encoding="utf-8") as f:
        graph_data = json.load(f)

    graph = json_graph.node_link_graph(graph_data, edges="links")

    with open(user_intents_path, "r", encoding="utf-8") as f:
        user_intents_data = json.load(f)

    node_ids = []
    embedding_texts = []

    for node_id, intent_data in user_intents_data.items():
        node_data = graph.nodes.get(node_id, {})

        page_purpose = node_data.get("page_purpose", "")
        active_tab = node_data.get("active_tab", "")
        active_subtab = node_data.get("active_subtab", "")
        user_intents = intent_data.get("user_intents", [])

        embedding_text = f"""
PAGE_PURPOSE:
{page_purpose}

ACTIVE_TAB:
{active_tab}

ACTIVE_SUBTAB:
{active_subtab}

USER_INTENTS:
{json.dumps(user_intents, ensure_ascii=False)}
""".strip()

        node_ids.append(node_id)
        embedding_texts.append(embedding_text)

    model = BGEM3FlagModel(
        bge_model_name,
        use_fp16=True,
    )

    outputs = model.encode(
        embedding_texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    embeddings = np.asarray(outputs["dense_vecs"], dtype=np.float32)

    # Normalize for cosine similarity.
    embeddings = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True),
        1e-12,
    )

    for i, node_id in enumerate(node_ids):
        user_intents_data[node_id]["embedding_text"] = embedding_texts[i]
        user_intents_data[node_id]["embedding"] = embeddings[i].tolist()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(user_intents_data, f, ensure_ascii=False, indent=4)

    return {
        "num_nodes": len(node_ids),
        "output_path": output_path,
        "embedding_dim": int(embeddings.shape[1]),
    }


if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    # i want to get config file from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_android.yaml")
    args = parser.parse_args()
    configs = Dynaconf(settings_files=[args.config], merge_enabled=True)
    configs = configs.default

    if configs.post_process.use_yibu_api:
        vlm_client = VLM_Yibu(configs, dbg)
    else:
        vlm_client = VLM(configs, dbg)


    image_utils = ImageUtils()

    with open(os.path.join(configs.logs.root, "graph.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    nx_graph = json_graph.node_link_graph(data, edges="links")

    node_intents = save_node_intents(nx_graph, vlm_client, configs, dbg)
    dbg.log("Stage 1 complete: node_intents.json", color="green")

    save_node_navigation_plans(nx_graph, node_intents, vlm_client, configs, dbg)
    dbg.log("Stage 2 complete: node_navigation_plans.json", color="green")

    add_node_embeddings_to_user_intents(
        configs.logs.root, "graph.json", "node_intents.json", "node_intents.json"
    )
    dbg.log("Stage 3 complete: embeddings written to node_intents.json", color="green")


    


