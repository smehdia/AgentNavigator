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
from tqdm import tqdm


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


def build_paths_for_node(nx_graph, node_id):
    root_node_ids = [node_id for node_id, node in nx_graph.nodes(data=True) if node.get("is_root", False)]
    paths = dict()
    for root_node_id in root_node_ids:
        node_path = nx.shortest_path(nx_graph, source=root_node_id, target=node_id)        
        screenshots = []
        page_summaries = []
        actions = []
        for i,node in enumerate(node_path):
            screenshot = cv2.imread(os.path.join(configs.logs.root, "screenshots", node + ".jpg"))
            screenshots.append(screenshot)
            page_summaries.append(nx_graph.nodes[node].get("page_summary", ""))
            if i < len(node_path) - 1:
                screenshots[i] = visualize_edge_on_screenshot(screenshots[i], nx_graph.get_edge_data(node_path[i], node_path[i + 1]))
                actions.append(simplify_edge_actions(nx_graph.get_edge_data(node_path[i], node_path[i + 1])))

        paths[root_node_id] = {
            "screenshots": screenshots,
            "page_summaries": page_summaries,
            "actions": actions

        }

    return paths



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

    node_intents = dict()

    for i, node_id in tqdm(enumerate(nx_graph.nodes()), total=len(nx_graph.nodes())):
        node_intents[node_id] = {
            "user_intents": [],
            "ui_navigation_memory": [],
        }

        paths = build_paths_for_node(nx_graph, node_id)

        if not paths:
            continue

        out_edges_all = [
            attrs["description"]
            for _, _, attrs in list(nx_graph.out_edges(node_id, data=True))
        ]

        if len(out_edges_all) > 1:
            non_scroll_swipe = [
                d for d in out_edges_all
                if not (
                    isinstance(d, str)
                    and (("scroll" in d.lower()) or ("swipe" in d.lower()))
                )
            ]
            out_edges = non_scroll_swipe if non_scroll_swipe else out_edges_all
        else:
            out_edges = out_edges_all

        # choose one visual path for user_intents
        primary_root_id, primary_path = min(
            paths.items(),
            key=lambda item: len(item[1].get("actions", []))
        )

        with dbg.time_block(f"Generating user intents for node {node_id}"):
            intent_output = vlm_client.get_node_user_intents(
                primary_path,
                out_edges
            )

            node_intents[node_id]["user_intents"] = intent_output.get(
                "user_intents",
                []
            )

        # generate route plans for all paths using text only
        with dbg.time_block(f"Generating navigation plans for node {node_id}"):
            navigation_plans = vlm_client.get_node_navigation_plans(paths)
            only_memories = [
                item["ui_navigation_memory"]
                for item in navigation_plans
            ]
            node_intents[node_id]["ui_navigation_memory"] = only_memories


        # deduplicate intents
        node_intents[node_id]["user_intents"] = list(dict.fromkeys(
            node_intents[node_id]["user_intents"]
        ))


    with open(os.path.join(configs.logs.root, "node_intents.json"), "w", encoding="utf-8") as f:
        json.dump(node_intents, f, ensure_ascii=False, indent=4)




    


