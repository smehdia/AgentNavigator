import os
import cv2
import json
import torch
import random
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


SCROLL, SWIPE, TAP, NONE = 1, 2, 3, 4

APP_DIR = "/home/mehdi/Desktop/github_mehdi/AgentNavigator/exploration/explored_apps/amazon/"

with open(os.path.join(APP_DIR, "graph.json"), "r") as f:
    graph_json = json.load(f)

with open(os.path.join(APP_DIR, "node_level_information.json"), "r") as f:
    node_level_information_json = json.load(f)

with open(os.path.join(APP_DIR, "edge_level_information.json"), "r") as f:
    edge_level_information_json = json.load(f)

with open(os.path.join(APP_DIR, "user_intents.json"), "r") as f:
    user_intents_json = json.load(f)

with open(os.path.join(APP_DIR, "node_navigation_plans.json"), "r") as f:
    node_navigation_plans_json = json.load(f)

networkx_graph = json_graph.node_link_graph(graph_json)

node_ids = [node["id"] for node in graph_json["nodes"]]
# Compute Floyd-Warshall distance dictionary with keys as (node_id1, node_id2)
distance_dict = {}
fw_dist = nx.floyd_warshall(networkx_graph)
for node_id1 in node_ids:
    for node_id2 in node_ids:
        distance_dict[(node_id1, node_id2)] = fw_dist[node_id1][node_id2]


def create_mask_for_interactive_elements(interactive_elements, screenshot):
    mask = np.zeros((screenshot.shape[0], screenshot.shape[1]), dtype=np.uint8)
    for element in interactive_elements:
        if 'boundingBox' in element.keys():
            if 'swipe' not in element['type'] and 'scroll' not in element['type']:
                mask[element['boundingBox'][1]:element['boundingBox'][3], element['boundingBox'][0]:element['boundingBox'][2]] = 255

    return mask

def shortest_path_from_root(node_id, networkx_graph):
    paths = []
    for root in (n for n, d in networkx_graph.nodes(data=True) if d['is_root']):
        try:
            paths.append(nx.shortest_path(networkx_graph, root, node_id))
        except nx.NetworkXNoPath:
            continue
    return min(paths, key=len)

def format_user_intents(user_intents):
    init_string = "For this UI Page the user has the following intents: \n"
    for i,intent in enumerate(user_intents):
        init_string += f"{i+1}. {intent} \n"
    return init_string

def format_node_navigation_plans(node_navigation_plans):
    """
    Format navigation memory plans in a readable string format.
    node_navigation_plans: a list of dicts. Each represents a navigation plan.
    Returns: str
    """
    if not isinstance(node_navigation_plans, list):
        raise TypeError("node_navigation_plans must be a list")
    output_lines = []

    for idx, plan in enumerate(node_navigation_plans, 1):
        output_lines.append("")
        output_lines.append(f"[Plan {idx}]")


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
                if isinstance(hint, dict):
                    low = str(hint.get("low_level", "")).strip()
                    high = str(hint.get("high_level", "")).strip()
                    if low and high:
                        output_lines.append(f"- Low: {low}\n  High: {high}")
                        added_hint = True
                    elif low:
                        output_lines.append(f"- Low: {low}")
                        added_hint = True
                    elif high:
                        output_lines.append(f"- High: {high}")
                        added_hint = True
                else:
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

    return "\n".join(output_lines).strip()


def get_node_data(node_id: str) -> dict[str, Any]:
    node = next(
        node
        for node in graph_json["nodes"]
        if node["id"] == node_id
    )

    screenshot = cv2.imread(
        os.path.join(
            APP_DIR,
            "screenshots",
            f"{node_id}.jpg",
        )
    )

    if screenshot is None:
        raise FileNotFoundError(
            f"Could not load screenshot for node {node_id}."
        )

    shortest_path = shortest_path_from_root(
        node_id,
        networkx_graph,
    )

    previous_screenshot = None
    previous_node_id = None

    action_type = NONE
    action_coordinate_center = None

    if len(shortest_path) >= 2:
        previous_node_id = shortest_path[-2]

        previous_screenshot = cv2.imread(
            os.path.join(
                APP_DIR,
                "screenshots",
                f"{previous_node_id}.jpg",
            )
        )

        edge = next(
            edge
            for edge in graph_json["links"]
            if (
                edge["source"] == previous_node_id
                and edge["target"] == node_id
            )
        )

        edge_type = edge.get("type", "").lower()

        if "swipe" in edge_type:
            action_type = SWIPE

        elif "scroll" in edge_type:
            action_type = SCROLL

        elif "tap" in edge_type:
            action_type = TAP

            bbox = edge.get("boundingBox")
            if bbox is None:
                raise ValueError(
                    f"Tap edge {previous_node_id} -> {node_id} "
                    "does not contain a boundingBox."
                )

            x1, y1, x2, y2 = bbox

            action_coordinate_center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
            )

    ui_bboxes = []

    for element in node.get("ui_elements", []):
        element_type = element.get("type", "").lower()

        if (
            "swipe" not in element_type
            and "scroll" not in element_type
            and "boundingBox" in element
        ):
            ui_bboxes.append(element["boundingBox"])

    intents = format_user_intents(
        user_intents_json[node_id]["user_intents"]
    )

    navigation_plans = format_node_navigation_plans(
        node_navigation_plans_json[node_id]
    )

    page_description = node_level_information_json[
        node_id
    ]["high_level"]

    canonical_layout = node.get(
        "canonical_page_layout",
        {},
    )

    # Modify these keys to match your actual JSON schema.
    active_tab = canonical_layout.get(
        "active_tab",
        "",
    )

    active_subtab = canonical_layout.get(
        "active_subtab",
        "",
    )

    return {
        "node_id": node_id,
        "previous_node_id": previous_node_id,

        "screenshot": screenshot,
        "screenshot_prev": previous_screenshot,

        "action_type": action_type,
        "action_coordinate_center":
            action_coordinate_center,

        "node_intents": intents,
        "node_navigation_plans": navigation_plans,

        "page_description": page_description,
        "active_tab": active_tab,
        "active_subtab": active_subtab,

        "ui_bboxes": ui_bboxes,
        "canonical_page_layout": canonical_layout,
        "shortest_path": shortest_path,
    }


def get_pair_data(
    debug: bool = False,
) -> tuple[dict, dict]:
    node_id1 = random.choice(node_ids)
    node_id2 = random.choice(node_ids)

    node1_info = get_node_data(node_id1)
    node2_info = get_node_data(node_id2)

    if debug:
        for index, node_info in enumerate(
            [node1_info, node2_info],
            start=1,
        ):
            mask = create_mask_for_interactive_elements(
                next(
                    node
                    for node in graph_json["nodes"]
                    if node["id"] == node_info["node_id"]
                )["ui_elements"],
                node_info["screenshot"],
            )

            cv2.imshow(
                f"mask_{index}",
                cv2.resize(mask, None, fx=0.5, fy=0.5),
            )

            cv2.imshow(
                f"screenshot_{index}",
                cv2.resize(
                    node_info["screenshot"],
                    None,
                    fx=0.5,
                    fy=0.5,
                ),
            )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return node1_info, node2_info