import os
import cv2
import json
import random
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph


SCROLL, SWIPE, TAP, NONE = 1, 2, 3, 4

# history_mode mixture for training robustness
DEFAULT_HISTORY_MODE_PROBS = {
    "valid": 0.50,      # real predecessor + correct action
    "none": 0.30,       # current screenshot only
    "corrupted": 0.20,  # wrong predecessor / wrong action / repeated current
}


def load_all_graphs(ROOT_DIR):

    dict_graph_info = {}
    for app in os.listdir(ROOT_DIR):
        app_dir = os.path.join(ROOT_DIR, app)
        try:
            with open(os.path.join(app_dir, "graph.json"), "r") as f:
                graph_json = json.load(f)
            with open(os.path.join(app_dir, "node_level_information.json"), "r") as f:
                node_level_information_json = json.load(f)
            with open(os.path.join(app_dir, "edge_level_information.json"), "r") as f:
                edge_level_information_json = json.load(f)
            with open(os.path.join(app_dir, "user_intents.json"), "r") as f:
                user_intents_json = json.load(f)
            with open(os.path.join(app_dir, "node_navigation_plans.json"), "r") as f:
                node_navigation_plans_json = json.load(f)
            networkx_graph = json_graph.node_link_graph(graph_json)
            node_ids = [node["id"] for node in graph_json["nodes"]]
            distance_dict = {}
            fw_dist = nx.floyd_warshall(networkx_graph)
            for node_id1 in node_ids:
                for node_id2 in node_ids:
                    distance_dict[(node_id1, node_id2)] = fw_dist[node_id1][node_id2]

            dict_graph_info[app] = {
                "graph_json": graph_json,
                "node_level_information_json": node_level_information_json,
                "edge_level_information_json": edge_level_information_json,
                "user_intents_json": user_intents_json,
                "node_navigation_plans_json": node_navigation_plans_json,
                "networkx_graph": networkx_graph,
                "node_ids": node_ids,
                "distance_dict": distance_dict,
            }
        except Exception as e:
            print(f"Error loading graph for {app}: {e}")
            continue
    return dict_graph_info



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


def get_node_data(node_id: str, app_info: dict, app_dir: str):
    graph_json = app_info["graph_json"]
    networkx_graph = app_info["networkx_graph"]
    user_intents_json = app_info["user_intents_json"]
    node_navigation_plans_json = app_info["node_navigation_plans_json"]
    node_level_information_json = app_info["node_level_information_json"]

    node = next(
        node
        for node in graph_json["nodes"]
        if node["id"] == node_id
    )

    screenshot = cv2.imread(
        os.path.join(
            app_dir,
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
                app_dir,
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

    intent_list = list(
        user_intents_json[node_id]["user_intents"]
    )
    intents = format_user_intents(intent_list)

    plans = node_navigation_plans_json[node_id]
    navigation_plans = format_node_navigation_plans(plans)

    waypoint_parts = []
    hint_parts = []
    for plan in plans:
        waypoints = plan.get("relevant_waypoint_sequence") or []
        if waypoints:
            waypoint_parts.append(
                " -> ".join(
                    str(w).strip()
                    for w in waypoints
                    if str(w).strip()
                )
            )
        for hint in plan.get("transition_hints") or []:
            if isinstance(hint, dict):
                low = str(hint.get("low_level", "")).strip()
                high = str(hint.get("high_level", "")).strip()
                if high and low:
                    hint_parts.append(f"High: {high}\nLow: {low}")
                elif high:
                    hint_parts.append(high)
                elif low:
                    hint_parts.append(low)
            else:
                text = str(hint).strip()
                if text:
                    hint_parts.append(text)

    page_description = node_level_information_json[
        node_id
    ]["high_level"]

    canonical_layout = node.get("canonical_page_layout", "")
    active_tab = node.get("active_tab") or ""
    active_subtab = node.get("active_subtab") or ""

    return {
        "node_id": node_id,
        "previous_node_id": previous_node_id,

        "screenshot": screenshot,
        "screenshot_prev": previous_screenshot,

        "action_type": action_type,
        "action_coordinate_center":
            action_coordinate_center,

        "node_intents": intents,
        "intent_list": intent_list,
        "node_navigation_plans": navigation_plans,
        "waypoint_text": " | ".join(waypoint_parts),
        "transition_hint_text": " | ".join(hint_parts),

        "page_description": page_description,
        "active_tab": active_tab,
        "active_subtab": active_subtab,

        "ui_bboxes": ui_bboxes,
        "canonical_page_layout": canonical_layout,
        "shortest_path": shortest_path,

        # Filled by apply_history_mode / get_data.
        "history_mode": "valid" if previous_screenshot is not None else "none",
        "supervise_transition": previous_screenshot is not None,
    }


def _normalize_history_mode_probs(history_mode_probs: dict | None) -> dict[str, float]:
    probs = dict(DEFAULT_HISTORY_MODE_PROBS)
    if history_mode_probs:
        probs.update(history_mode_probs)
    for key in ("valid", "none", "corrupted"):
        if key not in probs:
            raise KeyError(f"history_mode_probs missing required key {key!r}.")
        if probs[key] < 0:
            raise ValueError(f"history_mode_probs[{key!r}] must be >= 0.")
    total = probs["valid"] + probs["none"] + probs["corrupted"]
    if total <= 0:
        raise ValueError("history_mode_probs must sum to a positive value.")
    return {
        "valid": probs["valid"] / total,
        "none": probs["none"] / total,
        "corrupted": probs["corrupted"] / total,
    }


def _sample_history_mode(history_mode_probs: dict[str, float]) -> str:
    roll = random.random()
    if roll < history_mode_probs["valid"]:
        return "valid"
    if roll < history_mode_probs["valid"] + history_mode_probs["none"]:
        return "none"
    return "corrupted"


def _random_action(screenshot_shape) -> tuple[int, tuple[float, float] | None]:
    action_type = random.choice([SCROLL, SWIPE, TAP, NONE])
    if action_type != TAP:
        return action_type, None
    height, width = screenshot_shape[:2]
    return action_type, (
        random.uniform(0, max(width - 1, 1)),
        random.uniform(0, max(height - 1, 1)),
    )


def _load_node_screenshot(app_dir: str, node_id: str):
    path = os.path.join(app_dir, "screenshots", f"{node_id}.jpg")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not load screenshot for node {node_id}.")
    return image


def apply_history_mode(
    node_info: dict,
    app_info: dict,
    app_dir: str,
    history_mode_probs: dict | None = None,
) -> dict:
    """
    Rewrite predecessor/action fields according to a sampled history mode.

    Modes:
      valid     — keep real predecessor + action (falls back to none if root)
      none      — drop history; history_available will be False
      corrupted — keep history_available=True but make it unreliable:
                  wrong predecessor, wrong action, or repeated current screen
    """
    probs = _normalize_history_mode_probs(history_mode_probs)
    has_natural_history = node_info["screenshot_prev"] is not None
    mode = _sample_history_mode(probs)

    # Roots (and any sample without a real predecessor) cannot be "valid".
    if mode == "valid" and not has_natural_history:
        mode = "none"

    if mode == "none":
        node_info["screenshot_prev"] = None
        node_info["previous_node_id"] = None
        node_info["action_type"] = NONE
        node_info["action_coordinate_center"] = None
        node_info["history_mode"] = "none"
        node_info["supervise_transition"] = False
        return node_info

    if mode == "valid":
        node_info["history_mode"] = "valid"
        node_info["supervise_transition"] = True
        return node_info

    # ---- corrupted ----
    corruption = random.choice(
        ["wrong_predecessor", "wrong_action", "repeated_current"]
    )
    node_ids = app_info["node_ids"]
    current_id = node_info["node_id"]
    true_prev_id = node_info["previous_node_id"]

    if corruption == "repeated_current":
        node_info["screenshot_prev"] = node_info["screenshot"].copy()
        node_info["previous_node_id"] = current_id
        # Keep or randomize action; either is irrelevant to current.
        if random.random() < 0.5:
            action_type, xy = _random_action(node_info["screenshot"].shape)
            node_info["action_type"] = action_type
            node_info["action_coordinate_center"] = xy

    elif corruption == "wrong_predecessor":
        candidates = [
            node_id
            for node_id in node_ids
            if node_id != current_id and node_id != true_prev_id
        ]
        if not candidates:
            candidates = [node_id for node_id in node_ids if node_id != current_id]
        if candidates:
            wrong_id = random.choice(candidates)
            node_info["screenshot_prev"] = _load_node_screenshot(app_dir, wrong_id)
            node_info["previous_node_id"] = wrong_id
        else:
            node_info["screenshot_prev"] = node_info["screenshot"].copy()
            node_info["previous_node_id"] = current_id
        if random.random() < 0.5 or not has_natural_history:
            action_type, xy = _random_action(node_info["screenshot"].shape)
            node_info["action_type"] = action_type
            node_info["action_coordinate_center"] = xy

    else:  # wrong_action
        if not has_natural_history:
            # No real previous screen: use another node as a decoy predecessor.
            candidates = [node_id for node_id in node_ids if node_id != current_id]
            if candidates:
                decoy_id = random.choice(candidates)
                node_info["screenshot_prev"] = _load_node_screenshot(
                    app_dir, decoy_id
                )
                node_info["previous_node_id"] = decoy_id
            else:
                node_info["screenshot_prev"] = node_info["screenshot"].copy()
                node_info["previous_node_id"] = current_id
        action_type, xy = _random_action(node_info["screenshot"].shape)
        # Avoid accidentally sampling the true action type when possible.
        true_action = node_info["action_type"]
        if has_natural_history and action_type == true_action:
            alternatives = [a for a in (SCROLL, SWIPE, TAP, NONE) if a != true_action]
            action_type = random.choice(alternatives)
            if action_type == TAP:
                _, xy = _random_action(node_info["screenshot"].shape)
            else:
                xy = None
        node_info["action_type"] = action_type
        node_info["action_coordinate_center"] = xy

    node_info["history_mode"] = "corrupted"
    # Never train the transition predictor on intentionally wrong history.
    node_info["supervise_transition"] = False
    return node_info


def get_data(
    dict_graph_info,
    root_dir,
    selected_app,
    off_app_sample=False,
    debug=False,
    history_mode_probs: dict | None = None,
):
    # Select the application to sample from
    if off_app_sample:
        # Create a list of apps that are not the selected_app
        other_apps = [app for app in dict_graph_info.keys() if app != selected_app]
        if not other_apps:
            raise ValueError("No other apps available in dict_graph_info for off_app_sample.")
        app_name = random.choice(other_apps)
    else:
        app_name = selected_app

    # Get data for the chosen app
    app_info = dict_graph_info[app_name]
    app_dir = os.path.join(root_dir, app_name)
    node_ids = app_info["node_ids"]
    graph_json = app_info["graph_json"]

    node_id = random.choice(node_ids)
    node_info = get_node_data(node_id, app_info, app_dir)
    node_info["app_name"] = app_name
    node_info["on_graph"] = not off_app_sample
    node_info = apply_history_mode(
        node_info,
        app_info,
        app_dir,
        history_mode_probs=history_mode_probs,
    )

    if debug:
        # Assume index is node_id for display purposes
        index = node_id
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

    return node_info

