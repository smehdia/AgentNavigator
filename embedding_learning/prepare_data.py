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

def get_pair_data(debug=False):
    # select two random node_ids
    node_id1 = random.choice(node_ids)
    node_id2 = random.choice(node_ids)
    node1 = next(node for node in graph_json["nodes"] if node["id"] == node_id1)
    node2 = next(node for node in graph_json["nodes"] if node["id"] == node_id2)

    screenshot_1 = cv2.imread(os.path.join(APP_DIR, "screenshots", f"{node_id1}.jpg"))
    screenshot_2 = cv2.imread(os.path.join(APP_DIR, "screenshots", f"{node_id2}.jpg"))

    shortest_path_1 = shortest_path_from_root(node_id1, networkx_graph)
    shortest_path_2 = shortest_path_from_root(node_id2, networkx_graph)


    node_1_intents = format_user_intents(user_intents_json[node_id1]['user_intents'])
    node_2_intents = format_user_intents(user_intents_json[node_id2]['user_intents'])

    node_1_navigation_plans = format_node_navigation_plans(node_navigation_plans_json[node_id1])
    node_2_navigation_plans = format_node_navigation_plans(node_navigation_plans_json[node_id2])


    screenshot_1_prev = None
    action_type_1 = NONE
    action_coordinate_center = None
    if len(shortest_path_1) >= 2:
        node_id1_prev = shortest_path_1[-2]
        screenshot_1_prev = cv2.imread(os.path.join(APP_DIR, "screenshots", f"{node_id1_prev}.jpg"))
        # get edge between node_id1_prev and node_id1
        edge = next(edge for edge in graph_json["links"] if edge["source"] == node_id1_prev and edge["target"] == node_id1)
        if 'swipe' in edge['type']:
            action_type = SWIPE
        elif 'scroll' in edge['type']:
            action_type = SCROLL
        elif 'tap' in edge['type']:
            action_type = TAP
            x1, y1, x2, y2 = edge['boundingBox']
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            action_coordinate_center = (x_center, y_center)
    
    
    ui_elements_1 = node1['ui_elements']
    ui_elements_2 = node2['ui_elements']

    mask = create_mask_for_interactive_elements(ui_elements_1, screenshot_1)

    if debug:
        cv2.imshow("mask", cv2.resize(mask, None, fx=0.5, fy=0.5))
        cv2.imshow("screenshot_1", cv2.resize(screenshot_1, None, fx=0.5, fy=0.5))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    page_description1 = node_level_information_json[node_id1]
    page_description2 = node_level_information_json[node_id2]



    return screenshot_1, screenshot_2


screenshot_1, screenshot_2 = get_pair_data()


# model_path = "/home/mehdi/Desktop/MAI-UI-2B/"
# processor = AutoProcessor.from_pretrained(model_path)
# mai_ui_model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="cpu")

# from model import ScreenshotUIGraphEmbedder, UIGraphEmbedder, count_trainable_parameters, FrozenMAIUIVisualExtractor

# extractor = FrozenMAIUIVisualExtractor(processor, mai_ui_model)

# trainable = UIGraphEmbedder(token_dim=2048, pooled_dim=256, embedding_dim=256).to(mai_ui_model.device)
# embedder = ScreenshotUIGraphEmbedder(extractor, trainable)
# #


# # Later tap step:
out = embedder(screenshot_1, screenshot_2, 3, (10, 100))

print(out.current_element_logits)
# embedding = out.embedding  # [1, 256]





asd