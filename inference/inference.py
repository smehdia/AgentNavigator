
from dynaconf import Dynaconf

from Debugger import Debugger
from Driver.factory import build_driver

from ImageUtils import ImageUtils
from VLM import VLM
from Agents.factory import build_agent

import json
from networkx.readwrite import json_graph
import os
import cv2
import pickle
import argparse
import numpy as np 

import networkx as nx

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)


def sort_nodes_based_on_node_intent_similarity(query, node_intents, debugger, image_utils):

    with debugger.time_block("Computing query embedding:", color='green'):
        query_embedding = np.asarray(image_utils.text_model_embedder.encode(query, max_length=1024)['dense_vecs'])

    # compute the simlarity between the query embeddings and the node intents embeddings

    with debugger.time_block("Stage 1 retrieval:", color='green'):
        query = query_embedding.ravel().astype(np.float32)
        node_ids, intents, embs = [], [], []
        for node_id, data in node_intents.items():
            for intent, emb in zip(data['user_intents'], data['intent_embeddings']):
                node_ids.append(node_id)
                intents.append(intent)
                embs.append(emb)
        if not embs:
            return []
        E = np.asarray(embs, dtype=np.float32)
        q_norm = np.linalg.norm(query) + 1e-8
        e_norm = np.linalg.norm(E, axis=1) + 1e-8
        scores = (E @ query) / (e_norm * q_norm)
        results = list(zip(node_ids, intents, scores.tolist()))
        results.sort(key=lambda x: x[2], reverse=True)


    return results


def get_node_depths(G):
    """
    G: networkx graph
    Each node has attribute is_root=True for root nodes.

    Returns:
        dict: node_id -> depth from nearest root
    """
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
    """
    Simplified: Given a list of navigation plans (each a dict with
    'relevant_waypoint_sequence' and 'transition_hints') and a final_goal string,
    output one formatted string as described.

    If there are multiple plans, we enumerate as [Plan 1], [Plan 2], etc.

    Args:
        navigation_memory_plans: List[dict]
        final_goal: str

    Returns:
        str
    """

    if not isinstance(navigation_memory_plans, list):
        raise TypeError("navigation_memory_plans must be a list")
    if not isinstance(final_goal, str):
        raise TypeError("final_goal must be a string")

    output = [f"Final Goal:\n{final_goal.strip()}\n"]

    if not navigation_memory_plans:
        output.append("[Navigation Memory]\nNo plans available.")
        return "\n".join(output)

    output.append("[Navigation Memory]")
    for idx, plan in enumerate(navigation_memory_plans, 1):
        plan_title = f"[Plan {idx}]"
        output.append(plan_title)
        waypoints = plan.get("relevant_waypoint_sequence", []) or []
        hints = plan.get("transition_hints", []) or []

        # Format waypoints
        if waypoints and isinstance(waypoints, list):
            wp_str = " -> ".join(str(w).strip() for w in waypoints if str(w).strip())
        else:
            wp_str = ""
        output.append("Relevant waypoint sequence:")
        output.append(wp_str if wp_str else "(none)")

        # Format hints
        output.append("Transition hints:")
        if hints and isinstance(hints, list):
            for hint in hints:
                hint_str = str(hint).strip()
                if hint_str:
                    output.append(f"- {hint_str}")
        else:
            output.append("-")

        # Separate plans by an empty line except after last plan
        if idx < len(navigation_memory_plans):
            output.append("")

    return "\n".join(output).strip()



if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    image_utils = ImageUtils()
    

    # i want to get config file from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_android.yaml")
    args = parser.parse_args()

    configs = Dynaconf(settings_files=[args.config], merge_enabled=True)
    configs = configs.default


    agent = build_agent(model_name=configs.agent.model_name, url=configs.agent.url, agent_settings=configs.agent.settings, debugger=dbg)
    driver = build_driver(settings=configs.driver, agent=agent)
    vlm_client = VLM(configs, dbg)


    with open(os.path.join(configs.logs.root, "graph.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = json_graph.node_link_graph(data, edges="links")

    depths = get_node_depths(graph)


    with open(os.path.join(configs.logs.root, "node_intents.json"), "rb") as f:
        node_intents = json.load(f)

    if configs.query:
        user_query = configs.query
    else:
        user_query = input("Enter your query: ")


    relavance_scores_results = sort_nodes_based_on_node_intent_similarity(user_query, node_intents, dbg, image_utils)
    top_k = configs.top_k_in_first_stage_retrieval
    relavance_scores_results = relavance_scores_results[:top_k]

    candidates = []
    for result in relavance_scores_results:
        node_id = result[0]
        candidates.append({
            "node_id": result[0],
            "page_purpose": graph.nodes[node_id]['page_purpose'],
            "depth": depths[node_id],
            "user_intents": node_intents[node_id]['user_intents'],
            "best_matched_intent": result[1],
            "best_matched_intent_score": result[2],
            "ui_navigation_memory": node_intents[node_id]['ui_navigation_memory'],
        })

    top_k_node_ids, result = vlm_client.rerank_candidates(user_query, candidates, top_k=configs.top_k_retrieval_in_stage_2)

    # for node_id in top_k_node_ids:
    #     if os.path.exists(os.path.join(configs.logs.root, "screenshots", f"{node_id}.jpg")):
    #         screenshot = cv2.imread(os.path.join(configs.logs.root, "screenshots", f"{node_id}.jpg"))
    #         cv2.imshow('out1', cv2.resize(screenshot, None, fx=0.5, fy=0.5))
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #     else:
    #         print(f"Screenshot for node {node_id} not found")
    #         continue


    # top node
    top_node_id = top_k_node_ids[0]
    top_node_navigation_memory = node_intents[top_node_id]['ui_navigation_memory']
    top_node_navigation_memory = format_navigation_plan(top_node_navigation_memory, user_query)
    
    print(top_node_navigation_memory)

    driver.reset_to_start_page()
    finish_flag = False
    max_step = 10
    # reset memory of agent
    agent.clear_history()
    print("START NAVIGATION")
    for _ in range(10):
        step_result, _ = agent.step(top_node_navigation_memory, driver.take_screenshot())
        parsed = step_result[0] if isinstance(step_result, tuple) else step_result
        print(parsed)
        if str(getattr(parsed, "action_type", "") or "").strip().lower() in ("finished", "finish"):
            finish_flag = True
            break
        else:
            driver.execute_action(parsed)
        driver.wait()


