
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
import time
import pickle
import argparse
import numpy as np 
import gradio as gr


import networkx as nx

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy", 
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)

from FlagEmbedding import BGEM3FlagModel


def retrieve_nodes_from_user_intents_embeddings(
    root_path,
    query,
    embedded_intents_filename="node_intents.json",
    bge_model_name="BAAI/bge-m3",
    top_k=5,
):
    path = os.path.join(root_path, embedded_intents_filename)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    node_ids = []
    embeddings = []

    for node_id, item in data.items():
        if "embedding" not in item:
            continue

        node_ids.append(node_id)
        embeddings.append(item["embedding"])

    embeddings = np.asarray(embeddings, dtype=np.float32)

    model = BGEM3FlagModel(
        bge_model_name,
        use_fp16=True,
    )

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

def build_retrieval_query(user_goal: str) -> str:
    return f"""
PAGE_PURPOSE:
A UI page or screen for this goal: {user_goal}

USER_INTENTS:
{user_goal}
""".strip()




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
    Format navigation memory plans for UI-TARS.

    Notes:
    - navigation_memory_plans may contain multiple alternative plans from different roots.
    - The agent should first select the plan whose waypoints best match the current screenshot.
    - Once the final goal is reached, the agent should finish immediately.
    - If no plans are available, return an empty string so the caller can build the prompt without memory.
    """

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
        "There may be multiple alternative navigation plans, possibly generated from different roots. "
        "Before acting, compare the current screenshot with the waypoint sequences and transition hints, "
        "then follow the plan that best matches the current screen. "
        "If none of the plans match the current screenshot, rely on the screenshot instead of forcing a plan."
    )

    for idx, plan in enumerate(navigation_memory_plans, 1):
        output_lines.append("")
        output_lines.append(f"[Plan {idx}]")

        goal = plan.get("final_goal") if plan.get("final_goal") is not None else final_goal
        goal = goal.strip() if isinstance(goal, str) else ""

        output_lines.append(f"Final goal: {goal if goal else '(none)'}")

        # Optional root id, useful when plans come from different graph roots
        root_id = plan.get("root_id")
        if root_id:
            output_lines.append(f"Root id: {str(root_id).strip()}")

        # Waypoints
        waypoints = plan.get("relevant_waypoint_sequence", [])
        output_lines.append("\nRelevant waypoint sequence:")

        if isinstance(waypoints, list) and waypoints:
            wp_str = " -> ".join(str(w).strip() for w in waypoints if str(w).strip())
            output_lines.append(wp_str if wp_str else "(none)")
        else:
            output_lines.append("(none)")

        # Transition hints
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

        # Plan-specific usage instruction, if available
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


def pick_candidate(candidates, screenshots_dir, user_query="", vlm_reasoning=None):
    selection = {"node_id": None}
    def _screenshot_path(node_id):
        path = os.path.join(screenshots_dir, f"{node_id}.jpg")
        return path if os.path.exists(path) else None
    def _format_caption(c):
        lines = [
            f"**{c['node_id']}**",
            f"score={c['score']:.3f} | depth={c['depth']}",
            "",
            c.get("page_purpose", ""),
        ]
        if vlm_reasoning and c["node_id"] in vlm_reasoning:
            lines += ["", f"_VLM: {vlm_reasoning[c['node_id']]}_"]
        intents = c.get("user_intents") or []
        if intents:
            lines += ["", "**Intents:**"] + [f"- {i}" for i in intents[:5]]
        return "\n".join(lines)
    gallery_items = []
    for c in candidates:
        path = _screenshot_path(c["node_id"])
        if path:
            gallery_items.append((path, _format_caption(c)))
    radio_choices = [
        (
            f"{c['node_id']} | score={c['score']:.3f} | depth={c['depth']} | "
            f"{(c.get('page_purpose') or '')[:60]}",
            c["node_id"],
        )
        for c in candidates
    ]
    with gr.Blocks(title="Pick navigation target") as demo:
        gr.Markdown(f"### Query\n{user_query}")
        gr.Markdown("Pick the page that best matches your goal.")
        gr.Gallery(value=gallery_items, columns=3, height=500, object_fit="contain")
        radio = gr.Radio(
            choices=radio_choices,
            label="Select target page",
            value=radio_choices[0][1] if radio_choices else None,
        )
        out = gr.Textbox(label="Selected node_id", interactive=False)
        btn = gr.Button("Confirm and continue", variant="primary")
        def confirm(node_id):
            if not node_id:
                raise gr.Error("Select a candidate first.")
            selection["node_id"] = node_id
            demo.close()
            return node_id
        btn.click(confirm, inputs=radio, outputs=out)
    demo.launch(server_name="127.0.0.1", share=False)
    return selection["node_id"]


if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    image_utils = ImageUtils()
    

    # i want to get config file from user
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_harmony.yaml")
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


    if configs.use_memory_for_navigation:
        results = retrieve_nodes_from_user_intents_embeddings(
            root_path=configs.logs.root,
            query=build_retrieval_query(user_query),
            top_k=configs.top_k_in_first_stage_retrieval,
        )
        candidates = []
        for result in results:
            node_id = result["node_id"]
            candidates.append({
                "node_id": result["node_id"],
                "page_purpose": graph.nodes[node_id]['page_purpose'],
                "depth": depths[node_id],
                "score": result["score"],
                "user_intents": node_intents[node_id]['user_intents'],
                "ui_navigation_memory": node_intents[node_id]['ui_navigation_memory'],
            })

        top_k_node_ids, result = vlm_client.rerank_candidates(user_query, candidates, top_k=configs.top_k_retrieval_in_stage_2)

        # we filter candidates to only keep the one that node_ids are in top_k_node_ids
        candidates = [c for c in candidates if c["node_id"] in top_k_node_ids]
        selected_node_id = pick_candidate(candidates, os.path.join(configs.logs.root, "screenshots"), vlm_reasoning=result)

        navigation_memory = node_intents[selected_node_id]['ui_navigation_memory']
    else:
        navigation_memory = []


    navigation_memory = format_navigation_plan(navigation_memory, user_query)


    dbg.log("Using Memory for Navigation: ", configs.use_memory_for_navigation)
    dbg.log(f"Navigation memory: {navigation_memory}")

    driver.reset_to_start_page()
    agent.clear_history()
    finish_flag = False

    print("START NAVIGATION")
    for _ in range(getattr(configs, "max_agent_steps", 10)):
        s = time.time()
        step_result, _ = agent.step(navigation_memory, driver.take_screenshot())
        parsed = step_result[0] if isinstance(step_result, tuple) else step_result
        print(parsed)
        dbg.log(f"Time per step: {time.time() - s} seconds", color="green")
        if str(getattr(parsed, "action_type", "") or "").strip().lower() in ("finished", "finish"):
            finish_flag = True
            break
        else:
            driver.execute_action(parsed)
        driver.wait()


