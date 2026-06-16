
from dynaconf import Dynaconf

from Debugger import Debugger
from Driver.factory import build_driver

from VLM import VLM
from Agents.factory import build_agent

import json
from networkx.readwrite import json_graph
import os
import cv2
import time
import argparse
import numpy as np
import gradio as gr
import glob

import networkx as nx

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)

from FlagEmbedding import BGEM3FlagModel


def retrieve_nodes_from_user_intents_embeddings(
    embeddings,
    model,
    query,
    top_k=5,
):

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


def load_node_navigation_plans(logs_root: str) -> dict:
    """Load per-node navigation plans from node_navigation_plans.json."""
    path = os.path.join(logs_root, "node_navigation_plans.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def get_ui_navigation_memory_for_node(node_navigation_plans: dict, node_id: str) -> list:
    nav_entry = node_navigation_plans.get(node_id, {})
    if isinstance(nav_entry, dict):
        return nav_entry.get("ui_navigation_memory", []) or []
    return []


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


def get_task_prompts_dict_from_directory(input_dir):
    if input_dir is None:
        raise ValueError("input_dir is required in batch mode")
    task_directories = glob.glob(os.path.join(input_dir, "*"))
    task_prompts = dict()
    for task_directory in task_directories:
        prompts_file = os.path.join(task_directory, "prompts.json")
        with open(prompts_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)["prompts"]
            if len(prompts) > 0:
                task_prompts[os.path.split(task_directory)[-1]] = prompts[0]
    return task_prompts


def execute_single_task(task_prompt, configs, graph, depths, node_intents, node_navigation_plans, embeddings, embedding_model, vlm_client, agent, driver, dbg):

    if configs.use_memory_for_navigation:
        results = retrieve_nodes_from_user_intents_embeddings(
            embeddings=embeddings,
            model=embedding_model,
            query=build_retrieval_query(task_prompt),
            top_k=configs.top_k_in_first_stage_retrieval,
        )

        for result in results:
            node_id = result["node_id"]
            img = cv2.imread(os.path.join(configs.logs.root, "screenshots", f"{node_id}.jpg"))
            cv2.imshow('out', cv2.resize(img, None, fx=0.5, fy=0.5))
            cv2.waitKey(0)
            print(result)
        
        candidates = []
        for result in results:
            node_id = result["node_id"]
            node_entry = node_intents.get(node_id, {})
            summary = node_entry.get("enhanced_page_summary", {}) or {}

            candidates.append(
                {
                    "node_id": node_id,
                    "page_tag": summary.get("tag", ""),
                    "page_purpose": summary.get("page_purpose", ""),
                    "screen_type": summary.get("screen_type", ""),
                    "selected_navigation": summary.get("selected_navigation", ""),
                    "depth": depths.get(node_id),
                    "score": result.get("score"),
                    "user_intents": result.get("user_intents") or node_entry.get("user_intents", []),
                }
            )

        top_k_node_ids, result = vlm_client.rerank_candidates(task_prompt, candidates, top_k=configs.top_k_retrieval_in_stage_2)
        candidates = [c for c in candidates if c["node_id"] in top_k_node_ids]

        pick_candidate_index = getattr(configs, "pick_candidate_index", -1)
        print(f"pick_candidate_index: {pick_candidate_index}")
        if pick_candidate_index == -1:
            selected_node_id = pick_candidate(candidates, os.path.join(configs.logs.root, "screenshots"), user_query=task_prompt, vlm_reasoning=result)
        else:
            if not candidates:
                raise ValueError("No candidates available to pick.")
            if pick_candidate_index < 0 or pick_candidate_index >= len(candidates):
                raise IndexError(f"pick_candidate_index {pick_candidate_index} is out of range for {len(candidates)} candidates.")
            selected_node_id = candidates[pick_candidate_index]["node_id"]

        navigation_memory = get_ui_navigation_memory_for_node(node_navigation_plans, selected_node_id)
    else:
        navigation_memory = []

    navigation_memory = format_navigation_plan(navigation_memory, task_prompt)

    dbg.log("Using Memory for Navigation: ", configs.use_memory_for_navigation)
    dbg.log(f"Navigation memory: {navigation_memory}")

    driver.reset_to_start_page()
    agent.clear_history()
    finish_flag = False

    screenshots = []
    actions = []
    screenshots.append(driver.take_screenshot())

    print("START NAVIGATION")
    finish_flag = False
    for _ in range(getattr(configs.agent, "max_steps", 10)):
        s = time.time()
        step_result, _ = agent.step(navigation_memory, driver.take_screenshot())
        parsed = step_result[0] if isinstance(step_result, tuple) else step_result

        action_type = getattr(parsed, "action_type", None) or ""
        coords = None
        if hasattr(parsed, "sent_coords") and parsed.sent_coords and "point" in parsed.sent_coords:
            coords = parsed.sent_coords["point"]
        elif hasattr(parsed, "orig_coords") and parsed.orig_coords and "point" in parsed.orig_coords:
            coords = parsed.orig_coords["point"]
        thought = getattr(parsed, "thought", None) or ""

        actions.append({
            "type": action_type,
            "coordinate": coords,
            "thought": thought
        })

        print(f"Action: type={action_type}, coordinate={coords}, thought={thought}")

        dbg.log(f"Time per step: {time.time() - s} seconds", color="green")
        if str(action_type).strip().lower() in ("finished", "finish"):
            finish_flag = True
            return screenshots, actions, finish_flag
        else:
            driver.execute_action(parsed)
        driver.wait()
        screenshots.append(driver.take_screenshot())

    return screenshots, actions, finish_flag


def visualize_actions_on_screenshots(screenshots, actions):
    """
    Visualizes user actions as overlays on screenshots:
    - For 'click': draw a circle at the click coordinate.
    - For 'scroll'/'swipe': draw an arrow, connecting from previous to current coordinate.
    Does NOT visualize 'finished' actions.
    Returns a list of drawn (annotated) screenshots.

    Note:
    Number of screenshots is typically one more than number of actions (screenshot taken before *first* action).
    If lengths differ, we overlay up to the shortest and pad/truncate accordingly.
    """
    if not screenshots or not actions:
        return screenshots

    num_screenshots = len(screenshots)
    num_actions = len(actions)
    annotated = [img.copy() for img in screenshots]

    action_indices = [
        i for i in range(min(num_actions, num_screenshots))
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


if __name__ == "__main__":
    dbg = Debugger(palette="soft", indent_size=2, width=90)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/clock_android.yaml")
    args = parser.parse_args()

    configs = Dynaconf(settings_files=[args.config], merge_enabled=True)
    configs = configs.default

    with open(os.path.join(configs.logs.root, "graph.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    graph = json_graph.node_link_graph(data, edges="links")
    depths = get_node_depths(graph)
    with open(os.path.join(configs.logs.root, "node_intents.json"), "r", encoding="utf-8") as f:
        node_intents = json.load(f)
    node_navigation_plans = load_node_navigation_plans(configs.logs.root)

    agent = build_agent(model_name=configs.agent.model_name, url=configs.agent.url, agent_settings=configs.agent.settings, debugger=dbg)
    driver = build_driver(settings=configs.driver, agent=agent)
    vlm_client = VLM(configs, dbg)
    batch_mode = configs.get("batch_mode", False)
    output_dir = configs.get("output_dir", None)

    with open(os.path.join(configs.logs.root, "node_intents.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    node_ids, embeddings = zip(*[
        (node_id, item["embedding"])
        for node_id, item in data.items() if "embedding" in item
    ]) if data else ([], [])

    embeddings = np.asarray(embeddings, dtype=np.float32)
    embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    if batch_mode:
        if output_dir is None:
            raise ValueError("output_dir is required in batch mode")
        task_prompts = get_task_prompts_dict_from_directory(configs.input_dir)

        for task_name, task_prompt in task_prompts.items():
            os.makedirs(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}"), exist_ok=True)
            screenshots, actions, finish_flag = execute_single_task(task_prompt, configs, graph, depths, node_intents, node_navigation_plans, embeddings, embedding_model, vlm_client, agent, driver, dbg)
            annotated_screenshots = visualize_actions_on_screenshots(screenshots, actions)
            for i, screenshot in enumerate(annotated_screenshots):
                os.makedirs(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}", "annotated_screenshots"), exist_ok=True)
                cv2.imwrite(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}", "annotated_screenshots", f"{i}.jpg"), screenshot)
            for i, screenshot in enumerate(screenshots):
                cv2.imwrite(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}", f"{i}.jpg"), screenshot)
            with open(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}", "actions.json"), "w", encoding="utf-8") as f:
                json.dump(actions, f)
            with open(os.path.join(output_dir, f"{task_name}_{configs.pick_candidate_index}", "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"query": task_prompt}, f)

    else:
        if configs.query:
            user_query = configs.query
        else:
            user_query = input("Enter your query: ")

        screenshots, actions, finish_flag = execute_single_task(user_query, configs, graph, depths, node_intents, node_navigation_plans, embeddings, embedding_model, vlm_client, agent, driver, dbg)
        annotated_screenshots = visualize_actions_on_screenshots(screenshots, actions)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            for i, screenshot in enumerate(annotated_screenshots):
                cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), screenshot)
            with open(os.path.join(output_dir, "actions.json"), "w", encoding="utf-8") as f:
                json.dump(actions, f)
            with open(os.path.join(output_dir, "prompt.json"), "w", encoding="utf-8") as f:
                json.dump({"query": user_query}, f)
