
from dynaconf import Dynaconf

from Debugger import Debugger
from Driver.factory import build_driver

from VLM_Yibu import build_vlm_client
from Agents.factory import build_agent

import json
from networkx.readwrite import json_graph
import os
import re
import cv2
import time
import argparse
import numpy as np
import gradio as gr
import glob
import joblib


import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoModelForImageTextToText

import networkx as nx

for var in ["http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
            "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY"]:
    os.environ.pop(var, None)

from FlagEmbedding import BGEM3FlagModel

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "exploration"))
from train_localizer import ood_features, _letterbox

SIGLIP_ID = "google/siglip-base-patch16-224"
SMOLVLM_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
SIGLIP_DIM = 768  # first half of siglip_smolvlm_features.pt gallery_z
SMOL_DIM = 576
CONCAT_DIM = SIGLIP_DIM + SMOL_DIM



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


def load_node_navigation_plans(logs_root: str) -> dict:
    """Load per-node navigation plans from node_navigation_plans.json."""
    path = os.path.join(logs_root, "node_navigation_plans.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def get_ui_navigation_memory_for_node(node_navigation_plans: dict, node_id: str) -> list:
    return normalize_ui_navigation_memory(node_navigation_plans.get(node_id))


def build_retrieval_query(user_goal: str) -> str:
    return str(user_goal or "").strip()

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


def execute_single_task(task_prompt, configs, graph, depths, node_intents, node_level_information, node_navigation_plans, embeddings, embedding_model, vlm_client, agent, driver, localizer_embedder, ood_classifier, gallery_z, node_features, node_ids, target_hw, edge_level_information, dbg):

    if configs.use_memory_for_navigation:
        results = retrieve_nodes_from_user_intents_embeddings(
            embeddings=embeddings,
            model=embedding_model,
            query=build_retrieval_query(task_prompt),
            top_k=configs.top_k_in_first_stage_retrieval,
        )

        rerank_payload = build_rerank_candidates(
            task_prompt,
            results,
            node_level_information,
            node_intents,
            node_navigation_plans,
        )
        candidates = rerank_payload["candidates"]

        top_k_node_ids, rerank_result = vlm_client.rerank_candidates(
            task_prompt,
            candidates,
            top_k=configs.top_k_retrieval_in_stage_2,
        )
        

        candidates = [c for c in candidates if c["node_id"] in top_k_node_ids]
        candidates = [
            {
                **c,
                "score": c.get("retrieval_score", c.get("score", 0.0)),
                "depth": depths.get(c["node_id"], 0),
                "page_purpose": (
                    c.get("page_description", {}).get("high_level", "")
                    or c.get("page_description", {}).get("medium_level", "")
                ),
            }
            for c in candidates
        ]

        pick_candidate_index = getattr(configs, "pick_candidate_index", -1)
        print(f"pick_candidate_index: {pick_candidate_index}")
        if pick_candidate_index == -1:
            selected_node_id = pick_candidate(
                candidates,
                os.path.join(configs.logs.root, "screenshots"),
                user_query=task_prompt,
                vlm_reasoning=rerank_result.get("reasoning"),
            )
        else:
            if not candidates:
                raise ValueError("No candidates available to pick.")
            if pick_candidate_index < 0 or pick_candidate_index >= len(candidates):
                raise IndexError(f"pick_candidate_index {pick_candidate_index} is out of range for {len(candidates)} candidates.")
            selected_node_id = candidates[pick_candidate_index]["node_id"]


    driver.reset_to_start_page()
    agent.clear_history()
    finish_flag = False

    screenshots = []
    actions = []
    # Pre-normalize gallery once (used every on-graph step).
    G_gallery = np.asarray(node_features, dtype=np.float32)
    G_gallery = G_gallery / (np.linalg.norm(G_gallery, axis=1, keepdims=True) + 1e-8)

    print("START NAVIGATION")
    for step_i in range(getattr(configs.agent, "max_steps", 10)):
        t_step = time.time()

        t0 = time.time()
        screenshot = driver.take_screenshot()
        t_shot = time.time() - t0

        t0 = time.time()
        shot = _letterbox(screenshot, target_hw)
        # OOD only needs SigLIP; skip SmolVLM encode when off-graph.
        z = localizer_embedder.siglip_feat(shot)
        ood_feat = ood_features(z, gallery_z)
        ood_label = int(ood_classifier.predict(ood_feat.reshape(1, -1))[0])
        transition_hints = []
        top3 = []

        if ood_label == 1:
            dbg.log("On Graph", color="red")
            smol = localizer_embedder.smol_vision_feat(shot)
            screenshot_features = _l2(np.concatenate([z, smol], axis=0))
            q = np.asarray(screenshot_features, dtype=np.float32).reshape(-1)
            if q.size != G_gallery.shape[1]:
                raise RuntimeError(
                    f"feature dim mismatch: query={q.size}, gallery={G_gallery.shape[1]} "
                    f"(expected {CONCAT_DIM}). Re-run save_screenshot_features for this app."
                )
            q = q / (np.linalg.norm(q) + 1e-8)
            sims = G_gallery @ q
            top = np.argsort(-sims)[:3]
            top3 = [(node_ids[i], float(sims[i])) for i in top]
            top_ids = {nid for nid, _ in top3}

            if selected_node_id in top_ids:
                t_loc = time.time() - t0
                actions.append({
                    "type": "finished",
                    "coordinate": None,
                    "thought": "The selected node is already in the top 3 nodes.",
                })
                screenshots.append(screenshot)
                print(
                    f"[step {step_i + 1}] at_target  "
                    f"shot={t_shot:.2f}s loc={t_loc:.2f}s total={time.time() - t_step:.2f}s"
                )
                finish_flag = True
                return screenshots, actions, finish_flag

            for nid, score in top3:
                try:
                    path = nx.shortest_path(graph, nid, selected_node_id)
                    nxt = path[1] if len(path) > 1 else nid
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    nxt = None
                edge_info = edge_level_information.get("{}|{}".format(nid, nxt), [])
                if isinstance(edge_info, list):
                    edge_info = edge_info[0] if edge_info else {}
                if not isinstance(edge_info, dict):
                    edge_info = {}
                transition_hints.append({
                    "low_level": edge_info.get("low_level_action_description", ""),
                    "high_level": edge_info.get("high_level_action_description", ""),
                })
        else:
            dbg.log("Not on Graph", color="blue")

        t_loc = time.time() - t0
        navigation_plan = format_navigation_plan(task_prompt, transition_hints)

        t0 = time.time()
        step_result, _ = agent.step(navigation_plan, screenshot)
        t_agent = time.time() - t0
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
            "thought": thought,
        })
        screenshots.append(screenshot)

        print(f"Action: type={action_type}, coordinate={coords}")
        if top3:
            print("  top3:", [(n, round(s, 3)) for n, s in top3])

        if str(action_type).strip().lower() in ("finished", "finish"):
            print(
                f"[step {step_i + 1}] finish  "
                f"shot={t_shot:.2f}s loc={t_loc:.2f}s agent={t_agent:.2f}s "
                f"total={time.time() - t_step:.2f}s"
            )
            finish_flag = True
            return screenshots, actions, finish_flag

        t0 = time.time()
        driver.execute_action(parsed)
        t_act = time.time() - t0
        print(
            f"[step {step_i + 1}]  "
            f"shot={t_shot:.2f}s loc={t_loc:.2f}s agent={t_agent:.2f}s "
            f"act={t_act:.2f}s total={time.time() - t_step:.2f}s"
        )

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


def _l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-8)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Encoders:
    """Lazy SigLIP + SmolVLM vision encoders."""

    def __init__(self, device: str = "cuda"):

        self.device = device
        self._Image = Image

        self.siglip_proc = AutoProcessor.from_pretrained(SIGLIP_ID)
        self.siglip = AutoModel.from_pretrained(
            SIGLIP_ID, torch_dtype=torch.float16
        ).to(device)
        self.siglip.eval()

        self.smol_proc = AutoProcessor.from_pretrained(SMOLVLM_ID)
        self.smol = AutoModelForImageTextToText.from_pretrained(
            SMOLVLM_ID,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        ).to(device)
        self.smol.eval()
        self.image_token_id = getattr(self.smol.config, "image_token_id", None)

    def _pil(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._Image.fromarray(rgb)

    @torch.inference_mode()
    def siglip_feat(self, bgr: np.ndarray) -> np.ndarray:
        pil = self._pil(bgr)
        pixel_values = self.siglip_proc(images=pil, return_tensors="pt")["pixel_values"].to(
            self.device
        )
        # Always use pooled projection — never flatten patch tokens (B, 196, 768).
        pooled = self.siglip.vision_model(pixel_values=pixel_values).pooler_output
        z = self.siglip.visual_projection(pooled) if hasattr(self.siglip, "visual_projection") else pooled
        z = z[0].float().cpu().numpy().reshape(-1)
        if z.size != SIGLIP_DIM:
            raise RuntimeError(f"siglip_feat expected dim {SIGLIP_DIM}, got {z.size}")
        return _l2(z)

    @torch.inference_mode()
    def smol_vision_feat(self, bgr: np.ndarray) -> np.ndarray:
        """Mean-pool SmolVLM last-layer hidden states over image tokens only."""
        pil = self._pil(bgr)
        pil.thumbnail((512, 512))
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Screenshot."},
                ],
            }
        ]
        prompt = self.smol_proc.apply_chat_template(msgs, add_generation_prompt=False)
        inputs = self.smol_proc(text=prompt, images=[pil], return_tensors="pt")
        inputs = {
            k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()
        }
        out = self.smol(**inputs, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states[-1][0]
        input_ids = inputs["input_ids"][0]
        if self.image_token_id is not None:
            mask = input_ids == self.image_token_id
            if bool(mask.any()):
                z = hs[mask].mean(0).float().cpu().numpy().reshape(-1)
                return _l2(z)
        am = inputs["attention_mask"][0].bool()
        z = hs[am].mean(0).float().cpu().numpy().reshape(-1)
        return _l2(z)

    def concat_feat(self, bgr: np.ndarray) -> np.ndarray:
        sig = np.asarray(self.siglip_feat(bgr), dtype=np.float32).reshape(-1)
        smol = np.asarray(self.smol_vision_feat(bgr), dtype=np.float32).reshape(-1)
        if sig.size != SIGLIP_DIM or smol.size != SMOL_DIM:
            raise RuntimeError(
                f"concat_feat dims siglip={sig.size} (want {SIGLIP_DIM}), "
                f"smol={smol.size} (want {SMOL_DIM})"
            )
        return _l2(np.concatenate([sig, smol], axis=0))


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
    with open(os.path.join(configs.logs.root, "user_intents.json"), "r", encoding="utf-8") as f:
        node_intents = json.load(f)
    node_navigation_plans = load_node_navigation_plans(configs.logs.root)

    agent = build_agent(model_name=configs.agent.model_name, url=configs.agent.url, agent_settings=configs.agent.settings, debugger=dbg)
    driver = build_driver(settings=configs.driver, agent=agent)
    vlm_client = build_vlm_client(configs, dbg)
    batch_mode = configs.get("batch_mode", False)
    output_dir = configs.get("output_dir", None)

    with open(os.path.join(configs.logs.root, "user_intents.json"), "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(os.path.join(configs.logs.root, "node_level_information.json"), "r", encoding="utf-8") as f:
        node_level_information = json.load(f)

    node_ids, embeddings = zip(*[
        (node_id, item["embedding"])
        for node_id, item in data.items() if "embedding" in item
    ]) if data else ([], [])

    embeddings = np.asarray(embeddings, dtype=np.float32)
    embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    localizer_embedder = Encoders(device="cuda")
    ood_payload = joblib.load(os.path.join(configs.logs.root, "ood_classifier.joblib"))
    ood_classifier = ood_payload["model"]
    ood_threshold = ood_payload["threshold"]
    feat_payload = torch.load(
        os.path.join(configs.logs.root, "siglip_smolvlm_features.pt"),
        map_location="cpu",
        weights_only=False,
    )
    node_features = np.asarray(feat_payload["gallery_z"], dtype=np.float32)
    feat_node_ids = list(feat_payload["node_ids"])
    if node_features.ndim != 2 or node_features.shape[1] != CONCAT_DIM:
        raise RuntimeError(
            f"siglip_smolvlm_features.pt has dim {getattr(node_features, 'shape', None)}; "
            f"expected (*, {CONCAT_DIM}). Re-save features for this app with train_localizer."
        )
    # OOD profile uses SigLIP half of concat(siglip, smolvlm)
    gallery_z = node_features[:, :SIGLIP_DIM]
    target_hw = tuple(feat_payload["target_hw"])
    
    if configs.query:
        user_query = configs.query
    else:
        user_query = input("Enter your query: ")

    with open(os.path.join(configs.logs.root, "edge_level_information.json"), "r", encoding="utf-8") as f:
        edge_level_information = json.load(f)

    screenshots, actions, finish_flag = execute_single_task(user_query, configs, graph, depths, node_intents, node_level_information, node_navigation_plans, embeddings, embedding_model, vlm_client, agent, driver, localizer_embedder, ood_classifier, gallery_z, node_features, feat_node_ids, target_hw, edge_level_information, dbg)
    annotated_screenshots = visualize_actions_on_screenshots(screenshots, actions)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for i, screenshot in enumerate(annotated_screenshots):
            cv2.imwrite(os.path.join(output_dir, f"{i}.jpg"), screenshot)
        with open(os.path.join(output_dir, "actions.json"), "w", encoding="utf-8") as f:
            json.dump(actions, f)
        with open(os.path.join(output_dir, "prompt.json"), "w", encoding="utf-8") as f:
            json.dump({"query": user_query}, f)
