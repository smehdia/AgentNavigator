# NOTE: reference file from the PG-Agent graph-build + serving pipeline.
# Requires the upstream PG-Agent checkout ($PG_AGENT_ROOT) and a served VLM
# (we used Qwen2.5-VL-72B). See baselines/pg_agent/README.md before running.
#!/usr/bin/env python3
"""HTTP wrapper for PG-Agent's Odyssey workflow.

The upstream PG-Agent workflow is an offline benchmark script.  This server
adapts the same prompt/retrieval flow to AgentNavigator's live benchmark
contract:

  POST /next_action
  {
    "task": "...",
    "task_id": "...",
    "step": 1,
    "screenshot_url": "http://...",
    "history": [...],
    "graph_path": "pg_agent_odyssey_dataset/odyssey_library.json"
  }

It returns the PG-Agent action plus retrieved guidelines/prompts/raw responses
for auditability.  Run this on the host with the PG-Agent graph, bge-m3
embeddings, and the Qwen2.5-VL OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_utils import load_first_env_file
from build_graph import (
    DEFAULT_PG_AGENT_ROOT,
    OpenAICompatibleVisionClient,
    make_embedding_model,
    make_document,
    similarity_search,
)


def load_workflow_prompts(pg_agent_root: Path) -> Any:
    prompts_path = pg_agent_root / "workflow/odyssey/prompts.py"
    if not prompts_path.exists():
        raise FileNotFoundError(f"PG-Agent Odyssey workflow prompts not found: {prompts_path}")
    spec = importlib.util.spec_from_file_location("pg_agent_odyssey_workflow_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import PG-Agent workflow prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_graph(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    return {domain: {str(idx): item for idx, item in pages.items()} for domain, pages in raw.items()}


def document_transform(raw_document: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    search_document: dict[str, list[Any]] = {}
    for domain, pages in raw_document.items():
        search_document[domain] = []
        for idx in sorted(pages, key=lambda value: int(value)):
            item = pages[idx]
            search_document[domain].append(make_document(item["page_summary"], item))
    return search_document


def select_domain(raw_document: dict[str, Any], request: dict[str, Any]) -> str:
    requested = request.get("graph_domain") or request.get("domain") or request.get("app")
    if requested in raw_document:
        return str(requested)
    if requested:
        matches = [domain for domain in raw_document if str(domain).startswith(str(requested))]
        if len(matches) == 1:
            return matches[0]

    task_id = str(request.get("task_id") or "")
    matches = [domain for domain in raw_document if task_id.startswith(str(domain).split("_", 1)[0])]
    if len(matches) == 1:
        return matches[0]

    if len(raw_document) == 1:
        return next(iter(raw_document))
    raise ValueError(
        "graph has multiple domains; include graph_domain in request or use a single-app graph file"
    )


def bfs_goals(goal_list: list[str], idx: int | None, search_document: list[Any]) -> None:
    if idx is None:
        return
    queue = deque([(idx, 0)])
    visited = {idx}
    while queue:
        cur_node, cur_depth = queue.popleft()
        if cur_depth >= 3:
            continue
        next_page_list = search_document[cur_node].metadata["next_page_list"]
        for next_node in next_page_list:
            if next_node["actions"] == []:
                continue
            if next_node["goal"] not in goal_list:
                goal_list.append(next_node["goal"])
            node_idx = next_node["page_index"]
            if node_idx is not None and node_idx not in visited:
                visited.add(node_idx)
                queue.append((node_idx, cur_depth + 1))


def get_reference_actions(
    *,
    img_url: str,
    domain: str,
    goal: str,
    search_document: dict[str, list[Any]],
    embedding_model: Any,
    prompts: Any,
    client: OpenAICompatibleVisionClient,
    max_count: int = 10,
) -> tuple[str, list[dict[str, Any]]]:
    reference_actions = ""
    audit_records: list[dict[str, Any]] = []
    page_summary = client.chat([img_url], prompts.PAGE_SUMMARY_PROMPT)
    search_res = similarity_search(search_document[domain], page_summary, embedding_model)

    count = 0
    for res in search_res:
        for actions_chain in res.metadata["next_page_list"]:
            if len(actions_chain["actions"]) == 0:
                continue
            count += 1
            action_string = ", ".join(actions_chain["actions"])
            goal_list = [actions_chain["goal"]]
            bfs_goals(goal_list, actions_chain["page_index"], search_document[domain])
            goals_string = "; ".join(goal[:-1] if goal.endswith(".") else goal for goal in goal_list)
            reference_actions += prompts.REFERENCE_FORMAT.format(
                idx=count, actions=action_string, goals=goals_string
            )
            audit_records.append(
                {
                    "idx": count,
                    "source_page_index": res.metadata.get("index"),
                    "actions": actions_chain["actions"],
                    "goals": goal_list,
                    "next_page_index": actions_chain["page_index"],
                }
            )
            if count == max_count:
                return reference_actions, audit_records
    return reference_actions, audit_records


def history_text(history: list[dict[str, Any]]) -> str:
    if not history:
        return "<No previous step has been taken.>"
    lines: list[str] = []
    for idx, item in enumerate(history[-4:], 1):
        raw = item.get("pg_action", {}).get("raw")
        if isinstance(raw, dict):
            raw = raw.get("action") or raw.get("raw_action") or raw
        lines.append(f"Step{idx}: {raw}.")
    return "\n".join(lines)


def extract_action(response: str) -> tuple[str, str]:
    thought = ""
    action = response.strip()
    if "### Thought ###" in response and "### Action ###" in response:
        thought = response.split("### Thought ###", 1)[1].split("### Action ###", 1)[0].strip()
        action = response.split("### Action ###", 1)[1].strip()
    elif "### Action ###" in response:
        action = response.split("### Action ###", 1)[1].strip()
    return thought, action.splitlines()[0].strip()


def coordinate_frame_instruction(request: dict[str, Any]) -> str:
    width = request.get("screen_width")
    height = request.get("screen_height")
    if not width or not height:
        return ""
    return (
        "\n\n### Coordinate Frame ###\n"
        f"The screenshot resolution is {int(width)} pixels wide and {int(height)} pixels tall. "
        "For CLICK and LONG_PRESS, output x,y coordinates in this original screenshot coordinate system, "
        "not in a resized, cropped, or model-internal image coordinate system. "
        f"Valid x coordinates are 0 to {int(width) - 1}; valid y coordinates are 0 to {int(height) - 1}."
    )


class PGAgentOdysseyService:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.prompts = load_workflow_prompts(Path(args.pg_agent_root))
        self.client = OpenAICompatibleVisionClient(
            args.endpoint,
            args.model,
            timeout=args.timeout,
            api_key_env=args.api_key_env,
        )
        self.embedding_model = make_embedding_model(args.embedding_model, args.embedding_device)
        self.graph_cache: dict[str, tuple[dict[str, Any], dict[str, list[Any]]]] = {}
        self.global_plan_cache: dict[str, str] = {}

    def get_graph(self, graph_path: str) -> tuple[dict[str, Any], dict[str, list[Any]]]:
        path = str(Path(graph_path).resolve())
        if path not in self.graph_cache:
            raw = load_graph(Path(path))
            self.graph_cache[path] = (raw, document_transform(raw))
        return self.graph_cache[path]

    def next_action(self, request: dict[str, Any]) -> dict[str, Any]:
        requested_graph_path = request.get("graph_path") or request.get("graph_dir")
        graph_path = self.args.graph_path
        if requested_graph_path:
            requested_path = Path(str(requested_graph_path))
            if requested_path.exists():
                graph_path = requested_path
        if not graph_path:
            raise ValueError("request missing graph_path and server started without --graph-path")

        raw_graph, search_document = self.get_graph(str(graph_path))
        domain = select_domain(raw_graph, request)
        goal = str(request["task"])
        img_url = str(request.get("screenshot_data_url") or request["screenshot_url"])
        previous_step = history_text(request.get("history") or [])

        reference_actions, guideline_records = get_reference_actions(
            img_url=img_url,
            domain=domain,
            goal=goal,
            search_document=search_document,
            embedding_model=self.embedding_model,
            prompts=self.prompts,
            client=self.client,
            max_count=self.args.max_references,
        )

        observation_prompt = self.prompts.ODYSSEY_OBSERVATION_PROMT.replace("<goal>", goal).replace(
            "<history>", previous_step
        )
        observation = self.client.chat([img_url], observation_prompt)

        task_key = str(request.get("task_id") or goal)
        if int(request.get("step") or 1) <= 1 or task_key not in self.global_plan_cache:
            global_plan_prompt = self.prompts.ODYSSEY_GLOBAL_PLANNING_PROMT.replace("<goal>", goal)
            global_plan = self.client.chat([img_url], global_plan_prompt).split("### Global Plan ###")[-1].strip()
            self.global_plan_cache[task_key] = global_plan
        else:
            global_plan = self.global_plan_cache[task_key]

        plan_prompt = self.prompts.ODYSSEY_PLANNING_PROMT.replace("<goal>", goal)
        plan_prompt = plan_prompt.replace("<observation>", observation)
        plan_prompt = plan_prompt.replace("<global_plan>", global_plan)
        plan_prompt = plan_prompt.replace("<reference>", reference_actions)
        plan_prompt = plan_prompt.replace("<history>", previous_step)
        local_plan = self.client.chat([img_url], plan_prompt)

        execution_prompt = self.prompts.ODYSSEY_EXECUTION_PROMT.replace("<action_plan>", local_plan)
        execution_prompt = execution_prompt.replace("<reference>", reference_actions)
        execution_prompt += coordinate_frame_instruction(request)
        execution_response = self.client.chat([img_url], execution_prompt)
        thought, action = extract_action(execution_response)

        return {
            "action": action,
            "thought": thought,
            "domain": domain,
            "retrieved_guidelines": guideline_records,
            "pg_prompts": {
                "observation": observation_prompt,
                "planning": plan_prompt,
                "execution": execution_prompt,
            },
            "raw_pg_responses": {
                "observation": observation,
                "global_plan": global_plan,
                "planning": local_plan,
                "execution": execution_response,
                "reference_actions": reference_actions,
            },
        }


def make_handler(service: PGAgentOdysseyService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/next_action":
                self.send_error(404, "not found")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                request = json.loads(self.rfile.read(length).decode("utf-8"))
                response = service.next_action(request)
                payload = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception as exc:
                payload = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:
            if service.args.verbose:
                super().log_message(format, *args)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--pg-agent-root", type=Path, default=DEFAULT_PG_AGENT_ROOT)
    parser.add_argument("--graph-path", help="Default PG-Agent odyssey_library.json path")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--api-key-env", default="PG_AGENT_API_KEY")
    parser.add_argument("--env-file", type=Path, help="Optional .env file to load before reading --api-key-env")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--max-references", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    load_first_env_file(args.env_file, api_key_env=args.api_key_env, load_config=True)
    service = PGAgentOdysseyService(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    print(f"PG-Agent Odyssey server listening on http://{args.host}:{args.port}/next_action")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
