import argparse
import json
import os
import random
import re

from PIL import Image

# Verbatim MAI_MOBILE_SYS_PROMPT from AgentNavigator inference/Agents/MAI_UI/prompt.py
SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags."""

TERMINATE_ASSISTANT = (
    "<thinking>\n"
    "The screen already matches the user's goal, so I will terminate with success.\n"
    "</thinking>\n"
    '<tool_call>\n'
    '{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}\n'
    "</tool_call>"
)


def parse_edge_key(edge_key):
    if "|" not in edge_key:
        return None, None
    source_node, next_node = edge_key.split("|", 1)
    return source_node, next_node


def load_graph_edge_actions(graph_path):
    """Map (source, target) -> {action_key(int): {type, boundingBox, description}} from graph.json."""
    with open(graph_path, encoding="utf-8") as f:
        graph = json.load(f)
    edge_actions = {}
    for link in graph.get("links", []):
        pair = (link.get("source"), link.get("target"))
        edge_actions.setdefault(pair, {})[int(link.get("key", 0))] = {
            "type": str(link.get("type", "") or "").strip().lower(),
            "boundingBox": link.get("boundingBox"),
            "description": link.get("description", ""),
        }
    return edge_actions


def normalize_coordinate(bbox, width, height, scale=999):
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return [
        max(0, min(scale, round(cx / width * scale))),
        max(0, min(scale, round(cy / height * scale))),
    ]


def build_action_arguments(action_type, coordinate):
    if "scroll" in action_type:
        return {"action": "swipe", "direction": "down", "coordinate": coordinate}
    if "swipe" in action_type:
        return {"action": "swipe", "direction": "left", "coordinate": coordinate}
    return {"action": "click", "coordinate": coordinate}


def build_grounding_assistant(thought, arguments):
    thought = (thought or "").strip() or "I will interact with the target element."
    tool_call = json.dumps({"name": "mobile_use", "arguments": arguments}, separators=(",", ":"))
    return f"<thinking>\n{thought}\n</thinking>\n<tool_call>\n{tool_call}\n</tool_call>"


def validate_agent_output(text):
    if not text.strip():
        return None
    tool_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not tool_match:
        return None
    try:
        tool_call = json.loads(tool_match.group(1).strip())
    except json.JSONDecodeError:
        return None
    if not isinstance(tool_call.get("arguments"), dict):
        return None
    return text.strip()


def build_sample(
    app_graph_path,
    target_node,
    source_node,
    next_node,
    edge_key,
    user_intent,
    assistant_text,
    screenshot_path,
    sample_type,
):
    return {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_intent}]},
            {"role": "user", "content": [{"type": "image", "image": screenshot_path}]},
            {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
        ],
        "metadata": {
            "app_graph_path": app_graph_path,
            "target_node": target_node,
            "source_node": source_node,
            "next_node": next_node,
            "edge_key": edge_key,
            "user_intent": user_intent,
            "sample_type": sample_type,
        },
    }


def dedup_key(sample):
    meta = sample["metadata"]
    assistant = sample["messages"][-1]["content"][0]["text"]
    return (
        meta["user_intent"],
        meta["source_node"],
        meta["next_node"],
        meta["edge_key"],
        meta["sample_type"],
        assistant,
    )


def screen_size(screenshot_path, cache):
    if screenshot_path not in cache:
        with Image.open(screenshot_path) as img:
            cache[screenshot_path] = img.size  # (width, height)
    return cache[screenshot_path]


def collect_grounding_candidates(
    app_graph_path, edge_actions, edge_level_information, screenshots_dir, stats
):
    size_cache = {}
    candidates = []
    for edge_key, entries in edge_level_information.items():
        source_node, next_node = parse_edge_key(edge_key)
        if not source_node:
            stats["skipped_bad_edge_key"] += len(entries)
            continue

        graph_actions = edge_actions.get((source_node, next_node))
        if not graph_actions:
            stats["skipped_no_edge_info"] += len(entries)
            continue

        screenshot_path = os.path.join(screenshots_dir, f"{source_node}.jpg")
        if not os.path.exists(screenshot_path):
            stats["skipped_no_screenshot"] += len(entries)
            continue
        abs_screenshot = os.path.abspath(screenshot_path)
        width, height = screen_size(screenshot_path, size_cache)

        for entry in entries:
            action_key = int((entry.get("action_ids") or ["0"])[0])
            action = graph_actions.get(action_key)
            if action is None:
                stats["skipped_no_edge_info"] += 1
                continue

            text_pool = [
                (entry.get("low_level_action_description") or "").strip(),
                (entry.get("high_level_action_description") or "").strip(),
                (action.get("description") or "").strip(),
            ]
            text_pool = [t for t in text_pool if t]
            if not text_pool:
                stats["skipped_empty_entry"] += 1
                continue

            action_type = action["type"]
            bbox = action.get("boundingBox")
            if bbox and len(bbox) >= 4:
                coordinate = normalize_coordinate(bbox, width, height)
            elif "scroll" in action_type or "swipe" in action_type:
                coordinate = [499, 499]
            else:
                stats["skipped_no_bbox"] += 1
                continue

            candidates.append({
                "app_graph_path": app_graph_path,
                "target_node": next_node,
                "source_node": source_node,
                "next_node": next_node,
                "edge_key": edge_key,
                "text_pool": text_pool,
                "action_type": action_type,
                "coordinate": coordinate,
                "screenshot_path": abs_screenshot,
            })
    return candidates


def add_grounding_samples(candidates, target_count, samples, stats, rng):
    if target_count <= 0 or not candidates:
        return

    selected = rng.sample(candidates, target_count)
    for cand in selected:
        instruction = rng.choice(cand["text_pool"])
        thought = rng.choice(cand["text_pool"])
        arguments = build_action_arguments(cand["action_type"], cand["coordinate"])
        assistant_text = build_grounding_assistant(thought, arguments)
        sample = build_sample(
            cand["app_graph_path"],
            target_node=cand["target_node"],
            source_node=cand["source_node"],
            next_node=cand["next_node"],
            edge_key=cand["edge_key"],
            user_intent=instruction,
            assistant_text=assistant_text,
            screenshot_path=cand["screenshot_path"],
            sample_type="grounding",
        )
        samples.append(sample)
        stats["grounding_written"] += 1


def build_samples(app_graph_path, agent_data, user_intents, screenshots_dir):
    samples = []
    seen = set()
    stats = {
        "navigation_written": 0,
        "terminate_written": 0,
        "grounding_written": 0,
        "skipped_duplicate": 0,
        "skipped_no_screenshot": 0,
        "skipped_no_edge_info": 0,
        "skipped_no_bbox": 0,
        "skipped_bad_edge_key": 0,
        "skipped_empty_entry": 0,
        "skipped_unparseable": 0,
    }

    for target_node, edges in agent_data.items():
        if not isinstance(edges, dict):
            continue

        for edge_key, entries in edges.items():
            source_node, next_node = parse_edge_key(edge_key)
            if not source_node:
                stats["skipped_bad_edge_key"] += len(entries) if isinstance(entries, list) else 1
                continue

            screenshot_path = os.path.join(screenshots_dir, f"{source_node}.jpg")
            if not os.path.exists(screenshot_path):
                stats["skipped_no_screenshot"] += len(entries) if isinstance(entries, list) else 1
                continue
            abs_screenshot = os.path.abspath(screenshot_path)

            for entry in entries:
                user_instruction = (entry.get("user_instruction") or "").strip()
                agent_output = validate_agent_output(entry.get("agent_output") or "")
                if not user_instruction or agent_output is None:
                    stats["skipped_empty_entry" if not user_instruction else "skipped_unparseable"] += 1
                    continue

                sample = build_sample(
                    app_graph_path,
                    target_node=target_node,
                    source_node=source_node,
                    next_node=next_node,
                    edge_key=edge_key,
                    user_intent=user_instruction,
                    assistant_text=agent_output,
                    screenshot_path=abs_screenshot,
                    sample_type="navigation",
                )
                key = dedup_key(sample)
                if key in seen:
                    stats["skipped_duplicate"] += 1
                    continue
                seen.add(key)
                samples.append(sample)
                stats["navigation_written"] += 1

    for target_node, node_data in user_intents.items():
        intents = node_data.get("user_intents") or []
        if not intents:
            continue

        screenshot_path = os.path.join(screenshots_dir, f"{target_node}.jpg")
        if not os.path.exists(screenshot_path):
            stats["skipped_no_screenshot"] += len(intents)
            continue
        abs_screenshot = os.path.abspath(screenshot_path)

        for user_intent in intents:
            sample = build_sample(
                app_graph_path,
                target_node=target_node,
                source_node=target_node,
                next_node=target_node,
                edge_key="terminate",
                user_intent=user_intent,
                assistant_text=TERMINATE_ASSISTANT,
                screenshot_path=abs_screenshot,
                sample_type="terminate",
            )
            key = dedup_key(sample)
            if key in seen:
                stats["skipped_duplicate"] += 1
                continue
            seen.add(key)
            samples.append(sample)
            stats["terminate_written"] += 1

    return samples, seen, stats


def main():
    parser = argparse.ArgumentParser(
        description="Build MAI-UI fine-tuning samples from agent_data.json and user_intents.json.",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Explored app root (agent_data.json, user_intents.json, graph.json, screenshots/).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: <root>/training_data.jsonl).",
    )
    parser.add_argument(
        "--grounding-ratio",
        type=float,
        default=1.0,
        help="Fraction of navigation (planning) samples to add as grounding samples (0.0–1.0). "
        "Capped at the number of available edge actions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for grounding candidate/text selection.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.grounding_ratio <= 1.0:
        parser.error("--grounding-ratio must be between 0.0 and 1.0")

    root = os.path.abspath(args.root)

    agent_data_path = os.path.join(root, "agent_data.json")
    user_intents_path = os.path.join(root, "user_intents.json")
    screenshots_dir = os.path.join(root, "screenshots")
    output_path = args.output or os.path.join(root, "training_data.jsonl")

    for path in (agent_data_path, user_intents_path, screenshots_dir):
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    with open(agent_data_path, encoding="utf-8") as f:
        agent_data = json.load(f)
    with open(user_intents_path, encoding="utf-8") as f:
        user_intents = json.load(f)

    samples, seen, stats = build_samples(root, agent_data, user_intents, screenshots_dir)

    if args.grounding_ratio > 0:
        graph_path = os.path.join(root, "graph.json")
        edge_info_path = os.path.join(root, "edge_level_information.json")
        if not os.path.exists(graph_path) or not os.path.exists(edge_info_path):
            print("Skipping grounding: graph.json or edge_level_information.json not found")
        else:
            edge_actions = load_graph_edge_actions(graph_path)
            with open(edge_info_path, encoding="utf-8") as f:
                edge_level_information = json.load(f)
            candidates = collect_grounding_candidates(
                root, edge_actions, edge_level_information, screenshots_dir, stats
            )
            target_grounding = min(
                round(stats["navigation_written"] * args.grounding_ratio),
                len(candidates),
            )
            rng = random.Random(args.seed)
            add_grounding_samples(candidates, target_grounding, samples, stats, rng)
            print(f"Grounding candidates (edges): {len(candidates)}")
            print(f"Grounding target (min(navigation × {args.grounding_ratio}, edges)): {target_grounding}")

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    total = stats["navigation_written"] + stats["terminate_written"] + stats["grounding_written"]
    print(f"Root: {root}")
    print(f"Output: {output_path}")
    print(f"Navigation samples: {stats['navigation_written']}")
    print(f"Grounding samples: {stats['grounding_written']}")
    print(f"Terminate samples: {stats['terminate_written']}")
    print(f"Total samples: {total}")
    print(f"Skipped duplicate: {stats['skipped_duplicate']}")
    print(f"Skipped empty entry: {stats['skipped_empty_entry']}")
    print(f"Skipped unparseable agent_output: {stats['skipped_unparseable']}")
    print(f"Skipped bad edge key: {stats['skipped_bad_edge_key']}")
    print(f"Skipped no edge info: {stats['skipped_no_edge_info']}")
    print(f"Skipped no bbox: {stats['skipped_no_bbox']}")
    print(f"Skipped no screenshot: {stats['skipped_no_screenshot']}")


if __name__ == "__main__":
    main()
