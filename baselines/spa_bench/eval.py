#!/usr/bin/env python3
"""Run SPA-Bench evaluation on exported sessions, bypassing PaddleOCR dependency.

Uses GPT-4o for VLM evaluation with --skip_key_components semantics
(skips OCR-based key component matching, goes straight to VLM judge).

Required environment:
  OPENAI_API_KEY   (GPT-4o judge)

Example:
  python baselines/spa_bench/eval.py \
      --result-dir results/spa_bench/eval_no_graph \
      --task-ids chrome_2 clock_2 \
      --agent TAGNavAgent
"""
from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import requests


def get_dataset(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path, encoding="utf-8", header=0, index_col=0)
    for col in data.columns:
        data[col] = data[col].apply(_try_literal_eval)
    return data


def _try_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def get_screenshot_file_names(target_dir: str) -> list[str]:
    try:
        files = os.listdir(target_dir)
    except FileNotFoundError:
        return []
    screenshots = [f for f in files if f.endswith(".png") or f.endswith(".jpg")]
    screenshots.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return screenshots


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def generate_image_list(target_dir: str, detail: str = "high") -> list[dict]:
    image_list = []
    for fname in get_screenshot_file_names(target_dir):
        path = os.path.join(target_dir, fname)
        b64 = encode_image(path)
        image_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail},
        })
    return image_list


def sys_prompt() -> str:
    return (
        "You are an expert in evaluating smartphone operation tasks. "
        "Your primary role is to determine whether a task has been successfully "
        "completed based on a series of screenshots (provided in order of execution) "
        "and the corresponding task description.\n\n"
        "### Guidelines:\n"
        "1. **No Assumptions**: Evaluate solely based on the provided screenshots.\n"
        "2. **Subtask Completion**: A task is successful only when all its subtasks "
        "are successfully completed.\n"
        "3. **Common Reasons for Subtask Failure**:\n"
        "   - **Incomplete**: A subtask is not performed or achieved.\n"
        "   - **Incorrect Execution**: Screenshots don't align with instructions "
        "(wrong entity, wrong action).\n"
        "4. **Additional Actions**: Extra actions after completion are OK if they "
        "don't undo the task.\n"
        "5. **Order of Subtasks**: Any order is fine unless explicitly dependent.\n"
        "6. **Subtasks Completed Midway**: Still count as success if visible in "
        "intermediate screenshots.\n"
        "7. **Corrective Actions**: Initially failing subtasks corrected later count.\n"
        "8. **Focus on Overview**: Don't let minor irrelevant details distract.\n\n"
        "**Be strict and cautious. Use 1 to indicate success and 0 to indicate failure.**"
    )


def user_prompt(task_description: str) -> str:
    return (
        f"Now, here is a smartphone operation task description:\n"
        f"**{task_description}**\n\n"
        f"Please carefully determine whether the task has been correctly and "
        f"completely executed according to the provided screenshots.\n\n"
        f"Use the following format for your response:\n"
        f"Reason: <Brief description of why you believe the task was successful "
        f"or failed, starting with \"I believe this task is successful/failed\">\n"
        f"Result: <1 OR 0>"
    )


def gpt4o_evaluate(target_dir: str, task_description: str, max_retry: int = 3) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return {"result": -1, "reason": "OPENAI_API_KEY not set", "error": True}

    images = generate_image_list(target_dir)
    if not images:
        return {"result": -1, "reason": "no screenshots found", "error": True}

    content = [{"type": "text", "text": user_prompt(task_description)}]
    content.extend(images)

    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": sys_prompt()},
            {"role": "user", "content": content},
        ],
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    for attempt in range(max_retry):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            rj = resp.json()
            if "error" in rj:
                print(f"  GPT-4o API error: {rj['error'].get('message', rj['error'])}")
                if attempt < max_retry - 1:
                    import time
                    time.sleep(5)
                    continue
                return {"result": -1, "reason": str(rj["error"]), "error": True}

            text = rj["choices"][0]["message"]["content"]
            tokens = rj["usage"]["total_tokens"]

            pattern = r"Reason[:：]\s*(.*?)\s*Result[:：]\s*(\d)"
            match = re.search(pattern, text, re.DOTALL)
            if not match:
                pattern = r"(?:Reason[:：])?\s*(.*?)\s*Result[:：]\s*(\d)"
                match = re.search(pattern, text, re.DOTALL)

            if match:
                result = int(match.group(2).strip())
                reason = match.group(1).strip().replace("\n", " ")
            else:
                result = 0
                reason = f"unparseable response: {text[:200]}"

            return {
                "result": result,
                "reason": reason,
                "tokens": tokens,
                "raw_response": text,
            }
        except Exception as e:
            print(f"  attempt {attempt+1}/{max_retry} failed: {e}")
            if attempt < max_retry - 1:
                import time
                time.sleep(5)

    return {"result": -1, "reason": "all retries failed", "error": True}


def save_evaluation(csv_path: str, task_id: str, agent_name: str,
                    result: int, detail: dict,
                    reasoning_mode: str, action_mode: str) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8")
    df = df.set_index("task_identifier", drop=False).fillna("")
    prefix = f"{agent_name}_" if agent_name else ""
    eval_col = f"{reasoning_mode}_{action_mode}_{prefix}evaluation"
    detail_col = f"{reasoning_mode}_{action_mode}_{prefix}details"

    if eval_col not in df.columns:
        df[eval_col] = "N"
    if detail_col not in df.columns:
        df[detail_col] = "{}"

    if result == 1:
        df.at[task_id, eval_col] = "S"
    elif result == 0:
        df.at[task_id, eval_col] = "F"
    else:
        df.at[task_id, eval_col] = "E"

    df.at[task_id, detail_col] = json.dumps(detail, ensure_ascii=False)
    df.to_csv(csv_path, index=False, encoding="utf-8")


def evaluate_task(result_dir: str, task_id: str, agent_name: str,
                  reasoning_mode: str, action_mode: str) -> int:
    csv_path = os.path.join(result_dir, "results.csv")
    dataset = get_dataset(csv_path)

    target_dir = os.path.join(result_dir, task_id, agent_name)
    if not os.path.isdir(target_dir):
        target_dir = os.path.join(result_dir, task_id)

    screenshots = get_screenshot_file_names(target_dir)
    if not screenshots:
        print(f"  {task_id}: no screenshots")
        save_evaluation(csv_path, task_id, agent_name, -1, {}, reasoning_mode, action_mode)
        return -1

    task_description = dataset.loc[task_id, "task_description"]
    print(f"  {task_id}: {len(screenshots)} screenshots, evaluating with GPT-4o...")
    print(f"    task: {task_description[:100]}")

    eval_result = gpt4o_evaluate(target_dir, task_description)
    result = eval_result.get("result", -1)

    detail = {
        "fine_detect": int(result == 1),
        "fine_detect_reason": eval_result.get("reason", ""),
        "gpt_token_taken": eval_result.get("tokens", 0),
        "total_num": len(screenshots),
    }

    label = "SUCCESS" if result == 1 else "FAIL" if result == 0 else "ERROR"
    print(f"    -> {label}: {eval_result.get('reason', '')[:120]}")

    save_evaluation(csv_path, task_id, agent_name, result, detail, reasoning_mode, action_mode)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="Task IDs to evaluate. If omitted, evaluates all tasks with screenshots.")
    parser.add_argument("--agent", default="TAGNavAgent")
    parser.add_argument("--reasoning-mode", default="direct")
    parser.add_argument("--action-mode", default="no_action")
    args = parser.parse_args()

    result_dir = args.result_dir
    csv_path = os.path.join(result_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"ERROR: results.csv not found in {result_dir}")
        sys.exit(1)

    if args.task_ids:
        task_ids = args.task_ids
    else:
        task_ids = [
            d for d in os.listdir(result_dir)
            if os.path.isdir(os.path.join(result_dir, d))
            and d != "session_manifest.json"
            and get_screenshot_file_names(
                os.path.join(result_dir, d, args.agent)
                if os.path.isdir(os.path.join(result_dir, d, args.agent))
                else os.path.join(result_dir, d)
            )
        ]

    print(f"Evaluating {len(task_ids)} tasks in {result_dir}")
    results = {}
    for tid in sorted(task_ids):
        r = evaluate_task(result_dir, tid, args.agent, args.reasoning_mode, args.action_mode)
        results[tid] = r

    print(f"\n{'='*50}")
    print(f"Results: {sum(1 for v in results.values() if v == 1)}/{len(results)} success")
    for tid, r in sorted(results.items()):
        label = "S" if r == 1 else "F" if r == 0 else "E"
        print(f"  {tid}: {label}")


if __name__ == "__main__":
    main()
