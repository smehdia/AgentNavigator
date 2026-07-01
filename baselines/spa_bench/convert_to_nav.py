#!/usr/bin/env python3
"""Convert SPA-Bench task-completion goals into navigation-only goals.

Adapts the navigation-conversion policy for SPA-Bench tasks, which lack
ground-truth trajectories. Uses Qwen-VL-Max (text-only) or GPT-4o for the
conversion.

Required environment:
  DASHSCOPE_API_KEY   (when --backend dashscope)
  OPENAI_API_KEY      (when --backend openai)
  SPA_BENCH_ROOT      (optional; path to the upstream SPA-Bench checkout for the
                       default dataset path. Falls back to baselines/external/SPA-Bench.)

Example:
  python baselines/spa_bench/convert_to_nav.py \
      --apps clock calendar chrome airbnb amazon google_maps youtube \
      --out-dir spa_bench_subsets/nav_converted \
      --backend dashscope
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# baselines/spa_bench/<this file> -> repo root is parents[2]; baselines/ is parents[1].
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = Path(__file__).resolve().parents[1]


def spa_bench_root() -> Path:
    env = os.environ.get("SPA_BENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return BASELINES_DIR / "external" / "SPA-Bench"


SPA_NAV_PROMPT = """You are converting mobile app task-completion goals into navigation-only benchmark goals.

## Original Task
App: {app_label}
Goal: "{goal}"

## Conversion Policy

Convert the original task into the FIRST STABLE ACTIONABLE NAVIGATION DESTINATION needed to begin completing the task.
Do NOT simply prepend "Navigate to..." to the full original task.

The navigation goal must describe where the user should arrive, not the final side effect.
Use concise wording like:
- "Navigate to the alarm creation screen in Clock."
- "Navigate to the search results page for 'sunglasses' on Amazon."
- "Navigate to the event creation screen in Calendar."
- "Navigate to the Chrome bookmarks page."

## Key Rules

1. First objective only:
   If the original task chains multiple objectives, convert only the first actionable objective.
   Example: "Set an alarm for 10am on weekdays. Disable vibration for this alarm. Set another alarm for 11am on weekends."
   becomes "Navigate to the alarm creation screen in Clock."
   Ignore the later objectives.

2. Stop before task completion:
   Stop at the stable page, dialog, picker, or control where the first action can be performed.
   Do not include the final click/toggle/save.

3. Search tasks:
   For tasks that start with searching, the navigation target is the search results page.
   Example: "Get the search results for 'sunglasses'" -> "Navigate to the search results page for 'sunglasses' on Amazon."
   The search query IS part of the navigation (you need to type it to reach the results).

4. Stable app content can be a valid target:
   Built-in settings, pickers, dialogs, and fixed app pages are stable navigation targets.

5. Make the navigation goal specific:
   Preserve the stable app intent (the entity being searched, the setting being changed).
   Remove downstream completion wording ("save it", "add to cart", "subscribe").

## Output JSON
Respond with ONLY a JSON object:
{{
  "navigation_goal": "<concise navigation-only goal>",
  "primary_objective": "<the first actionable objective from the original task>",
  "target_screen_description": "<brief description of the target screen>",
  "conversion_type": "<one of: search_results, input_entry, stable_picker_or_dialog, stable_settings_or_control, stable_content_list, already_navigation>",
  "confidence": <0.0-1.0>,
  "needs_manual_review": <true or false>,
  "review_reason": "<brief reason, empty string if no review needed>"
}}"""


def convert_with_dashscope(app_label: str, goal: str) -> dict:
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    import dashscope
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is not set (see baselines/.env.example).")
    dashscope.api_key = api_key

    prompt = SPA_NAV_PROMPT.format(app_label=app_label, goal=goal)
    resp = dashscope.Generation.call(
        model="qwen-max-latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        result_format="message",
    )

    text = resp.output.choices[0].message.content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def convert_with_openai(app_label: str, goal: str) -> dict:
    import requests
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set (see baselines/.env.example).")
    prompt = SPA_NAV_PROMPT.format(app_label=app_label, goal=goal)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    rj = resp.json()
    text = rj["choices"][0]["message"]["content"].strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def convert_task(app_label: str, goal: str, backend: str, max_retry: int = 3) -> dict:
    for attempt in range(max_retry):
        try:
            if backend == "dashscope":
                return convert_with_dashscope(app_label, goal)
            else:
                return convert_with_openai(app_label, goal)
        except SystemExit:
            raise
        except Exception as e:
            print(f"    retry {attempt+1}/{max_retry}: {e}")
            time.sleep(3)
    return {"navigation_goal": f"Navigate to {app_label}.", "error": "all retries failed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", nargs="*",
                        default=["airbnb", "amazon", "calendar", "chrome",
                                 "clock", "google_maps", "youtube"])
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "spa_bench_subsets" / "nav_converted")
    parser.add_argument("--dataset", type=Path,
                        default=spa_bench_root() / "data" / "single-app-ENG.csv")
    parser.add_argument("--backend", choices=["dashscope", "openai"], default="dashscope")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset, encoding="utf-8")
    subset = df[df["task_app"].isin(args.apps)]
    print(f"Converting {len(subset)} tasks across {len(args.apps)} apps")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for _, row in subset.iterrows():
        tid = row["task_identifier"]
        app = row["task_app"]
        goal = row["task_description"]
        app_label = app.replace("_", " ").title()

        print(f"  {tid}: {goal[:70]}...")
        conversion = convert_task(app_label, goal, args.backend)
        nav_goal = conversion.get("navigation_goal", "")
        print(f"    -> {nav_goal}")

        result = {
            "task_identifier": tid,
            "task_app": app,
            "task_difficulty": int(row["task_difficulty"]),
            "golden_steps": int(row["golden_steps"]),
            "original_goal": goal,
            "navigation_goal": nav_goal,
            "adb_app": row["adb_app"],
            "adb_home_page": row["adb_home_page"],
            "conversion": conversion,
        }
        results.append(result)

        task_dir = args.out_dir / app / tid
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "prompts.json").write_text(
            json.dumps({"prompts": [nav_goal]}, indent=2) + "\n", encoding="utf-8")
        (task_dir / "meta.json").write_text(
            json.dumps({
                "schema": "agentnavigator_spa_nav_task_v1",
                "source": "SPA-Bench",
                "spa_task_identifier": tid,
                "spa_task_app": app,
                "spa_task_difficulty": str(row["task_difficulty"]),
                "golden_steps": int(row["golden_steps"]),
                "original_task_description": goal,
                "navigation_goal": nav_goal,
                "adb_app": row["adb_app"],
                "adb_home_page": row["adb_home_page"],
                "conversion_type": conversion.get("conversion_type", ""),
                "conversion_confidence": conversion.get("confidence", 0),
                "needs_manual_review": conversion.get("needs_manual_review", False),
            }, indent=2) + "\n", encoding="utf-8")

        time.sleep(0.5)

    manifest_path = args.out_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema": "agentnavigator_spa_nav_conversion_v1",
        "backend": args.backend,
        "apps": args.apps,
        "task_count": len(results),
        "tasks": results,
    }, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {len(results)} converted tasks to {args.out_dir}")
    print(f"Manifest: {manifest_path}")

    review_count = sum(1 for r in results if r["conversion"].get("needs_manual_review"))
    if review_count:
        print(f"\n{review_count} tasks flagged for manual review")


if __name__ == "__main__":
    main()
