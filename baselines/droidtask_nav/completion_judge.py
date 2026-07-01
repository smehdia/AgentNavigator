#!/usr/bin/env python3
"""Completion-first GPT judge for DroidTask-Nav benchmark runs.

Unlike the strict screenshot-pair judge, this evaluator treats the navigation
instruction as the source of truth. The converted GT screenshot is included as
context only, because some DroidTask-Nav v2 conversions point at an adjacent or
pre-action frame rather than the intended navigation target.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_ROOT = Path(os.environ.get(
    "DROIDTASK_NAV_TASKS_ROOT", str(REPO_ROOT / "DroidTask_nav_v2_DATA")))
FALLBACK_TASKS_ROOT = REPO_ROOT / "DroidTask_nav_reviewed_DATA"
VALID_VERDICTS = {"match", "partial", "mismatch"}
ISSUE_CATEGORIES = {"none", "environment", "target_or_judge", "navigation", "unclear"}

JUDGE_PROMPT = """\
You are judging a navigation-only mobile-app benchmark result.

Task: {prompt}
App: {app}
Target description: {target_description}
Target waypoint memory:
{target_memory}

Image 1 is the AGENT's final screen.
Image 2 is the converted DroidTask-Nav GT target screen. Image 2 is advisory
only: it may be stale, adjacent, or wrong for this navigation-only conversion.

Judge whether Image 1 satisfies the Task. Prefer the written task and target
description over Image 2 if they conflict.

Use these verdicts:
- match: the agent reached the requested page, tab, control, dialog, or screen
  where the user can perform the requested navigation goal.
- partial: the agent is in the right app and broad area, but one navigation
  step short, on a sibling page, wrong tab, wrong specific control, or ambiguous.
- mismatch: the agent is clearly on the wrong page/app, stuck, at home/launcher,
  or failed to reach the requested destination.

Also assign an issue_category:
- none: use only when verdict is match.
- environment: the final screen shows a missing default-app setup, missing
  seeded app/data, permissions/default-handler blocker, empty seeded corpus, or
  other state that prevents a fair navigation attempt.
- target_or_judge: Image 1 appears to satisfy the written task, but Image 2 or
  the converted GT target appears to represent a different/adjacent task state.
- navigation: the agent made a route/control/termination mistake.
- unclear: evidence is insufficient.

Return ONLY a JSON object:
{{
  "verdict": "match|partial|mismatch",
  "score": 1-5,
  "issue_category": "none|environment|target_or_judge|navigation|unclear",
  "gt_conflict": true|false,
  "justification": "<brief explanation>"
}}
"""


def resolve_tasks_root(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_TASKS_ROOT.is_dir():
        return DEFAULT_TASKS_ROOT
    return FALLBACK_TASKS_ROOT


def encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def resolve_image_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_file():
        return path
    parts = path.parts
    for marker in ("DroidTask_GT", "DroidTask"):
        if marker in parts:
            idx = parts.index(marker)
            tail = Path(*parts[idx + 1 :]) if marker == "DroidTask_GT" else Path(*parts[idx:])
            candidates = [
                REPO_ROOT / marker / tail if marker == "DroidTask_GT" else REPO_ROOT / tail,
                Path(os.environ.get("DROIDTASK_ROOT", str(REPO_ROOT / "DroidTask_GT" / "DroidTask"))) / tail.relative_to("DroidTask")
                if marker == "DroidTask" and tail.parts and tail.parts[0] == "DroidTask"
                else Path(os.environ.get("DROIDTASK_ROOT", str(REPO_ROOT / "DroidTask_GT" / "DroidTask"))) / tail,
            ]
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    return path


def parse_json(text: str) -> dict[str, Any]:
    if "```" in text:
        for part in text.split("```"):
            s = part.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("{"):
                text = s
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            return json.loads(text[start:end])
        raise


def parse_result(text: str) -> dict[str, Any]:
    try:
        data = parse_json(text)
    except Exception:
        return {
            "score": "",
            "verdict": "error",
            "issue_category": "unclear",
            "gt_conflict": "",
            "justification": f"parse-fail: {text[:200]}",
        }

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VALID_VERDICTS:
        verdict = "error"
    try:
        score: int | str = int(data.get("score"))
        score = max(1, min(5, score))
    except Exception:
        score = ""
    issue_category = str(data.get("issue_category", "")).strip().lower()
    if issue_category not in ISSUE_CATEGORIES:
        issue_category = "unclear"
    if verdict == "match" and issue_category != "target_or_judge":
        issue_category = "none"
    return {
        "score": score,
        "verdict": verdict,
        "issue_category": issue_category,
        "gt_conflict": bool(data.get("gt_conflict", False)),
        "justification": str(data.get("justification", "")),
    }


def find_source_app_dir(tasks_root: Path, slug: str) -> Path | None:
    direct = tasks_root / slug
    if direct.is_dir():
        return direct
    matches = sorted(
        p for p in tasks_root.iterdir()
        if p.is_dir() and p.name.startswith(f"droidtask_{slug}")
    )
    return matches[0] if len(matches) == 1 else None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def load_meta(tasks_root: Path, app_slug: str, task_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    app_dir = find_source_app_dir(tasks_root, app_slug)
    if app_dir is None:
        return None, None
    meta_path = app_dir / task_id / "meta.json"
    return meta_path, load_json(meta_path)


def prompt_from(run_summary: dict[str, Any] | None, meta: dict[str, Any] | None) -> str:
    if run_summary:
        prompt = run_summary.get("task_prompt") or run_summary.get("prompt")
        if prompt:
            return str(prompt)
    if meta:
        prompt = meta.get("prompt")
        if prompt:
            return str(prompt)
        prompts = meta.get("prompts")
        if prompts:
            return str(prompts[0])
    return ""


def target_description(meta: dict[str, Any] | None) -> str:
    if not meta:
        return ""
    nav = meta.get("droidtask_nav") or {}
    return str(
        meta.get("gt_nav_target_screen_description")
        or nav.get("target_screen_description")
        or nav.get("target_state_hint")
        or ""
    )


def base_row(app_slug: str, run_dir: Path, tasks_root: Path) -> dict[str, Any]:
    task_id = run_dir.name
    meta_path, meta = load_meta(tasks_root, app_slug, task_id)
    run_summary = load_json(run_dir / "summary.json")
    target_info = load_json(run_dir / "target_info.json") or {}
    gt_path = resolve_image_path(Path(str(meta.get("gt_nav_screenshot")))) if meta and meta.get("gt_nav_screenshot") else None
    memory = str(target_info.get("open_loop_memory") or "")
    if not memory:
        waypoints = target_info.get("target_waypoints") or []
        hints = target_info.get("target_hints") or []
        memory = "\n".join([*(str(w) for w in waypoints), *(str(h) for h in hints)])
    return {
        "app": app_slug,
        "task": task_id,
        "prompt": prompt_from(run_summary, meta),
        "target_description": target_description(meta),
        "target_memory": memory,
        "finished_reason": "" if not run_summary else str(run_summary.get("finished_reason", "")),
        "agent_png": str(run_dir / "agent_screenshot.png"),
        "gt_png": "" if gt_path is None else str(gt_path),
        "meta_path": "" if meta_path is None else str(meta_path),
        "score": "",
        "verdict": "error",
        "issue_category": "unclear",
        "gt_conflict": "",
        "justification": "",
    }


def judge_one(client, model: str, app_slug: str, run_dir: Path, tasks_root: Path) -> dict[str, Any]:
    row = base_row(app_slug, run_dir, tasks_root)
    agent_png = Path(row["agent_png"])
    gt_png = Path(row["gt_png"]) if row["gt_png"] else None
    if not agent_png.is_file():
        row["justification"] = "agent_screenshot.png missing"
        return row
    if gt_png is None or not gt_png.is_file():
        row["justification"] = "gt_nav_screenshot missing"
        return row

    prompt = JUDGE_PROMPT.format(
        app=row["app"],
        prompt=row["prompt"] or "(no task description)",
        target_description=row["target_description"] or "(not available)",
        target_memory=row["target_memory"] or "(not available)",
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_b64(agent_png)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_b64(gt_png)}"}},
    ]
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=240,
                temperature=0,
            )
            row.update(parse_result((response.choices[0].message.content or "").strip()))
            return row
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if "insufficient_quota" in msg_lower:
                row["justification"] = "judge-error: insufficient_quota"
                return row
            if "rate" in msg_lower or "429" in msg:
                time.sleep(2**attempt + 1)
                continue
            row["justification"] = f"judge-error: {exc}"
            return row
    row["justification"] = "judge-error: max retries"
    return row


def apps_from_root(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def write_outputs(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "app", "task", "score", "verdict", "issue_category", "gt_conflict",
        "justification", "finished_reason", "prompt", "target_description",
        "agent_png", "gt_png", "meta_path",
    ]
    rows = sorted(rows, key=lambda r: (r["app"], r["task"]))
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    by_app: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_app.setdefault(row["app"], []).append(row)

    lines = [
        "# DroidTask-Nav Completion Judge",
        "",
        "GPT-4o completion-first judge. `finished_action` is reported only as metadata and is not counted as a score.",
        "",
        "| App | Judged | Match | Partial | Mismatch | Env issue | GT/judge issue | Error | Match rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_valid = 0
    all_match = 0
    issue_totals = {category: 0 for category in ISSUE_CATEGORIES}
    for app in sorted(by_app):
        app_rows = by_app[app]
        valid = [r for r in app_rows if r["verdict"] in VALID_VERDICTS]
        matches = sum(1 for r in valid if r["verdict"] == "match")
        partial = sum(1 for r in valid if r["verdict"] == "partial")
        mismatch = sum(1 for r in valid if r["verdict"] == "mismatch")
        env = sum(1 for r in valid if r["issue_category"] == "environment")
        gt_issue = sum(1 for r in valid if r["issue_category"] == "target_or_judge" or r["gt_conflict"] is True)
        errors = sum(1 for r in app_rows if r["verdict"] == "error")
        all_valid += len(valid)
        all_match += matches
        for row in valid:
            issue_totals[row["issue_category"]] = issue_totals.get(row["issue_category"], 0) + 1
        rate = matches / len(valid) * 100 if valid else 0.0
        lines.append(
            f"| {app} | {len(valid)} | {matches} | {partial} | {mismatch} | "
            f"{env} | {gt_issue} | {errors} | {rate:.1f}% |"
        )
    total_rate = all_match / all_valid * 100 if all_valid else 0.0
    lines.extend([
        "",
        f"Aggregate completion match: {all_match}/{all_valid} ({total_rate:.1f}%).",
        "",
        "| Issue category | Count |",
        "| --- | ---: |",
    ])
    for category in ("environment", "target_or_judge", "navigation", "unclear", "none"):
        lines.append(f"| {category} | {issue_totals.get(category, 0)} |")
    out_csv.with_suffix(".summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Run root containing <app>/<task>/summary.json outputs.")
    parser.add_argument("--tasks-root", type=Path, default=None, help="DroidTask-Nav tree containing meta.json.")
    parser.add_argument("--apps", nargs="*", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="CSV path. Defaults to <root>/droidtask_nav_completion_judge.csv")
    parser.add_argument("--check-only", action="store_true", help="Check file presence without calling the judge model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    tasks_root = resolve_tasks_root(args.tasks_root).resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root not found: {root}")
    if not tasks_root.is_dir():
        raise SystemExit(f"Tasks root not found: {tasks_root}")

    rows: list[dict[str, Any]] = []
    apps = apps_from_root(root, args.apps)

    if args.check_only:
        for app in apps:
            app_dir = root / app
            task_dirs = sorted(d for d in app_dir.iterdir() if d.is_dir() and (d / "summary.json").is_file())
            for task_dir in task_dirs:
                row = base_row(app, task_dir, tasks_root)
                missing = []
                if not Path(row["agent_png"]).is_file():
                    missing.append("agent_screenshot.png")
                if not row["gt_png"] or not Path(row["gt_png"]).is_file():
                    missing.append("gt_nav_screenshot")
                row["verdict"] = "check_ok" if not missing else "error"
                row["justification"] = "" if not missing else "missing " + ", ".join(missing)
                rows.append(row)
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is required. Use --check-only for file checks.")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        for app in apps:
            app_dir = root / app
            if not app_dir.is_dir():
                print(f"SKIP {app}: no run directory")
                continue
            task_dirs = sorted(d for d in app_dir.iterdir() if d.is_dir() and (d / "summary.json").is_file())
            print(f"\n{app}: judging {len(task_dirs)} tasks")
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [
                    pool.submit(judge_one, client, args.judge_model, app, task_dir, tasks_root)
                    for task_dir in task_dirs
                ]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    print(
                        f"  {row['task']}: {row['verdict']} score={row['score']} "
                        f"{row['issue_category']} {row['justification'][:70]}"
                    )

    out_csv = Path(args.out).resolve() if args.out else root / "droidtask_nav_completion_judge.csv"
    write_outputs(rows, out_csv)
    print(f"\nCSV: {out_csv}")
    print(f"Summary: {out_csv.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
