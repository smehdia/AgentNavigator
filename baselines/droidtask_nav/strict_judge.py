#!/usr/bin/env python3
"""Strict vision judge for DroidTask-Nav benchmark runs.

The judge compares each task's final `agent_screenshot.png` against the
converted navigation target stored in `meta.json` as `gt_nav_screenshot`.
Scores 4-5 count as strict success, 3 as partial/failure, and 1-2 as failure.
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
FALLBACK_TASKS_ROOT = REPO_ROOT / "DroidTask_nav_collector_DATA"

JUDGE_PROMPT = """\
You are evaluating whether a mobile-app navigation agent reached the intended
target page or target control for a navigation-only task.

Task: {prompt}

Image 1 is the AGENT's final screen.
Image 2 is the GROUND-TRUTH DroidTask-Nav target screen.

Score from 1 to 5:
5 = exact target page/control reached.
4 = correct functional page/section reached; only minor visual, scroll, modal, or dynamic-content differences.
3 = partially correct: same broad area/app section, but wrong specific leaf, tab, field, or subpage.
2 = wrong section of the same app, or agent is clearly not at the requested destination.
1 = wrong app, launcher/home, stuck, or unrelated screen.

Be strict for score 4-5. A sibling settings page, nearby list item, or same broad
menu but wrong specific target is score 3, not success.

Respond with ONLY a JSON object:
{{"score": 1-5, "justification": "<brief explanation>"}}\
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


def parse_score(text: str) -> dict[str, Any]:
    if "```" in text:
        for part in text.split("```"):
            s = part.strip()
            if s.startswith("json"):
                s = s[4:].strip()
            if s.startswith("{"):
                text = s
                break
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if 0 <= start < end:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                return {"score": "", "verdict": "error", "justification": f"parse-fail: {text[:200]}"}
        else:
            return {"score": "", "verdict": "error", "justification": f"parse-fail: {text[:200]}"}
    try:
        score = int(data.get("score"))
    except Exception:
        return {"score": "", "verdict": "error", "justification": str(data.get("justification", ""))}
    score = max(1, min(5, score))
    verdict = "match" if score >= 4 else "partial" if score == 3 else "mismatch"
    return {"score": score, "verdict": verdict, "justification": str(data.get("justification", ""))}


def find_source_app_dir(tasks_root: Path, slug: str) -> Path | None:
    direct = tasks_root / slug
    if direct.is_dir():
        return direct
    matches = sorted(
        p for p in tasks_root.iterdir()
        if p.is_dir() and p.name.startswith(f"droidtask_{slug}")
    )
    return matches[0] if len(matches) == 1 else None


def load_meta(tasks_root: Path, app_slug: str, task_id: str) -> tuple[Path | None, dict[str, Any] | None]:
    app_dir = find_source_app_dir(tasks_root, app_slug)
    if app_dir is None:
        return None, None
    meta_path = app_dir / task_id / "meta.json"
    if not meta_path.is_file():
        return meta_path, None
    return meta_path, json.loads(meta_path.read_text())


def prompt_from(run_dir: Path, meta: dict[str, Any] | None) -> str:
    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            data = json.loads(summary.read_text())
            prompt = data.get("task_prompt") or data.get("prompt")
            if prompt:
                return str(prompt)
        except Exception:
            pass
    if meta:
        prompt = meta.get("prompt")
        if prompt:
            return str(prompt)
        prompts = meta.get("prompts")
        if prompts:
            return str(prompts[0])
    return ""


def base_row(app_slug: str, run_dir: Path, tasks_root: Path) -> dict[str, Any]:
    task_id = run_dir.name
    meta_path, meta = load_meta(tasks_root, app_slug, task_id)
    gt_path = resolve_image_path(Path(str(meta.get("gt_nav_screenshot")))) if meta and meta.get("gt_nav_screenshot") else None
    return {
        "app": app_slug,
        "task": task_id,
        "prompt": prompt_from(run_dir, meta),
        "agent_png": str(run_dir / "agent_screenshot.png"),
        "gt_png": "" if gt_path is None else str(gt_path),
        "meta_path": "" if meta_path is None else str(meta_path),
        "score": "",
        "verdict": "error",
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

    content = [
        {"type": "text", "text": JUDGE_PROMPT.format(prompt=row["prompt"] or "(no task description)")},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_b64(agent_png)}"}},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_b64(gt_png)}"}},
    ]
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=200,
                temperature=0,
            )
            row.update(parse_score((response.choices[0].message.content or "").strip()))
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
        "app", "task", "score", "verdict", "justification", "prompt",
        "agent_png", "gt_png", "meta_path",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["app"], r["task"])))

    by_app: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_app.setdefault(row["app"], []).append(row)

    lines = [
        "# DroidTask-Nav Strict Judge",
        "",
        "| App | Judged | Success | Partial | Error | Success rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    all_valid = 0
    all_success = 0
    for app in sorted(by_app):
        app_rows = by_app[app]
        valid = [r for r in app_rows if r["verdict"] in {"match", "partial", "mismatch"}]
        success = sum(1 for r in valid if r["verdict"] == "match")
        partial = sum(1 for r in valid if r["verdict"] == "partial")
        errors = sum(1 for r in app_rows if r["verdict"] == "error")
        all_valid += len(valid)
        all_success += success
        rate = success / len(valid) * 100 if valid else 0.0
        lines.append(f"| {app} | {len(valid)} | {success} | {partial} | {errors} | {rate:.1f}% |")
    total_rate = all_success / all_valid * 100 if all_valid else 0.0
    lines.extend(["", f"Aggregate strict success: {all_success}/{all_valid} ({total_rate:.1f}%)."])
    out_csv.with_suffix(".summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True,
                        help="Run root containing <app>/<task>/summary.json task outputs.")
    parser.add_argument("--tasks-root", type=Path, default=None,
                        help="DroidTask-Nav tree containing meta.json with gt_nav_screenshot.")
    parser.add_argument("--apps", nargs="*", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="CSV path. Defaults to <root>/droidtask_nav_strict_judge.csv")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check run/meta/GT file presence; do not call the judge model.")
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
            raise SystemExit("OPENAI_API_KEY is required to run strict judging. Use --check-only for file checks.")
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
                    print(f"  {row['task']}: {row['verdict']} score={row['score']} {row['justification'][:80]}")

    out_csv = Path(args.out).resolve() if args.out else root / "droidtask_nav_strict_judge.csv"
    write_outputs(rows, out_csv)
    print(f"\nCSV: {out_csv}")
    print(f"Summary: {out_csv.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
