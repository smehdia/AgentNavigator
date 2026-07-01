#!/usr/bin/env python3
"""DashScope/Qwen strict vision judge for PG-Agent benchmark outputs.

This mirrors ``gui_explorer_strict_judge.py`` but uses DashScope directly so it
can produce a fallback analysis artifact when OpenAI judging is unavailable.
Do not use its CSV as the canonical GPT-4o strict judge artifact unless the
paper explicitly switches judge providers.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.gui_explorer_strict_judge import (
    APP_TASK_DIRS,
    JUDGE_PROMPT,
    apps_from_root,
    find_gt_dir,
    find_gt_png,
    parse_score,
    prompt_for_task,
)


def setup_dashscope() -> Any:
    for var in [
        "http_proxy",
        "https_proxy",
        "ftp_proxy",
        "socks_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "FTP_PROXY",
        "SOCKS_PROXY",
    ]:
        os.environ.pop(var, None)

    import yaml
    import dashscope

    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    if not os.environ.get("DASHSCOPE_API_KEY"):
        raise SystemExit("set DASHSCOPE_API_KEY in your environment (see baselines/.env.example)")
    return dashscope


def encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def judge_one(dashscope: Any, model: str, app: str, run_dir: Path) -> dict[str, Any]:
    task_id = run_dir.name
    agent_png = run_dir / "agent_screenshot.png"
    gt_dir = find_gt_dir(app, task_id)
    gt_png = find_gt_png(gt_dir) if gt_dir else None
    prompt = prompt_for_task(run_dir, gt_dir)
    base = {
        "app": app,
        "task": task_id,
        "prompt": prompt,
        "agent_png": str(agent_png),
        "gt_png": "" if gt_png is None else str(gt_png),
        "score": "",
        "verdict": "error",
        "justification": "",
    }
    if app not in APP_TASK_DIRS:
        base["justification"] = "unsupported app"
        return base
    if not agent_png.exists():
        base["justification"] = "agent_screenshot.png missing"
        return base
    if gt_png is None:
        base["justification"] = "ground-truth frame_*.png missing"
        return base

    messages = [
        {
            "role": "user",
            "content": [
                {"text": JUDGE_PROMPT.format(prompt=prompt or "(no task description)")},
                {"image": f"data:image/png;base64,{encode_b64(agent_png)}"},
                {"image": f"data:image/png;base64,{encode_b64(gt_png)}"},
            ],
        }
    ]
    for attempt in range(5):
        try:
            resp = dashscope.MultiModalConversation.call(
                model=model,
                messages=messages,
                temperature=0.0,
            )
            if getattr(resp, "status_code", None) and resp.status_code != 200:
                raise RuntimeError(f"{resp.status_code}: {getattr(resp, 'message', resp)}")
            out = resp.output.choices[0].message.content
            if isinstance(out, list):
                text = "\n".join(item.get("text", "") for item in out if isinstance(item, dict)).strip()
            else:
                text = str(out).strip()
            base.update(parse_score(text))
            return base
        except Exception as exc:
            msg = str(exc)
            if "rate" in msg.lower() or "429" in msg or "Throttling" in msg:
                time.sleep(2**attempt + 1)
                continue
            base["justification"] = f"judge-error: {exc}"
            return base
    base["justification"] = "judge-error: max retries"
    return base


def write_summary(rows: list[dict[str, Any]], out_csv: Path, model: str) -> None:
    by_app: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_app.setdefault(row["app"], []).append(row)

    lines = [
        "# PG-Agent Qwen Strict Judge",
        "",
        f"Judge model: `{model}`",
        "",
        "| App | Judged | Success | Partial | Mismatch | Error | Success rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    total_valid = 0
    total_success = 0
    for app in sorted(by_app):
        app_rows = by_app[app]
        valid = [r for r in app_rows if r["verdict"] in {"match", "partial", "mismatch"}]
        success = sum(r["verdict"] == "match" for r in valid)
        partial = sum(r["verdict"] == "partial" for r in valid)
        mismatch = sum(r["verdict"] == "mismatch" for r in valid)
        errors = sum(r["verdict"] == "error" for r in app_rows)
        total_valid += len(valid)
        total_success += success
        rate = success / len(valid) * 100 if valid else 0.0
        lines.append(f"| {app} | {len(valid)} | {success} | {partial} | {mismatch} | {errors} | {rate:.1f}% |")
    total_rate = total_success / total_valid * 100 if total_valid else 0.0
    lines.extend(["", f"Aggregate strict success: {total_success}/{total_valid} ({total_rate:.1f}%)."])
    out_csv.with_suffix(".summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--apps", nargs="*", default=None)
    parser.add_argument("--judge-model", default="qwen-vl-max-latest")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    dashscope = setup_dashscope()
    rows: list[dict[str, Any]] = []
    for app in apps_from_root(root, args.apps):
        app_dir = root / app
        if not app_dir.is_dir():
            print(f"SKIP {app}: no run directory")
            continue
        task_dirs = sorted(d for d in app_dir.iterdir() if d.is_dir() and (d / "summary.json").exists())
        print(f"\n{app}: judging {len(task_dirs)} tasks")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_one, dashscope, args.judge_model, app, d) for d in task_dirs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"  {row['task']}: {row['verdict']} score={row['score']} {row['justification'][:80]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["app", "task", "score", "verdict", "justification", "prompt", "agent_png", "gt_png"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["app"], r["task"])))
    write_summary(rows, out, args.judge_model)
    print(f"\nCSV: {out}")
    print(f"Summary: {out.with_suffix('.summary.md')}")


if __name__ == "__main__":
    main()
