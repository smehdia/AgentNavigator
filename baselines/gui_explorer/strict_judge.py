#!/usr/bin/env python3
"""Strict GPT judge for GUI-Explorer navigation outputs.

Canonical scoring policy:
  - score 4 or 5 => strict success
  - score 3 => partial, counted as failure for strict success
  - score 1 or 2 => failure

This script judges final agent screenshots against the ground-truth final frame
from the navigation task data. It is designed for GUI-Explorer result roots, but
only assumes each task directory has `summary.json` and `agent_screenshot.png`.

Required environment:
  OPENAI_API_KEY     Used for GPT judging.
  TAGNAV_GT_ROOT     One or more roots (os.pathsep-separated) holding per-app
                     ground-truth task folders with frame_*.png. Falls back to
                     TAGNAV_TASKS_ROOT if unset. May also be passed via --gt-root.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_env_file(path: Path | None) -> None:
    """Minimal .env loader (KEY=VALUE lines). Does not overwrite existing vars."""
    candidates = [path] if path else []
    # default candidate: baselines/.env (two levels up from this file)
    candidates.append(Path(__file__).resolve().parents[1] / ".env")
    for candidate in candidates:
        if candidate and candidate.is_file():
            for line in candidate.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


APP_TASK_DIRS: dict[str, str] = {
    "airbnb": "airbnb_26.06",
    "amazon": "amazon_32.3.0.100",
    "calendar": "calendar_2026.05.0-864040481-release",
    "chrome": "chrome_144.0.7559.132",
    "clock": "clock_8.5",
    "ebay": "ebay_6.242.0.2",
    "google_maps": "google_maps_26.06.01.863982022",
    "instagram": "instagram_416.0.0.47.66",
    "linkedin": "linkedin_4.1.1168",
    "settings": "settings_16",
    "tiktok": "tiktok_43.8.3",
    "youtube": "youtube_21.05.264",
}


def gt_bases(cli_roots: list[str] | None) -> list[Path]:
    """Ground-truth base dirs: --gt-root, else $TAGNAV_GT_ROOT, else $TAGNAV_TASKS_ROOT."""
    if cli_roots:
        return [Path(p) for p in cli_roots]
    raw = os.environ.get("TAGNAV_GT_ROOT") or os.environ.get("TAGNAV_TASKS_ROOT") or ""
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


JUDGE_PROMPT = """\
You are evaluating whether a mobile-app navigation agent reached the intended
target page or target control for a user task.

Task: {prompt}

Image 1 is the AGENT's final screen.
Image 2 is the GROUND-TRUTH target screen.

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


def encode_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


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
    if score >= 4:
        verdict = "match"
    elif score == 3:
        verdict = "partial"
    else:
        verdict = "mismatch"
    return {"score": score, "verdict": verdict, "justification": str(data.get("justification", ""))}


def find_gt_dir(app: str, task_id: str, bases: list[Path]) -> Path | None:
    app_dir = APP_TASK_DIRS[app]
    for base in bases:
        candidate = base / app_dir / task_id
        if candidate.is_dir():
            return candidate
    return None


def find_gt_png(gt_dir: Path) -> Path | None:
    pngs = sorted(gt_dir.glob("frame_*.png"))
    return pngs[-1] if pngs else None


def prompt_for_task(run_dir: Path, gt_dir: Path | None) -> str:
    summary = run_dir / "summary.json"
    if summary.exists():
        try:
            data = json.loads(summary.read_text())
            prompt = data.get("task_prompt") or data.get("prompt")
            if prompt:
                return str(prompt)
        except Exception:
            pass
    if gt_dir:
        prompts = gt_dir / "prompts.json"
        if prompts.exists():
            try:
                data = json.loads(prompts.read_text())
                values = data.get("prompts") or [data.get("prompt", "")]
                return str(values[0] if values else "")
            except Exception:
                pass
    return ""


def judge_one(client, model: str, app: str, run_dir: Path, bases: list[Path]) -> dict[str, Any]:
    task_id = run_dir.name
    agent_png = run_dir / "agent_screenshot.png"
    gt_dir = find_gt_dir(app, task_id, bases)
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
    if not agent_png.exists():
        base["justification"] = "agent_screenshot.png missing"
        return base
    if gt_png is None:
        base["justification"] = "ground-truth frame_*.png missing"
        return base

    content = [
        {"type": "text", "text": JUDGE_PROMPT.format(prompt=prompt or "(no task description)")},
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
            parsed = parse_score((response.choices[0].message.content or "").strip())
            base.update(parsed)
            return base
        except Exception as exc:
            msg = str(exc)
            msg_lower = msg.lower()
            if "insufficient_quota" in msg_lower:
                base["justification"] = "judge-error: insufficient_quota"
                return base
            if "rate" in msg_lower or "429" in msg:
                time.sleep(2**attempt + 1)
                continue
            base["justification"] = f"judge-error: {exc}"
            return base
    base["justification"] = "judge-error: max retries"
    return base


def apps_from_root(root: Path, requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return sorted(app for app in APP_TASK_DIRS if (root / app).is_dir())


def is_valid_judge_row(row: dict[str, Any]) -> bool:
    try:
        score = int(float(row.get("score") or ""))
    except (TypeError, ValueError):
        return False
    return row.get("verdict") in {"match", "partial", "mismatch"} and 1 <= score <= 5


def load_existing_valid_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            app = str(row.get("app") or "")
            task = str(row.get("task") or "")
            if app and task and is_valid_judge_row(row):
                rows[(app, task)] = row
    return rows


def is_insufficient_quota_row(row: dict[str, Any]) -> bool:
    return "insufficient_quota" in str(row.get("justification") or "").lower()


def write_metadata(
    *,
    out_csv: Path,
    root: Path,
    apps: list[str],
    model: str,
    rows: list[dict[str, Any]],
    reused_rows: int,
    stopped_on_insufficient_quota: bool,
) -> Path:
    valid = [r for r in rows if r.get("verdict") in {"match", "partial", "mismatch"}]
    errors = [r for r in rows if r.get("verdict") == "error"]
    metadata = {
        "schema": "navigation_strict_judge_metadata_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "OpenAI",
        "judge_model": model,
        "root": str(root),
        "apps": apps,
        "csv": str(out_csv),
        "rows": len(rows),
        "valid_rows": len(valid),
        "error_rows": len(errors),
        "reused_rows": reused_rows,
        "stopped_on_insufficient_quota": stopped_on_insufficient_quota,
        "insufficient_quota_rows": sum(1 for row in rows if is_insufficient_quota_row(row)),
    }
    meta_path = out_csv.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return meta_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/gui_explorer")
    parser.add_argument("--apps", nargs="*", default=None)
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=None, help="CSV path. Defaults to <root>/strict_judge.csv")
    parser.add_argument(
        "--gt-root",
        nargs="*",
        default=None,
        help="Ground-truth root(s). Defaults to $TAGNAV_GT_ROOT or $TAGNAV_TASKS_ROOT.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse valid rows already present in --out and judge only missing/error rows.",
    )
    parser.add_argument(
        "--stop-on-insufficient-quota",
        action="store_true",
        help="Stop the batch as soon as OpenAI reports insufficient_quota.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional .env file to load before checking OPENAI_API_KEY. Defaults to baselines/.env.",
    )
    args = parser.parse_args()

    load_env_file(args.env_file)
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run strict judging.")

    bases = gt_bases(args.gt_root)
    if not bases:
        raise SystemExit(
            "No ground-truth root. Pass --gt-root or set TAGNAV_GT_ROOT / TAGNAV_TASKS_ROOT."
        )

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    root = Path(args.root)
    out = Path(args.out) if args.out else root / "strict_judge.csv"
    existing_rows = load_existing_valid_rows(out) if args.resume_existing else {}
    rows: list[dict[str, Any]] = []
    stop_judging = False
    reused_total = 0
    app_list = apps_from_root(root, args.apps)

    for app in app_list:
        if stop_judging:
            print(f"SKIP {app}: stopped after insufficient_quota")
            continue
        app_dir = root / app
        if not app_dir.is_dir():
            print(f"SKIP {app}: no run directory")
            continue
        task_dirs = sorted(d for d in app_dir.iterdir() if d.is_dir() and (d / "summary.json").exists())
        pending_dirs = [d for d in task_dirs if (app, d.name) not in existing_rows]
        reused = len(task_dirs) - len(pending_dirs)
        reused_total += reused
        rows.extend(existing_rows[(app, d.name)] for d in task_dirs if (app, d.name) in existing_rows)
        print(f"\n{app}: judging {len(pending_dirs)} tasks, reusing {reused}")
        if args.stop_on_insufficient_quota:
            for task_dir in pending_dirs:
                row = judge_one(client, args.judge_model, app, task_dir, bases)
                rows.append(row)
                print(f"  {row['task']}: {row['verdict']} score={row['score']} {row['justification'][:80]}")
                if is_insufficient_quota_row(row):
                    stop_judging = True
                    break
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge_one, client, args.judge_model, app, d, bases) for d in pending_dirs]
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                print(f"  {row['task']}: {row['verdict']} score={row['score']} {row['justification'][:80]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["app", "task", "score", "verdict", "justification", "prompt", "agent_png", "gt_png"]
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["app"], r["task"])))
    meta_path = write_metadata(
        out_csv=out,
        root=root,
        apps=app_list,
        model=args.judge_model,
        rows=rows,
        reused_rows=reused_total,
        stopped_on_insufficient_quota=stop_judging,
    )

    valid = [r for r in rows if r["verdict"] in {"match", "partial", "mismatch"}]
    strict = sum(1 for r in valid if r["verdict"] == "match")
    partial = sum(1 for r in valid if r["verdict"] == "partial")
    print(f"\nStrict judge: {strict}/{len(valid)} match ({strict / len(valid) * 100 if valid else 0:.1f}%), partial={partial}")
    print(f"CSV: {out}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
