#!/usr/bin/env python3
"""Summarize SPA-Bench evaluation outputs for the pilot subset (no-graph vs auto-BFS)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_common import load_json


def load_results(csv_path: Path, agent_name: str, reasoning_mode: str, action_mode: str) -> dict[str, dict]:
    prefix = f"{agent_name}_" if agent_name else ""
    eval_col = f"{reasoning_mode}_{action_mode}_{prefix}evaluation"
    detail_col = f"{reasoning_mode}_{action_mode}_{prefix}details"
    rows: dict[str, dict] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row["task_identifier"]
            detail_raw = row.get(detail_col, "") or "{}"
            try:
                details = json.loads(detail_raw)
            except json.JSONDecodeError:
                details = {"raw_details": detail_raw}
            rows[task_id] = {
                "evaluation": row.get(eval_col, ""),
                "details": details,
            }
    return rows


def is_success(code: str) -> bool:
    return code == "S"


def summarize_pair(subset_manifest: dict, no_graph_rows: dict, auto_rows: dict) -> dict:
    per_task = []
    per_app = defaultdict(lambda: {"no_graph_success": 0, "auto_bfs_success": 0, "attempts": 0})
    wins = Counter()

    for task in subset_manifest["tasks"]:
        task_id = task["task_id"]
        no_code = no_graph_rows.get(task_id, {}).get("evaluation", "")
        auto_code = auto_rows.get(task_id, {}).get("evaluation", "")
        no_success = is_success(no_code)
        auto_success = is_success(auto_code)
        if auto_success and not no_success:
            outcome = "win"
        elif no_success and not auto_success:
            outcome = "loss"
        else:
            outcome = "tie"
        wins[outcome] += 1
        per_task.append(
            {
                "task_id": task_id,
                "app": task["app"],
                "no_graph": no_code,
                "auto_bfs": auto_code,
                "outcome": outcome,
            }
        )
        app_row = per_app[task["app"]]
        app_row["attempts"] += 1
        app_row["no_graph_success"] += int(no_success)
        app_row["auto_bfs_success"] += int(auto_success)

    total = len(per_task)
    no_graph_success = sum(row["no_graph_success"] for row in per_app.values())
    auto_bfs_success = sum(row["auto_bfs_success"] for row in per_app.values())
    per_app_rows = []
    for app in sorted(per_app):
        row = per_app[app]
        attempts = row["attempts"]
        per_app_rows.append(
            {
                "app": app,
                "attempts": attempts,
                "no_graph_success": row["no_graph_success"],
                "auto_bfs_success": row["auto_bfs_success"],
                "no_graph_rate": row["no_graph_success"] / attempts if attempts else 0.0,
                "auto_bfs_rate": row["auto_bfs_success"] / attempts if attempts else 0.0,
            }
        )
    return {
        "schema": "agentnavigator_spa_pilot_summary_v1",
        "task_count": total,
        "aggregate": {
            "no_graph_success": no_graph_success,
            "auto_bfs_success": auto_bfs_success,
            "no_graph_rate": no_graph_success / total if total else 0.0,
            "auto_bfs_rate": auto_bfs_success / total if total else 0.0,
            "delta": (auto_bfs_success - no_graph_success) / total if total else 0.0,
        },
        "task_outcomes": {
            "win": wins.get("win", 0),
            "loss": wins.get("loss", 0),
            "tie": wins.get("tie", 0),
        },
        "per_app": per_app_rows,
        "per_task": per_task,
    }


def write_outputs(summary: dict, out_json: Path, out_md: Path) -> None:
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# SPA-Bench Pilot Summary",
        "",
        f"- Tasks: {summary['task_count']}",
        f"- No-graph: {summary['aggregate']['no_graph_success']}/{summary['task_count']} ({summary['aggregate']['no_graph_rate']*100:.1f}%)",
        f"- Auto-BFS: {summary['aggregate']['auto_bfs_success']}/{summary['task_count']} ({summary['aggregate']['auto_bfs_rate']*100:.1f}%)",
        f"- Delta: {summary['aggregate']['delta']*100:+.1f} pp",
        "",
        "## Outcomes",
        "",
        f"- Wins: {summary['task_outcomes'].get('win', 0)}",
        f"- Losses: {summary['task_outcomes'].get('loss', 0)}",
        f"- Ties: {summary['task_outcomes'].get('tie', 0)}",
        "",
        "## Per-App",
        "",
        "| App | Attempts | No-graph | Auto-BFS |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["per_app"]:
        lines.append(
            f"| {row['app']} | {row['attempts']} | "
            f"{row['no_graph_success']}/{row['attempts']} ({row['no_graph_rate']*100:.1f}%) | "
            f"{row['auto_bfs_success']}/{row['attempts']} ({row['auto_bfs_rate']*100:.1f}%) |"
        )
    lines.extend(
        [
            "",
            "## Per-Task",
            "",
            "| Task | App | No-graph | Auto-BFS | Outcome |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["per_task"]:
        lines.append(
            f"| {row['task_id']} | {row['app']} | {row['no_graph'] or '-'} | {row['auto_bfs'] or '-'} | {row['outcome']} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--no-graph-results", type=Path, required=True)
    parser.add_argument("--auto-bfs-results", type=Path, required=True)
    parser.add_argument("--agent-name", default="TAGNavAgent")
    parser.add_argument("--reasoning-mode", default="direct")
    parser.add_argument("--action-mode", default="with_action")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    subset_manifest = load_json(args.subset_manifest)
    no_rows = load_results(args.no_graph_results, args.agent_name, args.reasoning_mode, args.action_mode)
    auto_rows = load_results(args.auto_bfs_results, args.agent_name, args.reasoning_mode, args.action_mode)
    summary = summarize_pair(subset_manifest, no_rows, auto_rows)
    write_outputs(summary, args.out_json, args.out_md)
    print(f"Wrote SPA-Bench summary: {args.out_json}")


if __name__ == "__main__":
    main()
