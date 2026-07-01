#!/usr/bin/env python3
"""Summarize GUI-Explorer benchmark outputs across result roots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


APP_ORDER = [
    "airbnb",
    "amazon",
    "calendar",
    "chrome",
    "clock",
    "ebay",
    "google_maps",
    "instagram",
    "linkedin",
    "settings",
    "tiktok",
    "youtube",
]


def summarize_app(root: Path, app: str) -> dict[str, Any]:
    summary_path = root / app / "all_summaries.json"
    if not summary_path.exists():
        return {
            "root": str(root),
            "app": app,
            "status": "missing",
            "finished_action": "",
            "tasks": "",
            "success_rate": "",
            "max_steps": "",
            "errors": "",
        }
    data = json.loads(summary_path.read_text())
    finished = sum(1 for row in data if row.get("finished_reason") == "finished_action")
    errors = sum(1 for row in data if str(row.get("finished_reason", "")).startswith("error"))
    max_steps = sum(1 for row in data if row.get("finished_reason") == "max_steps")
    total = len(data)
    return {
        "root": str(root),
        "app": app,
        "status": "ok",
        "finished_action": finished,
        "tasks": total,
        "success_rate": round(finished / total, 4) if total else 0,
        "max_steps": max_steps,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["results/gui_explorer"],
    )
    parser.add_argument(
        "--out-csv",
        default="results/gui_explorer/gui_explorer_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        default="results/gui_explorer/gui_explorer_summary.md",
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        default=None,
        help="Apps to include. Defaults to apps with result directories unless --include-missing is set.",
    )
    parser.add_argument(
        "--include-missing",
        action="store_true",
        help="Include missing rows for requested/default apps.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for root_str in args.roots:
        root = Path(root_str)
        if args.apps:
            apps = args.apps
        elif args.include_missing:
            apps = APP_ORDER
        else:
            apps = [app for app in APP_ORDER if (root / app).exists()]
        for app in apps:
            rows.append(summarize_app(root, app))

    if not rows:
        raise SystemExit("No GUI-Explorer result directories found.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GUI-Explorer Result Summary",
        "",
        "| root | app | finished | tasks | success | max_steps | errors | status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {root} | {app} | {finished_action} | {tasks} | {success_rate} | "
            "{max_steps} | {errors} | {status} |".format(**row)
        )
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nWrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
