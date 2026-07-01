#!/usr/bin/env python3
"""Generate the fixed SPA-Bench pilot subset (Level-3, 7 apps)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# baselines/spa_bench/<this file> -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_common import (
    APP_CONFIG,
    PILOT_TASK_IDS,
    RETRIEVAL_PROMPTS,
    SPA_BENCH_SINGLE_APP_ENG_URL,
    load_spa_csv_rows,
)


DEFAULT_OUT_ROOT = REPO_ROOT / "spa_bench_subsets" / "single_app_eng_level3_7apps"


def path_for_manifest(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_subset_rows(csv_path: str | Path | None) -> list[dict[str, str]]:
    rows = load_spa_csv_rows(csv_path)
    rows_by_task = {row["task_identifier"]: row for row in rows}
    missing = [task_id for task_id in PILOT_TASK_IDS if task_id not in rows_by_task]
    if missing:
        raise SystemExit(f"Missing SPA-Bench rows for task ids: {', '.join(missing)}")
    selected = [rows_by_task[task_id] for task_id in PILOT_TASK_IDS]
    for row in selected:
        if row["task_difficulty"] != "3":
            raise SystemExit(f"Expected Level 3 task, got {row['task_identifier']} difficulty={row['task_difficulty']}")
    return selected


def materialize_subset(out_root: Path, rows: list[dict[str, str]]) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    apps_manifest: dict[str, dict] = {}
    tasks_manifest: list[dict] = []

    for row in rows:
        app = row["task_app"]
        app_cfg = APP_CONFIG[app]
        app_root = out_root / app_cfg["tasks_dir"]
        task_root = app_root / row["task_identifier"]
        task_root.mkdir(parents=True, exist_ok=True)

        prompt = row["task_description"]
        retrieval_prompt = RETRIEVAL_PROMPTS[row["task_identifier"]]
        meta = {
            "schema": "agentnavigator_spa_pilot_task_v1",
            "source": "SPA-Bench",
            "benchmark_mode": "internal_transfer_pilot",
            "spa_source_csv_url": SPA_BENCH_SINGLE_APP_ENG_URL,
            "spa_task_identifier": row["task_identifier"],
            "spa_task_app": row["task_app"],
            "spa_task_language": row["task_language"],
            "spa_task_difficulty": row["task_difficulty"],
            "golden_steps": int(row["golden_steps"]),
            "adb_app": row["adb_app"],
            "adb_home_page": row["adb_home_page"],
            "is_cross_app": row["is_cross_app"],
            "task_description": prompt,
            "retrieval_prompt": retrieval_prompt,
            "agentnavigator_app": app,
            "agentnavigator_tasks_dir": app_cfg["tasks_dir"],
            "auto_bfs_graph_pkl": app_cfg["graph_pkl"],
            "source_row": row,
        }
        prompts = {"prompts": [prompt]}
        (task_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (task_root / "prompts.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")

        apps_manifest.setdefault(
            app,
            {
                "app": app,
                "agentnavigator_tasks_dir": app_cfg["tasks_dir"],
                "tasks_root": path_for_manifest(app_root),
                "auto_bfs_graph_pkl": app_cfg["graph_pkl"],
                "task_ids": [],
            },
        )["task_ids"].append(row["task_identifier"])
        tasks_manifest.append(
            {
                "task_id": row["task_identifier"],
                "app": app,
                "tasks_root": path_for_manifest(app_root),
                "task_dir": path_for_manifest(task_root),
                "retrieval_prompt": retrieval_prompt,
                "golden_steps": int(row["golden_steps"]),
                "adb_app": row["adb_app"],
                "adb_home_page": row["adb_home_page"],
                "auto_bfs_graph_pkl": app_cfg["graph_pkl"],
            }
        )

    manifest = {
        "schema": "agentnavigator_spa_pilot_subset_v1",
        "benchmark_mode": "internal_transfer_pilot",
        "source_csv_url": SPA_BENCH_SINGLE_APP_ENG_URL,
        "subset_name": "single_app_eng_level3_7apps",
        "task_ids": PILOT_TASK_IDS,
        "retrieval_prompts": RETRIEVAL_PROMPTS,
        "apps": [apps_manifest[app] for app in sorted(apps_manifest)],
        "tasks": tasks_manifest,
    }
    (out_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", default=None, help="Optional local copy of single-app-ENG.csv")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    rows = build_subset_rows(args.csv_path)
    manifest = materialize_subset(args.out_root.resolve(), rows)
    print(f"Wrote SPA-Bench subset: {args.out_root}")
    print(f"Apps: {', '.join(app['app'] for app in manifest['apps'])}")
    print(f"Tasks: {len(manifest['tasks'])}")


if __name__ == "__main__":
    main()
