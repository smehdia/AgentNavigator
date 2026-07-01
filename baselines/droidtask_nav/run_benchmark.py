#!/usr/bin/env python3
"""DroidTask-Nav benchmark harness.

Prepares DroidTask navigation tasks with chosen graph targets and (optionally)
runs the shared navigation benchmark runner on them. It keeps the shared
`baselines/_common/nav_benchmark.py` runner unchanged and supplies explicit app
metadata, task roots, and processed graph PKLs.

This is part of the KG-RAG comparison: KG-RAG's upstream is not runnable from a
fresh clone (lab-internal services), so we run OUR agent on DroidTask -- KG-RAG's
home benchmark -- and compare against KG-RAG's published numbers. See README.md.

Paths/keys/hosts come from the environment (see baselines/.env.example):
  DROIDTASK_APPS_JSON     app metadata json (slug/package/activity) -- see droidtask_apps.example.json
  DROIDTASK_NAV_TASKS_ROOT converted DroidTask-Nav task tree (output of convert.py)
  EXPLORED_GRAPHS_ROOT    dir holding explored graphs (smt_<slug>_<graph_source>/...)
  ANDROID_DEVICE_SERIAL   adb serial of the target device/emulator
  NAV_BENCHMARK_RUNNER    nav runner (defaults to baselines/_common/nav_benchmark.py)
  PREPARE_TARGETS_SCRIPT  target prep (defaults to baselines/_common/prepare_targets.py)

Example:
  python baselines/droidtask_nav/run_benchmark.py \\
      --apps clock calendar --graph-source emulator-5554 \\
      --prepare-targets --run-benchmark --backbone qwen_vl --memory-mode open_loop
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPS_JSON = Path(os.environ.get(
    "DROIDTASK_APPS_JSON",
    str(Path(__file__).resolve().parent / "droidtask_apps.example.json"),
))
DEFAULT_TASKS_ROOT = Path(os.environ.get(
    "DROIDTASK_NAV_TASKS_ROOT", str(REPO_ROOT / "DroidTask_nav_v2_DATA")))
FALLBACK_TASKS_ROOT = REPO_ROOT / "DroidTask_nav_collector_DATA"
DEFAULT_RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", str(REPO_ROOT / "results")))
DEFAULT_GRAPHS_ROOT = Path(os.environ.get("EXPLORED_GRAPHS_ROOT", str(REPO_ROOT / "explored_apps")))
DEFAULT_DEVICE = os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554")

# Shared nav runner + target-prep helpers are vendored under baselines/_common/.
# Override via env if they live elsewhere in your checkout.
NAV_BENCHMARK_RUNNER = Path(os.environ.get(
    "NAV_BENCHMARK_RUNNER", str(REPO_ROOT / "baselines" / "_common" / "nav_benchmark.py")))
PREPARE_TARGETS_SCRIPT = Path(os.environ.get(
    "PREPARE_TARGETS_SCRIPT", str(REPO_ROOT / "baselines" / "_common" / "prepare_targets.py")))

SMOKE_APPS = ("clock", "calendar")


@dataclass(frozen=True)
class DroidTaskApp:
    slug: str
    package: str
    activity: str | None
    display_name: str = ""


def load_apps(path: Path) -> dict[str, DroidTaskApp]:
    data = json.loads(path.read_text())
    apps: dict[str, DroidTaskApp] = {}
    for item in data.get("apps", []):
        app = DroidTaskApp(
            slug=item["slug"],
            package=item["package"],
            activity=item.get("activity"),
            display_name=item.get("display_name", ""),
        )
        apps[app.slug] = app
    return apps


def resolve_tasks_root(path: Path | None) -> Path:
    if path is not None:
        return path
    if DEFAULT_TASKS_ROOT.is_dir():
        return DEFAULT_TASKS_ROOT
    return FALLBACK_TASKS_ROOT


def find_source_app_dir(tasks_root: Path, slug: str) -> Path:
    matches = sorted(
        p for p in tasks_root.iterdir()
        if p.is_dir() and p.name.startswith(f"droidtask_{slug}")
    )
    if not matches:
        raise FileNotFoundError(f"No DroidTask task directory for slug '{slug}' under {tasks_root}")
    if len(matches) > 1:
        names = ", ".join(p.name for p in matches)
        raise RuntimeError(f"Ambiguous DroidTask dirs for slug '{slug}': {names}")
    return matches[0]


def processed_pkl_for(slug: str, graph_source: str, graph_root: Path) -> Path:
    # Explored-graph dirs are named smt_<slug>_<graph_source>, where graph_source
    # is the device serial / tag the graph was explored on (e.g. an emulator serial).
    return graph_root / f"smt_{slug}_{graph_source}" / "node_information_dict_pure_visual_with_embeddings.pkl"


def copy_tasks(src: Path, dst: Path, clean: bool) -> int:
    if clean and dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for task_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        target = dst / task_dir.name
        shutil.copytree(task_dir, target, dirs_exist_ok=True)
        count += 1
    return count


def count_chosen_targets(tasks_dir: Path) -> int:
    return sum(1 for p in tasks_dir.iterdir() if p.is_dir() and (p / "chosen_target.json").is_file())


def count_summaries(run_dir: Path) -> int:
    if not run_dir.is_dir():
        return 0
    return sum(1 for p in run_dir.iterdir() if p.is_dir() and (p / "summary.json").is_file())


def count_task_dirs(tasks_dir: Path) -> int:
    return sum(1 for p in tasks_dir.iterdir() if p.is_dir())


def run_cmd(cmd: list[str], dry_run: bool) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def build_prepare_cmd(pkl: Path, tasks_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PREPARE_TARGETS_SCRIPT),
        "--pkl",
        str(pkl),
        "--tasks-root",
        str(tasks_dir),
        "--out-dir",
        str(tasks_dir),
        "--skip-dedup",
    ]


def build_runner_cmd(
    app: DroidTaskApp,
    tasks_dir: Path,
    pkl: Path,
    run_dir: Path,
    device: str,
    memory_mode: str,
    backbone: str,
    coord_max_side: int,
    max_steps: int,
    include_tasks: list[str] | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(NAV_BENCHMARK_RUNNER),
        "--app",
        f"droidtask_{app.slug}",
        "--device",
        device,
        "--app-package",
        app.package,
        "--tasks-root",
        str(tasks_dir),
        "--processed-pkl",
        str(pkl),
        "--memory-mode",
        memory_mode,
        "--backbone",
        backbone,
        "--coord-max-side",
        str(coord_max_side),
        "--max-steps",
        str(max_steps),
        "--out-dir",
        str(run_dir),
    ]
    if app.activity:
        cmd.extend(["--app-activity", app.activity])
    if include_tasks:
        cmd.append("--include-tasks")
        cmd.extend(include_tasks)
    return cmd


def write_summary(
    out_root: Path,
    rows: list[dict[str, Any]],
    graph_source: str,
    tasks_root: Path,
    device: str,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "graph_source": f"{graph_source} BFS",
        "tasks_root": str(tasks_root),
        "device": device,
        "apps": rows,
        "total_tasks": sum(r["tasks"] for r in rows),
        "total_chosen_targets": sum(r["chosen_targets"] for r in rows),
        "total_summaries": sum(r["summaries"] for r in rows),
    }
    (out_root / "droidtask_nav_harness_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# DroidTask-Nav Harness Summary",
        "",
        f"- Graph source: {summary['graph_source']}",
        f"- Tasks root: `{tasks_root}`",
        f"- Device: `{device}`",
        "",
        "| App | Tasks | Chosen targets | Summaries | Processed PKL |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['slug']} | {row['tasks']} | {row['chosen_targets']} | "
            f"{row['summaries']} | `{row['processed_pkl']}` |"
        )
    (out_root / "droidtask_nav_harness_summary.md").write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apps", nargs="*", default=list(SMOKE_APPS),
                        help="DroidTask app slugs to run. Defaults to clock calendar.")
    parser.add_argument("--all-apps", action="store_true", help="Run all apps in the apps json.")
    parser.add_argument("--tasks-root", type=Path, default=None,
                        help="Source DroidTask-Nav tree. Defaults to reviewed tree when present, else collector tree.")
    parser.add_argument("--apps-json", type=Path, default=DEFAULT_APPS_JSON)
    parser.add_argument("--tag", default="smoke")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--graph-source", default=DEFAULT_DEVICE,
                        help="Tag/serial the graph was explored on; used as the suffix in smt_<slug>_<graph_source>.")
    parser.add_argument("--graph-root", type=Path, default=DEFAULT_GRAPHS_ROOT)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--memory-mode", choices=["closed_loop", "open_loop", "none", "plan_edit"],
                        default="open_loop")
    parser.add_argument("--backbone", choices=["ui_tars", "qwen_vl", "qwen_vl_local"], default="ui_tars")
    parser.add_argument("--coord-max-side", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--include-tasks", nargs="*", default=None,
                        help="Optional task IDs to pass to the runner for every selected app.")
    parser.add_argument("--prepare-targets", action="store_true",
                        help="Run the target-prep script after materializing task copies.")
    parser.add_argument("--run-benchmark", action="store_true",
                        help="Run the navigation benchmark runner after task preparation.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete materialized per-app task copies before copying.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands and validate paths without running subprocesses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apps = load_apps(args.apps_json)
    selected_slugs = sorted(apps) if args.all_apps else args.apps
    unknown = [s for s in selected_slugs if s not in apps]
    if unknown:
        raise SystemExit(f"Unknown app slug(s): {', '.join(unknown)}")

    tasks_root = resolve_tasks_root(args.tasks_root).resolve()
    if not tasks_root.is_dir():
        raise SystemExit(f"Tasks root not found: {tasks_root}")

    out_root = (args.results_root / f"droidtask_nav_{args.tag}").resolve()
    materialized_root = out_root / "tasks_with_targets"
    runs_root = out_root / "runs"
    rows: list[dict[str, Any]] = []

    for slug in selected_slugs:
        app = apps[slug]
        src_app_dir = find_source_app_dir(tasks_root, slug)
        tasks_dir = materialized_root / slug
        if args.dry_run:
            task_count = count_task_dirs(src_app_dir)
        else:
            task_count = copy_tasks(src_app_dir, tasks_dir, clean=args.clean)

        pkl = processed_pkl_for(slug, args.graph_source, args.graph_root.resolve())
        if not pkl.is_file():
            raise SystemExit(f"Processed PKL not found for {slug}: {pkl}")

        action = "would be materialized" if args.dry_run else "materialized"
        print(f"\n{slug}: {task_count} tasks {action} at {tasks_dir}")
        if args.prepare_targets:
            rc = run_cmd(build_prepare_cmd(pkl, tasks_dir), dry_run=args.dry_run)
            if rc != 0:
                raise SystemExit(f"Target prep failed for {slug} with exit code {rc}")

        run_dir = runs_root / slug
        if args.run_benchmark:
            rc = run_cmd(
                build_runner_cmd(
                    app=app,
                    tasks_dir=tasks_dir,
                    pkl=pkl,
                    run_dir=run_dir,
                    device=args.device,
                    memory_mode=args.memory_mode,
                    backbone=args.backbone,
                    coord_max_side=args.coord_max_side,
                    max_steps=args.max_steps,
                    include_tasks=args.include_tasks,
                ),
                dry_run=args.dry_run,
            )
            if rc != 0:
                raise SystemExit(f"Benchmark failed for {slug} with exit code {rc}")

        rows.append({
            "slug": slug,
            "source_task_dir": str(src_app_dir),
            "materialized_tasks_dir": str(tasks_dir),
            "tasks": task_count,
            "chosen_targets": 0 if args.dry_run and not tasks_dir.exists() else count_chosen_targets(tasks_dir),
            "run_dir": str(run_dir),
            "summaries": 0 if args.dry_run and not run_dir.exists() else count_summaries(run_dir),
            "processed_pkl": str(pkl),
        })

    if args.dry_run:
        print("\nDry run complete; no task copies, benchmark outputs, or summaries were written.")
    else:
        write_summary(out_root, rows, args.graph_source, tasks_root, args.device)
        print(f"\nSummary written to {out_root / 'droidtask_nav_harness_summary.md'}")


if __name__ == "__main__":
    main()
