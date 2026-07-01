#!/usr/bin/env python3
"""Materialize a SPA-Bench-compatible evaluation session from pilot outputs.

Required environment:
  SPA_BENCH_ROOT   path to the upstream SPA-Bench checkout (provides framework.utils
                   and the default dataset). Falls back to baselines/external/SPA-Bench.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


# baselines/spa_bench/<this file> -> repo root is parents[2]; baselines/ is parents[1].
REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINES_DIR = Path(__file__).resolve().parents[1]


def spa_bench_root() -> Path:
    env = os.environ.get("SPA_BENCH_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return BASELINES_DIR / "external" / "SPA-Bench"


SPA_BENCH_ROOT = spa_bench_root()
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(SPA_BENCH_ROOT))

from pilot_common import load_json
from framework import utils as spa_utils  # provided by the upstream SPA-Bench checkout


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def copy_tree_contents(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for child in sorted(src.iterdir()):
        target = dst / child.name
        if child.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def populate_task_dir(result_root: Path, task_id: str, agent_name: str, pilot_task_dir: Path) -> None:
    target_dir = result_root / task_id / agent_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    copy_tree_contents(pilot_task_dir, target_dir)


def update_execution_columns(result_root: Path, specs: list[dict], agent_name: str, device: str) -> None:
    for spec in specs:
        task_id = spec["spa_task_identifier"]
        task_dir = result_root / task_id / agent_name
        run_manifest_path = task_dir / "run_manifest.json"
        if not run_manifest_path.is_file():
            continue
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        spa_utils.save_result__completed_execution(
            str(result_root),
            task_id,
            agent_name,
            task_completed=run_manifest.get("returncode", 1) == 0,
            exit_code=run_manifest.get("returncode", -1),
            device=device,
        )


def export_session(
    subset_manifest_path: Path,
    spec_path: Path,
    pilot_arm_root: Path,
    out_dir: Path,
    dataset_path: Path,
    agent_name: str,
    reasoning_mode: str,
    action_mode: str,
    device: str,
) -> None:
    subset_manifest = load_json(subset_manifest_path)
    specs = load_json(spec_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    spa_utils.setup_results_csv(
        str(out_dir),
        str(dataset_path),
        [agent_name],
        reasoning_mode,
        action_mode,
    )
    for spec in specs:
        task_id = spec["spa_task_identifier"]
        pilot_task_dir = pilot_arm_root / task_id
        if not pilot_task_dir.is_dir():
            continue
        populate_task_dir(out_dir, task_id, agent_name, pilot_task_dir)
    update_execution_columns(out_dir, specs, agent_name, device)
    manifest = {
        "schema": "agentnavigator_spa_export_session_v1",
        "subset_manifest": str(subset_manifest_path),
        "spec_path": str(spec_path),
        "pilot_arm_root": str(pilot_arm_root),
        "dataset_path": str(dataset_path),
        "agent_name": agent_name,
        "reasoning_mode": reasoning_mode,
        "action_mode": action_mode,
        "device": device,
        "task_ids": subset_manifest["task_ids"],
    }
    (out_dir / "session_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset-manifest", type=Path, required=True)
    parser.add_argument("--spec-path", type=Path, required=True)
    parser.add_argument("--pilot-arm-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=SPA_BENCH_ROOT / "data" / "single-app-ENG.csv")
    parser.add_argument("--agent-name", default="TAGNavAgent")
    parser.add_argument("--reasoning-mode", default="direct")
    parser.add_argument("--action-mode", default="with_action")
    parser.add_argument("--device", default=os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554"))
    args = parser.parse_args()

    export_session(
        resolve_repo_path(str(args.subset_manifest)),
        resolve_repo_path(str(args.spec_path)),
        resolve_repo_path(str(args.pilot_arm_root)),
        resolve_repo_path(str(args.out_dir)),
        args.dataset_path.resolve(),
        args.agent_name,
        args.reasoning_mode,
        args.action_mode,
        args.device,
    )
    print(f"Wrote SPA-Bench session: {args.out_dir}")


if __name__ == "__main__":
    main()
