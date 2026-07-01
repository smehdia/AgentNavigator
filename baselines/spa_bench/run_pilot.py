#!/usr/bin/env python3
"""Run the SPA-Bench pilot specs through the project's nav benchmark on a live device.

This wrapper consumes the run specs for the SPA-Bench pilot, calls the project's
own navigation benchmark runner once per task, and exports screenshots in a
SPA-Bench-friendly zero-indexed format.

Required environment / args:
  NAV_BENCHMARK_SCRIPT  path to the project's nav benchmark runner (the script that
                        executes one navigation task on device and writes step_*.png +
                        summary.json). Override with --nav-benchmark-script.
  ANDROID_DEVICE_SERIAL adb serial of the running emulator/device (or --device).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


# baselines/spa_bench/<this file> -> repo root is parents[2].
REPO_ROOT = Path(__file__).resolve().parents[2]


def nav_benchmark_script(cli_value: str | None) -> Path:
    # Defaults to the runner vendored under baselines/_common/; override via
    # --nav-benchmark-script or NAV_BENCHMARK_SCRIPT if yours lives elsewhere.
    raw = (cli_value or os.environ.get("NAV_BENCHMARK_SCRIPT")
           or str(REPO_ROOT / "baselines" / "_common" / "nav_benchmark.py"))
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.is_file():
        raise SystemExit(f"Nav benchmark runner not found: {path}")
    return path


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def load_specs(benchmark_manifest: Path | None, arm: str | None, spec_path: Path | None) -> tuple[str, list[dict]]:
    if spec_path is not None:
        resolved = resolve_repo_path(str(spec_path))
        return resolved.stem, load_json(resolved)
    if benchmark_manifest is None or arm is None:
        raise ValueError("Provide either --spec-path, or both --benchmark-manifest and --arm.")
    manifest = load_json(resolve_repo_path(str(benchmark_manifest)))
    arm_info = manifest["arms"].get(arm)
    if arm_info is None:
        raise ValueError(f"Arm not found in benchmark manifest: {arm}")
    resolved = resolve_repo_path(arm_info["spec_path"])
    return arm, load_json(resolved)


def filter_specs(specs: list[dict], include_tasks: list[str] | None, limit: int | None) -> list[dict]:
    out = specs
    if include_tasks:
        wanted = set(include_tasks)
        out = [spec for spec in out if spec["spa_task_identifier"] in wanted]
    if limit is not None:
        out = out[:limit]
    return out


def build_command(spec: dict, args: argparse.Namespace, raw_out_root: Path, nav_script: Path) -> list[str]:
    task_id = spec["spa_task_identifier"]
    command = [
        sys.executable,
        str(nav_script),
        "--app",
        spec["app"],
        "--device",
        args.device,
        "--memory-mode",
        spec["memory_mode"],
        "--max-steps",
        str(spec["max_steps"]),
        "--out-dir",
        str(raw_out_root),
        "--tasks-root",
        str(resolve_repo_path(spec["agentnavigator_tasks_root"])),
        "--include-tasks",
        spec["include_task_id"],
        "--ui-tars-url",
        args.ui_tars_url,
        "--adb-path",
        args.adb_path,
        "--backbone",
        args.backbone,
        "--keep-steps",
        str(args.keep_steps),
        "--sim-floor",
        str(args.sim_floor),
    ]
    if args.coord_max_side is not None:
        command.extend(["--coord-max-side", str(args.coord_max_side)])
    tap_home_apps = set(args.tap_home_apps or [])
    if args.tap_home_after_launch or spec["app"] in tap_home_apps:
        command.append("--tap-home-after-launch")
    if args.clear_between_tasks:
        command.append("--clear-between-tasks")
    if args.task_completion:
        command.append("--task-completion")
    if args.full_task_prompt:
        command.append("--full-task-prompt")
    processed_pkl = spec.get("processed_pkl")
    if processed_pkl:
        command.extend(["--processed-pkl", str(resolve_repo_path(processed_pkl))])
    if args.extra_run_nav_args:
        command.extend(args.extra_run_nav_args)
    if task_id != spec["include_task_id"]:
        raise ValueError(f"Spec task mismatch: {task_id} vs include_task_id={spec['include_task_id']}")
    return command


def export_screenshots(raw_task_dir: Path, export_dir: Path) -> list[str]:
    step_paths = sorted(raw_task_dir.glob("step_*.png"))
    final_path = raw_task_dir / "agent_screenshot.png"
    ordered_paths = list(step_paths)
    if final_path.is_file():
        ordered_paths.append(final_path)
    exported = []
    for idx, src in enumerate(ordered_paths):
        dst = export_dir / f"{idx}.png"
        shutil.copy2(src, dst)
        exported.append(dst.name)
    return exported


def copy_if_exists(src: Path, dst: Path) -> bool:
    if src.is_file():
        shutil.copy2(src, dst)
        return True
    return False


def export_task_outputs(raw_task_dir: Path, final_task_dir: Path, command: list[str], result: subprocess.CompletedProcess) -> dict:
    final_task_dir.mkdir(parents=True, exist_ok=True)
    exported = export_screenshots(raw_task_dir, final_task_dir)
    copied_files = {}
    copied_files["trajectory_jsonl"] = copy_if_exists(
        raw_task_dir / "trajectory.jsonl",
        final_task_dir / "agentnavigator_trajectory.jsonl",
    )
    copied_files["summary_json"] = copy_if_exists(
        raw_task_dir / "summary.json",
        final_task_dir / "agentnavigator_summary.json",
    )
    copied_files["target_info_json"] = copy_if_exists(
        raw_task_dir / "target_info.json",
        final_task_dir / "agentnavigator_target_info.json",
    )
    raw_summary_root = raw_task_dir.parent / "all_summaries.json"
    copied_files["all_summaries_json"] = copy_if_exists(
        raw_summary_root,
        final_task_dir / "agentnavigator_all_summaries.json",
    )
    manifest = {
        "schema": "agentnavigator_spa_pilot_run_v1",
        "task_id": raw_task_dir.name,
        "raw_task_dir": str(raw_task_dir),
        "command": command,
        "returncode": result.returncode,
        "screenshots": exported,
        "copied_files": copied_files,
        "stdout_path": "run_stdout.txt",
        "stderr_path": "run_stderr.txt",
    }
    (final_task_dir / "run_stdout.txt").write_text(result.stdout or "", encoding="utf-8")
    (final_task_dir / "run_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
    (final_task_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_one_spec(spec: dict, args: argparse.Namespace, arm_out_root: Path, nav_script: Path) -> dict:
    task_id = spec["spa_task_identifier"]
    final_task_dir = arm_out_root / task_id
    with tempfile.TemporaryDirectory(prefix=f"spa_pilot_{task_id}_") as tmp:
        raw_out_root = Path(tmp) / "agentnavigator_raw"
        raw_out_root.mkdir(parents=True, exist_ok=True)
        command = build_command(spec, args, raw_out_root, nav_script)
        if args.dry_run:
            manifest = {
                "schema": "agentnavigator_spa_pilot_run_v1",
                "task_id": task_id,
                "command": command,
                "returncode": None,
                "screenshots": [],
                "copied_files": {},
                "dry_run": True,
            }
            final_task_dir.mkdir(parents=True, exist_ok=True)
            (final_task_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            return manifest
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=args.per_task_timeout if args.per_task_timeout > 0 else None,
        )
        raw_task_dir = raw_out_root / task_id
        if not raw_task_dir.is_dir():
            raise RuntimeError(
                f"Expected raw task dir missing for {task_id}: {raw_task_dir}\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
        manifest = export_task_outputs(raw_task_dir, final_task_dir, command, result)
        if result.returncode != 0:
            raise RuntimeError(
                f"nav benchmark runner failed for {task_id} with returncode={result.returncode}\n"
                f"See {final_task_dir / 'run_stderr.txt'}"
            )
        return manifest


def build_index(arm: str, specs: list[dict], manifests: list[dict]) -> dict:
    task_index = []
    for spec, manifest in zip(specs, manifests):
        task_index.append(
            {
                "task_id": spec["spa_task_identifier"],
                "app": spec["app"],
                "memory_mode": spec["memory_mode"],
                "max_steps": spec["max_steps"],
                "returncode": manifest.get("returncode"),
                "screenshots": manifest.get("screenshots", []),
            }
        )
    return {
        "schema": "agentnavigator_spa_pilot_arm_index_v1",
        "arm": arm,
        "task_count": len(task_index),
        "tasks": task_index,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benchmark-manifest", type=Path)
    parser.add_argument("--arm", choices=["no_graph", "auto_bfs"])
    parser.add_argument("--spec-path", type=Path)
    parser.add_argument("--device", default=os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554"))
    parser.add_argument("--nav-benchmark-script", default=None,
                        help="Path to the project's nav benchmark runner (or set NAV_BENCHMARK_SCRIPT).")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "results" / "spa_bench_pilot")
    parser.add_argument("--include-tasks", nargs="*")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adb-path", default="adb")
    parser.add_argument("--ui-tars-url", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--backbone", choices=["ui_tars", "qwen_vl", "qwen_vl_local"], default="ui_tars")
    parser.add_argument("--keep-steps", type=int, default=3)
    parser.add_argument("--sim-floor", type=float, default=0.60)
    parser.add_argument("--coord-max-side", type=int)
    parser.add_argument("--per-task-timeout", type=int, default=0, help="Seconds; 0 disables timeout.")
    parser.add_argument("--tap-home-after-launch", action="store_true")
    parser.add_argument(
        "--tap-home-apps",
        nargs="*",
        default=["chrome"],
        help="Apps that should receive --tap-home-after-launch automatically.",
    )
    parser.add_argument("--clear-between-tasks", action="store_true")
    parser.add_argument("--task-completion", action="store_true")
    parser.add_argument("--full-task-prompt", action="store_true",
                        help="Use the full SPA-Bench task description as the agent prompt.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-run-nav-args",
        nargs=argparse.REMAINDER,
        help="Extra args appended verbatim to the nav benchmark runner after '--'.",
    )
    args = parser.parse_args()

    nav_script = nav_benchmark_script(args.nav_benchmark_script)
    arm, specs = load_specs(args.benchmark_manifest, args.arm, args.spec_path)
    specs = filter_specs(specs, args.include_tasks, args.limit)
    if not specs:
        raise SystemExit("No tasks selected.")

    arm_out_root = resolve_repo_path(str(args.out_root / arm))
    arm_out_root.mkdir(parents=True, exist_ok=True)
    manifests = []
    for spec in specs:
        task_id = spec["spa_task_identifier"]
        print(f"\n=== {arm} :: {task_id} ===")
        manifest = run_one_spec(spec, args, arm_out_root, nav_script)
        manifests.append(manifest)
        print(
            f"  exported {len(manifest.get('screenshots', []))} screenshots -> "
            f"{arm_out_root / task_id}"
        )

    index = build_index(arm, specs, manifests)
    index_path = arm_out_root / "arm_index.json"
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote arm index: {index_path}")


if __name__ == "__main__":
    main()
