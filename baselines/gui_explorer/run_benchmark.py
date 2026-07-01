"""
GUI-Explorer benchmark adapter.

Runs the GUI-Explorer agent (ACL'25) on our navigation tasks using their
pre-built knowledge base and RAG retrieval server. The upstream agent code is
loaded from an external checkout (not vendored); only orchestration, emulator
control, and trajectory logging live here.

Required environment (see baselines/.env.example):
  GUI_EXPLORER_ROOT      Path to an upstream GUI-explorer checkout.
  TAGNAV_TASKS_ROOT      Root holding per-app navigation task folders
                         (or pass --tasks-root). Each app has a task dir whose
                         subfolders each contain a prompts.json.
  ANDROID_DEVICE_SERIAL  adb serial of the running emulator/device
                         (or pass --device).
  GUI_EXPLORER_RAG_URL   Where the upstream retrieval server is reachable
                         (consumed by the upstream agent's own config). The
                         server + the shipped knowledge base must be running.

Prerequisites:
  1. The GUI-Explorer retrieval server is running and reachable at
     GUI_EXPLORER_RAG_URL (start it from the upstream checkout; it serves the
     shipped knowledge base — exploration is not re-run).
  2. An Android emulator/device is running with the target app installed.
  3. GUI_EXPLORER_ROOT points at the upstream checkout.

Usage:
  python baselines/gui_explorer/run_benchmark.py \
      --app clock --device emulator-5554 \
      --max-steps 15 --out-dir results/gui_explorer/clock
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

os.environ["UIAUTOMATOR2_LAUNCH_TIMEOUT"] = "60"

# Repo root = baselines/gui_explorer/<this file> -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_dir(env_name: str, *, required: bool = True) -> Path | None:
    """Resolve a directory from an env var (absolute, or relative to the repo root)."""
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        if required:
            raise SystemExit(
                f"{env_name} is not set. Copy baselines/.env.example to baselines/.env "
                f"and set {env_name} (see baselines/README.md)."
            )
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if required and not path.exists():
        raise SystemExit(f"{env_name} points to a missing path: {path}")
    return path


GUI_EXPLORER_ROOT = _resolve_dir("GUI_EXPLORER_ROOT")
sys.path.insert(0, str(GUI_EXPLORER_ROOT))

APP_REGISTRY = {
    "airbnb": {
        "package": "com.airbnb.android",
        "activity": "com.airbnb.android.feat.homescreen.HomeActivity",
        "tasks_dir": "airbnb_26.06",
    },
    "amazon": {
        "package": "com.amazon.mShop.android.shopping",
        "activity": "com.amazon.mShop.home.HomeActivity",
        "tasks_dir": "amazon_32.3.0.100",
    },
    "calendar": {
        "package": "com.google.android.calendar",
        "activity": None,
        "tasks_dir": "calendar_2026.05.0-864040481-release",
    },
    "chrome": {
        "package": "com.android.chrome",
        "activity": "com.google.android.apps.chrome.Main",
        "tasks_dir": "chrome_144.0.7559.132",
    },
    "clock": {
        "package": "com.google.android.deskclock",
        "activity": "com.android.deskclock.DeskClock",
        "tasks_dir": "clock_8.5",
    },
    "ebay": {
        "package": "com.ebay.mobile",
        "activity": ".home.impl.main.MainActivity",
        "tasks_dir": "ebay_6.242.0.2",
    },
    "google_maps": {
        "package": "com.google.android.apps.maps",
        "activity": "com.google.android.maps.MapsActivity",
        "tasks_dir": "google_maps_26.06.01.863982022",
    },
    "instagram": {
        "package": "com.instagram.android",
        "activity": ".activity.MainTabActivity",
        "tasks_dir": "instagram_416.0.0.47.66",
    },
    "linkedin": {
        "package": "com.linkedin.android",
        "activity": ".authenticator.LaunchActivityDefault",
        "tasks_dir": "linkedin_4.1.1168",
    },
    "settings": {
        "package": "com.android.settings",
        "activity": ".Settings",
        "tasks_dir": "settings_16",
    },
    "tiktok": {
        "package": "com.zhiliaoapp.musically",
        "activity": "com.ss.android.ugc.aweme.splash.SplashActivity",
        "tasks_dir": "tiktok_43.8.3",
    },
    "youtube": {
        "package": "com.google.android.youtube",
        "activity": "com.google.android.youtube.HomeActivity",
        "tasks_dir": "youtube_21.05.264",
    },
}


def adb_input_escape(text: str) -> str:
    """Escape text for `adb shell input text`.

    Android's `input text` treats `%s` as a space. Percent-encode the rest so
    simple search/form text has a reliable fallback when uiautomator2 send_keys
    fails.
    """
    return quote(text, safe="").replace("%20", "%s")


def patch_gui_explorer_text_input(device_serial: str):
    """Patch GUI-Explorer's Device.input_text with an ADB fallback.

    We avoid editing the external GUI-Explorer checkout. The patch is applied at
    runtime inside this adapter and preserves their uiautomator2 path first.
    """
    try:
        from utils.device import Device  # type: ignore
    except Exception as exc:
        print(f"WARNING: could not patch GUI-Explorer text input: {exc}")
        return

    original = Device.input_text

    def input_text_with_adb_fallback(self, text: str, smart_enter=True, clear_first=True):
        try:
            return original(self, text, smart_enter=smart_enter, clear_first=clear_first)
        except Exception as exc:
            print(f"WARNING: uiautomator2 send_keys failed, using adb input fallback: {exc}")
            if clear_first:
                subprocess.run(
                    ["adb", "-s", device_serial, "shell", "input", "keyevent", "KEYCODE_MOVE_END"],
                    capture_output=True,
                )
                for _ in range(80):
                    subprocess.run(
                        ["adb", "-s", device_serial, "shell", "input", "keyevent", "KEYCODE_DEL"],
                        capture_output=True,
                    )
            escaped = adb_input_escape(text)
            subprocess.run(
                ["adb", "-s", device_serial, "shell", "input", "text", escaped],
                capture_output=True,
                check=False,
            )
            if smart_enter:
                subprocess.run(
                    ["adb", "-s", device_serial, "shell", "input", "keyevent", "KEYCODE_ENTER"],
                    capture_output=True,
                )
            return None

    Device.input_text = input_text_with_adb_fallback


def force_stop_and_launch(device_serial: str, package: str, activity: str | None):
    """Force-stop and relaunch the app."""
    subprocess.run(
        ["adb", "-s", device_serial, "shell", "am", "force-stop", package],
        capture_output=True,
    )
    time.sleep(1)
    if activity:
        component = f"{package}/{activity}"
        subprocess.run(
            ["adb", "-s", device_serial, "shell", "am", "start", "-n", component],
            capture_output=True,
        )
    else:
        subprocess.run(
            [
                "adb", "-s", device_serial, "shell",
                "monkey", "-p", package, "-c",
                "android.intent.category.LAUNCHER", "1",
            ],
            capture_output=True,
        )
    time.sleep(3)


def take_screenshot(device_serial: str, out_path: str) -> bool:
    """Capture a screenshot from the device."""
    try:
        subprocess.run(
            ["adb", "-s", device_serial, "shell", "screencap", "-p", "/sdcard/screen.png"],
            capture_output=True, check=True, timeout=10,
        )
        subprocess.run(
            ["adb", "-s", device_serial, "pull", "/sdcard/screen.png", out_path],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except Exception:
        return False


def run_gui_explorer_on_task(
    agent,
    task_prompt: str,
    max_steps: int,
    device_serial: str,
    package: str,
    activity: str | None,
    task_out_dir: Path,
):
    """Run the GUI-Explorer agent on a single task."""
    task_out_dir.mkdir(parents=True, exist_ok=True)

    force_stop_and_launch(device_serial, package, activity)

    finished_reason = "max_steps"
    steps_taken = 0

    try:
        step_datas = agent.run(
            task_goal=task_prompt,
            max_rounds=max_steps,
            step_interval=2.0,
            step_data_output_dir=str(task_out_dir),
        )
        steps_taken = len(step_datas)

        for i, sd in enumerate(step_datas):
            if sd.get("raw_screenshot") is not None:
                from PIL import Image
                img = Image.fromarray(sd["raw_screenshot"])
                img.save(str(task_out_dir / f"step_{i+1:03d}.png"))

            converted = sd.get("converted_action")
            if hasattr(converted, "action_type") and converted.action_type == "status":
                if converted.goal_status == "infeasible":
                    finished_reason = "infeasible"
                else:
                    finished_reason = "finished_action"

    except Exception as e:
        finished_reason = f"error: {e}"
        print(f"  Error: {e}")

    take_screenshot(device_serial, str(task_out_dir / "agent_screenshot.png"))

    summary = {
        "task_prompt": task_prompt,
        "steps_taken": steps_taken,
        "finished_reason": finished_reason,
        "max_steps": max_steps,
        "agent": "gui_explorer",
        "backbone": "gpt-4o",
    }
    with open(task_out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run GUI-Explorer agent on our benchmark tasks")
    parser.add_argument("--app", required=True, choices=list(APP_REGISTRY.keys()))
    parser.add_argument(
        "--device",
        default=os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554"),
        help="adb device serial (default: $ANDROID_DEVICE_SERIAL or emulator-5554)",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tasks-root",
        default=os.environ.get("TAGNAV_TASKS_ROOT"),
        help="Root holding per-app task folders (default: $TAGNAV_TASKS_ROOT)",
    )
    args = parser.parse_args()

    if not args.tasks_root:
        raise SystemExit(
            "Task root not set. Pass --tasks-root or set TAGNAV_TASKS_ROOT "
            "(see baselines/.env.example)."
        )

    app_info = APP_REGISTRY[args.app]
    package = app_info["package"]
    activity = app_info["activity"]
    tasks_dir = Path(args.tasks_root) / app_info["tasks_dir"]

    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])
    print(f"Found {len(task_dirs)} tasks for {args.app} in {tasks_dir}")

    from MLLM_Agent.GUI_explorer import GUI_explorer
    patch_gui_explorer_text_input(args.device)
    agent = GUI_explorer(device_serial=args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for i, task_dir in enumerate(task_dirs):
        prompts_file = task_dir / "prompts.json"
        if not prompts_file.exists():
            print(f"Skipping {task_dir.name}: no prompts.json")
            continue

        with open(prompts_file) as f:
            prompts_data = json.load(f)

        prompt = prompts_data["prompts"][0]
        task_name = task_dir.name

        task_out = out_dir / task_name
        existing_summary = task_out / "summary.json"
        if existing_summary.exists():
            with open(existing_summary) as sf:
                s = json.load(sf)
            s["task_id"] = task_name
            all_summaries.append(s)
            print(f"\n[{i+1}/{len(task_dirs)}] {task_name}: SKIPPED (already done: {s['finished_reason']})")
            continue

        print(f"\n[{i+1}/{len(task_dirs)}] {task_name}: {prompt}")

        summary = run_gui_explorer_on_task(
            agent=agent,
            task_prompt=prompt,
            max_steps=args.max_steps,
            device_serial=args.device,
            package=package,
            activity=activity,
            task_out_dir=task_out,
        )
        summary["task_id"] = task_name
        all_summaries.append(summary)
        print(f"  -> {summary['finished_reason']} in {summary['steps_taken']} steps")

    with open(out_dir / "all_summaries.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    finished = sum(1 for s in all_summaries if s["finished_reason"] == "finished_action")
    total = len(all_summaries)
    print(f"\n=== Results: {finished}/{total} ({100*finished/total:.0f}%) finished_action ===")


if __name__ == "__main__":
    main()
