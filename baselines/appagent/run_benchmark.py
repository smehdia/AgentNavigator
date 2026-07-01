"""
AppAgent benchmark adapter.

Runs the AppAgent (CHI 2025) deployment policy on our navigation benchmark
tasks. Keeps AppAgent's original action space and prompts, but removes the
interactive CLI prompts from its stock run.py / task_executor.py.

Required environment (see baselines/.env.example):
  APPAGENT_ROOT         path to an upstream AppAgent checkout
                        (e.g. baselines/external/AppAgent)
  TAGNAV_TASKS_ROOT     root holding per-app navigation task folders
                        (or pass --tasks-root). Each app has a subdir named by
                        APP_REGISTRY[app]["tasks_dir"]; each task is a folder
                        containing prompts.json -> {"prompts": ["..."]}
  ANDROID_DEVICE_SERIAL adb serial of the target device (or pass --device)

AppAgent's model + API keys are read from the upstream AppAgent config.yaml
(MODEL / OPENAI_API_KEY / DASHSCOPE_API_KEY / ...). This adapter does not store
any keys.

Example:
  export APPAGENT_ROOT=baselines/external/AppAgent
  export TAGNAV_TASKS_ROOT=/path/to/navigation_tasks
  python baselines/appagent/run_benchmark.py \
      --app amazon --device emulator-5554 --max-steps 15 \
      --out-dir results/appagent/amazon
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Repo root = three levels up from baselines/appagent/run_benchmark.py
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_appagent_root() -> Path:
    raw = os.environ.get("APPAGENT_ROOT")
    if not raw:
        raise SystemExit(
            "APPAGENT_ROOT is not set. Copy baselines/.env.example to baselines/.env, "
            "clone AppAgent into baselines/external/AppAgent, and export APPAGENT_ROOT. "
            "See baselines/appagent/README.md."
        )
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise SystemExit(f"AppAgent root not found: {p} (check APPAGENT_ROOT)")
    return p


# Resolved in main() before first use.
APPAGENT_ROOT: Path = None  # type: ignore[assignment]

APP_REGISTRY: dict[str, dict[str, str | None]] = {
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


def import_appagent_modules() -> dict[str, Any]:
    if not APPAGENT_ROOT.exists():
        raise FileNotFoundError(f"AppAgent root not found: {APPAGENT_ROOT}")

    os.chdir(APPAGENT_ROOT)
    sys.path.insert(0, str(APPAGENT_ROOT / "scripts"))
    sys.path.insert(0, str(APPAGENT_ROOT))

    import prompts  # type: ignore
    from and_controller import AndroidController, list_all_devices, traverse_tree  # type: ignore
    from config import load_config  # type: ignore
    from model import OpenAIModel, QwenModel, parse_explore_rsp, parse_grid_rsp  # type: ignore
    from utils import draw_bbox_multi, draw_grid, print_with_color  # type: ignore

    return {
        "prompts": prompts,
        "AndroidController": AndroidController,
        "list_all_devices": list_all_devices,
        "traverse_tree": traverse_tree,
        "load_config": load_config,
        "OpenAIModel": OpenAIModel,
        "QwenModel": QwenModel,
        "parse_explore_rsp": parse_explore_rsp,
        "parse_grid_rsp": parse_grid_rsp,
        "draw_bbox_multi": draw_bbox_multi,
        "draw_grid": draw_grid,
        "print_with_color": print_with_color,
    }


def launch_app(device: str, package: str, activity: str | None) -> None:
    subprocess.run(["adb", "-s", device, "shell", "am", "force-stop", package], capture_output=True)
    time.sleep(1)
    if activity:
        component = f"{package}/{activity}"
        subprocess.run(["adb", "-s", device, "shell", "am", "start", "-n", component], capture_output=True)
    else:
        subprocess.run(
            [
                "adb",
                "-s",
                device,
                "shell",
                "monkey",
                "-p",
                package,
                "-c",
                "android.intent.category.LAUNCHER",
                "1",
            ],
            capture_output=True,
        )
    time.sleep(3)


def take_screenshot(device: str, out_path: Path) -> bool:
    try:
        subprocess.run(
            ["adb", "-s", device, "shell", "screencap", "-p", "/sdcard/screen.png"],
            capture_output=True,
            check=True,
            timeout=10,
        )
        subprocess.run(
            ["adb", "-s", device, "pull", "/sdcard/screen.png", str(out_path)],
            capture_output=True,
            check=True,
            timeout=10,
        )
        return True
    except Exception:
        return False


def make_model(mods: dict[str, Any], configs: dict[str, Any], model_override: str | None):
    model_kind = model_override or configs["MODEL"]
    if model_kind.lower() == "openai":
        return mods["OpenAIModel"](
            base_url=configs["OPENAI_API_BASE"],
            api_key=configs["OPENAI_API_KEY"],
            model=configs["OPENAI_API_MODEL"],
            temperature=configs["TEMPERATURE"],
            max_tokens=configs["MAX_TOKENS"],
        )
    if model_kind.lower() == "qwen":
        return mods["QwenModel"](api_key=configs["DASHSCOPE_API_KEY"], model=configs["QWEN_MODEL"])
    raise ValueError(f"Unsupported AppAgent model: {model_kind}")


def build_ui_doc(elem_list: list[Any], docs_dir: Path) -> str:
    ui_doc = ""
    for i, elem in enumerate(elem_list):
        doc_path = docs_dir / f"{elem.uid}.txt"
        if not doc_path.exists():
            continue
        doc_content = ast.literal_eval(doc_path.read_text())
        ui_doc += f"Documentation of UI element labeled with the numeric tag '{i + 1}':\n"
        if doc_content["tap"]:
            ui_doc += f"This UI element is clickable. {doc_content['tap']}\n\n"
        if doc_content["text"]:
            ui_doc += (
                "This UI element can receive text input. The text input is used for the "
                f"following purposes: {doc_content['text']}\n\n"
            )
        if doc_content["long_press"]:
            ui_doc += f"This UI element is long clickable. {doc_content['long_press']}\n\n"
        if doc_content["v_swipe"]:
            ui_doc += (
                "This element can be swiped directly without tapping. You can swipe vertically "
                f"on this UI element. {doc_content['v_swipe']}\n\n"
            )
        if doc_content["h_swipe"]:
            ui_doc += (
                "This element can be swiped directly without tapping. You can swipe horizontally "
                f"on this UI element. {doc_content['h_swipe']}\n\n"
            )
    if not ui_doc:
        return ""
    return (
        "\nYou also have access to the following documentations that describes the functionalities of UI "
        "elements you can interact on the screen. These docs are crucial for you to determine the target of your "
        "next action. You should always prioritize these documented elements for interaction:"
        + ui_doc
    )


def area_to_xy(area: int, subarea: str, rows: int, cols: int, width: int, height: int) -> tuple[int, int]:
    area -= 1
    row, col = area // cols, area % cols
    x_0, y_0 = col * (width // cols), row * (height // rows)
    offsets = {
        "top-left": (1, 1),
        "top": (2, 1),
        "top-right": (3, 1),
        "left": (1, 2),
        "center": (2, 2),
        "right": (3, 2),
        "bottom-left": (1, 3),
        "bottom": (2, 3),
        "bottom-right": (3, 3),
    }
    x_mul, y_mul = offsets.get(subarea, offsets["center"])
    return x_0 + (width // cols) * x_mul // 4, y_0 + (height // rows) * y_mul // 4


def run_task(
    *,
    mods: dict[str, Any],
    configs: dict[str, Any],
    mllm: Any,
    app_name: str,
    package: str,
    activity: str | None,
    device: str,
    task_id: str,
    task_prompt: str,
    task_out_dir: Path,
    max_steps: int,
    docs_mode: str,
    request_interval: float,
) -> dict[str, Any]:
    task_out_dir.mkdir(parents=True, exist_ok=True)
    launch_app(device, package, activity)

    controller = mods["AndroidController"](device)
    width, height = controller.get_device_size()
    if not width or not height:
        raise RuntimeError(f"Invalid device size for {device}: {width}x{height}")

    app_dir = APPAGENT_ROOT / "apps" / app_name
    docs_dir = app_dir / ("auto_docs" if docs_mode == "auto" else "demo_docs")
    use_docs = docs_mode != "none" and docs_dir.exists()
    prompts = mods["prompts"]

    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = task_out_dir / f"log_{task_id}_{timestamp}.jsonl"

    round_count = 0
    steps: list[dict[str, Any]] = []
    last_act = "None"
    task_complete = False
    grid_on = False
    rows, cols = 0, 0
    finished_reason = "max_steps"

    while round_count < max_steps:
        round_count += 1
        prefix = f"{task_id}_{round_count:03d}"
        screenshot_path = controller.get_screenshot(prefix, str(task_out_dir))
        xml_path = controller.get_xml(prefix, str(task_out_dir))
        if screenshot_path == "ERROR" or xml_path == "ERROR":
            finished_reason = "capture_error"
            break

        elem_list: list[Any] = []
        if grid_on:
            rows, cols = mods["draw_grid"](screenshot_path, str(task_out_dir / f"{prefix}_grid.png"))
            image = str(task_out_dir / f"{prefix}_grid.png")
            prompt = prompts.task_template_grid
        else:
            clickable_list: list[Any] = []
            focusable_list: list[Any] = []
            mods["traverse_tree"](xml_path, clickable_list, "clickable", True)
            mods["traverse_tree"](xml_path, focusable_list, "focusable", True)
            elem_list = clickable_list.copy()
            for elem in focusable_list:
                bbox = elem.bbox
                center = (bbox[0][0] + bbox[1][0]) // 2, (bbox[0][1] + bbox[1][1]) // 2
                close = False
                for e in clickable_list:
                    e_bbox = e.bbox
                    e_center = (e_bbox[0][0] + e_bbox[1][0]) // 2, (e_bbox[0][1] + e_bbox[1][1]) // 2
                    dist = ((center[0] - e_center[0]) ** 2 + (center[1] - e_center[1]) ** 2) ** 0.5
                    if dist <= configs["MIN_DIST"]:
                        close = True
                        break
                if not close:
                    elem_list.append(elem)

            labeled_path = task_out_dir / f"{prefix}_labeled.png"
            mods["draw_bbox_multi"](screenshot_path, str(labeled_path), elem_list, dark_mode=configs["DARK_MODE"])
            image = str(labeled_path)
            if use_docs:
                prompt = re.sub(r"<ui_document>", build_ui_doc(elem_list, docs_dir), prompts.task_template)
            else:
                prompt = re.sub(r"<ui_document>", "", prompts.task_template)

        prompt = re.sub(r"<task_description>", task_prompt, prompt)
        prompt = re.sub(r"<last_act>", last_act, prompt)
        status, rsp = mllm.get_model_response(prompt, [image])

        step_record = {
            "step": round_count,
            "image": image,
            "raw_screenshot": screenshot_path,
            "xml": xml_path,
            "response": rsp,
            "status": status,
        }
        with log_path.open("a") as logfile:
            logfile.write(json.dumps(step_record) + "\n")
        steps.append(step_record)

        if not status:
            finished_reason = f"model_error: {rsp}"
            break

        res = mods["parse_grid_rsp"](rsp) if grid_on else mods["parse_explore_rsp"](rsp)
        act_name = res[0]
        if act_name == "FINISH":
            task_complete = True
            finished_reason = "finished_action"
            break
        if act_name == "ERROR":
            finished_reason = "parse_error"
            break

        if act_name != "grid":
            last_act = res[-1]
            res = res[:-1]

        ret = None
        if act_name == "tap":
            _, area = res
            tl, br = elem_list[area - 1].bbox
            ret = controller.tap((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
        elif act_name == "text":
            _, input_str = res
            ret = controller.text(input_str)
        elif act_name == "long_press":
            _, area = res
            tl, br = elem_list[area - 1].bbox
            ret = controller.long_press((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2)
        elif act_name == "swipe":
            _, area, swipe_dir, dist = res
            tl, br = elem_list[area - 1].bbox
            ret = controller.swipe((tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2, swipe_dir, dist)
        elif act_name == "grid":
            grid_on = True
        elif act_name in {"tap_grid", "long_press_grid"}:
            _, area, subarea = res
            x, y = area_to_xy(area, subarea, rows, cols, width, height)
            ret = controller.tap(x, y) if act_name == "tap_grid" else controller.long_press(x, y)
        elif act_name == "swipe_grid":
            _, start_area, start_subarea, end_area, end_subarea = res
            start = area_to_xy(start_area, start_subarea, rows, cols, width, height)
            end = area_to_xy(end_area, end_subarea, rows, cols, width, height)
            ret = controller.swipe_precise(start, end)

        if ret == "ERROR":
            finished_reason = f"action_error: {act_name}"
            break
        if act_name != "grid":
            grid_on = False
        time.sleep(request_interval)

    agent_screenshot = task_out_dir / "agent_screenshot.png"
    take_screenshot(device, agent_screenshot)

    summary = {
        "task_id": task_id,
        "task_prompt": task_prompt,
        "steps_taken": len(steps),
        "finished_reason": "finished_action" if task_complete else finished_reason,
        "max_steps": max_steps,
        "agent": "appagent",
        "backbone": configs.get("QWEN_MODEL") if configs.get("MODEL") == "Qwen" else configs.get("OPENAI_API_MODEL"),
        "docs_mode": docs_mode,
        "docs_used": use_docs,
        "agent_screenshot": str(agent_screenshot),
        "log_path": str(log_path),
    }
    (task_out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AppAgent on our navigation benchmark tasks")
    parser.add_argument("--app", required=True, choices=sorted(APP_REGISTRY))
    parser.add_argument(
        "--device",
        default=os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554"),
        help="adb device serial (default: $ANDROID_DEVICE_SERIAL or emulator-5554)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--tasks-root",
        default=os.environ.get("TAGNAV_TASKS_ROOT"),
        help="Root of per-app navigation task folders (default: $TAGNAV_TASKS_ROOT)",
    )
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--request-interval", type=float, default=None)
    parser.add_argument("--docs", choices=["none", "auto", "demo"], default="none")
    parser.add_argument("--model", choices=["OpenAI", "Qwen"], default=None, help="Override AppAgent config MODEL")
    parser.add_argument("--task-id", action="append", default=None, help="Run only this task id; repeatable")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected tasks")
    parser.add_argument("--rerun", action="store_true", help="Rerun tasks even if summary.json exists")
    args = parser.parse_args()

    if not args.tasks_root:
        raise SystemExit("No tasks root. Pass --tasks-root or set TAGNAV_TASKS_ROOT (see baselines/.env.example).")

    global APPAGENT_ROOT
    APPAGENT_ROOT = resolve_appagent_root()

    mods = import_appagent_modules()
    configs = mods["load_config"](str(APPAGENT_ROOT / "config.yaml"))
    if args.model:
        configs["MODEL"] = args.model
    request_interval = args.request_interval
    if request_interval is None:
        request_interval = float(configs.get("REQUEST_INTERVAL", 3))

    devices = mods["list_all_devices"]()
    if args.device not in devices:
        raise SystemExit(f"Device {args.device} not found. Attached devices: {devices}")

    app_info = APP_REGISTRY[args.app]
    tasks_dir = Path(args.tasks_root) / str(app_info["tasks_dir"])
    if not tasks_dir.exists():
        raise SystemExit(f"Tasks directory not found: {tasks_dir}")

    task_dirs = sorted([d for d in tasks_dir.iterdir() if d.is_dir()])
    if args.task_id:
        wanted = set(args.task_id)
        task_dirs = [d for d in task_dirs if d.name in wanted]
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]
    if not task_dirs:
        raise SystemExit("No benchmark tasks selected")

    mllm = make_model(mods, configs, args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, Any]] = []
    print(f"Found {len(task_dirs)} tasks for {args.app} in {tasks_dir}")
    print(f"Running AppAgent with MODEL={configs['MODEL']}, docs={args.docs}, max_steps={args.max_steps}")

    for i, task_dir in enumerate(task_dirs, 1):
        prompts_file = task_dir / "prompts.json"
        if not prompts_file.exists():
            print(f"[{i}/{len(task_dirs)}] {task_dir.name}: SKIPPED (no prompts.json)")
            continue
        prompt = json.loads(prompts_file.read_text())["prompts"][0]
        task_out_dir = out_dir / task_dir.name
        existing_summary = task_out_dir / "summary.json"
        if existing_summary.exists() and not args.rerun:
            summary = json.loads(existing_summary.read_text())
            all_summaries.append(summary)
            print(f"[{i}/{len(task_dirs)}] {task_dir.name}: SKIPPED ({summary['finished_reason']})")
            continue

        print(f"\n[{i}/{len(task_dirs)}] {task_dir.name}: {prompt}")
        try:
            summary = run_task(
                mods=mods,
                configs=configs,
                mllm=mllm,
                app_name=args.app,
                package=str(app_info["package"]),
                activity=app_info["activity"],
                device=args.device,
                task_id=task_dir.name,
                task_prompt=prompt,
                task_out_dir=task_out_dir,
                max_steps=args.max_steps,
                docs_mode=args.docs,
                request_interval=request_interval,
            )
        except Exception as exc:
            task_out_dir.mkdir(parents=True, exist_ok=True)
            take_screenshot(args.device, task_out_dir / "agent_screenshot.png")
            summary = {
                "task_id": task_dir.name,
                "task_prompt": prompt,
                "steps_taken": 0,
                "finished_reason": f"error: {exc}",
                "max_steps": args.max_steps,
                "agent": "appagent",
                "docs_mode": args.docs,
                "docs_used": False,
                "agent_screenshot": str(task_out_dir / "agent_screenshot.png"),
            }
            (task_out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        all_summaries.append(summary)
        print(f"  -> {summary['finished_reason']} in {summary['steps_taken']} steps")

    (out_dir / "all_summaries.json").write_text(json.dumps(all_summaries, indent=2))
    finished = sum(1 for s in all_summaries if s["finished_reason"] == "finished_action")
    total = len(all_summaries)
    print(f"\n=== AppAgent finished_action: {finished}/{total} ({100 * finished / max(total, 1):.1f}%) ===")


if __name__ == "__main__":
    main()
