#!/usr/bin/env python3
"""PG-Agent benchmark adapter for AgentNavigator navigation tasks.

This runner keeps PG-Agent as an external baseline instead of mixing its page
graph into AgentNavigator memory modes.  It handles our emulator reset, live
screenshot capture, localhost image serving, action conversion, and canonical
trajectory logging.  The actual PG-Agent policy is loaded from an external
checkout through either:

  --pg-entrypoint package.module:function
      The function is called once per step with a request dict and must return
      either a PG-Agent action string or a dict containing an action.

  --pg-command-template "python ... --request {request_json}"
      The command is run once per step.  It receives a JSON request path through
      the formatted {request_json} placeholder and must print either JSON or an
      action string.

  --pg-server-url http://127.0.0.1:18000/next_action
      The request dict is POSTed to a remote PG-Agent service.  This is the
      faithful setup path for running PG-Agent policy/graph services on
      a remote GPU host while keeping emulator control and trajectory logging local.

Example:
  python scripts/pg_agent_benchmark.py \\
      --app clock --device emulator-5554 --max-steps 3 \\
      --out-dir results/pg_agent_smoke/clock \\
      --pg-agent-root "$PG_AGENT_ROOT" \\
      --pg-entrypoint pg_agent_adapter:next_action
"""

from __future__ import annotations

import argparse
import base64
import os
import importlib
import json
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Optional
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

BASELINES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BASELINES_ROOT.parent
sys.path.insert(0, str(BASELINES_ROOT))

from _common.nav_core import (  # noqa: E402  shared adb driver + app registry (see baselines/_common/)
    APP_REGISTRY,
    AdbDriver,
    discover_prompt,
    execute_action,
)


DEFAULT_PG_AGENT_ROOT = Path(os.environ.get("PG_AGENT_ROOT", str(BASELINES_ROOT / "external" / "PG-Agent")))


@dataclass
class PGAction:
    kind: str
    x: Optional[float] = None
    y: Optional[float] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    raw: Any = None


class ScreenshotServer:
    """Serve screenshots with PG-Agent's expected localhost image URLs."""

    def __init__(self, root: Path, host: str = "127.0.0.1", port: int = 6666):
        self.root = root.resolve()
        self.host = host
        self.port = port
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.thread: Optional[threading.Thread] = None

    def start(self) -> str:
        handler = self._handler()
        last_error: Optional[Exception] = None
        for port in range(self.port, self.port + 20):
            try:
                self.httpd = ThreadingHTTPServer((self.host, port), handler)
                self.port = port
                break
            except OSError as exc:
                last_error = exc
        if self.httpd is None:
            raise RuntimeError(f"could not start image server: {last_error}")
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return f"http://{self.host}:{self.port}"

    def _handler(self) -> type[SimpleHTTPRequestHandler]:
        root = str(self.root)

        class QuietHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=root, **kwargs)

            def log_message(self, format: str, *args: Any) -> None:
                return

        return QuietHandler

    def stop(self) -> None:
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None


def _loads_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def png_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_pg_action(response: Any) -> PGAction:
    """Parse common PG-Agent action forms.

    Supported examples:
      Click(123, 456)
      Type("hello")
      Scroll("down")
      Back
      Home
      Complete
      {"action": "Click", "x": 123, "y": 456}
      {"action_type": "Type", "text": "hello"}
    """
    raw = response
    response = _loads_maybe_json(response)
    if isinstance(response, dict):
        execution = response.get("execution")
        if isinstance(execution, str) and execution.strip():
            try:
                return parse_pg_action(execution)
            except ValueError:
                pass
        nested_action = response.get("action")
        if isinstance(nested_action, dict):
            nested = dict(nested_action)
            nested.setdefault("raw_response", response)
            return parse_pg_action(nested)
        for key in ("action", "action_type", "type", "name"):
            if response.get(key):
                name = str(response[key]).strip()
                break
        else:
            name = str(response.get("raw_action") or response.get("prediction") or "")
        lowered = name.lower()
        if lowered in {"click", "tap", "long_press", "long-press", "longpress"}:
            x = response.get("x")
            y = response.get("y")
            if x is None or y is None:
                point = response.get("point") or response.get("coordinate") or response.get("coordinates")
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    x, y = point[0], point[1]
            kind = "long_press" if "long" in lowered else "click"
            return PGAction(kind, float(x), float(y), raw=raw)
        if lowered in {"type", "input", "text"}:
            text = response.get("text", response.get("content", response.get("value", "")))
            return PGAction("type", text=str(text), raw=raw)
        if lowered in {"scroll", "swipe"}:
            direction = str(response.get("direction") or response.get("dir") or "down")
            return PGAction("scroll", direction=direction.lower(), raw=raw)
        if lowered in {"back", "press_back"}:
            return PGAction("back", raw=raw)
        if lowered in {"home", "press_home"}:
            return PGAction("home", raw=raw)
        if lowered in {"complete", "finished", "finish", "done"}:
            return PGAction("complete", raw=raw)
        if lowered in {"impossible"}:
            return PGAction("impossible", raw=raw)
        if name:
            return parse_pg_action(name)

    text = str(response).strip()
    action_line = text
    block = re.search(r"###Action###\s*(.+)", text, re.I | re.S)
    if block:
        action_line = block.group(1).strip().splitlines()[0].strip()
    m = re.search(r"(?:Action|Next action)\s*:\s*(.+)", text, re.I)
    if m:
        action_line = m.group(1).strip()
    action_line = action_line.strip().strip("`")
    lower = action_line.lower()

    if lower.startswith(("click", "tap", "long_press", "long-press", "longpress")):
        m = re.search(r"[-+]?\d*\.?\d+\s*,\s*[-+]?\d*\.?\d+", action_line)
        if not m:
            raise ValueError(f"PG-Agent click missing coordinates: {text[:160]}")
        x_s, y_s = re.split(r"\s*,\s*", m.group(0), maxsplit=1)
        kind = "long_press" if lower.startswith(("long_press", "long-press", "longpress")) else "click"
        return PGAction(kind, float(x_s), float(y_s), raw=raw)
    if lower.startswith(("type", "input")):
        m = re.search(r"['\"]([^'\"]*)['\"]", action_line)
        if not m:
            m = re.search(r"\((.*)\)", action_line)
        if m:
            text_value = m.group(1)
        elif ":" in action_line:
            text_value = action_line.split(":", 1)[1].strip()
        else:
            text_value = ""
        return PGAction("type", text=text_value, raw=raw)
    if lower.startswith(("scroll", "swipe")):
        direction = "down"
        for candidate in ("up", "down", "left", "right"):
            if re.search(rf"\b{candidate}\b", lower):
                direction = candidate
                break
        return PGAction("scroll", direction=direction, raw=raw)
    if re.fullmatch(r"(press_)?back\(?\)?", lower):
        return PGAction("back", raw=raw)
    if re.fullmatch(r"(press_)?home\(?\)?", lower):
        return PGAction("home", raw=raw)
    if re.fullmatch(r"(complete|finish|finished|done)\(?\)?", lower):
        return PGAction("complete", raw=raw)
    if re.fullmatch(r"(impossible)\(?\)?", lower):
        return PGAction("impossible", raw=raw)
    raise ValueError(f"unsupported PG-Agent action: {text[:160]}")


def pg_action_to_agentnavigator(
    action: PGAction,
    width: int,
    height: int,
    coordinate_width: Optional[int] = None,
    coordinate_height: Optional[int] = None,
) -> str:
    if action.kind == "click":
        if action.x is None or action.y is None:
            raise ValueError("click action missing x/y")
        x, y = action.x, action.y
        if 0 <= x <= 1 and 0 <= y <= 1:
            x *= width
            y *= height
        else:
            if coordinate_width and coordinate_width > 0 and 0 <= x <= coordinate_width:
                x = x / coordinate_width * width
            if coordinate_height and coordinate_height > 0 and 0 <= y <= coordinate_height:
                y = y / coordinate_height * height
        return f"click({int(round(x))}, {int(round(y))})"
    if action.kind == "long_press":
        if action.x is None or action.y is None:
            raise ValueError("long_press action missing x/y")
        x, y = action.x, action.y
        if 0 <= x <= 1 and 0 <= y <= 1:
            x *= width
            y *= height
        else:
            if coordinate_width and coordinate_width > 0 and 0 <= x <= coordinate_width:
                x = x / coordinate_width * width
            if coordinate_height and coordinate_height > 0 and 0 <= y <= coordinate_height:
                y = y / coordinate_height * height
        return f"long_press({int(round(x))}, {int(round(y))})"
    if action.kind == "type":
        escaped = (action.text or "").replace("\\", "\\\\").replace('"', '\\"')
        return f'type(content="{escaped}")'
    if action.kind == "scroll":
        return f"scroll({action.direction or 'down'})"
    if action.kind == "back":
        return "press_back()"
    if action.kind == "complete":
        return "finished()"
    if action.kind == "impossible":
        return "finished()"
    if action.kind == "home":
        return "press_home()"
    raise ValueError(f"unsupported action kind: {action.kind}")


def execute_pg_action(
    driver: AdbDriver,
    action: PGAction,
    width: int,
    height: int,
    coordinate_width: Optional[int] = None,
    coordinate_height: Optional[int] = None,
) -> dict[str, Any]:
    if action.kind == "impossible":
        return {"kind": "impossible", "finished": True, "converted_action": "impossible()"}
    if action.kind == "home":
        driver.home()
        return {"kind": "press_home", "finished": False}
    converted = pg_action_to_agentnavigator(
        action,
        width,
        height,
        coordinate_width=coordinate_width,
        coordinate_height=coordinate_height,
    )
    result = execute_action(driver, converted, width, height, max_side=0)
    result["converted_action"] = converted
    return result


class PGAgentBackend:
    def next_action(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError

    def reset(self) -> None:
        return


class EntrypointBackend(PGAgentBackend):
    def __init__(self, root: Path, entrypoint: str):
        if not root.exists():
            raise FileNotFoundError(f"PG-Agent root not found: {root}")
        sys.path.insert(0, str(root.resolve()))
        module_name, sep, func_name = entrypoint.partition(":")
        if not sep:
            raise ValueError("--pg-entrypoint must be package.module:function")
        module = importlib.import_module(module_name)
        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(f"PG-Agent entrypoint is not callable: {entrypoint}")
        self.func: Callable[[dict[str, Any]], Any] = func
        reset = getattr(module, "reset", None)
        self.reset_func = reset if callable(reset) else None

    def next_action(self, request: dict[str, Any]) -> Any:
        return self.func(request)

    def reset(self) -> None:
        if self.reset_func:
            self.reset_func()


class CommandBackend(PGAgentBackend):
    def __init__(self, command_template: str):
        self.command_template = command_template

    def next_action(self, request: dict[str, Any]) -> Any:
        request_path = Path(request["task_out_dir"]) / f"step_{request['step']:03d}_pg_request.json"
        request_path.write_text(json.dumps(request, indent=2))
        cmd = self.command_template.format(request_json=str(request_path))
        proc = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"PG-Agent command failed: {cmd}")
        return _loads_maybe_json(proc.stdout.strip())


class HTTPBackend(PGAgentBackend):
    def __init__(self, server_url: str, timeout: float = 300.0):
        self.server_url = server_url
        self.timeout = timeout

    def next_action(self, request: dict[str, Any]) -> Any:
        body = json.dumps(request).encode("utf-8")
        http_request = urlrequest.Request(
            self.server_url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(http_request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"PG-Agent HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"PG-Agent HTTP request failed: {exc.reason}") from exc
        return _loads_maybe_json(text.strip())


def make_backend(args: argparse.Namespace) -> PGAgentBackend:
    if args.pg_server_url:
        return HTTPBackend(args.pg_server_url, timeout=args.pg_http_timeout)
    if args.pg_entrypoint:
        return EntrypointBackend(Path(args.pg_agent_root), args.pg_entrypoint)
    if args.pg_command_template:
        return CommandBackend(args.pg_command_template)
    raise SystemExit(
        "Provide --pg-server-url, --pg-entrypoint, or --pg-command-template. "
        "This adapter intentionally does not vendor PG-Agent into AgentNavigator."
    )


def launch_app(driver: AdbDriver, package: str, activity: Optional[str], clear_data: bool) -> None:
    driver.home()
    if clear_data:
        driver.clear_data(package)
    else:
        driver.force_stop(package)
    time.sleep(1.0)
    driver.launch(package, activity)
    time.sleep(3.5)


def run_one_task(
    task_dir: Path,
    prompt: str,
    backend: PGAgentBackend,
    driver: AdbDriver,
    package: str,
    activity: Optional[str],
    out_root: Path,
    image_base_url: str,
    max_steps: int,
    graph_dir: Optional[Path],
    pg_agent_root: Path,
    guideline_mode: str,
    clear_data: bool,
    sleep_after_action: float,
    repeat_action_stop: int,
    pg_coordinate_width: Optional[int],
    pg_coordinate_height: Optional[int],
) -> dict[str, Any]:
    import cv2

    task_out = out_root / task_dir.name
    task_out.mkdir(parents=True, exist_ok=True)
    trajectory_path = task_out / "trajectory.jsonl"

    launch_app(driver, package, activity, clear_data=clear_data)
    backend.reset()

    history: list[dict[str, Any]] = []
    steps_taken = 0
    finished_reason = "max_steps"
    last_converted_action: Optional[str] = None
    repeated_action_count = 0

    with trajectory_path.open("w") as tlog:
        for step in range(1, max_steps + 1):
            try:
                screenshot = driver.screenshot_bgr()
            except Exception as exc:
                finished_reason = "screencap_fail"
                tlog.write(json.dumps({"step": step, "error": f"screencap_fail: {exc}"}) + "\n")
                break

            height, width = screenshot.shape[:2]
            shot_rel = Path(task_dir.name) / f"step_{step:03d}.png"
            shot_path = out_root / shot_rel
            cv2.imwrite(str(shot_path), screenshot)
            shot_url = f"{image_base_url}/{shot_rel.as_posix()}"

            request = {
                "task": prompt,
                "task_id": task_dir.name,
                "graph_domain": task_dir.parent.name,
                "step": step,
                "screenshot_path": str(shot_path),
                "screenshot_url": shot_url,
                "screenshot_data_url": png_data_url(shot_path),
                "screen_width": width,
                "screen_height": height,
                "history": history,
                "graph_dir": str(graph_dir) if graph_dir else None,
                "graph_path": str(graph_dir) if graph_dir else None,
                "guideline_mode": guideline_mode,
                "task_out_dir": str(task_out),
            }
            (task_out / f"step_{step:03d}_pg_request.json").write_text(json.dumps(request, indent=2))

            try:
                response = backend.next_action(request)
                (task_out / f"step_{step:03d}_pg_response.json").write_text(
                    json.dumps(response, indent=2, default=str)
                    if isinstance(response, (dict, list))
                    else json.dumps({"text": str(response)}, indent=2)
                )
                if isinstance(response, dict) and response.get("retrieved_guidelines") is not None:
                    (task_out / f"step_{step:03d}_guidelines.json").write_text(
                        json.dumps(response["retrieved_guidelines"], indent=2, default=str)
                    )
                if isinstance(response, dict) and response.get("pg_prompts") is not None:
                    (task_out / f"step_{step:03d}_pg_prompts.json").write_text(
                        json.dumps(response["pg_prompts"], indent=2, default=str)
                    )
                if isinstance(response, dict) and response.get("raw_pg_responses") is not None:
                    (task_out / f"step_{step:03d}_raw_pg_responses.json").write_text(
                        json.dumps(response["raw_pg_responses"], indent=2, default=str)
                    )
                pg_action = parse_pg_action(response)
                exec_result = execute_pg_action(
                    driver,
                    pg_action,
                    width,
                    height,
                    coordinate_width=pg_coordinate_width,
                    coordinate_height=pg_coordinate_height,
                )
            except Exception as exc:
                finished_reason = "pg_agent_fail"
                tlog.write(json.dumps({"step": step, "error": str(exc), "screenshot": str(shot_path)}) + "\n")
                break

            record = {
                "step": step,
                "task": prompt,
                "screenshot": str(shot_path),
                "screenshot_url": shot_url,
                "pg_action": {
                    "kind": pg_action.kind,
                    "x": pg_action.x,
                    "y": pg_action.y,
                    "text": pg_action.text,
                    "direction": pg_action.direction,
                    "raw": pg_action.raw,
                },
                "exec": exec_result,
            }
            tlog.write(json.dumps(record, default=str) + "\n")
            tlog.flush()
            history.append(record)
            steps_taken = step

            if exec_result.get("finished"):
                finished_reason = "impossible_action" if pg_action.kind == "impossible" else "finished_action"
                break
            converted_action = str(exec_result.get("converted_action") or "")
            if repeat_action_stop > 0 and converted_action:
                if converted_action == last_converted_action:
                    repeated_action_count += 1
                else:
                    last_converted_action = converted_action
                    repeated_action_count = 1
                if repeated_action_count >= repeat_action_stop:
                    finished_reason = "repeated_action_loop"
                    break
            time.sleep(sleep_after_action)

    final_screenshot_path = ""
    try:
        final = driver.screenshot_bgr()
        final_screenshot_path = str(task_out / "agent_screenshot.png")
        cv2.imwrite(final_screenshot_path, final)
    except Exception:
        pass

    summary = {
        "task_id": task_dir.name,
        "task_dir": str(task_dir),
        "prompt": prompt,
        "agent": "pg_agent",
        "pg_agent_root": str(pg_agent_root),
        "graph_dir": str(graph_dir) if graph_dir else None,
        "guideline_mode": guideline_mode,
        "steps_taken": steps_taken,
        "finished_reason": finished_reason,
        "agent_screenshot": final_screenshot_path,
    }
    (task_out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def iter_task_dirs(tasks_dir: Path, only_task: Optional[str], limit: Optional[int]) -> list[Path]:
    task_dirs = [p for p in sorted(tasks_dir.iterdir()) if p.is_dir()]
    if only_task:
        task_dirs = [p for p in task_dirs if p.name == only_task]
    if limit is not None:
        task_dirs = task_dirs[:limit]
    return task_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True, choices=sorted(APP_REGISTRY))
    parser.add_argument("--device", default=os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554"),
                        help="adb device serial (default: $ANDROID_DEVICE_SERIAL or emulator-5554)")
    parser.add_argument("--adb-path", default="adb", help="adb executable path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--tasks-root", default=os.environ.get("TAGNAV_TASKS_ROOT"),
                        help="root holding per-app task dirs (default: $TAGNAV_TASKS_ROOT)")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-task")
    parser.add_argument("--clear-data", action="store_true")
    parser.add_argument("--sleep-after-action", type=float, default=3.0)
    parser.add_argument(
        "--repeat-action-stop",
        type=int,
        default=0,
        help="Stop a task after the same converted action repeats this many consecutive times; 0 disables.",
    )
    parser.add_argument(
        "--pg-coordinate-width",
        type=int,
        help="Optional PG-Agent coordinate-frame width for scaling absolute click coordinates.",
    )
    parser.add_argument(
        "--pg-coordinate-height",
        type=int,
        help="Optional PG-Agent coordinate-frame height for scaling absolute click coordinates.",
    )
    parser.add_argument("--image-server-port", type=int, default=6666)
    parser.add_argument("--pg-agent-root", default=str(DEFAULT_PG_AGENT_ROOT))
    parser.add_argument("--pg-server-url", default=os.environ.get("PG_AGENT_SERVER_URL"),
                        help="PG-Agent HTTP endpoint, usually .../next_action (default: $PG_AGENT_SERVER_URL)")
    parser.add_argument("--pg-http-timeout", type=float, default=300.0)
    parser.add_argument("--pg-entrypoint", help="External PG-Agent callable: package.module:function")
    parser.add_argument("--pg-command-template", help="External command; use {request_json} placeholder")
    parser.add_argument("--graph-dir", type=Path, help="PG-Agent page graph directory to use for guideline retrieval")
    parser.add_argument(
        "--guideline-mode",
        choices=["full", "no_guidelines", "planning_only", "decision_only"],
        default="full",
    )
    args = parser.parse_args()

    if not args.tasks_root:
        raise SystemExit("set --tasks-root or TAGNAV_TASKS_ROOT (see baselines/.env.example)")
    app_info = APP_REGISTRY[args.app]
    tasks_dir = Path(args.tasks_root) / app_info["tasks_dir"]
    if not tasks_dir.exists():
        raise SystemExit(f"tasks directory not found: {tasks_dir}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args)
    driver = AdbDriver(args.adb_path, args.device)
    server = ScreenshotServer(out_root, port=args.image_server_port)
    image_base_url = server.start()

    summaries: list[dict[str, Any]] = []
    try:
        task_dirs = iter_task_dirs(tasks_dir, args.only_task, args.limit)
        print(f"Found {len(task_dirs)} tasks for {args.app} in {tasks_dir}")
        for idx, task_dir in enumerate(task_dirs, 1):
            prompt = discover_prompt(str(task_dir))
            if not prompt:
                print(f"[{idx}/{len(task_dirs)}] {task_dir.name}: skipped, no prompt")
                continue
            print(f"[{idx}/{len(task_dirs)}] {task_dir.name}: {prompt}")
            summary = run_one_task(
                task_dir=task_dir,
                prompt=prompt,
                backend=backend,
                driver=driver,
                package=app_info["package"],
                activity=app_info.get("activity"),
                out_root=out_root,
                image_base_url=image_base_url,
                max_steps=args.max_steps,
                graph_dir=args.graph_dir,
                pg_agent_root=Path(args.pg_agent_root),
                guideline_mode=args.guideline_mode,
                clear_data=args.clear_data,
                sleep_after_action=args.sleep_after_action,
                repeat_action_stop=args.repeat_action_stop,
                pg_coordinate_width=args.pg_coordinate_width,
                pg_coordinate_height=args.pg_coordinate_height,
            )
            summaries.append(summary)
            print(f"  -> {summary['finished_reason']} in {summary['steps_taken']} steps")
    finally:
        server.stop()

    (out_root / "all_summaries.json").write_text(json.dumps(summaries, indent=2))
    finished = sum(1 for item in summaries if item.get("finished_reason") == "finished_action")
    total = len(summaries)
    rate = (100 * finished / total) if total else 0.0
    print(f"Finished-action rate: {finished}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
