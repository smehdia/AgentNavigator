# NOTE: reference file from the PG-Agent graph-build + serving pipeline.
# Requires the upstream PG-Agent checkout ($PG_AGENT_ROOT) and a served VLM
# (we used Qwen2.5-VL-72B). See baselines/pg_agent/README.md before running.
#!/usr/bin/env python3
"""Start or describe the PG-Agent Odyssey HTTP server.

This is the serving stage between graph construction and live benchmarking.
By default it writes a manifest and prints the server command. Use `--execute`
to start the server as a background process with pid/log files.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_PATH = Path("results/pg_agent_experiment/canonical_graph_build/odyssey_library.json")


def shell_join(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def build_server_command(args: argparse.Namespace) -> list[str | Path]:
    command: list[str | Path] = [
        args.python_bin,
        "baselines/pg_agent/odyssey_server.py",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--pg-agent-root",
        args.pg_agent_root,
        "--graph-path",
        args.graph_path,
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--embedding-model",
        args.embedding_model,
        "--embedding-device",
        args.embedding_device,
        "--max-references",
        str(args.max_references),
    ]
    if args.env_file:
        command.extend(["--env-file", args.env_file])
    return command


def stop_existing(pid_path: Path) -> bool:
    if not pid_path.exists():
        return False
    try:
        pid = int(pid_path.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_path.unlink(missing_ok=True)
        return False
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.25)
    os.kill(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)
    return True


def start_server(args: argparse.Namespace, command: list[str | Path]) -> int:
    args.out_root.mkdir(parents=True, exist_ok=True)
    log_file = args.log_path.open("a")
    proc = subprocess.Popen(
        [str(part) for part in command],
        cwd=REPO_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    args.pid_path.write_text(str(proc.pid))
    time.sleep(args.startup_wait)
    if proc.poll() is not None:
        raise RuntimeError(f"PG-Agent server exited early with code {proc.returncode}; see {args.log_path}")
    return proc.pid


def write_manifest(args: argparse.Namespace, command: list[str | Path], pid: int | None, stopped: bool) -> Path:
    manifest: dict[str, Any] = {
        "schema": "agentnavigator_pg_agent_server_run_v1",
        "execute": args.execute,
        "python_bin": str(args.python_bin),
        "stopped": stopped,
        "host": args.host,
        "port": args.port,
        "url": f"http://{args.host}:{args.port}/next_action",
        "pid": pid,
        "pid_path": str(args.pid_path),
        "log_path": str(args.log_path),
        "graph_path": str(args.graph_path),
        "pg_agent_root": str(args.pg_agent_root),
        "endpoint": args.endpoint,
        "model": args.model,
        "api_key_env": args.api_key_env,
        "env_file": None if args.env_file is None else str(args.env_file),
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
        "max_references": args.max_references,
        "startup_wait": args.startup_wait,
        "command": shell_join(command),
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_root / "server_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Start the server in the background")
    parser.add_argument("--stop", action="store_true", help="Stop the server recorded in --pid-path")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable to put in the generated server command")
    parser.add_argument("--out-root", type=Path, default=Path("results/pg_agent_experiment/canonical_server"))
    parser.add_argument("--pid-path", type=Path)
    parser.add_argument("--log-path", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--pg-agent-root", type=Path, default=Path("../PG-Agent"))
    parser.add_argument("--graph-path", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--api-key-env", default="PG_AGENT_API_KEY")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--max-references", type=int, default=10)
    parser.add_argument("--startup-wait", type=float, default=2.0)
    args = parser.parse_args()

    args.pid_path = args.pid_path or args.out_root / "pg_agent_server.pid"
    args.log_path = args.log_path or args.out_root / "pg_agent_server.log"
    command = build_server_command(args)
    stopped = stop_existing(args.pid_path) if args.stop else False
    pid = None
    if args.execute and not args.stop:
        pid = start_server(args, command)
    manifest_path = write_manifest(args, command, pid, stopped)
    print(shell_join(command))
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
