"""
Inference GUI — FastAPI backend for the React demo wizard.

Run (dev):
    cd inference && uvicorn gui_demo.inference_gui:app --reload --port 8765

Run (production / single-port demo):
    cd inference/gui_demo/web && npm run build
    cd inference && uvicorn gui_demo.inference_gui:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

for var in [
    "http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY",
]:
    os.environ.pop(var, None)

from Agents.factory import build_agent
from Debugger import Debugger
from Driver.factory import build_driver
from FlagEmbedding import BGEM3FlagModel
from VLM_Yibu import build_vlm_client

from .pipeline import (
    encode_image_jpeg_b64,
    format_candidate_label,
    get_navigation_memory_for_node,
    load_artifacts,
    run_navigation_loop,
    run_retrieval,
    save_trajectory_mp4,
    trajectory_mp4_filename,
    visualize_actions_on_screenshots,
)

GUI_DEMO_DIR = Path(__file__).resolve().parent
INFERENCE_DIR = GUI_DEMO_DIR.parent
CONFIGS_DIR = INFERENCE_DIR / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "outlook_android.yaml"
WEB_DIST = GUI_DEMO_DIR / "web" / "dist"

dbg = Debugger(palette="soft", indent_size=2, width=90)

TRAJECTORY_MP4_DIR = os.environ.get("AGENTNAV_SAVE_TRAJECTORY_MP4_DIR", "").strip()
TRAJECTORY_MP4_FPS = float(os.environ.get("AGENTNAV_TRAJECTORY_MP4_FPS", "1") or "1")


def _candidates_for_trajectory_mp4() -> list[dict]:
    items: list[dict] = []
    for c in session.candidates:
        items.append(
            {
                **c,
                "page_label": format_candidate_label(c),
                "vlm_reasoning": session.vlm_reasoning.get(c.get("node_id", ""), ""),
            }
        )
    return items


def _maybe_save_trajectory_mp4(
    screenshots: list,
    actions: list,
    *,
    query: str,
    finish_flag: bool,
    final_screenshot=None,
    candidates: Optional[list] = None,
    selected_node_id: Optional[str] = None,
    logs_root: Optional[str] = None,
) -> Optional[str]:
    if not TRAJECTORY_MP4_DIR or not screenshots or not actions:
        return None

    output_path = Path(TRAJECTORY_MP4_DIR) / trajectory_mp4_filename(query, session.active_config_id)
    try:
        return save_trajectory_mp4(
            screenshots,
            actions,
            output_path,
            final_screenshot=final_screenshot if finish_flag else None,
            fps=TRAJECTORY_MP4_FPS,
            query=query,
            candidates=candidates,
            selected_node_id=selected_node_id,
            logs_root=logs_root,
        )
    except Exception as exc:
        dbg.log(f"Failed to save trajectory MP4: {exc}", color="yellow")
        return None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ConfigUpdate(BaseModel):
    config: dict[str, Any]


class ConfigSelectRequest(BaseModel):
    config_id: str = Field(..., description="Config stem name, e.g. outlook_android")


class RetrieveRequest(BaseModel):
    query: str


class SelectNodeRequest(BaseModel):
    node_id: str


class NavigationMemoryRequest(BaseModel):
    node_id: str
    query: str


class ExecuteStartRequest(BaseModel):
    query: str
    node_id: Optional[str] = None
    reset_to_start_page: bool = True
    model_thinking: bool = True


def _apply_model_thinking(agent: Any, enabled: bool) -> None:
    client = getattr(agent, "agent_client", None)
    if client is not None and hasattr(client, "set_model_thinking"):
        client.set_model_thinking(enabled)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


class InferenceSession:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.config: dict[str, Any] = {}
        self.driver = None
        self.agent = None
        self.ctx = None
        self.embedding_model = None
        self.vlm_client = None
        self.candidates: list[dict] = []
        self.vlm_reasoning: dict = {}
        self.selected_node_id: Optional[str] = None
        self.current_query: str = ""
        self.device_connected: bool = False
        self.resources_loaded: bool = False
        self.agent_ok: bool = False
        self.device_ok: bool = False
        self.agent_check_message: str = ""
        self.device_check_message: str = ""
        self.agent_model_id: str = ""
        self.load_status: dict[str, str] = {}
        self.active_config_id: str = ""
        self._device_stream_active: bool = False
        self._execution_lock = threading.Lock()
        self._executing: bool = False
        self._stop_requested: bool = False
        self.reset_to_start_page: bool = True
        self.model_thinking: bool = True

    def driver_settings(self) -> dict:
        d = dict(self.config.get("driver", {}))
        return d

    def agent_settings(self) -> dict:
        return dict(self.config.get("agent", {}).get("settings", {}))

    def logs_root(self) -> str:
        root = self.config.get("logs", {}).get("root", "")
        if root and not os.path.isabs(root):
            return str((INFERENCE_DIR / root).resolve())
        return str(root)


session = InferenceSession()


def _load_yaml_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = data.get("default", data)
    cfg["batch_mode"] = False
    return cfg


def _resolve_config_path(config_id: str) -> Path:
    config_id = str(config_id or "").strip()
    if not config_id:
        raise HTTPException(400, "config_id is required")

    candidate = CONFIGS_DIR / config_id
    if candidate.suffix != ".yaml":
        candidate = CONFIGS_DIR / f"{config_id}.yaml"

    if not candidate.is_file():
        raise HTTPException(404, f"Config not found: {config_id}")

    resolved = candidate.resolve()
    if CONFIGS_DIR.resolve() not in resolved.parents:
        raise HTTPException(400, "Invalid config path")

    return resolved


def _list_available_configs() -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for path in sorted(CONFIGS_DIR.glob("*.yaml")):
        try:
            cfg = _load_yaml_config(path)
            app_name = str(cfg.get("app", {}).get("name") or path.stem.replace("_", " ").title())
        except Exception:
            app_name = path.stem.replace("_", " ").title()
        items.append(
            {
                "id": path.stem,
                "label": app_name,
                "filename": path.name,
            }
        )
    return items


def _clear_runtime_state() -> None:
    session.ctx = None
    session.embedding_model = None
    session.vlm_client = None
    session.candidates = []
    session.vlm_reasoning = {}
    session.selected_node_id = None
    session.current_query = ""
    session.resources_loaded = False
    session.load_status = {}
    session._device_stream_active = False
    session.device_connected = False
    session.driver = None
    session.agent = None
    session.agent_ok = False
    session.device_ok = False
    session.agent_check_message = ""
    session.device_check_message = ""
    session.agent_model_id = ""
    session._executing = False
    session._stop_requested = False


def _apply_config_file(path: Path) -> dict:
    session.config = _load_yaml_config(path)
    session.active_config_id = path.stem
    _clear_runtime_state()
    return session.config


def _normalize_agent_url(url: str) -> str:
    agent_url = str(url).strip().rstrip("/")
    if not agent_url.endswith("/v1"):
        agent_url = f"{agent_url}/v1"
    return agent_url


def _check_agent_health(url: str) -> tuple[bool, str, str]:
    agent_url = _normalize_agent_url(url)
    try:
        result = subprocess.run(
            ["curl", "-sSf", f"{agent_url}/models"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=5,
        )
        if result.returncode != 0:
            return False, "", f"Agent server unreachable at {agent_url}"
        data = json.loads(result.stdout.decode().strip())
        model_id = ""
        if isinstance(data, dict) and isinstance(data.get("data"), list) and data["data"]:
            model_id = str(data["data"][0].get("id") or "")
        return True, model_id, f"Connected — model: {model_id or 'unknown'}"
    except Exception as exc:
        return False, "", f"Agent check failed: {exc}"


def _config_object():
    class _Cfg:
        pass

    def _populate(obj, data):
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    sub = _Cfg()
                    _populate(sub, v)
                    setattr(obj, k, sub)
                else:
                    setattr(obj, k, v)

    cfg = _Cfg()
    _populate(cfg, session.config)
    return cfg


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="AgentNavigator Inference GUI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8765"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup() -> None:
    _apply_config_file(DEFAULT_CONFIG_PATH)


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------


@app.get("/api/configs")
def list_configs() -> dict:
    return {
        "configs": _list_available_configs(),
        "active_config_id": session.active_config_id,
    }


@app.post("/api/config/select")
def select_config(body: ConfigSelectRequest) -> dict:
    path = _resolve_config_path(body.config_id)
    config = _apply_config_file(path)
    return {
        "ok": True,
        "config_id": session.active_config_id,
        "config": config,
    }


@app.get("/api/config/default")
def get_default_config() -> dict:
    if session.config:
        return {
            "config": session.config,
            "config_id": session.active_config_id,
        }
    config = _apply_config_file(DEFAULT_CONFIG_PATH)
    return {"config": config, "config_id": session.active_config_id}


@app.get("/api/config")
def get_config() -> dict:
    return {"config": session.config, "config_id": session.active_config_id}


@app.put("/api/config")
def update_config(body: ConfigUpdate) -> dict:
    body.config["batch_mode"] = False
    session.config = body.config
    return {"ok": True, "config": session.config, "config_id": session.active_config_id}


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


@app.post("/api/checks/run")
def run_checks() -> dict:
    agent_url = session.config.get("agent", {}).get("url", "")
    session.agent_ok, session.agent_model_id, session.agent_check_message = _check_agent_health(agent_url)

    try:
        driver = build_driver(settings=session.driver_settings(), agent=None)
        session.device_ok = driver.check_device()
        fg = None
        if session.device_ok:
            try:
                fg = driver.get_foreground_package()
            except Exception:
                fg = None
        expected = session.driver_settings().get("appPackage", "")
        if session.device_ok:
            msg = f"Device online ({session.driver_settings().get('device_id', 'unknown')})"
            if fg:
                msg += f" — foreground: {fg}"
                if expected and fg != expected:
                    msg += f" (expected {expected})"
            session.device_check_message = msg
        else:
            session.device_check_message = "Device not reachable via ADB/HDC"
    except Exception as exc:
        session.device_ok = False
        session.device_check_message = f"Device check failed: {exc}"

    return {
        "agent": {
            "ok": session.agent_ok,
            "model_id": session.agent_model_id,
            "message": session.agent_check_message,
        },
        "device": {
            "ok": session.device_ok,
            "message": session.device_check_message,
        },
    }


@app.get("/api/checks/status")
def checks_status() -> dict:
    return {
        "agent": {"ok": session.agent_ok, "model_id": session.agent_model_id, "message": session.agent_check_message},
        "device": {"ok": session.device_ok, "message": session.device_check_message},
    }


# ---------------------------------------------------------------------------
# Device connect / disconnect
# ---------------------------------------------------------------------------


@app.post("/api/device/connect")
def device_connect() -> dict:
    if not session.device_ok:
        raise HTTPException(400, "Run device check first — device not OK")

    try:
        session.driver = build_driver(settings=session.driver_settings(), agent=None)
        if not session.driver.check_device():
            raise HTTPException(400, "Device no longer reachable")
        session.driver.run_application()
        session.driver.wait()
        session.device_connected = True
        if session.agent:
            session.driver.agent = session.agent
        return {"ok": True, "message": "Connected and app launched"}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Connect failed: {exc}") from exc


@app.post("/api/device/disconnect")
def device_disconnect() -> dict:
    session._device_stream_active = False
    session.device_connected = False
    session.driver = None
    return {"ok": True}


@app.websocket("/ws/device")
async def ws_device(websocket: WebSocket) -> None:
    await websocket.accept()
    session._device_stream_active = True
    try:
        while session._device_stream_active:
            if session.driver and session.device_connected and not session._executing:
                try:
                    img = await asyncio.to_thread(session.driver.take_screenshot)
                    b64 = encode_image_jpeg_b64(img, quality=70)
                    await websocket.send_json({"type": "frame", "image_b64": b64})
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
            else:
                await websocket.send_json({"type": "waiting", "message": "Device not connected"})
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass
    finally:
        session._device_stream_active = False


# ---------------------------------------------------------------------------
# Resource loading (SSE stream)
# ---------------------------------------------------------------------------


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/resources/status")
def resources_status() -> dict:
    return {
        "loaded": session.resources_loaded,
        "items": session.load_status,
    }


@app.get("/api/resources/load/stream")
async def resources_load_stream() -> StreamingResponse:
    async def generate():
        session.load_status = {}
        session.resources_loaded = False

        steps = [
            ("graph", "Loading graph.json"),
            ("node_intents", "Loading user_intents.json"),
            ("node_level_information", "Loading node_level_information.json"),
            ("node_navigation_plans", "Loading node_navigation_plans.json"),
            ("embeddings", "Building embedding matrix"),
            ("bge_model", "Loading BGE-M3 embedding model"),
            ("vlm", "Initializing VLM client"),
            ("agent", "Connecting to agent server"),
        ]

        for key, label in steps:
            session.load_status[key] = "loading"
            yield _sse_event("progress", {"key": key, "label": label, "status": "loading"})

        try:
            logs_root = session.logs_root()
            session.ctx = await asyncio.to_thread(load_artifacts, logs_root)
            session.load_status["graph"] = "done"
            yield _sse_event("progress", {"key": "graph", "label": "Graph loaded", "status": "done"})
            session.load_status["node_intents"] = "done"
            yield _sse_event("progress", {"key": "node_intents", "label": "User intents loaded", "status": "done"})
            session.load_status["node_level_information"] = "done"
            yield _sse_event(
                "progress",
                {
                    "key": "node_level_information",
                    "label": "Node level information loaded",
                    "status": "done",
                },
            )
            session.load_status["node_navigation_plans"] = "done"
            yield _sse_event(
                "progress",
                {
                    "key": "node_navigation_plans",
                    "label": "Navigation plans loaded",
                    "status": "done",
                },
            )
            session.load_status["embeddings"] = "done"
            yield _sse_event("progress", {"key": "embeddings", "label": "Embedding matrix ready", "status": "done"})

            session.embedding_model = await asyncio.to_thread(
                lambda: BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
            )
            session.load_status["bge_model"] = "done"
            yield _sse_event("progress", {"key": "bge_model", "label": "BGE-M3 loaded", "status": "done"})

            cfg = _config_object()
            session.vlm_client = await asyncio.to_thread(build_vlm_client, cfg, dbg)
            vlm_label = (
                "VLM client ready (Yibu)"
                if getattr(getattr(cfg, "vlm", None), "use_yibu_api", False)
                else "VLM client ready"
            )
            session.load_status["vlm"] = "done"
            yield _sse_event("progress", {"key": "vlm", "label": vlm_label, "status": "done"})

            if not session.agent_ok:
                ok, model_id, msg = _check_agent_health(session.config.get("agent", {}).get("url", ""))
                if not ok:
                    session.load_status["agent"] = "error"
                    yield _sse_event("error", {"message": msg})
                    return
                session.agent_ok = ok
                session.agent_model_id = model_id

            agent_cfg = session.config.get("agent", {})
            session.agent = await asyncio.to_thread(
                build_agent,
                agent_cfg.get("model_name", "mai_ui"),
                agent_cfg.get("url", ""),
                session.agent_settings(),
                dbg,
            )
            _apply_model_thinking(
                session.agent,
                bool(session.agent_settings().get("model_thinking", True)),
            )
            session.load_status["agent"] = "done"
            yield _sse_event("progress", {"key": "agent", "label": "Agent connected", "status": "done"})

            if session.driver:
                session.driver.agent = session.agent

            session.resources_loaded = True
            yield _sse_event("complete", {"ok": True})
        except Exception as exc:
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Screenshots from exploration artifacts
# ---------------------------------------------------------------------------


@app.get("/api/screenshots/{node_id}")
def get_screenshot(node_id: str) -> FileResponse:
    path = os.path.join(session.logs_root(), "screenshots", f"{node_id}.jpg")
    if not os.path.isfile(path):
        raise HTTPException(404, f"Screenshot not found for {node_id}")
    return FileResponse(path, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Retrieval & selection
# ---------------------------------------------------------------------------


@app.post("/api/retrieve")
async def retrieve(body: RetrieveRequest) -> dict:
    if not session.resources_loaded:
        raise HTTPException(400, "Load resources first")
    if not body.query.strip():
        raise HTTPException(400, "Query is required")

    session.current_query = body.query.strip()
    cfg = _config_object()

    if not getattr(cfg, "use_memory_for_navigation", False):
        session.candidates = []
        session.vlm_reasoning = {}
        return {"candidates": [], "use_memory": False}

    candidates, vlm_reasoning = await asyncio.to_thread(
        run_retrieval,
        session.current_query,
        session.ctx,
        cfg,
        session.embedding_model,
        session.vlm_client,
    )
    session.candidates = candidates
    session.vlm_reasoning = vlm_reasoning

    result = []
    for c in candidates:
        result.append(
            {
                "node_id": c["node_id"],
                "score": c["score"],
                "depth": c["depth"],
                "page_tag": c.get("page_tag", ""),
                "page_purpose": c.get("page_purpose", ""),
                "page_label": format_candidate_label(c),
                "user_intents": c.get("user_intents", []),
                "vlm_reasoning": vlm_reasoning.get(c["node_id"], ""),
                "screenshot_url": f"/api/screenshots/{c['node_id']}",
            }
        )
    return {"candidates": result, "use_memory": True, "query": session.current_query}


@app.post("/api/select-node")
def select_node(body: SelectNodeRequest) -> dict:
    session.selected_node_id = body.node_id
    return {"ok": True, "node_id": body.node_id}


@app.post("/api/navigation-memory")
def navigation_memory(body: NavigationMemoryRequest) -> dict:
    if not session.resources_loaded or not session.ctx:
        raise HTTPException(400, "Resources not loaded")
    node_id = body.node_id or session.selected_node_id
    if not node_id:
        raise HTTPException(400, "No node selected")
    query = body.query or session.current_query
    cfg = _config_object()
    if getattr(cfg, "use_memory_for_navigation", False) and node_id:
        memory = get_navigation_memory_for_node(session.ctx, node_id, query)
    else:
        from .pipeline import format_navigation_plan
        memory = format_navigation_plan([], query)
    return {"memory": memory, "node_id": node_id, "query": query}


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@app.post("/api/execute/start")
def execute_start(body: ExecuteStartRequest) -> dict:
    if not session.resources_loaded:
        raise HTTPException(400, "Load resources first")
    if not session.driver or not session.device_connected:
        raise HTTPException(400, "Connect device first")
    if not session.agent:
        raise HTTPException(400, "Agent not loaded")
    if session._executing:
        raise HTTPException(409, "Execution already in progress")

    query = body.query or session.current_query
    node_id = body.node_id or session.selected_node_id
    cfg = _config_object()

    if getattr(cfg, "use_memory_for_navigation", False) and node_id:
        navigation_memory = get_navigation_memory_for_node(session.ctx, node_id, query)
    else:
        from .pipeline import format_navigation_plan
        navigation_memory = format_navigation_plan([], query)

    session.current_query = query
    session.selected_node_id = node_id
    session.reset_to_start_page = body.reset_to_start_page
    session.model_thinking = body.model_thinking
    _apply_model_thinking(session.agent, body.model_thinking)
    session._stop_requested = False
    return {"ok": True, "message": "Connect to /ws/execution to stream steps"}


@app.post("/api/execute/stop")
def execute_stop() -> dict:
    if not session._executing:
        return {"ok": False, "message": "No navigation in progress"}
    session._stop_requested = True
    return {"ok": True, "message": "Stop requested — will halt after current step"}


@app.websocket("/ws/execution")
async def ws_execution(websocket: WebSocket) -> None:
    await websocket.accept()

    if not session.resources_loaded or not session.driver or not session.agent:
        await websocket.send_json({"type": "error", "message": "Not ready — load resources and connect device"})
        await websocket.close()
        return

    if session._executing:
        await websocket.send_json({"type": "error", "message": "Execution already in progress"})
        await websocket.close()
        return

    query = session.current_query
    node_id = session.selected_node_id
    cfg = _config_object()

    if getattr(cfg, "use_memory_for_navigation", False) and node_id:
        navigation_memory = get_navigation_memory_for_node(session.ctx, node_id, query)
    else:
        from .pipeline import format_navigation_plan
        navigation_memory = format_navigation_plan([], query)

    max_steps = getattr(getattr(cfg, "agent", None), "max_steps", 10) or 10
    _apply_model_thinking(session.agent, session.model_thinking)
    step_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    session._stop_requested = False

    def on_step(payload: dict) -> None:
        loop.call_soon_threadsafe(step_queue.put_nowait, payload)

    def should_stop() -> bool:
        return session._stop_requested

    async def run_nav() -> tuple:
        session._executing = True
        try:
            return await asyncio.to_thread(
                run_navigation_loop,
                navigation_memory,
                session.agent,
                session.driver,
                max_steps,
                on_step,
                session.reset_to_start_page,
                should_stop,
            )
        finally:
            session._executing = False
            session._stop_requested = False

    nav_task = asyncio.create_task(run_nav())

    collected_screenshots: list = []
    collected_actions: list = []
    step_num = 0

    try:
        await websocket.send_json({"type": "started", "query": query, "node_id": node_id})

        while True:
            if nav_task.done() and step_queue.empty():
                break
            try:
                payload = await asyncio.wait_for(step_queue.get(), timeout=0.5)
            except asyncio.TimeoutError:
                continue

            step_num += 1
            action = payload["action"]
            screenshot = payload["screenshot"]
            collected_screenshots.append(screenshot)
            collected_actions.append(action)

            annotated_list = visualize_actions_on_screenshots([screenshot], [action])
            annotated = annotated_list[0] if annotated_list else screenshot

            timing = payload.get("timing") or action.get("timing") or {}
            await websocket.send_json(
                {
                    "type": "step",
                    "step": step_num,
                    "thought": action.get("thought", ""),
                    "action_type": action.get("type", ""),
                    "coordinate": action.get("coordinate"),
                    "screenshot_b64": encode_image_jpeg_b64(screenshot),
                    "annotated_b64": encode_image_jpeg_b64(annotated),
                    "driver_screenshot_s": timing.get("driver_screenshot_s"),
                    "model_prediction_s": timing.get("model_prediction_s"),
                    "other_processing_s": timing.get("other_processing_s"),
                }
            )

            if str(action.get("type", "")).strip().lower() in ("finished", "finish"):
                break

        screenshots, actions, finish_flag, stopped = await nav_task

        final_shot = screenshots[-1] if finish_flag and len(screenshots) > len(collected_actions) else None
        trajectory_mp4_path = _maybe_save_trajectory_mp4(
            collected_screenshots,
            collected_actions,
            query=query,
            finish_flag=finish_flag,
            final_screenshot=final_shot,
            candidates=_candidates_for_trajectory_mp4()
            if getattr(cfg, "use_memory_for_navigation", False) and session.candidates
            else None,
            selected_node_id=node_id,
            logs_root=session.logs_root(),
        )

        if stopped:
            await websocket.send_json(
                {
                    "type": "stopped",
                    "total_steps": len(actions),
                    "message": "Navigation stopped by user",
                    "trajectory_mp4": trajectory_mp4_path,
                }
            )
        else:
            if final_shot is not None:
                await websocket.send_json(
                    {
                        "type": "step",
                        "step": step_num + 1,
                        "thought": "Navigation complete",
                        "action_type": "finish",
                        "coordinate": None,
                        "screenshot_b64": encode_image_jpeg_b64(final_shot),
                        "annotated_b64": encode_image_jpeg_b64(final_shot),
                    }
                )

            await websocket.send_json(
                {
                    "type": "done",
                    "success": finish_flag,
                    "total_steps": len(actions),
                    "trajectory_mp4": trajectory_mp4_path,
                }
            )
    except WebSocketDisconnect:
        nav_task.cancel()
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    finally:
        session._executing = False


# ---------------------------------------------------------------------------
# Static files (React build)
# ---------------------------------------------------------------------------

if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    _NO_CACHE_HEADERS = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    def _spa_file_response(path: Path, *, no_cache: bool = False) -> FileResponse:
        if no_cache:
            return FileResponse(path, headers=_NO_CACHE_HEADERS)
        return FileResponse(path)

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            raise HTTPException(404)
        index = WEB_DIST / "index.html"
        file_path = WEB_DIST / full_path
        if file_path.is_file():
            return _spa_file_response(file_path, no_cache=file_path.name == "index.html")
        return _spa_file_response(index, no_cache=True)

else:

    @app.get("/")
    def root_placeholder():
        return {
            "message": "AgentNavigator Inference GUI API is running.",
            "hint": "Build the React frontend: cd inference/gui_demo/web && npm install && npm run build",
            "dev": "Or run Vite dev server on :5173 with API on :8765",
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gui_demo.inference_gui:app", host="0.0.0.0", port=8765, reload=True)
