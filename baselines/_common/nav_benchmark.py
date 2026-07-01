#!/usr/bin/env python3
"""Shared per-task navigation benchmark runner.

This is the runner used by the droidtask-nav and spa-bench-nav benchmarks; the
baseline adapters invoke it as a subprocess. It drives our navigation agent over
a set of tasks and supports three memory modes for fair same-runner comparison:
  closed_loop  -- per-step localization + dynamic memory rebuild
  open_loop    -- target's pre-stored navigation memory, fixed for all steps
  none         -- no memory_text passed to the agent

Third-party deps: numpy (required); opencv-python, dashscope and openai are
imported lazily by the code paths that use them.

Required environment / prereqs:
  - An agent server (e.g. UI-TARS-1.5-7B via vLLM) reachable at --ui-tars-url
    (default http://localhost:8000/v1/chat/completions).
  - DASHSCOPE_API_KEY in env (embeddings / Qwen-VL paths).
  - OPENAI_API_KEY in env if using the OpenAI-compatible path.
  - An adb device serial reachable (--device, or ANDROID_DEVICE_SERIAL).

Example:
  python baselines/_common/nav_benchmark.py \\
      --app chrome --device emulator-5554 \\
      --memory-mode open_loop --sim-floor 0.60 \\
      --max-steps 15 --out-dir runs/chrome_open
"""
import argparse
import base64
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# This file lives at <repo>/baselines/_common/nav_benchmark.py, so the repo root
# is three levels up. Data roots (graphs/tasks) default under it but are normally
# overridden by --tasks-root / --processed-pkl or the TAGNAV_TASKS_ROOT env var.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
# Also add AgentNavigator root for legacy imports (UI_TARS_1_5, VLM, etc.)
sys.path.insert(0, os.path.join(os.path.dirname(REPO_ROOT), "AgentNavigator"))

# ---------------------------------------------------------------------------
# App registry
# ---------------------------------------------------------------------------

APP_REGISTRY: Dict[str, Dict[str, Any]] = {
    "amazon": {
        "package": "com.amazon.mShop.android.shopping",
        "activity": "com.amazon.mShop.home.HomeActivity",
        "tasks_dir": "amazon_32.3.0.100",
    },
    "calendar": {
        "package": "com.google.android.calendar",
        "activity": None,
        "tasks_dir": "calendar_2026.05.0-864040481-release",
        "processed_pkl_name": "google_calendar",
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
        "processed_pkl_name": None,
    },
    "tiktok": {
        "package": "com.zhiliaoapp.musically",
        "activity": "com.ss.android.ugc.aweme.splash.SplashActivity",
        "tasks_dir": "tiktok_43.8.3",
    },
    "youtube": {
        "package": "com.google.android.youtube",
        "activity": "com.google.android.apps.youtube.app.WatchWhileActivity",
        "tasks_dir": "youtube_21.05.264",
    },
    "airbnb": {
        "package": "com.airbnb.android",
        "activity": ".activities.HomeActivity",
        "tasks_dir": "airbnb_26.06",
    },
}

def _resolve_app_paths(app_name: str, graph_type: Optional[str] = None) -> Dict[str, Any]:
    """Resolve processed pkl path from app name and graph type."""
    info = dict(APP_REGISTRY[app_name])
    legacy_pkl_name = info.pop("processed_pkl_name", None)

    if graph_type and graph_type != "none":
        pkl_path = os.path.join(REPO_ROOT, "graphs", graph_type, f"{app_name}.pkl")
        info["processed_pkl"] = pkl_path if os.path.isfile(pkl_path) else None
    elif "processed_pkl" not in info:
        # Legacy fallback: resolve from explored_apps_processed/
        dirname = legacy_pkl_name if legacy_pkl_name is not None else app_name
        info["processed_pkl"] = os.path.join(
            REPO_ROOT, "explored_apps_processed", dirname,
            "node_information_dict_pure_visual_with_embeddings.pkl",
        ) if dirname else None
    return info


# ---------------------------------------------------------------------------
# DashScope helpers
# ---------------------------------------------------------------------------

# DashScope Qwen-VL model id used by the --backbone qwen_vl path. Override via env if
# your account lacks access to the "-latest" alias: some keys get HTTP 403 on
# "qwen-vl-max-latest" (set QWEN_VL_MODEL=qwen-vl-max, the stable fallback).
QWEN_VL_MODEL = os.environ.get("QWEN_VL_MODEL", "qwen-vl-max-latest")


def _setup_dashscope():
    for var in [
        "http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
        "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY",
    ]:
        os.environ.pop(var, None)
    import dashscope
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    return dashscope


def _ensure_dashscope_key():
    # Key comes from the environment only (DASHSCOPE_API_KEY); no config-file fallback.
    import dashscope
    if os.environ.get("DASHSCOPE_API_KEY") or getattr(dashscope, "api_key", None):
        return
    print("ERROR: DASHSCOPE_API_KEY not found in environment")
    sys.exit(2)


def _embed_text(text: str) -> np.ndarray:
    import dashscope
    resp = dashscope.TextEmbedding.call(model="text-embedding-v4", input=text)
    if getattr(resp, "status_code", None) != 200:
        raise RuntimeError(f"text-embedding-v4 failed: {getattr(resp, 'message', None)}")
    return np.array(resp.output["embeddings"][0]["embedding"], dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + eps))


def _bgr_to_data_uri(bgr, fmt: str = "JPEG", quality: int = 80) -> str:
    import cv2
    ext = ".jpg" if fmt.upper() == "JPEG" else ".png"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if fmt.upper() == "JPEG" else []
    ok, buf = cv2.imencode(ext, bgr, params)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _describe_screenshot(screenshot_bgr) -> str:
    import dashscope
    data_uri = _bgr_to_data_uri(screenshot_bgr, fmt="JPEG", quality=80)
    system = (
        "You describe an Android UI screen in one short phrase for retrieval. "
        "Focus on the active interaction layer (modal/sheet/menu over background). "
        "Use lowercase. Do not describe path or earlier screens. "
        "Output ONLY the phrase, no JSON, no quotes, no prefix."
    )
    resp = dashscope.MultiModalConversation.call(
        model=QWEN_VL_MODEL,
        messages=[
            {"role": "system", "content": [{"text": system}]},
            {"role": "user", "content": [
                {"image": data_uri},
                {"text": "Describe this screen in one short phrase."},
            ]},
        ],
        temperature=0.0,
    )
    out = resp.output.choices[0].message.content
    if isinstance(out, list):
        text = "\n".join(it.get("text", "") for it in out if isinstance(it, dict)).strip()
    else:
        text = str(out).strip()
    text = text.strip().strip('"\'')
    if not text:
        raise RuntimeError("empty page description from qwen-vl")
    return text


# ---------------------------------------------------------------------------
# ADB driver
# ---------------------------------------------------------------------------

class AdbDriver:
    def __init__(self, adb_path, device, screencap_path="/sdcard/_uitars_cap.png"):
        self.adb = adb_path
        self.device = device
        self.cap = screencap_path

    def _run(self, *args, capture=True, timeout=30):
        cmd = [self.adb, "-s", self.device] + list(args)
        return subprocess.run(cmd, capture_output=capture, timeout=timeout)

    def shell(self, line, timeout=30):
        return self._run("shell", line, timeout=timeout)

    def screenshot_bgr(self):
        import cv2
        local_path = os.path.join(tempfile.gettempdir(), "_uitars_local.png")
        self.shell(f"screencap -p {self.cap}", timeout=15)
        self._run("pull", self.cap, local_path, timeout=20)
        img = cv2.imread(local_path)
        if img is None:
            raise RuntimeError(f"failed to read screenshot at {local_path}")
        return img

    def force_stop(self, pkg):
        self.shell(f"am force-stop {pkg}", timeout=10)

    def clear_data(self, pkg):
        self.shell(f"pm clear {pkg}", timeout=15)

    def launch(self, pkg, activity):
        if activity:
            self.shell(f"am start -n {pkg}/{activity}", timeout=15)
            try:
                r = self.shell("dumpsys window 2>/dev/null | grep -m1 mCurrentFocus", timeout=5)
                out = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "")
                if pkg not in out:
                    self.shell(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1", timeout=15)
            except Exception:
                self.shell(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1", timeout=15)
        else:
            self.shell(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1", timeout=15)

    def home(self):
        self.shell("input keyevent KEYCODE_HOME", timeout=5)

    def back(self):
        self.shell("input keyevent KEYCODE_BACK", timeout=5)

    def tap(self, x, y):
        self.shell(f"input tap {int(x)} {int(y)}", timeout=5)

    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        self.shell(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}", timeout=5)

    def text(self, text):
        safe = (text or "").replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
        self.shell(f"input text '{safe}'", timeout=10)

    def get_screen_size(self):
        r = self.shell("wm size", timeout=5)
        s = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "")
        m = re.search(r"(\d+)\s*x\s*(\d+)", s)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def get_ui_elements(self, max_elements: int = 30) -> Optional[str]:
        """Dump UI hierarchy and extract actionable elements as a compact string.

        Returns a text block like:
            [Button "Save" bounds=(800,100,1000,200)]
            [EditText "Title" bounds=(50,200,500,280)]
            [CheckBox "Vibrate" checked=true bounds=(50,400,100,450)]
        """
        try:
            self.shell("uiautomator dump /sdcard/_ui_dump.xml", timeout=10)
            r = self._run("shell", "cat /sdcard/_ui_dump.xml", timeout=10)
            xml_str = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "")
        except Exception:
            return None
        if not xml_str or "<hierarchy" not in xml_str:
            return None

        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return None

        elements = []
        for node in root.iter("node"):
            clickable = node.get("clickable", "false") == "true"
            editable = "EditText" in (node.get("class") or "")
            checkable = node.get("checkable", "false") == "true"
            scrollable = node.get("scrollable", "false") == "true"
            if not (clickable or editable or checkable or scrollable):
                continue

            text = node.get("text", "").strip()
            desc = node.get("content-desc", "").strip()
            cls = (node.get("class") or "").split(".")[-1]
            bounds = node.get("bounds", "")
            checked = node.get("checked", "")
            label = text or desc or ""
            if not label and cls in ("ImageView", "ImageButton", "View"):
                continue

            parts = [cls]
            if label:
                parts.append(f'"{label}"')
            if checked in ("true", "false") and checkable:
                parts.append(f"checked={checked}")
            if bounds:
                parts.append(f"bounds={bounds}")
            elements.append("[" + " ".join(parts) + "]")

            if len(elements) >= max_elements:
                break

        return "\n".join(elements) if elements else None

    def tap_system_dialog_button(self, labels: Tuple[str, ...]) -> bool:
        """Tap a visible system-dialog button by text/content-desc."""
        try:
            self.shell("uiautomator dump /sdcard/_ui_dump.xml", timeout=10)
            r = self._run("shell", "cat /sdcard/_ui_dump.xml", timeout=10)
            xml_str = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "")
        except Exception:
            return False
        if not xml_str or "<hierarchy" not in xml_str:
            return False

        import xml.etree.ElementTree as ET
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError:
            return False

        wanted = {label.strip().lower() for label in labels}
        for node in root.iter("node"):
            text = (node.get("text") or node.get("content-desc") or "").strip().lower()
            if text not in wanted:
                continue
            bounds = node.get("bounds", "")
            m = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds)
            if not m:
                continue
            x1, y1, x2, y2 = map(int, m.groups())
            self.tap((x1 + x2) // 2, (y1 + y2) // 2)
            return True
        return False


# ---------------------------------------------------------------------------
# Qwen-VL local agent (vLLM-served, 7B-class backbone)
# ---------------------------------------------------------------------------

class QwenVLLocalAgent:
    """GUI navigation agent using a locally-served Qwen2.5-VL via vLLM.

    Same interface as QwenVLAgent but hits a local OpenAI-compatible endpoint
    instead of the DashScope API.
    """

    MAX_SIDE = 1280

    def __init__(self, server_url: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen2.5-VL-7B-Instruct",
                 keep_steps: int = 3):
        base = server_url.rstrip("/")
        if base.endswith("/chat/completions"):
            base = base.rsplit("/chat/completions", 1)[0]
        if not base.endswith("/v1"):
            base = base + "/v1"
        self.server_url = base
        self.model_name = model_name
        self.keep_steps = keep_steps
        self.history: List[dict] = []
        self._img_w = 0
        self._img_h = 0

    def clean_history(self):
        self.history.clear()

    def _resize_and_encode(self, bgr) -> Tuple[str, int, int]:
        import cv2
        h, w = bgr.shape[:2]
        if self.MAX_SIDE <= 0 or max(h, w) <= self.MAX_SIDE:
            rw, rh = w, h
            resized = bgr
        else:
            scale = min(1.0, self.MAX_SIDE / max(h, w))
            rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(bgr, (rw, rh))
        ok, buf = cv2.imencode(".jpg", resized,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", rw, rh

    def step(self, user_instruction: str, current_screenshot_bgr,
             memory_text=None, ui_elements=None):
        import openai

        data_uri, rw, rh = self._resize_and_encode(current_screenshot_bgr)
        self._img_w = rw
        self._img_h = rh

        system = (
            f"You are a GUI agent controlling an Android phone. "
            f"The screenshot is {rw}x{rh} pixels.\n"
            f"Output your reasoning, then exactly one action.\n\n"
            f"Format:\n"
            f"Thought: <reasoning>\n"
            f"Action: click(x, y) | scroll(up|down|left|right) | "
            f"type(\"text\") | press_back() | finished()\n\n"
            f"Coordinates are pixel positions on the {rw}x{rh} image. "
            f"Output ONLY one Thought + one Action."
        )

        user_content = [
            {"type": "image_url",
             "image_url": {"url": data_uri}},
        ]
        text_block = f"Task: {user_instruction}"
        if memory_text:
            text_block += f"\n\n{memory_text}"
        if ui_elements:
            text_block += (
                f"\n\nUI elements on screen (from accessibility tree):\n"
                f"{ui_elements}"
            )
        text_block += "\n\nWhat is the next action?"
        user_content.append({"type": "text", "text": text_block})

        messages = [{"role": "system", "content": system}]
        for h in self.history:
            messages.append(h)
        messages.append({"role": "user", "content": user_content})

        client = openai.OpenAI(base_url=self.server_url, api_key="unused")
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

        text = resp.choices[0].message.content.strip()

        action_match = re.search(r"Action:\s*(.+)", text)
        action_str = (action_match.group(1).strip() if action_match
                      else text.split("\n")[-1].strip())
        thought_match = re.search(r"Thought:\s*(.+)", text)
        thought = thought_match.group(1).strip() if thought_match else ""

        self.history.append({"role": "user", "content": user_content})
        self.history.append({"role": "assistant",
                             "content": text})
        if len(self.history) > self.keep_steps * 2:
            self.history = self.history[-(self.keep_steps * 2):]

        return text, action_str, thought


class UITarsBenchmarkAdapter:
    """Adapter from UITarsMobileHistoryAgent's ParsedAction API to benchmark API.

    Outputs orig_coords so execute_action can map them 1:1 when coord_max_side=0.
    """

    def __init__(self, inner):
        self.inner = inner

    def clean_history(self):
        if hasattr(self.inner, "assistant_history"):
            self.inner.assistant_history.clear()

    @staticmethod
    def _first_coord(coords: dict) -> Optional[Tuple[int, int]]:
        for key in ("point", "start_point", "start_box", "box"):
            pt = coords.get(key)
            if pt:
                return pt
        return None

    def step(self, user_instruction: str, current_screenshot_bgr,
             memory_text=None, ui_elements=None):
        instruction = str(user_instruction or "")
        if memory_text:
            instruction += f"\n\n{memory_text}"
        if ui_elements:
            instruction += (
                "\n\nUI elements on screen (from accessibility tree):\n"
                f"{ui_elements}"
            )
        result = self.inner.step(instruction, current_screenshot_bgr)
        # Unwrap nested tuples to find the ParsedAction object
        # MAI_UI returns ((ParsedAction, ...), ResizeMeta), UI-TARS returns ParsedAction directly
        parsed = result
        while isinstance(parsed, (tuple, list)):
            parsed = parsed[0]
        action = str(getattr(parsed, "action_type", "") or "").strip().lower()
        coords = getattr(parsed, "orig_coords", {}) or {}
        params = getattr(parsed, "params", {}) or {}
        if action in ("click", "tap"):
            pt = self._first_coord(coords)
            if pt:
                action_str = f"click({int(pt[0])}, {int(pt[1])})"
            else:
                action_str = "click()"
        elif action in ("long_press", "longpress", "long-press"):
            pt = self._first_coord(coords)
            if pt:
                action_str = f"long_press({int(pt[0])}, {int(pt[1])})"
            else:
                action_str = "long_press()"
        elif action in ("finished", "finish"):
            action_str = "finished()"
        elif action in ("press_back", "back"):
            action_str = "press_back()"
        elif action in ("press_home", "home"):
            action_str = "press_home()"
        elif action == "type":
            text = str(params.get("content", ""))
            action_str = f'type("{text}")'
        elif action == "scroll":
            direction = str(params.get("direction", "down") or "down")
            action_str = f"scroll({direction})"
        else:
            action_str = str(getattr(parsed, "raw_text", "") or "")
        return (
            str(getattr(parsed, "raw_text", "") or ""),
            action_str,
            str(getattr(parsed, "thought", "") or ""),
        )


# Qwen-VL-Max agent (API-based, 72B-class backbone)
# ---------------------------------------------------------------------------

class QwenVLAgent:
    """GUI navigation agent using qwen-vl-max-latest via DashScope API.

    Drop-in replacement for UI_TARS — same step()/clean_history() interface.
    Uses a larger model (~72B class) that should be more robust to prompt
    perturbation than UI-TARS-1.5-7B.
    """

    MAX_SIDE = 1280

    def __init__(self, keep_steps: int = 3):
        self.keep_steps = keep_steps
        self.history: List[dict] = []
        self._img_w = 0
        self._img_h = 0

    def clean_history(self):
        self.history.clear()

    def _resize_and_encode(self, bgr) -> Tuple[str, int, int]:
        import cv2
        h, w = bgr.shape[:2]
        if self.MAX_SIDE <= 0 or max(h, w) <= self.MAX_SIDE:
            rw, rh = w, h
            resized = bgr
        else:
            scale = min(1.0, self.MAX_SIDE / max(h, w))
            rw, rh = max(1, int(w * scale)), max(1, int(h * scale))
            resized = cv2.resize(bgr, (rw, rh))
        ok, buf = cv2.imencode(".jpg", resized,
                               [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}", rw, rh

    def step(self, user_instruction: str, current_screenshot_bgr,
             memory_text=None, ui_elements=None):
        import dashscope

        data_uri, rw, rh = self._resize_and_encode(current_screenshot_bgr)
        self._img_w = rw
        self._img_h = rh

        system = (
            f"You are a GUI agent controlling an Android phone. "
            f"The screenshot is {rw}x{rh} pixels.\n"
            f"Output your reasoning, then exactly one action.\n\n"
            f"Format:\n"
            f"Thought: <reasoning>\n"
            f"Action: click(x, y) | scroll(up|down|left|right) | "
            f"type(\"text\") | press_back() | finished()\n\n"
            f"Coordinates are pixel positions on the {rw}x{rh} image. "
            f"Output ONLY one Thought + one Action."
        )

        user_parts = [{"image": data_uri}]
        text_block = f"Task: {user_instruction}"
        if memory_text:
            text_block += f"\n\n{memory_text}"
        if ui_elements:
            text_block += (
                f"\n\nUI elements on screen (from accessibility tree):\n"
                f"{ui_elements}"
            )
        text_block += "\n\nWhat is the next action?"
        user_parts.append({"text": text_block})

        messages = [{"role": "system", "content": [{"text": system}]}]
        for h in self.history:
            messages.append(h)
        messages.append({"role": "user", "content": user_parts})

        resp = dashscope.MultiModalConversation.call(
            model=QWEN_VL_MODEL,
            messages=messages,
            temperature=0.0,
        )

        out = resp.output.choices[0].message.content
        if isinstance(out, list):
            text = "\n".join(
                it.get("text", "") for it in out if isinstance(it, dict)
            ).strip()
        else:
            text = str(out).strip()

        action_match = re.search(r"Action:\s*(.+)", text)
        action_str = (action_match.group(1).strip() if action_match
                      else text.split("\n")[-1].strip())
        thought_match = re.search(r"Thought:\s*(.+)", text)
        thought = thought_match.group(1).strip() if thought_match else ""

        self.history.append({"role": "user", "content": user_parts})
        self.history.append({"role": "assistant",
                             "content": [{"text": text}]})
        if len(self.history) > self.keep_steps * 2:
            self.history = self.history[-(self.keep_steps * 2):]

        return text, action_str, thought

    def get_model_input_size(self, orig_w, orig_h):
        scale = min(1.0, self.MAX_SIDE / max(orig_w, orig_h))
        return max(1, int(orig_w * scale)), max(1, int(orig_h * scale))


# ---------------------------------------------------------------------------
# Action parsing + execution
# ---------------------------------------------------------------------------

def model_input_size(orig_w, orig_h, max_side=768):
    if max_side <= 0:
        return orig_w, orig_h
    scale = min(1.0, float(max_side) / max(orig_w, orig_h))
    return max(1, int(orig_w * scale)), max(1, int(orig_h * scale))


def execute_action(driver, action_str, orig_w, orig_h, max_side=768):
    if not action_str:
        return {"kind": "noop", "finished": False, "error": "empty_action"}
    a = action_str.lower()
    if "finished" in a:
        return {"kind": "finished", "finished": True}
    if "click" in a:
        m = re.search(r"\((\d+)\s*,\s*(\d+)", action_str)
        if not m:
            return {"kind": "click", "finished": False, "error": "unparseable_coords"}
        rx, ry = int(m.group(1)), int(m.group(2))
        in_w, in_h = model_input_size(orig_w, orig_h, max_side=max_side)
        tap_x = int(rx / max(1, in_w) * orig_w)
        tap_y = int(ry / max(1, in_h) * orig_h)
        driver.tap(tap_x, tap_y)
        return {"kind": "click", "raw_xy": [rx, ry], "tap_xy": [tap_x, tap_y], "finished": False}
    if "press_home" in a:
        driver.home()
        return {"kind": "press_home", "finished": False}
    if "press_back" in a or ("back" in a and "scroll" not in a):
        driver.back()
        return {"kind": "press_back", "finished": False}
    if "scroll" in a:
        direction = "down"
        if "up" in a: direction = "up"
        elif "left" in a: direction = "left"
        elif "right" in a: direction = "right"
        cx, cy = orig_w // 2, orig_h // 2
        step = max(200, min(orig_w, orig_h) // 3)
        if direction == "down":
            driver.swipe(cx, cy + step, cx, cy - step)
        elif direction == "up":
            driver.swipe(cx, cy - step, cx, cy + step)
        elif direction == "left":
            driver.swipe(cx + step, cy, cx - step, cy)
        elif direction == "right":
            driver.swipe(cx - step, cy, cx + step, cy)
        return {"kind": "scroll", "direction": direction, "finished": False}
    if "type" in a:
        m = re.search(r"type\s*\(\s*content\s*=\s*[\"']([^\"']*)[\"']", action_str, re.I)
        if not m:
            m = re.search(r"type\s*\(\s*[\"']([^\"']*)[\"']", action_str, re.I)
        if m:
            driver.text(m.group(1))
            return {"kind": "type", "text": m.group(1), "finished": False}
        return {"kind": "type", "finished": False, "error": "unparseable_text"}
    if "long_press" in a or "long-press" in a or "longpress" in a:
        m = re.search(r"\((\d+)\s*,\s*(\d+)", action_str)
        if m:
            rx, ry = int(m.group(1)), int(m.group(2))
            in_w, in_h = model_input_size(orig_w, orig_h, max_side=max_side)
            tap_x = int(rx / max(1, in_w) * orig_w)
            tap_y = int(ry / max(1, in_h) * orig_h)
            driver.swipe(tap_x, tap_y, tap_x, tap_y, duration_ms=1000)
            return {"kind": "long_press", "tap_xy": [tap_x, tap_y], "finished": False}
        return {"kind": "long_press", "finished": False, "error": "unparseable_coords"}
    if "wait" in a:
        time.sleep(1.0)
        return {"kind": "wait", "finished": False}
    return {"kind": "unhandled", "finished": False, "error": f"unhandled: {action_str[:120]}"}


# ---------------------------------------------------------------------------
# Prompt + open-loop memory discovery
# ---------------------------------------------------------------------------

def _first_prompt(prompts_json_path: str) -> Optional[str]:
    with open(prompts_json_path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        arr = data.get("prompts") or data.get("tasks") or data.get("items") or []
    elif isinstance(data, list):
        arr = data
    else:
        arr = []
    if not arr:
        return None
    item = arr[0]
    if isinstance(item, str):
        return item.strip() or None
    if isinstance(item, dict):
        for k in ("prompt", "text", "instruction", "query", "user_query", "task", "description"):
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def discover_prompt(task_dir: str) -> Optional[str]:
    chosen = os.path.join(task_dir, "chosen_target.json")
    if os.path.isfile(chosen):
        with open(chosen) as f:
            ct = json.load(f)
        p = ct.get("prompt")
        if isinstance(p, str) and p.strip():
            return p.strip()
    pj = os.path.join(task_dir, "prompts.json")
    if os.path.isfile(pj):
        return _first_prompt(pj)
    return None


def _dedup_ordered(items: List[str]) -> List[str]:
    """Remove duplicate strings preserving first-occurrence order (case-insensitive)."""
    seen: set = set()
    out: List[str] = []
    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _format_memory_from_nm(prompt: str, nm: dict, style: str = "default") -> Optional[str]:
    wpts = _dedup_ordered(nm.get("relevant_waypoint_sequence") or [])
    hints = _dedup_ordered(nm.get("transition_hints") or [])
    if not (wpts or hints):
        return None

    if style == "default":
        lines = [f"[Navigation Memory]", f"Final goal: {prompt}", ""]
        if wpts:
            lines += [f"Relevant waypoint sequence:", " -> ".join(wpts), ""]
        if hints:
            lines += ["Transition hints:"] + [f"- {h}" for h in hints]
        return "\n".join(lines)

    elif style == "numbered":
        lines = [f'"""', f"Task: {prompt}", "Plan:"]
        for i, h in enumerate(hints):
            lines.append(f"  Step {i+1}: {h}")
        lines += ["Begin with Step 1.", '"""']
        return "\n".join(lines)

    elif style == "waypoint_hints":
        lines = [f'"""', f"Task: {prompt}"]
        if wpts:
            lines.append(f"Start at: {wpts[0]}")
        for i, h in enumerate(hints):
            lines.append(f"  → {h}")
            if i + 1 < len(wpts):
                lines.append(f"Arrive at: {wpts[i+1]}")
        lines.append('"""')
        return "\n".join(lines)

    elif style == "natural":
        sentences = []
        if wpts and hints:
            sentences.append(f"Starting from the {wpts[0]}, {hints[0].lower()}.")
            for i in range(1, len(hints)):
                wp = wpts[i] if i < len(wpts) else "the next screen"
                sentences.append(f"Once on the {wp}, {hints[i].lower()}.")
            sentences.append(f"This will take you to the {wpts[-1]}.")
        paragraph = " ".join(sentences)
        return f'"""\nTask: {prompt}\nNavigation guide: {paragraph}\n"""'

    elif style == "waypoint_only":
        if wpts:
            path = " → ".join(wpts)
            return f'"""\nTask: {prompt}\nRoute: {path}\nNavigate through each waypoint in order.\n"""'
        return None

    elif style == "arrow":
        if hints:
            steps_str = " -> ".join(hints)
            first = hints[0][0].lower() + hints[0][1:] if hints[0] else ""
            thought = (
                f"The task is to {prompt.lower().rstrip('.')}. "
                f"To achieve this, I need to navigate through: {steps_str}. "
                f"I will start by {first}."
            )
            return f'"""\nExample 1:\nThought: {thought}\n"""'
        return None

    else:
        # Fallback to default
        return _format_memory_from_nm(prompt, nm, style="default")


def _discover_prestored_target(task_dir: str) -> Tuple[Optional[Tuple[int, int]], Optional[dict]]:
    """Read the pre-stored target (hid, nid) and navigation_memory from task dir."""
    chosen = os.path.join(task_dir, "chosen_target.json")
    if os.path.isfile(chosen):
        with open(chosen) as f:
            ct = json.load(f)
        hid = ct.get("hid")
        nid = ct.get("nid")
        nm = ct.get("navigation_memory") or {}
        if hid is not None and nid is not None:
            return (hid, nid), nm

    nrj = os.path.join(task_dir, "node_retrieval.json")
    if os.path.isfile(nrj):
        with open(nrj) as f:
            d = json.load(f)
        bn = d.get("best_node") or {}
        hid = bn.get("hid")
        nid = bn.get("nid")
        nm = bn.get("navigation_memory") or {}
        if hid is not None and nid is not None:
            return (hid, nid), nm

    return None, None


def discover_open_loop_memory(task_dir: str, prompt: str, memory_format: str = "default") -> Optional[str]:
    """Read the pre-stored memory from the task directory."""
    chosen = os.path.join(task_dir, "chosen_target.json")
    if os.path.isfile(chosen):
        with open(chosen) as f:
            ct = json.load(f)
        nm = ct.get("navigation_memory") or {}
        mem = _format_memory_from_nm(prompt, nm, style=memory_format)
        if mem:
            return mem

    mem_path = os.path.join(task_dir, "memory_text.json")
    if os.path.isfile(mem_path) and memory_format == "default":
        # Only use pre-stored text for default format; other formats need to be generated
        with open(mem_path) as f:
            mt = json.load(f)
        if isinstance(mt, str):
            return mt
        if isinstance(mt, dict):
            return mt.get("memory_text") or mt.get("text")

    nrj = os.path.join(task_dir, "node_retrieval.json")
    if os.path.isfile(nrj):
        with open(nrj) as f:
            d = json.load(f)
        nm = (d.get("best_node") or {}).get("navigation_memory") or {}
        return _format_memory_from_nm(prompt, nm, style=memory_format)
    return None


def _is_system_dialog_text(text: str) -> bool:
    t = (text or "").lower()
    return any(s in t for s in (
        "isn't responding",
        "is not responding",
        "not responding dialog",
        "app not responding",
        "app unresponsiveness",
        "unresponsive system dialog",
    ))


def _is_system_dialog_node(meta: Dict[str, Any]) -> bool:
    nm = meta.get("navigation_memory") or {}
    parts = [
        str(meta.get("visual_page_summary", "")),
        str(meta.get("intent_from_current_screen", "")),
        " ".join(map(str, nm.get("relevant_waypoint_sequence") or [])),
        " ".join(map(str, nm.get("transition_hints") or [])),
    ]
    return _is_system_dialog_text(" ".join(parts))


# ---------------------------------------------------------------------------
# ClosedLoopMemoryStore
# ---------------------------------------------------------------------------

class ClosedLoopMemoryStore:
    """Embedding-backed localization and memory builder.

    Uses the processed pkl for embedding-based matching and waypoint trimming
    to produce dynamic per-step memory text. No graph topology needed — the
    navigation_memory stored per-node already encodes paths.
    """

    def __init__(self, processed_pkl_path: str, sim_floor: float = 0.60):
        with open(processed_pkl_path, "rb") as f:
            self._nodes: Dict[Tuple[int, int], Dict[str, Any]] = pickle.load(f)
        before = len(self._nodes)
        self._nodes = {
            key: meta for key, meta in self._nodes.items()
            if not _is_system_dialog_node(meta)
        }
        removed = before - len(self._nodes)
        if removed:
            print(f"[memory-store] filtered {removed} system-dialog/ANR nodes")
        self.sim_floor = sim_floor
        self._summary_index = self._build_summary_index()

    def _build_summary_index(self) -> Dict[Tuple[int, int], str]:
        """Pre-compute a clean one-line summary per node for waypoint matching."""
        index = {}
        for key, meta in self._nodes.items():
            raw = (meta.get("visual_page_summary") or "").strip()
            first_line = re.split(r"\s*\n\s*", raw)[0].strip() if raw else ""
            index[key] = first_line.lower()
        return index

    # --- target retrieval (2-stage) ----------------------------------------

    def find_target(self, prompt: str) -> Tuple[Optional[Tuple[int, int]], Optional[dict]]:
        """Return (target_key, navigation_memory) using 2-stage retrieval."""
        from node_retrieval import stage1_retrieval, qwen_select_best_node

        candidates = stage1_retrieval(prompt, self._nodes)
        if not candidates:
            return None, None

        candidates_for_llm = []
        for i, c in enumerate(candidates):
            candidates_for_llm.append({
                "index": i,
                "visual_page_summary": str(c.get("visual_page_summary", "")).strip(),
                "intent_from_current_screen": str(c.get("intent_from_current_screen", "")).strip(),
                "summary_score": c.get("summary_score"),
                "intent_score": c.get("intent_score"),
                "matched_sources": c.get("matched_sources"),
            })

        try:
            result = qwen_select_best_node(prompt, candidates_for_llm)
        except Exception as e:
            print(f"[memory-store] qwen rerank failed: {e}, using cosine top-1")
            result = None

        if result and "best_index" in result:
            best = candidates[result["best_index"]]
        else:
            best = candidates[0]

        key = (best["hid"], best["nid"])
        nm = best.get("navigation_memory") or {}
        print(f"[memory-store] target: hid={key[0]} nid={key[1]} "
              f"score={best.get('best_embedding_score', '?'):.3f}")
        return key, nm

    # --- per-step localization ---------------------------------------------

    def localize(self, screenshot_bgr) -> Optional[Tuple[Tuple[int, int], float, str]]:
        """Localize via VLM page description + dual-embedding cosine match."""
        try:
            page_desc = _describe_screenshot(screenshot_bgr)
        except Exception as e:
            print(f"[localize] vlm-describe failed: {e}")
            return None
        try:
            q = _embed_text(page_desc)
        except Exception as e:
            print(f"[localize] embedding failed: {e}")
            return None

        best_key = None
        best_score = -1.0
        for key, meta in self._nodes.items():
            for field in ("visual_page_summary_embedding", "intent_from_current_screen_embedding"):
                emb = meta.get(field)
                if emb is None:
                    continue
                s = _cosine(q, emb)
                if s > best_score:
                    best_score = s
                    best_key = key
        if best_key is None:
            return None
        return best_key, best_score, page_desc

    # --- closed-loop annotation (additive, not subtractive) -----------------

    def build_localization_annotation(
        self,
        curr_key: Tuple[int, int],
        page_desc: str,
        target_nm: Optional[dict],
        target_key: Optional[Tuple[int, int]] = None,
        is_stuck: bool = False,
    ) -> Optional[str]:
        """Build a rich annotation describing the agent's current state.

        APPENDED to the unchanged open-loop memory text. Three signals:
        1. Current position + progress on the waypoint plan
        2. Next-action hint from transition hints
        3. Near-target or stuck warnings
        """
        curr_summary = self._summary_index.get(curr_key, "")
        if not curr_summary:
            return None

        waypoints = list((target_nm or {}).get("relevant_waypoint_sequence") or [])
        hints = list((target_nm or {}).get("transition_hints") or [])

        lines = [f"\nCurrent position: {page_desc}"]

        # Track forward progress — only match waypoints at or after the last matched index
        if not hasattr(self, "_progress_idx"):
            self._progress_idx = 0

        matched_idx = -1
        if curr_summary and waypoints:
            # First try to match from current progress point forward
            for i in range(self._progress_idx, len(waypoints)):
                wp_lower = waypoints[i].lower().strip()
                if not wp_lower:
                    continue
                words_wp = set(wp_lower.split())
                words_curr = set(curr_summary.split())
                if not words_wp:
                    continue
                overlap = len(words_wp & words_curr) / len(words_wp)
                if overlap >= 0.6:
                    matched_idx = i
                    self._progress_idx = i
                    break

            if matched_idx >= 0:
                remaining = len(waypoints) - matched_idx - 1
                if remaining > 0:
                    lines.append(
                        f"Progress: step {matched_idx+1} of {len(waypoints)}, "
                        f"~{remaining} steps remaining"
                    )
                else:
                    lines.append(
                        f"Progress: you appear to be at or near the "
                        f"final waypoint in the plan"
                    )

        if matched_idx >= 0 and matched_idx < len(hints):
            lines.append(f"Next action: {hints[matched_idx]}")
        elif self._progress_idx < len(hints):
            lines.append(f"Next action: {hints[self._progress_idx]}")

        if target_key and curr_key[0] == target_key[0]:
            lines.append(
                "Note: you appear to be on the target page. "
                "If the screen matches the task goal, use finished()."
            )
        elif matched_idx >= 0 and matched_idx >= len(waypoints) - 2:
            lines.append(
                "Note: you are very close to the target. "
                "Check if the current screen already satisfies the task."
            )

        if is_stuck:
            lines.append(
                "Warning: you seem to be repeating actions. "
                "Try a different approach or check if you have already "
                "reached the target."
            )

        return "\n".join(lines)


    # --- plan editing (subtractive — rewrites the memory) --------------------

    def _embed_waypoints(self, waypoints: List[str]) -> List[np.ndarray]:
        """Embed waypoint labels. Cached per-call to avoid redundant API hits."""
        if not hasattr(self, "_wp_emb_cache"):
            self._wp_emb_cache: Dict[str, np.ndarray] = {}
        out = []
        for wp in waypoints:
            key = wp.strip().lower()
            if key not in self._wp_emb_cache:
                try:
                    self._wp_emb_cache[key] = _embed_text(wp)
                except Exception:
                    self._wp_emb_cache[key] = np.zeros(1)
            out.append(self._wp_emb_cache[key])
        return out

    def _match_waypoint_by_embedding(
        self, page_desc: str, waypoints: List[str],
        wp_embeddings: List[np.ndarray], threshold: float = 0.55,
    ) -> int:
        """Find the best-matching waypoint by embedding similarity.

        Returns the index of the last matched waypoint (highest index with
        sim >= threshold), or -1 if none match. Preferring the latest match
        means we trim the most completed waypoints.
        """
        try:
            q = _embed_text(page_desc)
        except Exception:
            return -1
        best_idx = -1
        best_sim = -1.0
        for i, emb in enumerate(wp_embeddings):
            if emb.size <= 1:
                continue
            s = _cosine(q, emb)
            if s >= threshold and s > best_sim:
                best_sim = s
                best_idx = i
        return best_idx

    def build_edited_plan(
        self,
        prompt: str,
        curr_key: Tuple[int, int],
        page_desc: str,
        target_nm: Optional[dict],
        target_key: Optional[Tuple[int, int]] = None,
        is_stuck: bool = False,
    ) -> Optional[str]:
        """Build a REPLACEMENT memory text with completed waypoints trimmed.

        Fixes over v1:
        - Deduplicates waypoints/hints before processing
        - Uses embedding similarity for waypoint matching (not word overlap)
        - No off-path penalty — unmatched positions get the full plan
          with current position context, not a "go back" instruction
        """
        waypoints = _dedup_ordered(
            (target_nm or {}).get("relevant_waypoint_sequence") or [])
        hints = _dedup_ordered(
            (target_nm or {}).get("transition_hints") or [])
        if not waypoints:
            return None

        # Trim waypoints past the target: find the waypoint closest to the
        # goal (by embedding sim to the prompt) and drop everything after it.
        # This removes noise from merged alternate-trajectory suffixes.
        if len(waypoints) > 2:
            wp_embs = self._embed_waypoints(waypoints)
            if not hasattr(self, "_prompt_emb_cache"):
                self._prompt_emb_cache: Dict[str, np.ndarray] = {}
            pk = prompt.strip().lower()
            if pk not in self._prompt_emb_cache:
                try:
                    self._prompt_emb_cache[pk] = _embed_text(prompt)
                except Exception:
                    self._prompt_emb_cache[pk] = np.zeros(1)
            prompt_emb = self._prompt_emb_cache[pk]
            if prompt_emb.size > 1:
                best_target_idx = -1
                best_target_sim = -1.0
                for i, emb in enumerate(wp_embs):
                    if emb.size <= 1:
                        continue
                    s = _cosine(prompt_emb, emb)
                    if s > best_target_sim:
                        best_target_sim = s
                        best_target_idx = i
                if best_target_idx >= 0 and best_target_idx < len(waypoints) - 1:
                    waypoints = waypoints[:best_target_idx + 1]
                    hints = hints[:best_target_idx + 1]

        # Near target — tell model to check and finish
        if target_key and curr_key[0] == target_key[0]:
            return (
                f"[Navigation Memory]\n"
                f"Final goal: {prompt}\n\n"
                f"You have reached the target page. "
                f"Verify the screen matches the task goal, then use finished()."
            )

        # Match current position to waypoint sequence via embedding similarity
        wp_embeddings = self._embed_waypoints(waypoints)
        matched_idx = self._match_waypoint_by_embedding(
            page_desc, waypoints, wp_embeddings)

        # On-path: trim completed waypoints
        if matched_idx >= 0:
            remaining_wpts = waypoints[matched_idx + 1:]
            remaining_hints = hints[matched_idx + 1:] if matched_idx + 1 < len(hints) else []
            if not remaining_wpts:
                return (
                    f"[Navigation Memory]\n"
                    f"Final goal: {prompt}\n\n"
                    f"You are at the last waypoint. "
                    f"Check if the current screen satisfies the task, then use finished()."
                )
            lines = [
                f"[Navigation Memory]",
                f"Final goal: {prompt}",
                f"",
                f"Current position: {page_desc}",
                f"",
                f"Remaining waypoints:",
                " -> ".join(remaining_wpts),
            ]
            if remaining_hints:
                lines += ["", "Next steps:"]
                lines += [f"- {h}" for h in remaining_hints]
            if is_stuck:
                lines.append(
                    "\nYou seem stuck. Try a different approach."
                )
            return "\n".join(lines)

        # No match: show full plan with current position context (no penalty)
        lines = [
            f"[Navigation Memory]",
            f"Final goal: {prompt}",
            f"",
            f"Current position: {page_desc}",
            f"",
            f"Waypoint sequence:",
            " -> ".join(waypoints),
        ]
        if hints:
            lines += ["", "Transition hints:"]
            lines += [f"- {h}" for h in hints]
        if is_stuck:
            lines.append(
                "\nYou seem stuck. Try a different approach."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stuck detector
# ---------------------------------------------------------------------------

class StuckDetector:
    def __init__(self, action_repeat_threshold: int = 2, hid_stall_threshold: int = 3):
        self._action_repeat = action_repeat_threshold
        self._hid_stall = hid_stall_threshold
        self._recent_actions: List[str] = []
        self._recent_hids: List[Optional[int]] = []

    def update(self, action_str: str, curr_hid: Optional[int]):
        self._recent_actions.append((action_str or "").strip().lower())
        self._recent_hids.append(curr_hid)

    def is_stuck(self) -> bool:
        n = self._action_repeat
        if len(self._recent_actions) >= n:
            tail = self._recent_actions[-n:]
            if all(a == tail[0] for a in tail) and tail[0]:
                return True
        m = self._hid_stall
        if len(self._recent_hids) >= m:
            tail = self._recent_hids[-m:]
            if tail[0] is not None and all(h == tail[0] for h in tail):
                return True
        return False

    def reset(self):
        self._recent_actions.clear()
        self._recent_hids.clear()


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_one_task(
    task_dir: str,
    ui_tars,
    driver: AdbDriver,
    memory_store: Optional[ClosedLoopMemoryStore],
    app_pkg: str,
    app_activity: Optional[str],
    max_steps: int,
    out_dir: str,
    memory_mode: str,
    sim_floor: float,
    coord_max_side: int = 768,
    clear_between_tasks: bool = False,
    tap_home_after_launch: bool = False,
    task_completion: bool = False,
    memory_format: str = "default",
):
    import cv2
    task_id = os.path.basename(task_dir.rstrip("/"))
    prompt = discover_prompt(task_dir)
    if prompt is None:
        return {"task_id": task_id, "error": "no_prompt"}

    out_task = os.path.join(out_dir, task_id)
    os.makedirs(out_task, exist_ok=True)
    traj_path = os.path.join(out_task, "trajectory.jsonl")
    summary_path = os.path.join(out_task, "summary.json")

    open_loop_fallback = discover_open_loop_memory(task_dir, prompt, memory_format=memory_format)
    mode_effective = memory_mode
    target_key = None
    target_nm = None

    # Read pre-stored target from the task directory (same target for both modes)
    target_key, target_nm = _discover_prestored_target(task_dir)
    if target_key is not None:
        print(f"  [{task_id}] pre-stored target: hid={target_key[0]} nid={target_key[1]}")

    if memory_mode in ("closed_loop", "plan_edit") and (memory_store is None or target_nm is None):
        print(f"  [{task_id}] no memory store or no pre-stored target; degrading to open_loop")
        mode_effective = "open_loop"

    fixed_memory = open_loop_fallback

    # Save target info for inspection
    target_info_path = os.path.join(out_task, "target_info.json")
    os.makedirs(out_task, exist_ok=True)
    with open(target_info_path, "w") as f:
        json.dump({
            "prompt": prompt,
            "target_hid": target_key[0] if target_key else None,
            "target_nid": target_key[1] if target_key else None,
            "target_waypoints": (target_nm or {}).get("relevant_waypoint_sequence"),
            "target_hints": (target_nm or {}).get("transition_hints"),
            "open_loop_memory": fixed_memory,
            "memory_mode": memory_mode,
        }, f, indent=2)

    # Copy target node screenshot into output for easy visual comparison
    import shutil
    _target_copied = False
    chosen_path = os.path.join(task_dir, "chosen_target.json")
    if os.path.isfile(chosen_path):
        try:
            with open(chosen_path) as f:
                ct = json.load(f)
            src_screenshot = ct.get("source_screenshot")
            if src_screenshot and os.path.isfile(src_screenshot):
                ext = os.path.splitext(src_screenshot)[1] or ".jpg"
                shutil.copy2(src_screenshot, os.path.join(out_task, f"target_node{ext}"))
                _target_copied = True
        except Exception:
            pass
    if not _target_copied:
        for candidate in ("node_retrieval.png", "node_retrieval.jpg"):
            src = os.path.join(task_dir, candidate)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(out_task, "target_node.jpg"))
                break

    # Reset app
    driver.home()
    driver.force_stop(app_pkg)
    time.sleep(1.0)
    driver.launch(app_pkg, app_activity)
    time.sleep(3.5)
    if tap_home_after_launch:
        driver.tap(30, 30)
        time.sleep(2.0)
    ui_tars.clean_history()
    if memory_store is not None:
        memory_store._progress_idx = 0

    finished_reason = "max_steps"
    steps_taken = 0
    stuck_detector = StuckDetector()
    consecutive_fallbacks = 0
    localize_log: List[Dict[str, Any]] = []

    with open(traj_path, "w") as tlog:
        for step_idx in range(1, max_steps + 1):
            try:
                shot = driver.screenshot_bgr()
            except Exception as e:
                tlog.write(json.dumps({"step": step_idx, "error": f"screencap-fail: {e}"}) + "\n")
                finished_reason = "screencap_fail"
                break
            orig_h, orig_w = shot.shape[:2]

            memory_text = None
            curr_key = None
            curr_sim = None
            page_desc = None

            if mode_effective == "plan_edit" and memory_store is not None and target_key is not None:
                if step_idx == 1:
                    memory_text = fixed_memory
                    localize_log.append({
                        "step": step_idx,
                        "page_desc": None,
                        "curr_hid": None, "curr_nid": None,
                        "sim": None,
                        "used_plan_edit": False,
                        "skipped": "step_1_warmup",
                    })
                elif (loc := memory_store.localize(shot)) is not None:
                    curr_key, curr_sim, page_desc = loc
                    if curr_sim >= sim_floor:
                        edited = memory_store.build_edited_plan(
                            prompt, curr_key, page_desc, target_nm,
                            target_key=target_key,
                            is_stuck=stuck_detector.is_stuck(),
                        )
                        memory_text = edited if edited else fixed_memory
                        consecutive_fallbacks = 0
                    else:
                        memory_text = fixed_memory
                        consecutive_fallbacks += 1
                else:
                    memory_text = fixed_memory
                    consecutive_fallbacks += 1

                if step_idx > 1:
                    if consecutive_fallbacks >= 3:
                        print(f"  [{task_id}] step {step_idx}: 3 consecutive localization failures, "
                              f"degrading to open_loop")
                        mode_effective = "open_loop"
                        memory_text = fixed_memory

                    localize_log.append({
                        "step": step_idx,
                        "page_desc": page_desc,
                        "curr_hid": curr_key[0] if curr_key else None,
                        "curr_nid": curr_key[1] if curr_key else None,
                        "sim": curr_sim,
                        "used_plan_edit": memory_text != fixed_memory,
                    })

            elif mode_effective == "closed_loop" and memory_store is not None and target_key is not None:
                annotation = None
                loc = memory_store.localize(shot)
                if loc is not None:
                    curr_key, curr_sim, page_desc = loc
                    if curr_sim >= sim_floor:
                        annotation = memory_store.build_localization_annotation(
                            curr_key, page_desc, target_nm,
                            target_key=target_key,
                            is_stuck=stuck_detector.is_stuck(),
                        )
                        if fixed_memory and annotation:
                            memory_text = fixed_memory + annotation
                        else:
                            memory_text = fixed_memory
                        consecutive_fallbacks = 0
                    else:
                        memory_text = fixed_memory
                        consecutive_fallbacks += 1
                else:
                    memory_text = fixed_memory
                    consecutive_fallbacks += 1

                if consecutive_fallbacks >= 3:
                    print(f"  [{task_id}] step {step_idx}: 3 consecutive localization failures, "
                          f"degrading to open_loop")
                    mode_effective = "open_loop"
                    memory_text = fixed_memory

                localize_log.append({
                    "step": step_idx,
                    "page_desc": page_desc,
                    "curr_hid": curr_key[0] if curr_key else None,
                    "curr_nid": curr_key[1] if curr_key else None,
                    "sim": curr_sim,
                    "used_fallback": annotation is None,
                })
            elif mode_effective == "open_loop":
                memory_text = fixed_memory
            # else: mode_effective == "none", memory_text stays None

            if _is_system_dialog_text(page_desc or ""):
                shot_path = os.path.join(out_task, f"step_{step_idx:03d}.png")
                cv2.imwrite(shot_path, shot)
                dismissed = driver.tap_system_dialog_button(("Close app", "OK"))
                if not dismissed:
                    driver.back()
                time.sleep(1.0)
                driver.force_stop(app_pkg)
                time.sleep(0.5)
                driver.launch(app_pkg, app_activity)
                time.sleep(3.5)
                ui_tars.clean_history()
                stuck_detector.reset()
                if memory_store is not None:
                    memory_store._progress_idx = 0
                tlog.write(json.dumps({
                    "step": step_idx,
                    "thought": "system dialog recovery",
                    "action_str": "recover_from_system_dialog()",
                    "exec": {
                        "kind": "system_dialog_recovery",
                        "dismissed": dismissed,
                        "finished": False,
                    },
                    "screenshot": shot_path,
                    "memory_mode_effective": mode_effective,
                    "localization": {
                        "page_desc": page_desc,
                        "curr_hid": curr_key[0] if curr_key else None,
                        "curr_nid": curr_key[1] if curr_key else None,
                        "sim": curr_sim,
                        "on_path": False,
                    },
                    "memory_text": memory_text,
                }) + "\n")
                tlog.flush()
                steps_taken = step_idx
                continue

            ui_elements = None
            if task_completion:
                ui_elements = driver.get_ui_elements(max_elements=30)

            try:
                step_kwargs = {"memory_text": memory_text}
                if ui_elements and hasattr(ui_tars, "step"):
                    try:
                        import inspect
                        sig = inspect.signature(ui_tars.step)
                        if "ui_elements" in sig.parameters:
                            step_kwargs["ui_elements"] = ui_elements
                    except Exception:
                        pass
                model_out, action_str, thought = ui_tars.step(
                    prompt, shot, **step_kwargs,
                )
            except Exception as e:
                tlog.write(json.dumps({"step": step_idx, "error": f"ui_tars-fail: {e}"}) + "\n")
                finished_reason = "ui_tars_fail"
                break

            res = execute_action(driver, action_str, orig_w, orig_h,
                                 max_side=coord_max_side)

            shot_path = os.path.join(out_task, f"step_{step_idx:03d}.png")
            cv2.imwrite(shot_path, shot)

            # Write per-step memory file for easy inspection
            if memory_text:
                mem_path = os.path.join(out_task, f"step_{step_idx:03d}_memory.txt")
                with open(mem_path, "w") as mf:
                    mf.write(memory_text)

            tlog.write(json.dumps({
                "step": step_idx,
                "thought": thought,
                "action_str": action_str,
                "exec": res,
                "screenshot": shot_path,
                "memory_mode_effective": mode_effective,
                "localization": {
                    "page_desc": page_desc,
                    "curr_hid": curr_key[0] if curr_key else None,
                    "curr_nid": curr_key[1] if curr_key else None,
                    "sim": curr_sim,
                    "on_path": (curr_key is not None and curr_sim is not None
                                and curr_sim >= sim_floor),
                },
                "memory_text": memory_text,
            }) + "\n")
            tlog.flush()
            steps_taken = step_idx

            if res.get("finished"):
                finished_reason = "finished_action"
                break

            stuck_detector.update(action_str, curr_key[0] if curr_key else None)
            if stuck_detector.is_stuck():
                print(f"  [{task_id}] step {step_idx}: stuck detected, pressing back")
                driver.back()
                stuck_detector.reset()
                time.sleep(1.0)
            else:
                time.sleep(3.0)

    # Final screenshot
    final_screenshot_path = ""
    try:
        final_shot = driver.screenshot_bgr()
        final_screenshot_path = os.path.join(out_task, "agent_screenshot.png")
        cv2.imwrite(final_screenshot_path, final_shot)
    except Exception:
        pass

    summary = {
        "task_id": task_id,
        "task_dir": task_dir,
        "prompt": prompt,
        "memory_mode_requested": memory_mode,
        "memory_mode_effective": mode_effective,
        "target_hid": target_key[0] if target_key else None,
        "target_nid": target_key[1] if target_key else None,
        "localize_log": localize_log,
        "steps_taken": steps_taken,
        "finished_reason": finished_reason,
        "agent_screenshot": final_screenshot_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--app", required=True,
                   help=f"App short name ({', '.join(sorted(APP_REGISTRY))}) "
                        f"or explicit paths via --tasks-root etc.")
    p.add_argument("--device", required=True, help="adb device serial")
    p.add_argument("--memory-mode", choices=["closed_loop", "open_loop", "none", "plan_edit"],
                   default="open_loop")
    p.add_argument("--memory-format",
                   choices=["default", "numbered", "waypoint_hints", "natural",
                            "waypoint_only", "arrow"],
                   default="default",
                   help="Memory text format style for open_loop mode. "
                        "default = [Navigation Memory] block, "
                        "numbered = Step 1/2/3, "
                        "waypoint_hints = Start at → Arrive at, "
                        "natural = paragraph, "
                        "waypoint_only = screen names only, "
                        "arrow = legacy Example 1: Thought: format")
    p.add_argument("--sim-floor", type=float, default=0.60,
                   help="Localization similarity floor (default 0.60)")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--adb-path",
                   default="adb")
    p.add_argument("--ui-tars-url",
                   default="http://localhost:8000/v1/chat/completions")
    p.add_argument("--local-model-name",
                   default=None,
                   help="OpenAI model name for --backbone qwen_vl_local. "
                        "Defaults to Qwen2.5-VL-7B-Instruct.")
    p.add_argument("--keep-steps", type=int, default=3)
    p.add_argument("--include-tasks", nargs="*", default=None)
    p.add_argument("--exclude-tasks", nargs="*", default=None)
    p.add_argument("--clear-between-tasks", action="store_true",
                   help="Use 'pm clear' instead of 'force-stop' between tasks (resets app data)")
    p.add_argument("--tap-home-after-launch", action="store_true",
                   help="Tap home icon (top-left) after app launch to clear restored tabs (useful for Chrome)")

    p.add_argument("--backbone", choices=["ui_tars", "mai_ui", "qwen_vl", "qwen_vl_local"],
                   default="ui_tars",
                   help="Action model backbone: ui_tars (local 7B), mai_ui (MAI-UI-8B), qwen_vl (API 72B), or qwen_vl_local (local 7B Qwen)")
    p.add_argument("--coord-max-side", type=int, default=None,
                   help="Override coord_max_side (image longest-side resize). "
                        "Default: 0 (full res) for ui_tars, 1280 for qwen_vl. "
                        "Use 0 for full device resolution (no downscale).")

    p.add_argument("--task-completion", action="store_true",
                   help="DroidTask mode: add XML accessibility hints per step "
                        "for task-completion benchmarks (not for navigation-only)")

    p.add_argument("--graph-type",
                   choices=["none", "random", "auto_bfs", "manual"],
                   default=None,
                   help="Graph condition for memory. Resolves PKL from graphs/<type>/<app>.pkl")

    # Override paths (used when --app doesn't resolve correctly)
    p.add_argument("--tasks-root", default=None)
    p.add_argument("--processed-pkl", default=None)
    p.add_argument("--graph-pkl", default=None)
    p.add_argument("--app-package", default=None)
    p.add_argument("--app-activity", default=None)

    args = p.parse_args()

    _setup_dashscope()
    _ensure_dashscope_key()

    # Resolve app info
    if args.app in APP_REGISTRY:
        info = _resolve_app_paths(args.app, graph_type=args.graph_type)
    else:
        info = {"package": args.app_package, "activity": args.app_activity,
                "tasks_dir": None, "processed_pkl": args.processed_pkl,
                "graph_pkl": args.graph_pkl}

    if args.tasks_root:
        tasks_root = args.tasks_root
    elif info.get("tasks_dir"):
        env_root = os.environ.get("TAGNAV_TASKS_ROOT")
        if env_root and os.path.isdir(os.path.join(env_root, info["tasks_dir"])):
            tasks_root = os.path.join(env_root, info["tasks_dir"])
        else:
            tasks_root = os.path.join(REPO_ROOT, "tasks", info["tasks_dir"])
    else:
        tasks_root = None
    if args.graph_type == "none" and args.memory_mode != "none":
        args.memory_mode = "none"
    processed_pkl = args.processed_pkl or info.get("processed_pkl")
    graph_pkl = args.graph_pkl or info.get("graph_pkl")
    app_pkg = args.app_package or info["package"]
    app_activity = args.app_activity or info.get("activity")

    if not app_pkg:
        print("ERROR: --app-package required when --app is not in registry")
        sys.exit(2)
    if not os.path.isdir(tasks_root):
        print(f"ERROR: tasks root not found: {tasks_root}")
        print("       Benchmark tasks are not included in the minimal repo checkout.")
        print("       Provide --tasks-root, or create tasks/<app_tasks_dir>/ with task files")
        print("       containing chosen_target.json, memory_text.json, or node_retrieval.json.")
        sys.exit(2)

    # Load action backbone
    driver = AdbDriver(adb_path=args.adb_path, device=args.device)
    sz = driver.get_screen_size()
    print(f"Device {args.device} screen size: {sz}")

    if args.backbone in ("qwen_vl", "qwen_vl_local"):
        if args.backbone == "qwen_vl_local":
            default_max_side = QwenVLLocalAgent.MAX_SIDE
        else:
            default_max_side = QwenVLAgent.MAX_SIDE
        if args.coord_max_side is not None:
            if args.coord_max_side == 0:
                coord_max_side = max(sz) if sz else default_max_side
            else:
                coord_max_side = args.coord_max_side
        else:
            coord_max_side = default_max_side
        if args.backbone == "qwen_vl_local":
            local_model_name = args.local_model_name or "Qwen2.5-VL-7B-Instruct"
            agent = QwenVLLocalAgent(server_url=args.ui_tars_url,
                                     model_name=local_model_name,
                                     keep_steps=args.keep_steps)
            agent.MAX_SIDE = coord_max_side
            print(f"Backbone: {local_model_name} local (vLLM, coord_max_side={coord_max_side})")
        else:
            agent = QwenVLAgent(keep_steps=args.keep_steps)
            agent.MAX_SIDE = coord_max_side
            print(f"Backbone: qwen-vl-max (API, coord_max_side={coord_max_side})")
    elif args.backbone == "mai_ui":
        # MAI-UI-8B via refactored exploration code
        _mai_sys_path = os.path.join(os.path.dirname(REPO_ROOT), "AgentNavigator", "refactored", "exploration")
        if _mai_sys_path not in sys.path:
            sys.path.insert(0, _mai_sys_path)
        from Agents.MAI_UI.MAI_UI import MAI_UI
        _base = str(args.ui_tars_url or "").strip()
        if _base.endswith("/chat/completions"):
            _base = _base[: -len("/chat/completions")]
        # MAI_UI expects base_url ending with /v1
        if not _base.endswith("/v1"):
            _base = _base.rstrip("/") + "/v1"
        if args.coord_max_side is not None:
            coord_max_side = args.coord_max_side
        else:
            coord_max_side = 0
        _mai_client = MAI_UI(
            llm_base_url=_base,
            runtime_conf={"history_n": args.keep_steps, "resize_factor": 0.5},
        )
        agent = UITarsBenchmarkAdapter(_mai_client)
        print(f"Backbone: MAI-UI-8B (local, coord_max_side={coord_max_side})")
    else:
        from UI_TARS_1_5 import UITarsMobileHistoryAgent
        _base = str(args.ui_tars_url or "").strip()
        if _base.endswith("/chat/completions"):
            _base = _base[: -len("/chat/completions")]
        if _base.endswith("/v1"):
            _base = _base[: -len("/v1")]
        if args.coord_max_side is not None:
            coord_max_side = args.coord_max_side
        else:
            coord_max_side = 0
        agent = UITarsBenchmarkAdapter(UITarsMobileHistoryAgent(
            base_url=_base,
            model="UI-TARS-1.5-7B",
            keep_last_assistant_turns=args.keep_steps,
            max_image_side=coord_max_side,
        ))
        print(f"Backbone: UI-TARS-1.5-7B (local, coord_max_side={coord_max_side})")

    # Load memory store
    memory_store = None
    if args.memory_mode in ("closed_loop", "open_loop", "plan_edit") and processed_pkl:
        if not os.path.isfile(processed_pkl):
            print(f"WARN: processed pkl not found: {processed_pkl}")
        else:
            print(f"Loading processed pkl: {processed_pkl}")
            memory_store = ClosedLoopMemoryStore(
                processed_pkl, sim_floor=args.sim_floor,
            )
            print(f"  processed nodes: {len(memory_store._nodes)}")

    if args.memory_mode in ("open_loop", "closed_loop", "plan_edit"):
        expected_memory_files = (
            "chosen_target.json",
            "memory_text.json",
            "node_retrieval.json",
        )
        has_task_memory = False
        for root, _dirs, files in os.walk(tasks_root):
            if any(name in files for name in expected_memory_files):
                has_task_memory = True
                break
        if not has_task_memory:
            print(
                "WARN: no task-side navigation memory files found under "
                f"{tasks_root}"
            )
            print(
                "      Expected at least one of: "
                + ", ".join(expected_memory_files)
            )
            print(
                "      Runs can still execute, but memory_text will be null and "
                "no step_*_memory.txt files will be written."
            )

    # Discover tasks
    tasks = sorted(d for d in os.listdir(tasks_root)
                   if os.path.isdir(os.path.join(tasks_root, d)))
    if args.include_tasks:
        tasks = [t for t in tasks if t in set(args.include_tasks)]
    if args.exclude_tasks:
        tasks = [t for t in tasks if t not in set(args.exclude_tasks)]
    print(f"\nRunning {len(tasks)} tasks from {tasks_root}")
    print(f"Mode: {args.memory_mode} | sim_floor: {args.sim_floor} | "
          f"max_steps: {args.max_steps}")

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for tid in tasks:
        td = os.path.join(tasks_root, tid)
        print(f"\n=== {tid} ===")
        try:
            s = run_one_task(
                td, agent, driver, memory_store,
                app_pkg, app_activity,
                args.max_steps, args.out_dir,
                memory_mode=args.memory_mode,
                sim_floor=args.sim_floor,
                coord_max_side=coord_max_side,
                clear_between_tasks=args.clear_between_tasks,
                tap_home_after_launch=args.tap_home_after_launch,
                task_completion=args.task_completion,
                memory_format=getattr(args, 'memory_format', 'default'),
            )
            summaries.append(s)
            print(f"  {tid}: steps={s.get('steps_taken', '?')} "
                  f"reason={s.get('finished_reason', '?')} "
                  f"mode={s.get('memory_mode_effective', '?')}")
        except KeyboardInterrupt:
            print("Interrupted.")
            break
        except Exception as e:
            print(f"  {tid}: FAIL {e}")
            summaries.append({"task_id": tid, "error": str(e)})

    with open(os.path.join(args.out_dir, "all_summaries.json"), "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nDone. {len(summaries)} tasks. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
