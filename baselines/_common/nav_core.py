"""Shared adb + task plumbing for the baseline benchmark adapters.

This module is a trimmed, self-contained extraction from the TAG-Nav navigation
benchmark runner. It provides the small pieces the baseline adapters share so each
adapter does not re-implement device control or task discovery:

  - APP_REGISTRY      : app name -> {package, activity, tasks_dir}
  - AdbDriver         : thin adb wrapper (screenshot, tap/swipe/text, launch, UI dump)
  - execute_action    : map a UI-TARS-style action string to a device action
  - discover_prompt   : read a task's prompt from chosen_target.json / prompts.json

Third-party deps:
  - `opencv-python` (cv2) is imported lazily, ONLY inside AdbDriver.screenshot_bgr().
    Adapters that capture screenshots another way do not need it.
  - Everything else is stdlib.

No machine-specific paths, hosts, serials, or keys are hardcoded here. The device
serial is supplied by the caller (see ANDROID_DEVICE_SERIAL in baselines/.env.example).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# App registry: app name -> package / launch activity / task-folder name.
# tasks_dir values are pinned app-version identifiers (not personal data); the
# actual task folders live under your TAGNAV_TASKS_ROOT.
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
        "activity": "com.google.android.apps.youtube.app.WatchWhileActivity",
        "tasks_dir": "youtube_21.05.264",
    },
    "airbnb": {
        "package": "com.airbnb.android",
        "activity": ".activities.HomeActivity",
        "tasks_dir": "airbnb_26.06",
    },
}


# ---------------------------------------------------------------------------
# adb device driver
# ---------------------------------------------------------------------------

class AdbDriver:
    def __init__(self, adb_path: str = "adb", device: Optional[str] = None,
                 screencap_path: str = "/sdcard/_uitars_cap.png"):
        self.adb = adb_path
        self.device = device or os.environ.get("ANDROID_DEVICE_SERIAL", "emulator-5554")
        self.cap = screencap_path

    def _run(self, *args, capture: bool = True, timeout: int = 30):
        cmd = [self.adb, "-s", self.device] + list(args)
        return subprocess.run(cmd, capture_output=capture, timeout=timeout)

    def shell(self, line: str, timeout: int = 30):
        return self._run("shell", line, timeout=timeout)

    def screenshot_bgr(self):
        """Capture the screen as an OpenCV BGR ndarray. Requires opencv-python."""
        import cv2  # lazy: only needed if you actually capture screenshots here
        local_path = os.path.join(tempfile.gettempdir(), "_uitars_local.png")
        self.shell(f"screencap -p {self.cap}", timeout=15)
        self._run("pull", self.cap, local_path, timeout=20)
        img = cv2.imread(local_path)
        if img is None:
            raise RuntimeError(f"failed to read screenshot at {local_path}")
        return img

    def force_stop(self, pkg: str):
        self.shell(f"am force-stop {pkg}", timeout=10)

    def clear_data(self, pkg: str):
        self.shell(f"pm clear {pkg}", timeout=15)

    def launch(self, pkg: str, activity: Optional[str]):
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

    def swipe(self, x1, y1, x2, y2, duration_ms: int = 300):
        self.shell(f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}", timeout=5)

    def text(self, text: str):
        safe = (text or "").replace(" ", "%s").replace("'", "\\'").replace('"', '\\"')
        self.shell(f"input text '{safe}'", timeout=10)

    def get_screen_size(self) -> Optional[Tuple[int, int]]:
        r = self.shell("wm size", timeout=5)
        s = (r.stdout.decode("utf-8", errors="replace") if r.stdout else "")
        m = re.search(r"(\d+)\s*x\s*(\d+)", s)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def get_ui_elements(self, max_elements: int = 30) -> Optional[str]:
        """Dump the UI hierarchy and return actionable elements as a compact text block."""
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
# Action execution: map a UI-TARS-style action string to a device action.
# Coordinates from the model are in the model's input resolution and are scaled
# back to the device's native resolution via model_input_size().
# ---------------------------------------------------------------------------

def model_input_size(orig_w: int, orig_h: int, max_side: int = 768) -> Tuple[int, int]:
    if max_side <= 0:
        return orig_w, orig_h
    scale = min(1.0, float(max_side) / max(orig_w, orig_h))
    return max(1, int(orig_w * scale)), max(1, int(orig_h * scale))


def execute_action(driver: "AdbDriver", action_str: str, orig_w: int, orig_h: int,
                   max_side: int = 768) -> Dict[str, Any]:
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
        if "up" in a:
            direction = "up"
        elif "left" in a:
            direction = "left"
        elif "right" in a:
            direction = "right"
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
# Task prompt discovery
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
    """Return the task's natural-language goal from chosen_target.json or prompts.json."""
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


__all__ = [
    "APP_REGISTRY",
    "AdbDriver",
    "model_input_size",
    "execute_action",
    "discover_prompt",
]
