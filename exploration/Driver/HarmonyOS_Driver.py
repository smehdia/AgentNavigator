import json
import re
import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np
import xml.etree.ElementTree as ET

from Driver.BaseDriver import BaseDriver


class HarmonyDriver(BaseDriver):
    def _hdc_prefix(self):
        if not self.device_id:
            raise ValueError("HarmonyDriver requires device_id for hdc.")
        return ["hdc", "-t", self.device_id]

    def _hdc_out(self, args, timeout=20) -> bytes:
        return subprocess.check_output(self._hdc_prefix() + args, stderr=subprocess.STDOUT, timeout=timeout)

    def _hdc_run(self, args, timeout=20) -> None:
        subprocess.run(self._hdc_prefix() + args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout)

    def check_device(self) -> bool:
        try:
            out = subprocess.check_output(["hdc", "list", "targets"], stderr=subprocess.STDOUT, timeout=5).decode(
                "utf-8", "ignore"
            )
            return bool(self.device_id and self.device_id in out)
        except Exception:
            return False


    def is_keyboard_open(self) -> bool:
        _IME_PKG_RE = re.compile(
            r'package="[^"]*(?:inputmethod|\.ime\.|keyboard|honeyboard|input-method)',
            re.I,
        )
        _IME_CLASS_RE = re.compile(
            r'class="[^"]*(?:keyboard|inputmethod|softinput|keyboardpanel)',
            re.I,
        )

        def keyboard_visible_in_xml(xml: str) -> bool:
            if not xml or "</hierarchy>" not in xml:
                return False
            return bool(_IME_PKG_RE.search(xml) or _IME_CLASS_RE.search(xml))
    
        try:
            return keyboard_visible_in_xml(self.get_xml_layout())
        except Exception:
            return False

    def take_screenshot(self):
        if self.is_keyboard_open():
            self.back()
        # Minimal: use hdc file recv if available; fall back to uitest screenshot.
        remote = "/data/local/tmp/__agentnav.png"
        local = str(self.output_dir / "__agentnav.png")
        self._hdc_run(["shell", "snapshot_display", "-f", remote], timeout=20)
        self._hdc_run(["file", "recv", remote, local], timeout=30)
        img = cv2.imread(local)
        if img is None:
            raise RuntimeError("Failed to read screenshot from Harmony device.")
        return img

    def close_application(self) -> None:
        bundle = self.settings["appPackage"]  # Harmony uses bundle name in appPackage field
        self._hdc_run(["shell", "aa", "force-stop", bundle], timeout=10)

    def get_xml_layout(self) -> str:
        """
        Dump HarmonyOS layout via uitest and normalize to Android-style hierarchy XML.
        """
        remote = "/data/local/tmp/window_dump.xml"
        # uitest dumpLayout writes either XML or JSON to the given path.
        self._hdc_run(["shell", "uitest", "dumpLayout", "-p", remote], timeout=30)
        raw = self._hdc_out(["shell", "cat", remote], timeout=20).decode("utf-8", "ignore")

        xml_ok = self._normalize_hierarchy_xml(raw)
        if xml_ok.strip():
            return xml_ok

        converted = self._harmony_json_to_android_xml(raw)
        if converted.strip():
            return converted
        return ""

    @staticmethod
    def _normalize_hierarchy_xml(raw: str) -> str:
        s = (raw or "").strip()
        return s if s.startswith("<?xml") else ""

    @staticmethod
    def _harmony_json_to_android_xml(raw: str) -> str:
        """
        Convert HarmonyOS uitest JSON tree into Android-like XML (`hierarchy` + `node` elements).
        """
        if not raw:
            return ""
        try:
            data = json.loads(raw)
        except Exception:
            return ""

        def _as_bool_str(v) -> str:
            s = str(v if v is not None else "").strip().lower()
            return "true" if s in ("1", "true", "yes") else "false"

        def _to_android_node(node) -> ET.Element:
            if isinstance(node, dict):
                attrs = node.get("attributes", {}) if isinstance(node.get("attributes", {}), dict) else {}
                children = node.get("children", []) if isinstance(node.get("children", []), list) else []
            else:
                attrs, children = {}, []

            klass = str(attrs.get("class") or attrs.get("type") or "node")
            out = ET.Element("node")
            out.set("class", klass)
            out.set("text", str(attrs.get("text") or attrs.get("originalText") or ""))
            out.set("content-desc", str(attrs.get("description") or attrs.get("accessibilityId") or ""))
            out.set("resource-id", str(attrs.get("id") or attrs.get("key") or ""))
            out.set("package", str(attrs.get("bundleName") or ""))
            out.set("bounds", str(attrs.get("bounds") or ""))
            out.set("clickable", _as_bool_str(attrs.get("clickable")))
            out.set("long-clickable", _as_bool_str(attrs.get("longClickable")))
            out.set("enabled", _as_bool_str(attrs.get("enabled")))
            out.set("focusable", _as_bool_str(attrs.get("focused")))
            out.set("focused", _as_bool_str(attrs.get("focused")))
            out.set("checkable", _as_bool_str(attrs.get("checkable")))
            out.set("checked", _as_bool_str(attrs.get("checked")))
            out.set("selected", _as_bool_str(attrs.get("selected")))
            out.set("scrollable", _as_bool_str(attrs.get("scrollable")))

            for ch in children:
                out.append(_to_android_node(ch))
            return out

        roots = data if isinstance(data, list) else [data]
        hierarchy = ET.Element("hierarchy")
        for r in roots:
            hierarchy.append(_to_android_node(r))
        return ET.tostring(hierarchy, encoding="unicode")

    def get_current_app_id(self) -> Optional[str]:
        try:
            out = self._hdc_out(["shell", "aa", "dump", "-a"], timeout=10).decode("utf-8", "ignore")
            m = re.search(r"bundleName:\s*([\w.]+)", out)
            return m.group(1) if m else None
        except Exception:
            return None

    def click(self, x: int, y: int) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "click", str(int(x)), str(int(y))], timeout=10)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 500) -> None:
        # uitest uses duration in ms via swipe
        self._hdc_run(
            ["shell", "uitest", "uiInput", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(duration_ms))],
            timeout=15,
        )

    def type_text(self, text: str) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "text", str(text)], timeout=15)

    def back(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "back"], timeout=10)

    def home(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "home"], timeout=10)

    def get_screen_size(self) -> Tuple[int, int]:
        out = self._hdc_out(["shell", "hidumper", "-s", "DisplayManagerService", "-a", "dumpDisplayInfo"], timeout=10).decode(
            "utf-8", "ignore"
        )
        m = re.search(r"width\s*=\s*(\d+).*height\s*=\s*(\d+)", out, re.DOTALL)
        if not m:
            # fallback common phone
            return 1080, 2400
        return int(m.group(1)), int(m.group(2))

    def run_application(self) -> None:
        self._hdc_run(["shell", "am", "start", "-W", self.settings['appPackage'] + "/" + self.settings['appActivity']], timeout=10)