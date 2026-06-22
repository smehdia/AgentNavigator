import json
import re
import subprocess
from typing import Optional, Tuple

import cv2
import numpy as np
import xml.etree.ElementTree as ET

import logging
from Driver.BaseDriver import BaseDriver


class HarmonyDriver(BaseDriver):
    def __init__(self, settings: dict, agent=None) -> None:
        super().__init__(settings, agent)

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
        remote = "/data/local/tmp/__agentnav.jpeg"
        local = "__agentnav.jpeg"
        self._hdc_run(["shell", "snapshot_display", "-f", remote], timeout=20)
        self._hdc_run(["file", "recv", remote, local], timeout=30)
        img = cv2.imread(local)
        if img is None:
            raise RuntimeError("Failed to read screenshot from Harmony device.")
        return img
        
    def get_foreground_package(self) -> str | None:
        try:
            out = self._hdc_out(
                [
                    "shell",
                    (
                        "uitest dumpLayout -p /data/local/tmp/window_dump.json >/dev/null && "
                        "cat /data/local/tmp/window_dump.json | "
                        "grep -oE '\"bundleName\":\"[^\"]+\"|\"bundleName\": \"[^\"]+\"' | "
                        "sed -E 's/.*\"bundleName\"[ ]*:[ ]*\"([^\"]+)\".*/\\1/' | "
                        "grep -vE 'com\\.ohos\\.sceneboard|com\\.ohos\\.systemui|com\\.ohos\\.launcher' | "
                        "head -n 1"
                    ),
                ],
                timeout=10,
            ).decode("utf-8", "ignore").strip()

            return out or None

        except Exception:
            return None

    def close_application(self) -> None:
        bundle = self.settings["appPackage"]  
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
        # uitest uiInput swipe expects swipeVelocityPps_ (px/s, 200–40000), not duration_ms.
        distance = max(abs(int(x2) - int(x1)), abs(int(y2) - int(y1)))
        ms = max(int(duration_ms), 1)
        if distance > 0:
            velocity_pps = int(round(distance / (ms / 1000.0)))
        else:
            velocity_pps = 600  # Harmony default for same-point gestures (e.g. long press)
        velocity_pps = max(200, min(40000, velocity_pps))
        self._hdc_run(
            ["shell", "uitest", "uiInput", "swipe", str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(velocity_pps)],
            timeout=15,
        )

    def type_text(self, text: str) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "text", str(text)], timeout=15)

    def back(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "Back"], timeout=10)

    def home(self) -> None:
        self._hdc_run(["shell", "uitest", "uiInput", "keyEvent", "Home"], timeout=10)

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
        pkg = self.settings["appPackage"]
        ability = self.settings.get("appActivity", "EntryAbility")
        self._hdc_run(["shell", "aa", "start", "-a", ability, "-b", pkg], timeout=10)


    def get_app_version(self):
        pkg = self.settings["appPackage"]

        def extract_all_fields(text: str, field: str):
            pattern = rf'"{re.escape(field)}"\s*:\s*("([^"]*)"|-?\d+|true|false|null)'
            values = []

            for m in re.finditer(pattern, text):
                raw = m.group(1)

                if raw.startswith('"') and raw.endswith('"'):
                    value = raw[1:-1]
                elif raw == "true":
                    value = True
                elif raw == "false":
                    value = False
                elif raw == "null":
                    value = None
                elif re.fullmatch(r"-?\d+", raw):
                    value = int(raw)
                else:
                    value = raw

                values.append(value)

            return values

        def last_valid(text: str, field: str):
            values = extract_all_fields(text, field)

            for v in reversed(values):
                if v not in (None, "", 0):
                    return v

            return values[-1] if values else None

        try:
            dump = self._hdc_out(
                ["shell", "bm", "dump", "-n", pkg],
                timeout=20,
            ).decode("utf-8", "ignore")

            version_name = last_valid(dump, "versionName")
            version_code = last_valid(dump, "versionCode")
            main_ability = last_valid(dump, "mainAbility")
            main_element_name = last_valid(dump, "mainElementName")

            module_names = sorted(set(
                m for m in extract_all_fields(dump, "moduleName")
                if m not in (None, "", 0)
            ))

            ability_name = main_element_name or main_ability
            module_name = "entry" if "entry" in module_names else (
                module_names[0] if module_names else None
            )

            return {
                "package_name": pkg,
                "version_name": version_name,
                "version_code": version_code,
                "main_ability": main_ability,
                "main_element_name": main_element_name,
                "module_names": module_names,
                "entry": {
                    "bundle_name": pkg,
                    "module_name": module_name,
                    "ability_name": ability_name,
                    "launch_target": (
                        f"{pkg}/{module_name}/{ability_name}"
                        if module_name and ability_name
                        else None
                    ),
                    "launch_command": (
                        f"hdc -t {self.device_id} shell aa start -b {pkg} -m {module_name} -a {ability_name}"
                        if module_name and ability_name
                        else None
                    ),
                },
            }

        except Exception as e:
            logging.exception("Failed to get Harmony app version/info")
            return {
                "package_name": pkg,
                "version_name": None,
                "version_code": None,
                "main_ability": None,
                "main_element_name": None,
                "module_names": [],
                "entry": None,
                "error": str(e),
            }

    def is_keyboard_open(self) -> bool:
        ime_bundle_re = re.compile(r"(?i)(inputmethod|\.ime\.|keyboard|input-method|honeyboard|anco)")
        ime_class_re = re.compile(r"(?i)(keyboard|inputmethod|softinput|keyboardpanel|inputpanel|ime)")
        ime_xml_pkg_re = re.compile(
            r'package="[^"]*(?:inputmethod|\.ime\.|keyboard|honeyboard|input-method|anco)"',
            re.I,
        )
        ime_xml_class_re = re.compile(
            r'class="[^"]*(?:keyboard|inputmethod|softinput|keyboardpanel|inputpanel|ime)"',
            re.I,
        )
        def is_visible(v) -> bool:
            s = str(v if v is not None else "").strip().lower()
            return s in ("", "1", "true", "yes")  # missing visible => treat as visible
        def node_is_ime(attrs: dict) -> bool:
            bundle = str(attrs.get("bundleName") or "")
            klass = str(attrs.get("class") or attrs.get("type") or "")
            return bool(ime_bundle_re.search(bundle) or ime_class_re.search(klass))
        def walk_hierarchy(node) -> bool:
            if not isinstance(node, dict):
                return False
            attrs = node.get("attributes") or {}
            if is_visible(attrs.get("visible")) and node_is_ime(attrs):
                return True
            for ch in node.get("children") or []:
                if walk_hierarchy(ch):
                    return True
            return False
        # 1) Primary: uitest JSON via hmdriver2 (same source as layout dump)
        try:
            data = self._hm_driver.dump_hierarchy()
            roots = data if isinstance(data, list) else [data]
            if any(walk_hierarchy(r) for r in roots):
                return True
        except Exception:
            pass
        # 2) Secondary: WindowManager dump (no Android-style dumpsys input_method)
        try:
            out = self._hm_driver.shell("hidumper -s WindowManagerService -a '-a'").output
            if re.search(r"(?i)(inputmethod|soft.?keyboard|keyboardpanel|input.?method)", out):
                if re.search(r"(?i)(visible|shown|foreground|active).*(true|1)", out):
                    return True
        except Exception:
            pass
        # 3) Fallback: normalized XML from get_xml_layout()
        try:
            xml = self.get_xml_layout()
            if xml and "</hierarchy>" in xml:
                return bool(ime_xml_pkg_re.search(xml) or ime_xml_class_re.search(xml))
        except Exception:
            pass
        return False