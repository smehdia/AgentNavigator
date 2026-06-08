"""
VLM backend using yibuapi.com (OpenAI-compatible chat/completions).

Same public API as VLM.py (class VLM_Yibu). Text embeddings still use Alibaba DashScope.
"""

import os
import re
import cv2
import copy
import json
import math
import textwrap
import time
import base64
import random
import string
import backoff
import logging
import numpy as np
from PIL import Image
from io import BytesIO
import copy
import logging
import sys

import dashscope
import networkx as nx
import requests
import urllib3

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_API_LOGGER_PKG_DIR = os.path.join(_SCRIPT_DIR, "api_logger")
if os.path.isdir(_API_LOGGER_PKG_DIR) and _API_LOGGER_PKG_DIR not in sys.path:
    sys.path.insert(0, _API_LOGGER_PKG_DIR)

try:
    from local_api_logger import log_completion, set_log_dir

    _API_LOGGER_AVAILABLE = True
except ImportError:
    _API_LOGGER_AVAILABLE = False
    log_completion = None
    set_log_dir = None

for _proxy_var in (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy",
):
    os.environ.pop(_proxy_var, None)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_yibu_session = requests.Session()
_yibu_session.trust_env = False
_logger = logging.getLogger(__name__)

def resize_and_encode_to_base64(img, target_size=1344, jpeg_quality=90):
    """
    Resize an image so that its longest side is approximately target_size,
    preserving aspect ratio and rounding dimensions to multiples of 28.

    No square padding is applied. The "pad_value" argument is kept for compatibility but not used.

    Args:
        img (np.ndarray): Input image with shape (H, W, C).
        target_size (int): The desired size for the longest side (default=1344).
        pad_value (int): Not used.

    Returns:
        out_img (np.ndarray): The resized image (no square padding applied).
        scale (float): The scaling factor used for resizing.
        pad (tuple): Tuple (top, bottom, left, right), always (0, 0, 0, 0).
    """
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(round((w * scale)))
    new_h = int(round((h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    return resized, scale, f"data:image/jpeg;base64,{b64}"


def parse_json_from_model_response(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Could not parse model JSON response: empty text")
    # Remove markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    def _balanced_json_slice(s):
        start = next((i for i, ch in enumerate(s) if ch in "{["), -1)
        if start < 0: return None
        stack, in_str, esc = [], False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"': in_str = True; continue
            if ch in "{[": stack.append(ch)
            elif ch in "}]":
                if not stack: return None
                top = stack.pop()
                if (top, ch) not in {("{", "}"), ("[", "]")}:
                    return None
                if not stack:
                    return s[start:i + 1]
        return None

    candidates = [raw]
    balanced = _balanced_json_slice(raw)
    if balanced and balanced != raw:
        candidates.append(balanced)
    candidates.append(re.sub(r",\s*([}\]])", r"\1", raw))
    if balanced:
        candidates.append(re.sub(r",\s*([}\]])", r"\1", balanced))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not parse model JSON response. error={last_error}; preview={raw[:500]!r}")


class _YibuUsage:
    def __init__(self, usage: dict):
        prompt_details = usage.get("prompt_tokens_details") or {}
        image_tokens = int(prompt_details.get("image_tokens") or 0)
        text_tokens = int(prompt_details.get("text_tokens") or 0)
        if not text_tokens and not image_tokens:
            text_tokens = int(usage.get("prompt_tokens") or 0)
        self.input_tokens_details = {"image_tokens": image_tokens, "text_tokens": text_tokens}
        self.output_tokens_details = {
            "text_tokens": int(usage.get("completion_tokens") or 0),
        }


class _YibuMessage:
    def __init__(self, content):
        self.content = content


class _YibuChoice:
    def __init__(self, message_content):
        self.message = _YibuMessage(message_content)


class _YibuOutput:
    def __init__(self, choices):
        self.choices = choices


class YibuMultimodalResponse:
    """DashScope-compatible wrapper around yibu OpenAI chat completion JSON."""

    def __init__(self, data: dict):
        self._data = data
        self.usage = _YibuUsage(data.get("usage") or {})
        content = data["choices"][0]["message"]["content"]
        self.output = _YibuOutput([_YibuChoice(content)])

    def __getitem__(self, key):
        if key == "output":
            msg = self._data["choices"][0]["message"]
            return {"choices": [{"message": {"content": msg.get("content")}}]}
        return self._data[key]


class VLM_Yibu:
    def __init__(self, configs, debugger):
        self.total_calls = {}
        self.token_usage_details = {}
        self.yibu_api_key = str(configs.vlm.yibu_api_key).strip()
        self.yibu_base_url = str(
            getattr(configs.vlm, "yibu_base_url", "https://yibuapi.com")
        ).rstrip("/")
        self.enable_thinking = bool(getattr(configs.vlm, "enable_thinking", False))
        self.enable_high_resolution = bool(
            getattr(configs.vlm, "vl_high_resolution_images", True)
        )
        connect_timeout = int(getattr(configs.vlm, "connect_timeout", 30))
        read_timeout = int(getattr(configs.vlm, "read_timeout", 180))
        self._request_timeout = (connect_timeout, read_timeout)
        self._request_retries = int(getattr(configs.vlm, "request_retries", 5))
        self._retryable_statuses = {429, 500, 502, 503, 504}
        self._api_log_user = str(
            getattr(configs.vlm, "api_log_user", None)
            or getattr(configs.vlm, "yibu_log_user", None)
            or "yibu"
        )
        # Embeddings stay on Alibaba DashScope (separate key if configured).
        dashscope.api_key = str(
            getattr(configs.vlm, "dashscope_api_key", configs.vlm.alibaba_api_key)
        ).strip()
        dashscope.base_http_api_url = str(
            getattr(
                configs.vlm,
                "dashscope_base_url",
                "https://dashscope-intl.aliyuncs.com/api/v1",
            )
        )
        self.debugger = debugger
        self.configs = configs

        if _API_LOGGER_AVAILABLE:
            log_dir = getattr(
                configs.vlm,
                "api_log_dir",
                os.path.join(_SCRIPT_DIR, "api_logs"),
            )
            set_log_dir(log_dir)

    @staticmethod
    def _dashscope_messages_to_openai(messages: list) -> list:
        openai_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, str):
                openai_messages.append({"role": role, "content": content})
                continue
            if not isinstance(content, list):
                openai_messages.append({"role": role, "content": str(content)})
                continue
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if "image" in part:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": part["image"],
                                "detail": "high",
                            },
                        }
                    )
                elif "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
            openai_messages.append({"role": role, "content": parts or ""})
        return openai_messages

    @staticmethod
    def _payload_for_api_log(payload: dict) -> dict:
        logged = copy.deepcopy(payload)
        for msg in logged.get("messages", []):
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "image_url":
                    continue
                url = (part.get("image_url") or {}).get("url", "")
                if isinstance(url, str) and url.startswith("data:"):
                    part["image_url"]["url"] = "<base64 omitted>"
        return logged

    def _log_yibu_call(self, payload: dict, response_data: dict, elapsed_s: float) -> None:
        if not _API_LOGGER_AVAILABLE or not response_data.get("usage"):
            return
        try:
            log_completion(
                model=payload.get("model", ""),
                request_data=self._payload_for_api_log(payload),
                response_data=response_data,
                user=self._api_log_user,
                duration_ms=elapsed_s * 1000.0,
            )
        except Exception:
            _logger.warning("local_api_logger log_completion failed", exc_info=True)

    def _multimodal_conversation_call(self, model: str, messages: list, **kwargs):
        """OpenAI-compatible yibu chat/completions (replaces DashScope MultiModalConversation)."""
        url = f"{self.yibu_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.yibu_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": self._dashscope_messages_to_openai(messages),
            "enable_thinking": kwargs.get("enable_thinking", self.enable_thinking),
            "vl_high_resolution_images": kwargs.get(
                "vl_high_resolution_images", self.enable_high_resolution
            ),
        }
        if kwargs.get("response_format"):
            payload["response_format"] = kwargs["response_format"]
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]
        if kwargs.get("seed") is not None:
            payload["seed"] = kwargs["seed"]
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]

        last_error: Exception | None = None
        for attempt in range(1, self._request_retries + 1):
            start = time.perf_counter()
            try:
                response = _yibu_session.post(
                    url,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=self._request_timeout,
                )
                elapsed = time.perf_counter() - start

                if response.status_code in self._retryable_statuses:
                    body_preview = (response.text or "")[:500]
                    last_error = requests.HTTPError(
                        f"{response.status_code} from yibuapi (attempt {attempt}/{self._request_retries}): {body_preview}",
                        response=response,
                    )
                    _logger.warning(
                        "Yibu API %s for model=%s (attempt %d/%d), retrying: %s",
                        response.status_code,
                        model,
                        attempt,
                        self._request_retries,
                        body_preview,
                    )
                    if attempt < self._request_retries:
                        time.sleep(min(30, attempt * 3))
                        continue
                    response.raise_for_status()

                response.raise_for_status()
                data = response.json()
                self._log_yibu_call(payload, data, elapsed)
                return YibuMultimodalResponse(data)

            except (requests.ConnectTimeout, requests.ReadTimeout) as exc:
                elapsed = time.perf_counter() - start
                last_error = exc
                _logger.warning(
                    "Yibu API timeout for model=%s after %.2fs (attempt %d/%d): %s",
                    model,
                    elapsed,
                    attempt,
                    self._request_retries,
                    exc,
                )
                if attempt < self._request_retries:
                    time.sleep(min(30, attempt * 3))
                    continue

            except requests.HTTPError as exc:
                last_error = exc
                status = getattr(exc.response, "status_code", None)
                if status in self._retryable_statuses and attempt < self._request_retries:
                    time.sleep(min(30, attempt * 3))
                    continue
                raise

        assert last_error is not None
        raise last_error

    def extract_elements_from_page(self, screenshot, application_description="", action_history=None):
        """
        Extract actionable UI elements from a screenshot.
        This function uses only the screenshot, not XML layout information.

        Output schema is unchanged:
        {
            "elements": [
                {
                    "type": "icon|button|nav|card|scroll|swipe|input",
                    "bbox": [x1, y1, x2, y2],   
                    "description": "..."
                }
            ]
        }
        """
        history_lines = [str(item).strip() for item in (action_history or []) if str(item or "").strip() and "scroll" not in str(item).lower() and "swipe" not in str(item).lower()]
        history_block = (
            "\n".join(f"- {line}" for line in history_lines)
            if history_lines
            else "- None"
        )

        history_context = (
            "Recent non-scroll/swipe actions that led here "
            "(use only if screenshot is ambiguous; visible foreground always wins):\n"
            f"{history_block}\n"
        )

        resized, scale, data_uri = resize_and_encode_to_base64(screenshot, target_size=self.configs.vlm.screenshot_longest_side_for_element_extraction_model, jpeg_quality=self.configs.vlm.jpeg_quality_for_element_extraction_model)
        app_context_line = (
            f"Application context: {self.configs.app.description}\n"
            if self.configs.app.description
            else "Application context: General-purpose mobile app exploration.\n"
        )
        
        active_surface_rules = (
            "ACTIVE SURFACE / Z-ORDER RULES:\n"
            "Before output, silently choose the single topmost interactive surface receiving taps now.\n"
            "This is a z-order decision, not only a bbox decision.\n"
            "Foreground surfaces include modal, dialog, alert, bottom sheet, action sheet, popup, menu, dropdown, picker, drawer, flyout, expanded FAB menu, or any panel over a dimmed/blurred/darkened/de-emphasized background.\n"
            "If any foreground surface exists, the underlying page is inactive even if its buttons, tabs, cards, toolbar, bottom nav, or text are still visible.\n"
            "Return only controls visually attached to that topmost surface itself: same panel/sheet/dialog/menu/elevation/layer.\n"
            "Never return lower-z background controls behind an overlay, even if visible, important, or overlapping the foreground region.\n"
            "If uncertain whether a candidate is foreground or background, exclude it.\n"
            "Only if no foreground surface exists, use the main page as the active surface.\n"
        )

        schema_rules = (
            "OUTPUT SCHEMA:\n"
            'Return strictly valid JSON exactly as {"elements": [...]} and nothing else.\n'
            "Each element must have exactly: type, bbox, description.\n"
            "type must be one of: icon, button, nav, card, scroll, swipe, input.\n"
            "bbox must be [x1,y1,x2,y2] normalized to 0..1000.\n"
            "description must be a short action sentence with relative position. For icons, mention visual form, e.g. gear icon, avatar, three-dot icon.\n"
            "Do not output active_layer, surface_id, z_order, confidence, rationale, or extra fields.\n"
        )

        extraction_rules = (
            "EXTRACTION ORDER:\n"
            "On the active surface only, extract: (1) top chrome of that surface, (2) scope tabs/chips, (3) remaining app-level actionable controls.\n"
            "Be sensitive to small icons, icon-only buttons, colored icons, and visual symbols, but only if they are on the active surface.\n"
            "False positives from inactive background layers are worse than missing ambiguous elements.\n"
        )

        exclusion_rules = (
            "EXCLUDE:\n"
            "Back/up/close/cancel/dismiss/X controls that only exit or close; ads/promotional/sponsored/recommendation cards; decorative visuals; detailed personal content; keyboard/search/query/chat entry bars; sample prompt chips; emojis; dial/key/num pads; media controls; delete/remove/trash/sign-out/delete-account actions.\n"
            "Goal is app-level navigation/persistent chrome: section tabs, bottom nav, top utility icons, overflow/menu, profile/account, settings/about, global actions.\n"
            "If those controls are behind a modal/sheet/menu/drawer/popup, exclude them.\n"
            "Data lists/dashboards: do not return repeating item rows/cards, per-item toggles, per-row plus/chevron/overflow, inbox rows, alarms, orders, playlists, messages, or feed cards.\n"
            "Settings/preferences exception: if the active surface is a uniform preferences list, include preference rows and their switch/slider/segmented controls.\n"
            "Cards: include only stable app-navigation cards such as category hubs or settings/menu destinations, not feed/content/product/promotional cards unless app context says product cards are primary navigation.\n"
            "Expanded dropdown/accordion/menu: include actionable subitems that change scope/destination; skip informational or duplicate lines.\n"
            "Horizontal scope chips/tabs: include each tappable peer only if the row belongs to the active surface.\n"
            "Utility icons in header/search area: include if tappable and on active surface; exclude if the header/search area is behind an overlay.\n"
        )

        z_order_examples = (
            "Z-ORDER MISTAKES TO AVOID:\n"
            "- Dialog open: do not return app buttons/tabs/cards/toolbar behind it.\n"
            "- Bottom sheet open: do not return bottom nav/list rows behind it.\n"
            "- Overflow menu open: do not return toolbar/page controls behind it.\n"
            "- Dropdown open: do not return controls behind it unless they are part of the dropdown.\n"
            "- Dimmed/greyed/blurred/darkened background means background UI is inactive.\n"
        )

        # ------------------------------------------------------------------
        # System message
        # ------------------------------------------------------------------

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "Extract actionable UI elements from a mobile screenshot for automated app navigation. "
                + "Only output elements on the single topmost active interaction surface. "
                + "Do not output inactive background/lower-z UI. "
                + 'Return strictly valid JSON {"elements": [...]} only.\n\n'
                + active_surface_rules
                + "\n"
                + schema_rules
                + "\n"
                + extraction_rules
                + "\n"
                + exclusion_rules
                + "\n"
                + z_order_examples
            ),
        }

        # ------------------------------------------------------------------
        # User prompt
        # ------------------------------------------------------------------

        prompt = (
            app_context_line
            + history_context
            + 'Return JSON only: {"elements": [...]}.\n'
            + "Keep the schema unchanged: no active_layer, surface_id, z_order, confidence, or explanations.\n\n"

            + "Silently do this before output:\n"
            + "1. Identify the single topmost active interaction surface by z-order, not bbox alone.\n"
            + "2. If a modal/dialog/sheet/menu/popup/dropdown/drawer/picker/overlay is open, the main page behind it is inactive.\n"
            + "3. Reject candidates from the background page even if visible, recognizable, tappable when overlay closes, or spatially overlapping the foreground.\n"
            + "4. Reject dimmed, blurred, darkened, occluded, scrimmed, lower-z, or visually-underlayed controls.\n"
            + "5. Output only actionable controls visually attached to the active surface itself.\n"
            + "6. If foreground/background membership is uncertain, exclude the candidate.\n\n"

            + "For each output element provide only: type, bbox, description.\n"
            + "Allowed types: icon, button, nav, card, scroll, swipe, input.\n"
            + "bbox must be normalized [x1,y1,x2,y2] in 0..1000.\n"
            + "Description must be short and include relative position; for icons mention visual form.\n\n"

            + "Include on active surface: top utility icons, overflow/menu, compose/edit-as-chrome, profile/account/avatar, bottom nav, main section tabs, scope chips/tabs, settings/about/global navigation.\n"
            + "Do not include these if they are behind an overlay.\n\n"

            + "Exclude: back/up/close/cancel/dismiss/X; ads/promos/banners; decorative visuals; search/query/chat/keyboard entry bars; sample prompt chips; dial/key/num pads; media controls; delete/remove/trash/sign-out/delete-account; repeating item rows/cards and their inline toggles/plus/chevrons/overflow.\n"
            + "Settings exception: if the active surface is a uniform preferences/settings list, include preference rows and switches/sliders/segments.\n"
            + "Expanded menus/dropdowns/accordions: include actionable destination/scope subitems only.\n"
            + "For horizontal scope chips/tabs on the active surface, include each tappable peer.\n\n"

            + "Common mistakes: do not return page buttons behind a dialog; do not return bottom nav behind a bottom sheet; do not return toolbar icons behind an overflow menu; do not return controls behind a dropdown; dimmed background is inactive.\n"
        )

        user_content = [
            {"image": data_uri},
            {"text": prompt},
        ]

        user_msg = {
            "role": "user",
            "content": user_content,
        }

        messages = [system_msg, user_msg]
        self.prompt = messages

        resp = self._multimodal_conversation_call(
            model=self.configs.vlm.model_name_for_elements_extraction,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42
        )

        image_input_tokens = resp.usage.input_tokens_details['image_tokens']
        text_input_tokens = resp.usage.input_tokens_details['text_tokens']
        output_tokens = resp.usage.output_tokens_details['text_tokens']

        if self.configs.vlm.model_name_for_elements_extraction in self.token_usage_details.keys():
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["image_input_tokens"] += image_input_tokens
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[self.configs.vlm.model_name_for_elements_extraction] = {
                "image_input_tokens": image_input_tokens,
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens
            }

        raw = resp.output.choices[0].message.content
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list) else str(raw).strip()
        )
        elements = (parse_json_from_model_response(raw_text) or {}).get("elements", [])
        H0, W0 = screenshot.shape[:2]
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        def map_back(b):
            if not (b and len(b) == 4): return None
            try: x1, y1, x2, y2 = map(float, b)
            except: return None
            Hp = int(round(H0 * scale)) + pad_top + pad_bottom
            Wp = int(round(W0 * scale)) + pad_left + pad_right
            x1, x2 = [(v * Wp / 1000.0 - pad_left) / scale for v in (x1, x2)]
            y1, y2 = [(v * Hp / 1000.0 - pad_top) / scale for v in (y1, y2)]
            x1, x2 = sorted([max(0, min(W0 - 1, x1)), max(0, min(W0 - 1, x2))])
            y1, y2 = sorted([max(0, min(H0 - 1, y1)), max(0, min(H0 - 1, y2))])
            return [x1, y1, x2, y2]

        def maybe_add_swipe(items):
            """
            Conditionally add a synthetic 'swipe' control to the items list if it looks like
            a horizontal control strip is present.
            
            Flow:
            1. If "swipe" is already present, do nothing and return.
            2. Collect controls of type nav/button. If less than 2, do nothing.
            3. Group controls into approximate horizontal rows (using vertical center and vertical tolerance).
            4. For each row, find candidates that:
               - are not too high vertically (near the top)
               - are not too "tall" (the row is horizontally oriented)
               - span a wide enough range horizontally
            5. Pick the best row, and synthesize a "swipe" element for it.
            6. Return the list, with the swipe element added if applicable.
            """
            if any(e.get("type") == "swipe" for e in items):
                return items
            controls = [e for e in items if e.get("type") in {"nav", "button"}]
            if len(controls) < 2:
                return items
            # Sort controls by vertical center
            controls.sort(key=lambda e: 0.5 * (e["bbox"][1] + e["bbox"][3]))
            rows, tol = [], 0.05 * H0
            for e in controls:
                cy = 0.5 * (e["bbox"][1] + e["bbox"][3])
                for row in rows:
                    # Group controls in the same horizontal row
                    if abs(cy - row["cy"]) <= tol:
                        row["items"].append(e)
                        row["cy"] = sum(0.5 * (it["bbox"][1] + it["bbox"][3]) for it in row["items"]) / len(row["items"])
                        break
                else:
                    rows.append({"cy": cy, "items": [e]})
            # Try to find the best row that looks like a swipe strip
            best = max(
                (
                    (
                        len(r["items"])
                        + (2 if (0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) < 0.35 * H0) else 0)
                        + (1 if (0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) < 0.22 * H0) else 0),
                        {
                            "type": "swipe",
                            "bbox": [
                                min(e["bbox"][0] for e in r["items"]),
                                min(e["bbox"][1] for e in r["items"]),
                                max(e["bbox"][2] for e in r["items"]),
                                max(e["bbox"][3] for e in r["items"]),
                            ],
                            "description": "Swipe left or right on this control strip.",
                        }
                    )
                    for r in rows
                    if len(r["items"]) >= 2
                    # Only rows not too low, not too tall, and wide enough
                    and 0.5 * (min(e["bbox"][1] for e in r["items"]) + max(e["bbox"][3] for e in r["items"])) <= 0.82 * H0
                    and (max(e["bbox"][3] for e in r["items"]) - min(e["bbox"][1] for e in r["items"])) <= 0.16 * H0
                    and (max(e["bbox"][2] for e in r["items"]) - min(e["bbox"][0] for e in r["items"])) >= 0.22 * W0
                ),
                key=lambda pair: pair[0],
                default=(None, None),
            )
            if best[1]:
                items.append(best[1])
            return items
       

        def maybe_add_scroll(items):
            """
            Conditionally add a synthetic 'scroll' control to the items list if it looks like
            there is a large vertical area with navigable controls.
            
            Flow:
            1. If "scroll" is already present, do nothing and return.
            2. Collect controls of types nav/button/card/icon. If less than 4 found, do nothing.
            3. If the vertical span of these controls is less than 55% of the screen, do nothing.
            4. Otherwise, generate a scroll bbox that covers most of the vertical region (not full height, clipped a bit at top/bottom).
            5. Only add scroll if the vertical span is wide enough.
            6. Add the "scroll" element and return the list.
            """
            if any(e.get("type") == "scroll" for e in items):
                return items
            c = [e for e in items if e.get("type") in {"nav", "button", "card", "icon"}]
            if len(c) < 4:
                return items
            top, bot = min(e["bbox"][1] for e in c), max(e["bbox"][3] for e in c)
            if (bot - top) < 0.55 * H0:
                return items
            y1, y2 = max(0.10 * H0, top), min(0.96 * H0, bot)
            x1, x2 = 0.03 * W0, 0.97 * W0
            if y2 - y1 < 0.30 * H0:
                return items
            items.append({"type": "scroll", "bbox": [x1, y1, x2, y2], "description": "Scroll vertically to view more content."})
            return items

        mapped_elements = []
        for e in elements:
            if not isinstance(e, dict): continue
            bbox = map_back(e.get("bbox"))
            if not bbox: continue
            desc = str(e.get("description", e.get("action_description", ""))).strip()
            mapped_elements.append({"type": str(e.get("type", "")).strip().lower(), "bbox": bbox, "description": desc})

        mapped_elements = maybe_add_swipe(mapped_elements)
        mapped_elements = maybe_add_scroll(mapped_elements)


        return {"elements": mapped_elements, "image_input_tokens": image_input_tokens, "text_input_tokens": text_input_tokens, "output_tokens": output_tokens}
   

    def extract_page_summary(
        self,
        screenshot,
        application_description="",
        action_history=None,
        interactive_action_descriptions=None,
    ):
        """
        Extract compact structured page identity for screenshot matching.

        Returns:
            {
                "active_tab": str | None,
                "active_subtab": str | None,
                "page_purpose": str,
                "image_input_tokens": int,
                "text_input_tokens": int,
                "output_tokens": int,
            }
        """

        history_lines = [
            str(item).strip()
            for item in (action_history or [])
            if (
                str(item or "").strip()
                and "scroll" not in str(item).lower()
                and "swipe" not in str(item).lower()
            )
        ]

        history_block = (
            "\n".join(f"- {line}" for line in history_lines)
            if history_lines
            else "- None"
        )

        resized, scale, data_uri = resize_and_encode_to_base64(
            screenshot,
            target_size=self.configs.vlm.screenshot_longest_side_for_page_summary_model,
            jpeg_quality=self.configs.vlm.jpeg_quality_for_page_summary_model,
        )

        interactive_actions_json = json.dumps(
            {
                "interactive_actions": [
                    dict(
                        {
                            "description": (
                                str(
                                    item.get("description")
                                    or item.get("action_description")
                                    or ""
                                ).strip()
                                if isinstance(item, dict)
                                else str(item or "").strip()
                            )
                        },
                        **(
                            {
                                "type": str(
                                    item.get("type")
                                    or item.get("vlm_type")
                                    or ""
                                )
                                .strip()
                                .lower()
                            }
                            if (
                                isinstance(item, dict)
                                and str(
                                    item.get("type")
                                    or item.get("vlm_type")
                                    or ""
                                ).strip()
                            )
                            else {}
                        ),
                    )
                    for item in (interactive_action_descriptions or [])
                    if (
                        (
                            isinstance(item, dict)
                            and str(
                                item.get("description")
                                or item.get("action_description")
                                or ""
                            ).strip()
                        )
                        or (
                            not isinstance(item, dict)
                            and str(item or "").strip()
                        )
                    )
                ]
            },
            ensure_ascii=False,
        )

        app_description = (
            str(application_description).strip()
            or str(getattr(self.configs.app, "description", "") or "").strip()
        )

        app_context_line = (
            f"Application context: {app_description}\n"
            if app_description
            else "Application context: General-purpose mobile app exploration.\n"
        )

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "You extract compact page identity information for mobile UI screenshot matching.\n"
                + "Use the screenshot as the primary evidence.\n"
                + "Use recent non-scroll/swipe action history only to disambiguate the selected tab, subtab, route, or page purpose when the screenshot alone is unclear.\n"
                + "Use the provided interactive action descriptions only as supporting evidence.\n"
                + "Describe only the active foreground surface the user can interact with now.\n"
                + "If a modal, dialog, sheet, menu, popup, or scrim-backed overlay is open, treat that overlay as the active surface, not the dimmed page behind it.\n"
                + "Identify active_tab as the currently selected top-level tab or main navigation section, if visible or strongly implied. Use null if unclear.\n"
                + "Identify active_subtab as the selected subtab, filter, segmented control, secondary tab, or active subsection, if visible or strongly implied. Use null if none or unclear.\n"
                + "Write page_purpose as one concise sentence describing what the active screen is used for.\n"
                + "page_purpose should capture the functional purpose of the current screen, not just list visible UI elements.\n"
                + "Do not invent tabs, subtabs, routes, controls, or purposes that are not supported by the screenshot, action history, or provided action list.\n"
                + "Return strictly valid JSON and nothing else using exactly this schema:\n"
                + "{"
                + '"active_tab": string or null, '
                + '"active_subtab": string or null, '
                + '"page_purpose": string'
                + "}.\n"
            ),
        }

        prompt = (
            app_context_line
            + "Recent non-scroll/swipe action descriptions that led to this screen:\n"
            + f"{history_block}\n\n"
            + "Known interactive actions on this node:\n"
            + f"{interactive_actions_json}\n\n"
            + "Analyze the active foreground screen for screenshot matching.\n"
            + "Extract:\n"
            + "1. active_tab: the currently selected main navigation tab or main section, or null if not visible/clear.\n"
            + "2. active_subtab: the selected subtab, filter, segmented control, secondary tab, or subsection, or null if none is visible/clear.\n"
            + "3. page_purpose: one concise sentence describing what this active screen is used for.\n\n"
            + "Rules:\n"
            + "- The screenshot is the main evidence.\n"
            + "- Use action history only when the visual state is ambiguous.\n"
            + "- If a popup/dialog/sheet/menu is open, summarize that foreground layer as the active screen.\n"
            + "- Do not describe hidden or background pages as active.\n"
            + "- Do not invent tab or subtab names if no selected state is visible or strongly implied.\n"
            + "- page_purpose should be useful for matching this screenshot against future screenshots.\n\n"
            + "Return JSON only in this exact format:\n"
            + "{"
            + '"active_tab": null, '
            + '"active_subtab": null, '
            + '"page_purpose": "..."'
            + "}"
        )

        user_msg = {
            "role": "user",
            "content": [
                {"image": data_uri},
                {"text": prompt},
            ],
        }

        messages = [system_msg, user_msg]
        self.prompt = messages

        resp = self._multimodal_conversation_call(
            model=self.configs.vlm.model_name_for_page_summary,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        image_input_tokens = resp.usage.input_tokens_details["image_tokens"]
        text_input_tokens = resp.usage.input_tokens_details["text_tokens"]
        output_tokens = resp.usage.output_tokens_details["text_tokens"]

        model_name = self.configs.vlm.model_name_for_page_summary

        if model_name in self.token_usage_details.keys():
            self.token_usage_details[model_name]["image_input_tokens"] += image_input_tokens
            self.token_usage_details[model_name]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[model_name]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[model_name] = {
                "image_input_tokens": image_input_tokens,
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens,
            }

        raw = resp.output.choices[0].message.content

        if isinstance(raw, list):
            raw_text = "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in raw
            ).strip()
        else:
            raw_text = str(raw).strip()

        content = parse_json_from_model_response(raw_text)

        active_tab = content.get("active_tab")
        active_subtab = content.get("active_subtab")
        page_purpose = str(content.get("page_purpose", "") or "").strip()

        if isinstance(active_tab, str):
            active_tab = active_tab.strip() or None

        if isinstance(active_subtab, str):
            active_subtab = active_subtab.strip() or None

        if not page_purpose:
            page_purpose = "Unknown active screen purpose."

        return {
            "active_tab": active_tab,
            "active_subtab": active_subtab,
            "page_purpose": page_purpose,
            "image_input_tokens": image_input_tokens,
            "text_input_tokens": text_input_tokens,
            "output_tokens": output_tokens,
        }


    def extract_text_embedding(self, text):
        """
        Single-string text embedding (DashScope ``text-embedding-v4``).

        Args:
            text: String (may be None or empty).

        Returns:
            np.ndarray of shape (D,) float32, the embedding vector for the text.
        """
        import numpy as np

        t = "" if text is None else str(text)
        resp = dashscope.TextEmbedding.call(
            model="text-embedding-v4",
            input=[t],
        )
        if getattr(resp, "status_code", None) != 200 or not getattr(resp, "output", None):
            raise RuntimeError(
                f"TextEmbedding status={getattr(resp, 'status_code', None)} "
                f"code={getattr(resp, 'code', None)} message={getattr(resp, 'message', None)} "
                f"output={getattr(resp, 'output', None)}"
            )
        out = resp.output
        if not isinstance(out, dict):
            raise KeyError(f"Unexpected TextEmbedding output type: {type(out)!r}")
        embs = out.get("embeddings")
        if not isinstance(embs, list) or len(embs) != 1:
            raise KeyError(
                f"Unexpected TextEmbedding embeddings: len={len(embs) if isinstance(embs, list) else None}, "
                f"expected 1, out={out!r}"
            )
        e = embs[0]
        if not isinstance(e, dict) or "embedding" not in e:
            raise KeyError(f"Unexpected embedding entry: {e!r}")
        arr = np.array(e["embedding"], dtype=np.float32)
        return arr



    def guess_undo_action(self, current_screenshot, parent_screenshot, forward_action):
        """
        Suggest one undo gesture from the current (child) screen back to the parent screen.

        Returns ``(undo_mode, parsed_action)`` where ``parsed_action`` is a ``ParsedAction``
        executable by the driver, or ``None`` when no gesture is needed or none is possible.
        """
        from Agents.MAI_UI.MAI_UI import ParsedAction

        forward_action = forward_action if isinstance(forward_action, dict) else {}
        fwd_type = str(forward_action.get("type", "") or "").strip().lower()
        fwd_desc = str(
            forward_action.get("description")
            or forward_action.get("action_description")
            or ""
        ).strip()

        bbox = forward_action.get("boundingBox") 
        parent_screenshot_vis = parent_screenshot
        if bbox:
            x1, y1, x2, y2 = bbox
            parent_screenshot_vis = cv2.rectangle(parent_screenshot.copy(), (x1, y1), (x2, y2), (0, 255, 0), 5)

        def _map_bbox_on_current(b):
            if not (b and len(b) == 4):
                return None
            try:
                x1, y1, x2, y2 = map(float, b)
            except Exception:
                return None
            Hp = int(round(H0 * scale_cur))
            Wp = int(round(W0 * scale_cur))
            x1, x2 = [(v * Wp / 1000.0) / scale_cur for v in (x1, x2)]
            y1, y2 = [(v * Hp / 1000.0) / scale_cur for v in (y1, y2)]
            x1, x2 = sorted([max(0, min(W0 - 1, x1)), max(0, min(W0 - 1, x2))])
            y1, y2 = sorted([max(0, min(H0 - 1, y1)), max(0, min(H0 - 1, y2))])
            return [x1, y1, x2, y2]

        def _finish(mode, action_type="unknown", *, direction=None, point=None, label=""):
            if mode in ("none", "on_parent", "scroll_unchanged"):
                return mode, None
            orig = {}
            if point is not None:
                orig["point"] = (int(round(point[0])), int(round(point[1])))
            params = {}
            if direction:
                params["direction"] = direction
            text = label or mode
            return mode, ParsedAction(
                raw_text=text,
                thought=text,
                action_type=action_type,
                params=params,
                orig_coords=orig,
            )

        if current_screenshot is None or parent_screenshot is None:
            return "none", None

        H0, W0 = current_screenshot.shape[:2]
        _, scale_cur, uri_cur = resize_and_encode_to_base64(
            current_screenshot, target_size=self.configs.graph.exploration_modes.local_bfs.screenshot_longest_side_for_guess_undo_vlm_model, jpeg_quality=self.configs.graph.exploration_modes.local_bfs.jpeg_quality_for_guess_undo_vlm_model
        )
        _, scale_par, uri_par = resize_and_encode_to_base64(
            parent_screenshot_vis, target_size=self.configs.graph.exploration_modes.local_bfs.screenshot_longest_side_for_guess_undo_vlm_model, jpeg_quality=self.configs.graph.exploration_modes.local_bfs.jpeg_quality_for_guess_undo_vlm_model
        )

        app_context_line = (
            f"Application context: {self.configs.app.description}\n"
            if getattr(self.configs.app, "description", None)
            else "Application context: General-purpose mobile app exploration.\n"
        )

        forward_json = json.dumps(
            {"type": fwd_type or "unknown", "description": fwd_desc or "(none)"},
            ensure_ascii=False,
        )

        system_msg = {
            "role": "system",
            "content": (
                app_context_line
                + "You choose the single best UNDO gesture to return from the CURRENT screen to the PARENT screen.\n"
                + "Image 1 is CURRENT (after the forward action). "
                + "Image 2 is PARENT (target state), with the action bounding box (if any) visualized as a bright green rectangle. "
                + "The forward action that was taken is provided as JSON.\n"
                + "Return strictly valid JSON only.\n\n"
                + "RULES (apply in order; pick the first matching case):\n"
                + "1) If ONLY the main bottom/top navigation bar selection changed between parent and current "
                + "(same page body, only active nav tab differs), undo_mode=main_nav_tab and action is a click on "
                + "the nav item that matches the PARENT screenshot's selected tab (bbox on image 1).\n"
                + "2) If the forward action was scroll down (type scroll/scroll_down) and the screenshots differ "
                + "in scrollable content, undo_mode=scroll_up with action_type=scroll direction=up (bbox optional).\n"
                + "   If the forward action was swipe left and a horizontal strip/carousel changed, undo_mode=swipe_right "
                + "with action_type=scroll direction=right; prefer bbox on the same strip region on image 1.\n"
                + "   If BOTH scroll-down and swipe-left effects are visible, return scroll_up first (only one action).\n"
                + "3) If a menu/accordion/expandable row opened on current vs parent, undo_mode=collapse_menu and "
                + "action is click on the expanded row/header to collapse (bbox on image 1).\n"
                + "4) If a modal/dialog/sheet blocks return to parent, or no safe undo exists, undo_mode=none and action=null.\n"
                + "5) If ONLY a subtab/filter/segment/chip row selection changed (not main nav), undo_mode=tap_subtab "
                + "and action is click the subtab that matches PARENT on image 1.\n"
                + "6) If current already matches parent (same page), undo_mode=on_parent and action=null.\n"
                + "7) Otherwise: tap_back if a back/up/close control is visible on image 1, else system_back.\n\n"
                + "Image 2 always has the forward action's bounding box shown as a bright green rectangle, if the action had a bounding box. Use this as reference for what was acted on in the parent.\n\n"
                + "Schema:\n"
                + '{"undo_mode": string, "reason": string, "action": null or '
                + '{"action_type": "click"|"scroll"|"press_back", '
                + '"direction": "up"|"down"|"left"|"right"|null, '
                + '"bbox": [x1,y1,x2,y2] normalized 0..1000 on IMAGE 1 or null, '
                + '"description": string}}.\n'
                + "undo_mode must be one of: none, on_parent, main_nav_tab, scroll_up, swipe_right, "
                + "collapse_menu, tap_subtab, tap_back, system_back, scroll_unchanged.\n"
                + "For scroll undo use action_type=scroll with direction up or right. For taps use action_type=click.\n"
                + "For system_back use action_type=press_back and bbox=null.\n"
            ),
        }

        user_msg = {
            "role": "user",
            "content": [
                {"image": uri_cur},
                {"image": uri_par},
                {
                    "text": (
                        app_context_line
                        + "Forward action that was taken from PARENT to CURRENT:\n"
                        + f"{forward_json}\n\n"
                        + "Image 1 = CURRENT screen (execute undo here).\n"
                        + "Image 2 = PARENT screen (target after undo, with the forward action visualized as green box if applicable).\n\n"
                        + "Decide the single best undo. Return JSON only."
                    ),
                },
            ],
        }

        model_name = self.configs.graph.exploration_modes.local_bfs.guess_undo_vlm_type

        resp = self._multimodal_conversation_call(
            model=model_name,
            messages=[system_msg, user_msg],
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        usage = getattr(resp, "usage", None)
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None) or {}
            image_input_tokens = (
                details.get("image_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "image_tokens", 0)
            )
            text_input_tokens = (
                details.get("text_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "text_tokens", 0)
            )
            out_details = getattr(usage, "output_tokens_details", None) or {}
            output_tokens = (
                out_details.get("text_tokens", 0)
                if isinstance(out_details, dict)
                else getattr(out_details, "text_tokens", 0)
            )
            if model_name in self.token_usage_details:
                self.token_usage_details[model_name]["image_input_tokens"] += image_input_tokens
                self.token_usage_details[model_name]["text_input_tokens"] += text_input_tokens
                self.token_usage_details[model_name]["output_tokens"] += output_tokens
            else:
                self.token_usage_details[model_name] = {
                    "image_input_tokens": image_input_tokens,
                    "text_input_tokens": text_input_tokens,
                    "output_tokens": output_tokens,
                }

        raw = resp.output.choices[0].message.content
        raw_text = (
            "".join(
                chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                for chunk in raw
            ).strip()
            if isinstance(raw, list)
            else str(raw).strip()
        )


        try:
            content = parse_json_from_model_response(raw_text) or {}
        except Exception:
            return "none", None

        undo_mode = str(content.get("undo_mode", "none") or "none").strip().lower()
        reason = str(content.get("reason", "") or "").strip()
        act = content.get("action")

        if undo_mode in ("none", "on_parent", "scroll_unchanged"):
            return undo_mode, None

        if not isinstance(act, dict):
            act = {}

        action_type = str(act.get("action_type", "") or "").strip().lower()
        direction = str(act.get("direction", "") or "").strip().lower() or None
        desc = str(act.get("description", "") or "").strip() or reason or undo_mode
        bbox = _map_bbox_on_current(act.get("bbox"))
        point = None
        if bbox is not None:
            point = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        elif undo_mode == "swipe_right":
            point = _fwd_tap_xy()

        if undo_mode == "scroll_up":
            return _finish(
                undo_mode,
                "scroll",
                direction="up",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode == "swipe_right":
            return _finish(
                undo_mode,
                "scroll",
                direction="right",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode == "system_back":
            return _finish(
                undo_mode,
                "press_back",
                label=f"contextual undo (vlm): {desc}",
            )
        if undo_mode in ("main_nav_tab", "collapse_menu", "tap_subtab", "tap_back"):
            if point is None:
                return "none", None
            return _finish(
                undo_mode,
                "click",
                point=point,
                label=f"contextual undo (vlm): {desc}",
            )

        if action_type == "press_back":
            return _finish("system_back", "press_back", label=desc)
        if action_type == "scroll":
            d = direction if direction in ("up", "down", "left", "right") else "up"
            mode = "scroll_up" if d == "up" else "swipe_right" if d == "right" else undo_mode
            return _finish(mode, "scroll", direction=d, point=point, label=desc)
        if action_type == "click" and point is not None:
            return _finish(undo_mode or "click", "click", point=point, label=desc)

        return "none", None


    def prioritize_node_for_bfs(self, graph, top_k=5, MAX_LIMIT_OF_NODES_FOR_LLM=100):
        import json

        nodes = {}

        def get_depth(graph, node_id):
            g = graph.nx_graph
            tid = str(node_id)
            roots = [nid for nid, n in graph.nodes.items() if getattr(n, "is_root", False)]
            if not roots:
                roots = [nid for nid in g.nodes if g.in_degree(nid) == 0]
            min_depth = None
            for r in roots:
                rs = str(r)
                if rs == tid:
                    return 0
                try:
                    p = nx.shortest_path(g, source=rs, target=tid)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                d = len(p) - 1
                if min_depth is None or d < min_depth:
                    min_depth = d
            return min_depth if min_depth is not None else -1

        for node in graph.nodes.values():
            unexplored_actions = [action for action in node.ui_elements if not action.get("explored")]
            if (len(unexplored_actions) == 0) or (len(node.ui_elements) == 0):
                continue

            unexplored_ratio = len(unexplored_actions) / len(node.ui_elements)

            ui_element_descriptions = [
                {
                    "description": action.get("description", ""),
                    "explored": bool(action.get("explored", False)),
                }
                for action in node.ui_elements
            ]

            nodes[node.node_id] = {
                "page_summary": node.page_summary,
                "ui_element_descriptions": ui_element_descriptions,
                "unexplored_num_actions": len(unexplored_actions),
                "explored_num_actions": len(node.ui_elements) - len(unexplored_actions),
                "unexplored_ratio": unexplored_ratio,
                "num_visits": node.num_visits,
                "depth": get_depth(graph, node.node_id),
            }

        # Only keep up to MAX_LIMIT_OF_NODES_FOR_LLM
        if len(nodes) > MAX_LIMIT_OF_NODES_FOR_LLM:
            sampled_node_ids = random.sample(list(nodes.keys()), MAX_LIMIT_OF_NODES_FOR_LLM)
            nodes = {nid: nodes[nid] for nid in sampled_node_ids}

        # Prepare LLM prompt
        system_prompt = (
            "You are a mobile app graph exploration planner.\n"
            "Your job is to prioritize candidate nodes for BFS-style app exploration.\n"
            "You must rank nodes based on exploration value, navigation cost, and risk.\n"
            "Definitions:\n"
            "- Exploration value is high when a node has many unexplored actions, a high unexplored ratio, and a page purpose likely to reveal new app states.\n"
            "- Broad navigation pages such as Home, Browse, Categories, Menu, Dashboard, Search, and Account overview are usually valuable.\n"
            "Ranking rules:\n"
            "1. Prefer nodes with many unexplored actions.\n"
            "2. Prefer nodes whose unexplored UI elements look likely to reveal new states.\n"
            "3. Prefer nodes with high unexplored_ratio, but do not rank only by ratio.\n"
            "4. Prefer shallow nodes when exploration value is similar.\n"
            "5. Penalize nodes with many visits unless they still have many useful unexplored actions.\n"
            "6. Prefer broad navigation/category/home/dashboard pages.\n"
            "7. Penalize repetitive chat/assistant pages unless they are clearly useful.\n"
            "8. Do not choose a node with zero unexplored actions.\n"
            "9. Return only valid JSON.\n"
            "10. Do not include markdown or extra explanation.\n"
            "Output schema:\n"
            "{\n"
            "  \"ranked_nodes\": [\n"
            "    {\n"
            "      \"node_id\": \"string\",\n"
            "      \"rank\": 1,\n"
            "      \"reason\": \"brief reason\"\n"
            "    }\n"
            "  ]\n"
            "}\n"
        )

        # Pick a current_node_id, for BFS this can be root or most recently visited
        if hasattr(graph, "current_node_id"):
            current_node_id = graph.current_node_id
        elif hasattr(graph, "last_node_id"):
            current_node_id = graph.last_node_id
        elif graph.nodes:
            current_node_id = next(iter(graph.nodes))
        else:
            current_node_id = ""

        user_prompt = (
            f"Current node:\n{current_node_id}\n\n"
            "Task:\n"
            f"Rank the following candidate nodes for the next BFS exploration step. Only return the top {top_k} ranked nodes in your output.\n\n"
            "Candidate node fields:\n"
            "- page_summary: semantic description of the page\n"
            "- ui_element_descriptions: descriptions of all actionable UI elements and whether each was explored\n"
            "- unexplored_num_actions: number of actions not yet explored\n"
            "- explored_num_actions: number of actions already explored\n"
            "- unexplored_ratio: unexplored_num_actions / total actions\n"
            "- num_visits: how many times this node has been visited\n"
            "- depth: shortest-path depth from root\n\n"
            f"Candidate nodes:\n{json.dumps(nodes, indent=2)}\n\n"
            f"Select the best next node and return only the top {top_k} ranked nodes as a list in the output.\n\n"
            "Remember:\n"
            "- Favor nodes that are likely to reveal new states.\n"
            "- Avoid risky or low-value pages when better alternatives exist.\n"
            "- Balance unexplored actions, depth, visits, and page type.\n"
            "- Each reason: one short sentence (under 12 words).\n"
            "- Return only valid JSON."
        )

        response = self._multimodal_conversation_call(
            model=self.configs.graph.exploration_modes.model_llm_bfs.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format='message',
            temperature=0.1,
            seed=42,
            top_p=0.9,
            max_tokens=max(192, top_k * 96 + 32),
        )

        model_name = self.configs.graph.exploration_modes.model_llm_bfs.model_name
        usage = getattr(response, "usage", None)
        if usage is not None:
            details = getattr(usage, "input_tokens_details", None) or {}
            text_input_tokens = (
                details.get("text_tokens", 0)
                if isinstance(details, dict)
                else getattr(details, "text_tokens", 0)
            )
            out_details = getattr(usage, "output_tokens_details", None) or {}
            output_tokens = (
                out_details.get("text_tokens", 0)
                if isinstance(out_details, dict)
                else getattr(out_details, "text_tokens", 0)
            )
            if model_name in self.token_usage_details:
                self.token_usage_details[model_name]["text_input_tokens"] += text_input_tokens
                self.token_usage_details[model_name]["output_tokens"] += output_tokens
            else:
                self.token_usage_details[model_name] = {
                    "text_input_tokens": text_input_tokens,
                    "output_tokens": output_tokens,
                }

        raw = response.output.choices[0].message.content
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )
        return parse_json_from_model_response(raw_text)


    def compare_navigtional_purpose(
        self,
        query_page_purpose,
        query_action_history_descriptions,
        candidates,
    ):
        system_prompt = """
    You are a mobile app graph deduplication judge.

    Decide whether the query screen has the same page/navigational purpose as one of the candidate nodes.

    Use:
    - page_purpose as the main signal
    - action_path as supporting context

    Same purpose examples:
    - search results for shoes vs search results for laptops
    - product detail page for item A vs item B
    - filter modal from one search results page vs filter modal from another
    - cart with 1 item vs cart with 3 items

    Different purpose examples:
    - search results page vs product detail page
    - product detail page vs reviews page
    - cart page vs checkout page
    - filter modal vs sort modal
    - normal page vs modal/dialog/drawer

    Be conservative. If no candidate clearly has the same purpose, return null.

    Return only valid JSON with this exact schema:
    {
    "matched_node_id": "node id or null",
    "same_purpose": true,
    "confidence": 0.0,
    "reason": "short reason"
    }
    """.strip()

        user_prompt = json.dumps(
            {
                "query": {
                    "page_purpose": query_page_purpose,
                    "action_path": query_action_history_descriptions,
                },
                "candidates": candidates,
                "rule": "If confidence is below 0.75, return matched_node_id as null and same_purpose as false.",
            },
            ensure_ascii=False,
            indent=2,
        )

        response = self._multimodal_conversation_call(
            model=self.configs.graph.where_am_i_global.find_node_page_purpose_emphasize.compare_navigational_purpose_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            result_format="message",
            temperature=0,
            top_p=1,
            seed=42,
        )

        text_input_tokens = response.usage.input_tokens_details["text_tokens"]
        output_tokens = response.usage.output_tokens_details["text_tokens"]

        usage_key = self.configs.graph.exploration_modes.model_llm_bfs.model_name

        if usage_key in self.token_usage_details:
            self.token_usage_details[usage_key]["text_input_tokens"] += text_input_tokens
            self.token_usage_details[usage_key]["output_tokens"] += output_tokens
        else:
            self.token_usage_details[usage_key] = {
                "text_input_tokens": text_input_tokens,
                "output_tokens": output_tokens,
            }

        raw = response["output"]["choices"][0]["message"]["content"]
        raw_text = (
            "".join(chunk.get("text", "") if isinstance(chunk, dict) else str(chunk) for chunk in raw).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )

        result = parse_json_from_model_response(raw_text)

        if result.get("matched_node_id") not in candidates:
            result["matched_node_id"] = None
            result["same_purpose"] = False

        conf_thresh = self.configs.graph.where_am_i_global.find_node_page_purpose_emphasize.page_purpose_similarity_threshold
        if float(result.get("confidence", 0.0)) < conf_thresh:
            result["matched_node_id"] = None
            result["same_purpose"] = False

        return result

    def get_node_user_intents(self, path, out_edges):
        """
        Generate user_intents for the final active surface of one visual path.

        Input:
            path: {
                "screenshots": [...],
                "page_summaries": [...],
                "actions": [...]
            }
            out_edges: outgoing edges/descriptions from the final node

        Output:
            {
                "user_intents": [...]
            }
        """
        def clean_actions(actions):
            cleaned_steps = []

            for step_edges in actions:
                cleaned_edges = []

                if isinstance(step_edges, dict):
                    edge_items = step_edges.values()
                elif isinstance(step_edges, list):
                    edge_items = step_edges
                else:
                    edge_items = []

                for attrs in edge_items:
                    if not isinstance(attrs, dict):
                        continue

                    cleaned_edges.append({
                        "type": attrs.get("type", ""),
                        "description": attrs.get("description", "")
                    })

                cleaned_steps.append(cleaned_edges)

            return cleaned_steps

        def clean_outgoing_edges(out_edges):
            descriptions = []

            for edge in out_edges:
                if isinstance(edge, str):
                    desc = edge.strip()
                    if desc:
                        descriptions.append(desc)
                    continue

                if len(edge) == 3:
                    _, _, attrs = edge
                elif len(edge) == 4:
                    _, _, _, attrs = edge
                else:
                    continue

                if not isinstance(attrs, dict):
                    continue

                desc = attrs.get("description", "")
                if desc:
                    descriptions.append(desc)

            return list(dict.fromkeys(descriptions))

        system_prompt = """
    You are a precise mobile UI user-intent generator.

    You are given one visual navigation path inside an already-open mobile app.

    The FINAL screenshot is the main target.
    Earlier screenshots are only route context.
    Outgoing actions are only possible next actions from the final page.

    Task:
    Generate only user_intents for the FINAL ACTIVE SURFACE.

    Core interpretation rules:
    - Focus primarily on the FINAL screenshot/state.
    - Use earlier screenshots only to understand how the final state was reached.
    - Use page summaries and action descriptions as structured evidence.
    - Use visual action annotations only to understand which controls were used.
    - Use outgoing actions only to understand available next actions from the final page.
    - Do not treat outgoing actions as proof of the final page's intent.
    - Do not treat visual action annotations as app UI content.
    - Do not output coordinates, bounding boxes, node ids, raw edge dictionaries, overlay colors, or usage instructions.
    - The app is already open. Never include instructions such as "open the app", "launch the app", "go to home screen", or "start from the launcher".

    Active-surface rule:
    - The current node/state is defined by the TOPMOST ACTIVE UI SURFACE in the FINAL screenshot.
    - If a modal, dialog, bottom sheet, overflow menu, popup menu, dropdown, picker, drawer, search overlay, permission prompt, or focused panel is open, focus on that active surface as the final page/state.
    - The dimmed or underlying page is only background context when an active overlay is present.
    - Do NOT generate intents for the underlying page when an active overlay is present.
    - If no overlay or focused active surface is present, use the main visible page/tab as the final page/state.

    User intent rules:
    - Return one ranked list of natural-language user intents that the CURRENT FINAL ACTIVE SURFACE itself can satisfy.
    - Each intent must be a complete sentence describing a user navigation goal.
    - For normal pages, tabs, settings screens, lists, management screens, configuration screens, detail pages, browsing screens, and forms, phrase intents as destination-style navigation goals.
    - Prefer "Go to the page to ..." or "Navigate to the page to ..." over direct functional phrasing.
    - Do NOT write direct functional intents such as "Manage configured alarms.", "Change the temperature unit.", "Block JavaScript for a specific site.", or "Select AM or PM for the new alarm time." when the final active surface is a page, form, or settings screen where that task can be done.
    - Instead, write destination-style intents such as:
    "Go to the page to manage configured alarms."
    "Navigate to the page to change the temperature unit."
    "Go to the setting for blocking JavaScript for a specific site."
    "Go to the page to change AM or PM for a new alarm."

    Control-inside-page wording rule:
    - Visible controls such as switches, radio buttons, AM/PM selectors, checkboxes, dropdowns, text fields, sliders, buttons, or rows may be part of the page's purpose.
    - However, if those controls are embedded inside a normal page, form, or settings screen, do not phrase the intent as directly operating that control.
    - Phrase the intent as navigating to the page, form, screen, or setting where that control can be changed.
    - Only use direct control-level phrasing when the topmost active surface itself is a picker, dialog, dropdown menu, popup menu, confirmation prompt, permission prompt, or focused control panel.

    When direct functional phrasing is allowed:
    - Use direct functional phrasing only when the topmost active surface itself is an immediate-action surface.
    - Immediate-action surfaces include dialogs, confirmation prompts, standalone menus, popup menus, dropdown menus, expanded pickers, permission prompts, and focused panels.
    - A normal page, form, or settings screen containing selectable controls is NOT an immediate-action surface.

    Strict outgoing-action rule:
    - outgoing_actions are provided only to prevent confusing current-state purpose with next-page destinations.
    - Do NOT include destination-specific intents that require taking an outgoing edge.
    - Do NOT create "Go to X" intents from outgoing actions.
    - If an outgoing action opens a new page, do not treat that new page's purpose as an intent of the current active state.

    Before finalizing:
    - Identify the topmost active UI surface in the FINAL screenshot.
    - user_intents must be complete natural-language sentences.
    - For page-like states, user_intents must be destination-style navigation goals.
    - For forms and settings screens, user_intents must describe navigating to the form or setting where controls can be changed.
    - For embedded controls inside a page/form/settings screen, user_intents must not be phrased as directly operating the control.
    - For immediate-action surfaces, user_intents may be direct functional goals.
    - user_intents must remain one single ranked list.
    - user_intents must not include destination-specific intents requiring an outgoing edge.

    Return valid JSON only with exactly this schema:
    {
    "user_intents": [
        "..."
    ]
    }
    """.strip()

        user_prompt = f"""
    You are given one visual navigation path to a FINAL mobile app page/state.

    Important:
    - The app is already open.
    - Do NOT include any instruction to open, launch, or start the app.
    - The screenshots are provided in order.
    - The LAST screenshot is the FINAL target state.
    - The LAST screenshot has higher resolution and should receive the most attention.
    - Earlier screenshots are lower-resolution route context only.
    - Some screenshots contain visualized action annotations showing the action taken from that page.
    - These annotations are not app UI content; use them only to understand the route.
    - page_summaries and actions are ordered from the start page to the final state.
    - actions[i] describes the transition from page_summaries[i] to page_summaries[i + 1].

    Critical active-surface instruction:
    - In the FINAL screenshot, first identify the topmost active UI surface.
    - If a modal, dialog, bottom sheet, overflow menu, popup menu, dropdown menu, expanded picker, drawer, search overlay, permission prompt, or focused panel is open, focus on that active surface.
    - The underlying page is only background context when an active overlay is present.
    - Do NOT generate intents for the underlying page when the active surface is a modal/menu/dialog/picker.

    Critical destination-intent instruction:
    - user_intents must describe what page, form, screen, tab, setting, or active surface the user wants to navigate to.
    - For normal pages, tabs, settings screens, lists, management screens, configuration screens, detail pages, browsing screens, and forms, every intent must be destination-style.
    - Do NOT phrase intents as directly doing the task when the final state is a page/form/settings screen.
    - For embedded controls inside a page/form/settings screen, user_intents must describe navigating to the place where the control can be changed.

    Critical distinction:
    - actions describe how the final state was reached.
    - outgoing_actions describe actions available FROM the final state.
    - outgoing_actions are NOT necessarily the main purpose of the final state.
    - Do NOT include destination-specific intents that require taking an outgoing edge.

    Ordered page summaries:
    {path["page_summaries"]}

    Ordered action descriptions used to reach the final state:
    {clean_actions(path["actions"])}

    Outgoing actions available from the final state:
    {clean_outgoing_edges(out_edges)}

    Generate JSON for the FINAL ACTIVE SURFACE only.

    Output requirements:
    - Return only user_intents.
    - user_intents must be one ranked list only.
    - user_intents must be complete natural-language sentences.
    - For page-like states, form states, and settings states, user_intents must be destination-style navigation goals.
    - Do NOT generate direct control-operation intents unless the control is the topmost active picker/dialog/menu/focused panel.
    - Do NOT include destination-specific intents that require taking an outgoing edge.
    - Do not output coordinates, bounding boxes, node ids, raw edge dictionaries, visual overlay descriptions, or usage instructions.

    Return JSON only.
    """.strip()

        screenshot_uris = []

        for i, screenshot in enumerate(path["screenshots"]):
            if screenshot is None:
                raise FileNotFoundError(f"Missing screenshot at path step {i}")

            is_final = i == len(path["screenshots"]) - 1

            max_side = (
                self.configs.post_process.vlm_processing_img_size_for_last_screenshot
                if is_final
                else self.configs.post_process.vlm_processing_img_size_for_other_screenshots
            )

            quality = (
                self.configs.post_process.vlm_processing_img_quality_for_last_screenshot
                if is_final
                else self.configs.post_process.vlm_processing_img_quality_for_other_screenshots
            )

            _, _, data_uri = resize_and_encode_to_base64(
                screenshot,
                target_size=max_side,
                jpeg_quality=quality,
            )

            screenshot_uris.append(data_uri)

        user_content = [{"text": user_prompt}]

        for i, data_uri in enumerate(screenshot_uris):
            is_final = i == len(screenshot_uris) - 1

            label = (
                f"Screenshot {i + 1}/{len(screenshot_uris)}: "
                + (
                    "FINAL TARGET PAGE/STATE. Focus mainly on this image."
                    if is_final
                    else "Lower-resolution route context only."
                )
            )

            user_content.append({"text": label})
            user_content.append({"image": data_uri})

        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = self._multimodal_conversation_call(
            model=self.configs.post_process.vlm_model_name,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        raw = response.output.choices[0].message.content
        raw_text = (
            "".join(
                item.get("text", "")
                for item in raw
                if isinstance(item, dict)
            ).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )

        output = parse_json_from_model_response(raw_text)

        return {
            "user_intents": output.get("user_intents", [])
        }




    def get_node_navigation_plans(self, paths):
        """
        Generate route/navigation plans for all paths using only page_summaries and actions,
        but call Qwen through MultiModalConversation for qwen3.6-plus style usage.

        Input:
            paths:
                dict[root_id -> path]
            where each path has:
                {
                    "page_summaries": [...],
                    "actions": [...]
                }

        Output:
            [
                {
                    "root_id": root_id,
                    "ui_navigation_memory": {
                        "relevant_waypoint_sequence": [...],
                        "transition_hints": [...]
                    }
                },
                ...
            ]
        """
        def clean_actions(actions):
            cleaned_steps = []

            for step_edges in actions:
                cleaned_edges = []

                if isinstance(step_edges, dict):
                    edge_items = step_edges.values()
                elif isinstance(step_edges, list):
                    edge_items = step_edges
                else:
                    edge_items = []

                for attrs in edge_items:
                    if not isinstance(attrs, dict):
                        continue

                    cleaned_edges.append({
                        "type": attrs.get("type", ""),
                        "description": attrs.get("description", "")
                    })

                cleaned_steps.append(cleaned_edges)

            return cleaned_steps

        system_prompt = """
    You are a precise mobile UI navigation-memory generator.

    You are given one or more in-app navigation paths represented only by:
    - ordered page summaries
    - ordered action descriptions between consecutive pages

    Task:
    Generate compact route memory for each path.

    Do NOT generate user_intents.
    Do NOT infer new page purposes beyond the provided page summaries.
    Do NOT include app launch, phone home screen, or OS-level states.
    Do NOT include coordinates, node ids, screenshots, raw edge dictionaries, visual overlay descriptions, or usage instructions.

    For each path, generate:
    - relevant_waypoint_sequence
    - transition_hints

    Rules for relevant_waypoint_sequence:
    - Use semantic page/state names, not raw actions.
    - The final waypoint must be the final page/state from the last page summary.
    - The first waypoint should be the earliest relevant in-app screen from the provided path.
    - Never include "Open app", "Launch app", "Phone home screen", or OS-level states.
    - Every waypoint must be inside the same app.
    - Every waypoint must move toward the final page/state.
    - Remove duplicate pages, loops, retries, and backtracking.
    - Do not include child-page detours if the final page is a parent page.
    - The waypoint sequence should describe how to reach the final page/state, not what can be done after reaching it.

    Rules for transition_hints:
    - Convert action descriptions into robust agent instructions.
    - Mention rough stable regions only if useful, such as top-right menu, bottom tab, settings list, search field, modal, or bottom sheet.
    - Mention scrolling only when genuinely needed to reveal the next target.
    - If the path contains "scroll then click X", rewrite it as:
    "Look for X in the list; scroll or swipe if it is not visible."
    - Avoid exact coordinates, bounding boxes, fragile visual details, and overlay colors.
    - Never include "open the app" or "launch the app".
    - Do not include actions that happen after the final page/state is reached.
    - Do not write usage instructions or describe what the page can be used for.

    Return valid JSON only with exactly this schema:
    {
    "navigation_plans": [
        {
        "root_id": "...",
        "ui_navigation_memory": {
            "relevant_waypoint_sequence": [
            "..."
            ],
            "transition_hints": [
            "..."
            ]
        }
        }
    ]
    }
    """.strip()

        if isinstance(paths, dict):
            path_items = list(paths.items())
        else:
            path_items = [(str(i), path) for i, path in enumerate(paths)]

        compact_paths = []

        for root_id, path in path_items:
            compact_paths.append({
                "root_id": str(root_id),
                "page_summaries": path.get("page_summaries", []),
                "actions": clean_actions(path.get("actions", [])),
            })

        user_prompt = f"""
    You are given multiple in-app navigation paths to the same FINAL node/page/state.

    Each path may start from a different root or starting context.

    Generate route memory for each path independently.

    Do NOT generate user_intents.
    Do NOT use screenshots.
    Do NOT include app launch or phone home screen steps.

    Paths:
    {json.dumps(compact_paths, ensure_ascii=False, indent=2)}

    Return JSON only.
    """.strip()

        messages = [
            {
                "role": "system",
                "content": [{"text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"text": user_prompt}],
            },
        ]

        response = self._multimodal_conversation_call(
            model=self.configs.post_process.vlm_model_name,
            messages=messages,
            response_format={"type": "json_object"},
            vl_high_resolution_images=True,
            temperature=0.0,
            seed=42,
            top_p=0.9,
        )

        raw = response.output.choices[0].message.content
        raw_text = (
            "".join(
                item.get("text", "")
                for item in raw
                if isinstance(item, dict)
            ).strip()
            if isinstance(raw, list)
            else str(raw or "").strip()
        )

        output = parse_json_from_model_response(raw_text)

        return output.get("navigation_plans", [])
