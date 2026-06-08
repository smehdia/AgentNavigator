"""Thin facade over MAI navigation / grounding agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image

from Agents.utils import (
    ResizeMeta,
    action_with_resize_dims,
    mai_coord_to_orig,
    prepare_mai_ui_image,
    reverse_mai_swipe_direction,
)

from .mai_grounding_agent import MAIGroundingAgent
from .mai_navigation_agent import (
    MAIUINaivigationAgent,
    parse_action_to_structure_output,
    parse_tagged_text,
)


@dataclass
class ParsedAction:
    raw_text: str
    thought: str
    action_type: str
    params: Dict[str, str] = field(default_factory=dict)
    sent_coords: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    orig_coords: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def _to_parsed(raw: str, action_json: Optional[Dict[str, Any]], meta: ResizeMeta) -> ParsedAction:
    if not action_json or action_json.get("action") is None:
        return ParsedAction(raw_text=raw or "", thought="", action_type="unknown", params={})

    action = str(action_json.get("action", "") or "").strip().lower()
    try:
        thought = str(parse_tagged_text(raw).get("thinking") or "")
    except Exception:
        thought = ""

    params: Dict[str, str] = {}
    sent_coords: Dict[str, Tuple[int, int]] = {}
    orig_coords: Dict[str, Tuple[int, int]] = {}

    def put(key: str, coord: Optional[list]) -> None:
        s, o = mai_coord_to_orig(coord, meta)
        if s:
            sent_coords[key] = s
        if o:
            orig_coords[key] = o

    if action in ("click", "long_press", "double_click"):
        put("point", action_json.get("coordinate"))
        at = "long_press" if action == "long_press" else "click"
        return ParsedAction(
            raw_text=raw, thought=thought, action_type=at,
            params=params, sent_coords=sent_coords, orig_coords=orig_coords,
        )
    if action == "swipe":
        params["direction"] = reverse_mai_swipe_direction(str(action_json.get("direction", "up")))
        put("point", action_json.get("coordinate"))
        return ParsedAction(
            raw_text=raw, thought=thought, action_type="scroll",
            params=params, sent_coords=sent_coords, orig_coords=orig_coords,
        )
    if action == "drag":
        put("start_point", action_json.get("start_coordinate"))
        put("end_point", action_json.get("end_coordinate"))
        return ParsedAction(
            raw_text=raw, thought=thought, action_type="drag",
            params=params, sent_coords=sent_coords, orig_coords=orig_coords,
        )
    if action == "type":
        params["content"] = str(action_json.get("text", "") or "")
        return ParsedAction(raw_text=raw, thought=thought, action_type="type", params=params)
    if action == "system_button":
        btn = str(action_json.get("button", "") or "").strip().lower()
        if btn == "back":
            return ParsedAction(raw_text=raw, thought=thought, action_type="press_back", params=params)
        if btn == "home":
            return ParsedAction(raw_text=raw, thought=thought, action_type="press_home", params=params)
        return ParsedAction(raw_text=raw, thought=thought, action_type="unknown", params=params)
    if action in ("terminate", "answer"):
        params["content"] = str(
            action_json.get("status" if action == "terminate" else "text", "") or ""
        )
        return ParsedAction(raw_text=raw, thought=thought, action_type="finished", params=params)
    if action == "wait":
        return ParsedAction(raw_text=raw, thought=thought, action_type="wait", params=params)
    if action == "open":
        params["app_name"] = str(action_json.get("text", "") or "")
        return ParsedAction(raw_text=raw, thought=thought, action_type="open_app", params=params)
    return ParsedAction(raw_text=raw, thought=thought, action_type=action or "unknown", params=params)


def _build_mai_runtime_conf(runtime_conf: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_conf = {
        "history_n": 3,
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
        "max_tokens": 512,
    }
    conf = {**default_conf, **(runtime_conf or {})}
    ctx = int(conf.pop("max_context_length", 2048))
    max_out = int(conf.get("max_tokens", 512))
    if ctx <= 4096:
        conf["max_tokens"] = min(max_out, 512)
    else:
        conf["max_tokens"] = min(max_out, max(512, ctx // 4))
    return conf


class MAI_UI:
    def __init__(
        self,
        llm_base_url: str,
        model_name: str = "MAI-UI-8B",
        runtime_conf: Optional[Dict[str, Any]] = None,
        mcp_tools: Optional[List[Dict[str, Any]]] = None,
        debugger=None,
    ) -> None:
        conf = _build_mai_runtime_conf(runtime_conf)
        self._navigation = MAIUINaivigationAgent(
            llm_base_url=llm_base_url,
            model_name=model_name,
            runtime_conf=conf,
            mcp_tools=mcp_tools,
        )
        self._grounding = MAIGroundingAgent(
            llm_base_url=llm_base_url,
            model_name=model_name,
            runtime_conf=conf,
        )
        self.debugger = debugger

    def reset(self) -> None:
        self._navigation.reset()

    def step(
        self,
        instruction: str,
        image: Union[np.ndarray, Image.Image],
    ) -> Tuple[ParsedAction, int, int]:
        pil, meta = prepare_mai_ui_image(image)

        raw, action_json = self._navigation.predict(instruction.strip(), {"screenshot": pil})
        if not raw or raw == "llm client error" or not action_json:
            return action_with_resize_dims(
                ParsedAction(raw_text=raw or "", thought="", action_type="unknown", params={}),
                meta,
            ), meta
        try:
            if action_json.get("action") is None:
                action_json = parse_action_to_structure_output(raw)["action_json"]
        except Exception:
            pass
        return action_with_resize_dims(_to_parsed(raw, action_json, meta), meta), meta

    def grounding_action(
        self,
        image: Union[np.ndarray, Image.Image],
        action_description: str,
    ) -> Tuple[ParsedAction, int, int]:
        pil, meta = prepare_mai_ui_image(image)

        raw, result = self._grounding.predict(action_description.strip(), pil)
        if not raw or raw == "llm client error" or not result:
            return action_with_resize_dims(
                ParsedAction(raw_text=raw or "", thought="", action_type="unknown", params={}),
                meta,
            ), meta
        sent_coords: Dict[str, Tuple[int, int]] = {}
        orig_coords: Dict[str, Tuple[int, int]] = {}
        s, o = mai_coord_to_orig(result.get("coordinate"), meta)
        if s:
            sent_coords["point"] = s
        if o:
            orig_coords["point"] = o
        return action_with_resize_dims(
            ParsedAction(
                raw_text=raw,
                thought=str(result.get("thinking") or ""),
                action_type="click" if orig_coords else "unknown",
                params={},
                sent_coords=sent_coords,
                orig_coords=orig_coords,
            ),
            meta,
        ), meta


MAIUINavigationAgent = MAI_UI
