"""UI-TARS 1.5 agent: OpenAI-compatible chat with message history and action parsing."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from openai import OpenAI
from PIL import Image

from Agents.utils import (
    ResizeMeta,
    action_with_resize_dims,
    prepare_ui_tars_jpeg_data_url,
    ui_tars_extract_coords_from_text,
)

from .action_parser import add_box_token, parse_action_to_structure_output
from .prompt import GROUNDING_DOUBAO, MOBILE_USE_DOUBAO


@dataclass
class ParsedAction:
    raw_text: str
    thought: str
    action_type: str
    params: Dict[str, str] = field(default_factory=dict)
    sent_coords: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    orig_coords: Dict[str, Tuple[int, int]] = field(default_factory=dict)


def _normalize_llm_base_url(url: str) -> str:
    u = str(url or "").strip().rstrip("/")
    if not u:
        raise ValueError("llm_base_url is empty")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", u):
        u = "http://" + u.lstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


class UITARS_1_5:
    def __init__(
        self,
        llm_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: str = "empty",
        runtime_conf: Optional[Dict[str, Any]] = None,
        jpeg_quality: int = 90,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        debugger: Optional[Debugger] = None,
    ) -> None:
        llm_base_url = llm_base_url or base_url
        model_name = model_name or model or "ByteDance-Seed/UI-TARS-1.5-7B"
        if not llm_base_url:
            raise ValueError("llm_base_url (or base_url) is required")
        default_conf = {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 512,
            "stream": False,
        }
        self.runtime_conf = {**default_conf, **(runtime_conf or {})}
        self.llm_base_url = _normalize_llm_base_url(llm_base_url)
        self.model_name = model_name
        self.jpeg_quality = jpeg_quality
        self.client = OpenAI(base_url=self.llm_base_url, api_key=api_key)
        self.messages: List[Dict[str, Any]] = []
        self.history_n = int(self.runtime_conf.get("history_n", 5))
        self.assistant_history: List[str] = []
        self.debugger = debugger

    def _recent_assistant_history(self) -> List[str]:
        if self.history_n <= 0:
            return []
        return self.assistant_history[-self.history_n :]

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = copy.deepcopy(messages)
        for message in out:
            if message.get("role") == "assistant" and isinstance(message.get("content"), str):
                message["content"] = add_box_token(message["content"])
        return out

    def complete(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        *,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> str:
        msgs = self.prepare_messages(messages if messages is not None else self.messages)
        stream = self.runtime_conf["stream"] if stream is None else stream
        chat = self.client.chat.completions.create(
            model=self.model_name,
            messages=msgs,
            temperature=kwargs.pop("temperature", self.runtime_conf["temperature"]),
            max_tokens=kwargs.pop("max_tokens", self.runtime_conf["max_tokens"]),
            top_p=kwargs.pop("top_p", self.runtime_conf["top_p"]),
            stream=stream,
            **kwargs,
        )
        if stream:
            return "".join(
                (chunk.choices[0].delta.content or "") for chunk in chat
            )
        return (chat.choices[0].message.content or "").strip()

    def load_messages(self, messages: List[Dict[str, Any]]) -> None:
        self.messages = copy.deepcopy(messages)

    def parse_response(self, text: str, meta: ResizeMeta) -> ParsedAction:
        actions = parse_action_to_structure_output(text)
        if not actions:
            raise ValueError(f"No actions parsed from model output:\n{text}")
        action = actions[0]
        thought = str(action.get("thought") or "")
        action_type = str(action.get("action_type") or "")
        params = {k: str(v) for k, v in (action.get("action_inputs") or {}).items() if v is not None}
        sent_coords, orig_coords = ui_tars_extract_coords_from_text(text, meta)

        if action_type == "click" and "point" not in orig_coords and orig_coords:
            orig_coords["point"] = next(iter(orig_coords.values()))

        return ParsedAction(
            raw_text=text,
            thought=thought,
            action_type=action_type,
            params=params,
            sent_coords=sent_coords,
            orig_coords=orig_coords,
        )

    def step(
        self,
        instruction: str,
        image: Union[np.ndarray, Image.Image],
        language: str = "English",
        *,
        messages: Optional[List[Dict[str, Any]]] = None,
        append_history: bool = True,
    ) -> Tuple[ParsedAction, int, int]:
        data_url, meta = prepare_ui_tars_jpeg_data_url(
            image, jpeg_quality=self.jpeg_quality
        )

        if messages is not None:
            turn_messages = copy.deepcopy(messages)
            turn_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Current screen screenshot. Reply with exactly one "
                                "Thought: ... / Action: ... block."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )
        else:
            system_content = MOBILE_USE_DOUBAO.format(
                language=(language.strip() or "English"),
                instruction=instruction.strip(),
            )
            turn_messages = [{"role": "system", "content": system_content}]
            recent = self._recent_assistant_history()
            if recent:
                turn_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Prior steps (Thought + Action). Use with the new screenshot "
                            "for the single next step only.\n\n"
                            + "\n\n---\n\n".join(recent)
                        ),
                    }
                )
            turn_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Current screen screenshot. Reply with exactly one "
                                "Thought: ... / Action: ... block."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            )

        raw_text = self.complete(turn_messages)
        parsed = self.parse_response(raw_text, meta)
        if append_history:
            self.assistant_history.append(raw_text)
            if self.history_n > 0:
                self.assistant_history = self.assistant_history[-self.history_n :]
            else:
                self.assistant_history = []
            if messages is not None:
                self.messages = turn_messages + [{"role": "assistant", "content": raw_text}]
        return action_with_resize_dims(parsed, meta), meta

    def grounding_action(
        self,
        image: Union[np.ndarray, Image.Image],
        action_description: str,
    ) -> Tuple[ParsedAction, int, int]:
        data_url, meta = prepare_ui_tars_jpeg_data_url(
            image, jpeg_quality=self.jpeg_quality
        )

        messages = [
            {
                "role": "system",
                "content": GROUNDING_DOUBAO.format(instruction=action_description.strip()),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Find an action with the following description:\n"
                            f"{action_description.strip()}\n\n"
                            "Ground this action on the screenshot and return the target action with coordinates."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ]
        parsed = self.parse_response(self.complete(messages), meta)
        return action_with_resize_dims(parsed, meta), meta


