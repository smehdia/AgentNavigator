"""
VLM backend using yibuapi.com (OpenAI-compatible chat/completions).

Mirror of inference/VLM.py: same rerank_candidates contract, Yibu API instead of DashScope.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

import requests
import urllib3

for _proxy_var in (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "all_proxy",
):
    os.environ.pop(_proxy_var, None)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_yibu_session = requests.Session()
_yibu_session.trust_env = False
_logger = logging.getLogger(__name__)


def parse_json_from_model_response(text):
    raw = (text or "").strip()
    if not raw:
        raise ValueError("Could not parse model JSON response: empty text")
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    def _balanced_json_slice(s):
        start = next((i for i, ch in enumerate(s) if ch in "{["), -1)
        if start < 0:
            return None
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
            if ch == '"':
                in_str = True
                continue
            if ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    return None
                top = stack.pop()
                if (top, ch) not in {("{", "}"), ("[", "]")}:
                    return None
                if not stack:
                    return s[start : i + 1]
        return None

    last_error = None
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
        except Exception as exc:
            last_error = exc
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


class VLM_Yibu:
    def __init__(self, configs, debugger):
        self.total_calls = {}
        self.token_usage_details = {}
        self.yibu_api_key = str(configs.vlm.yibu_api_key).strip()
        self.yibu_base_url = str(
            getattr(configs.vlm, "yibu_base_url", "https://yibuapi.com")
        ).rstrip("/")
        self.rerank_model = str(
            getattr(configs.vlm, "yibu_rerank_model", "qwen3.5-flash")
        ).strip()
        self.enable_thinking = bool(getattr(configs.vlm, "enable_thinking", False))
        connect_timeout = int(getattr(configs.vlm, "connect_timeout", 30))
        read_timeout = int(getattr(configs.vlm, "read_timeout", 180))
        self._request_timeout = (connect_timeout, read_timeout)
        self._request_retries = int(getattr(configs.vlm, "request_retries", 5))
        self._retryable_statuses = {429, 500, 502, 503, 504}
        self.debugger = debugger
        self.configs = configs

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
                            "image_url": {"url": part["image"], "detail": "high"},
                        }
                    )
                elif "text" in part:
                    parts.append({"type": "text", "text": part["text"]})
            openai_messages.append({"role": role, "content": parts or ""})
        return openai_messages

    def _multimodal_conversation_call(self, model: str, messages: list, **kwargs):
        url = f"{self.yibu_base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.yibu_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": self._dashscope_messages_to_openai(messages),
            "enable_thinking": kwargs.get("enable_thinking", self.enable_thinking),
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
            try:
                response = _yibu_session.post(
                    url,
                    headers=headers,
                    json=payload,
                    verify=False,
                    timeout=self._request_timeout,
                )
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
                return YibuMultimodalResponse(response.json())

            except (requests.ConnectTimeout, requests.ReadTimeout) as exc:
                last_error = exc
                _logger.warning(
                    "Yibu API timeout for model=%s (attempt %d/%d): %s",
                    model,
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

    def rerank_candidates(self, user_query, candidates, top_k=3):
        if not candidates:
            return [], {"top_k_node_ids": [], "reasoning": {}}

        top_k = min(top_k, len(candidates))

        compact_candidates = []
        for c in candidates:
            page_description = c.get("page_description", {}) or {}
            compact_candidates.append(
                {
                    "node_id": c.get("node_id"),
                    "retrieval_score": c.get("retrieval_score", 0.0),
                    "page_description": {
                        "high_level": page_description.get("high_level", ""),
                        "medium_level": page_description.get("medium_level", ""),
                        "low_level": page_description.get("low_level", ""),
                    },
                    "user_intents": c.get("user_intents", []),
                    "ui_navigation_memory": c.get("ui_navigation_memory", []),
                }
            )

        system_prompt = """
    You are a strict reranker for mobile UI navigation.

    You are given:
    - user query
    - candidate target nodes
    - each candidate has page_description, user_intents, retrieval_score, and ui_navigation_memory

    Select the best target node ids.

    Ranking rules:
    - Prefer candidates whose user_intents directly match the user query.
    - Use page_description to resolve ambiguity.
    - ui_navigation_memory is only route evidence, not main relevance evidence.
    - retrieval_score is only a tie-breaker.
    - Prefer the actual destination page/state, not an intermediate page that only leads there.
    - If the query asks for a specific page, choose the most specific matching page.
    - If the query asks for search history, choose search history, not recommendation/history feed.
    - If the query asks for invoice or receipt history, choose invoices/receipts, not generic history.
    - If the query asks for settings, choose actual settings/account configuration.
    - If the query asks for filters or labels, choose the filter/label surface.
    - Penalize candidates that match words but not the actual user intent.
    - Penalize overlays/modals unless the query asks for that overlay or its direct function.
    - Only select node_ids from candidates.
    - Return JSON only.
    """.strip()

        user_prompt = f"""
    User query:
    {user_query}

    Candidates:
    {json.dumps(compact_candidates, ensure_ascii=False, indent=2)}

    Select exactly {top_k} best candidate nodes.

    Return exactly:

    {{
    "top_k_node_ids": [
        "node_id"
    ],
    "reasoning": {{
        "node_id": "brief reason"
    }}
    }}

    Do not include markdown or extra text.
    """.strip()

        response = self._multimodal_conversation_call(
            model=self.rerank_model,
            messages=[
                {"role": "system", "content": [{"text": system_prompt}]},
                {"role": "user", "content": [{"text": user_prompt}]},
            ],
            response_format={"type": "json_object"},
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        content = response.output.choices[0].message.content
        if isinstance(content, list):
            text = "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        else:
            text = str(content or "")

        result = parse_json_from_model_response(text.strip())

        valid_ids = {c["node_id"] for c in compact_candidates if c.get("node_id")}

        selected = []
        for node_id in result.get("top_k_node_ids", []):
            if node_id in valid_ids and node_id not in selected:
                selected.append(node_id)

        if len(selected) < top_k:
            for c in compact_candidates:
                node_id = c.get("node_id")
                if node_id in valid_ids and node_id not in selected:
                    selected.append(node_id)
                if len(selected) >= top_k:
                    break

        selected = selected[:top_k]

        reasoning = result.get("reasoning", {})
        if not isinstance(reasoning, dict):
            reasoning = {}

        result = {
            "top_k_node_ids": selected,
            "reasoning": {
                node_id: str(reasoning.get(node_id, "Selected by reranker.")).strip()
                for node_id in selected
            },
        }

        return result["top_k_node_ids"], result


def build_vlm_client(configs, debugger):
    """Return VLM_Yibu when configs.vlm.use_yibu_api else Alibaba DashScope VLM."""
    if getattr(getattr(configs, "vlm", None), "use_yibu_api", False):
        return VLM_Yibu(configs, debugger)
    from VLM import VLM

    return VLM(configs, debugger)
