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
import dashscope
import networkx as nx
from dashscope import MultiModalConversation, Generation



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


class VLM:
    def __init__(self, configs, debugger):
        self.total_calls = {}
        self.token_usage_details = {}
        dashscope.api_key = configs.vlm.alibaba_api_key 
        dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
        self.debugger = debugger
        self.configs = configs


    def rerank_candidates(
        self,
        user_query,
        candidates,
        top_k=3,
    ):
        import json
        from dashscope import MultiModalConversation

        if not candidates:
            return [], {"top_k_node_ids": [], "reasoning": {}}

        top_k = min(top_k, len(candidates))

        compact_candidates = []

        for c in candidates:
            page_description = c.get("page_description", {}) or {}

            compact_candidates.append({
                "node_id": c.get("node_id"),
                "retrieval_score": c.get("retrieval_score", 0.0),
                "page_description": {
                    "high_level": page_description.get("high_level", ""),
                    "medium_level": page_description.get("medium_level", ""),
                    "low_level": page_description.get("low_level", ""),
                },
                "user_intents": c.get("user_intents", []),
                "ui_navigation_memory": c.get("ui_navigation_memory", []),
            })

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

        response = MultiModalConversation.call(
            model="qwen3.6-plus",
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

        if response.status_code != 200:
            raise RuntimeError(
                f"VLM call failed: status_code={response.status_code}, "
                f"code={getattr(response, 'code', None)}, "
                f"message={getattr(response, 'message', None)}"
            )

        content = response.output.choices[0].message.content

        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
            )
        else:
            text = content

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").removesuffix("```").strip()
        elif text.startswith("```"):
            text = text.removeprefix("```").removesuffix("```").strip()

        result = json.loads(text)

        valid_ids = {
            c["node_id"]
            for c in compact_candidates
            if c.get("node_id")
        }

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
                node_id: str(
                    reasoning.get(node_id, "Selected by reranker.")
                ).strip()
                for node_id in selected
            },
        }

        return result["top_k_node_ids"], result