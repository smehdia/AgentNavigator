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
        top_k = min(top_k, len(candidates))

        system_msg = {
            "role": "system",
            "content": [
                {
                    "text": (
                        "You are a strict reranking module for Android UI navigation.\n"
                        "Your task is to select the best target nodes for a user's navigation query.\n\n"
                        "Each candidate may contain:\n"
                        "- node_id: unique node identifier\n"
                        "- page_purpose: what the page/screen is mainly for\n"
                        "- depth: navigation depth from root; lower means easier to reach\n"
                        "- score: embedding similarity score between the query and the node\n"
                        "- user_intents: possible user goals supported by this node\n"
                        "- ui_navigation_memory: waypoint and transition hints\n\n"
                        "Ranking rules:\n"
                        "1. Prefer candidates whose page_purpose directly satisfies the user query.\n"
                        "2. Match the specificity of the user query:\n"
                        "   - If the query is broad or generic, prefer the broad parent page.\n"
                        "   - If the query is specific, prefer the most specific page that directly satisfies it.\n"
                        "3. Do not select a more specific sub-setting page unless the query explicitly mentions that specific setting.\n"
                        "4. Then prefer candidates whose user_intents directly match the query.\n"
                        "5. Relevance and specificity are more important than depth or embedding score.\n"
                        "6. Use depth only as a tie-breaker when relevance and specificity are similar.\n"
                        "7. Penalize menus, dialogs, popups, temporary overlays, and intermediate screens unless the query explicitly asks for them.\n"
                        "8. Penalize candidates that are only indirectly related, even if their embedding score is high.\n"
                        "9. Only select node_ids from the provided candidates.\n\n"
                        "Examples:\n"
                        "- For 'go to settings', prefer the main Settings page over Alarm settings, Timer settings, World clock style, or Date/time settings.\n"
                        "- For 'change world clock style', prefer the specific World clock style/settings page over the generic Settings page.\n"
                        "- For 'open alarm settings', prefer Alarm settings over the generic Settings page.\n\n"
                        "You must output valid JSON only."
                    )
                }
            ],
        }

        compact_candidates = []
        for c in candidates:
            compact_candidates.append(
                {
                    "node_id": c.get("node_id"),
                    "page_purpose": c.get("page_purpose", ""),
                    "depth": c.get("depth"),
                    "score": c.get("score"),
                    "user_intents": c.get("user_intents", []),
                    "ui_navigation_memory": c.get("ui_navigation_memory", []),
                }
            )

        prompt = (
            "User query:\n"
            f"{user_query}\n\n"
            "Specificity guidance:\n"
            "- Broad query means the user asks for a general destination, such as settings, alarms, timer, stopwatch, profile, or help.\n"
            "- Specific query means the user asks for a particular option, preference, subpage, configuration, or action.\n"
            "- For a broad query, select the broadest directly matching page.\n"
            "- For a specific query, select the most specific page that directly satisfies the requested goal.\n\n"
            f"Select exactly {top_k} best candidate nodes.\n\n"
            "Candidate nodes:\n"
            f"{json.dumps(compact_candidates, indent=2, ensure_ascii=False)}\n\n"
            "Return JSON in exactly this format:\n"
            "{\n"
            '  "top_k_node_ids": ["node_id_1", "node_id_2"],\n'
            '  "reasoning": {\n'
            '    "node_id_1": "brief reason",\n'
            '    "node_id_2": "brief reason"\n'
            "  }\n"
            "}\n\n"
            f"The list top_k_node_ids must contain exactly {top_k} node_ids.\n"
            "Only use node_ids that appear in the provided candidates.\n"
            "Do not include markdown or extra text."
        )

        user_msg = {
            "role": "user",
            "content": [
                {"text": prompt},
            ],
        }

        messages = [system_msg, user_msg]
        self.prompt = messages

        resp = MultiModalConversation.call(
            model="qwen3.6-plus",
            messages=messages,
            response_format={"type": "json_object"},
            enable_thinking=False,
            temperature=0.0,
            top_p=1.0,
            seed=42,
        )

        text = resp.output.choices[0].message.content[0]["text"]
        result = json.loads(text)

        return result["top_k_node_ids"], result