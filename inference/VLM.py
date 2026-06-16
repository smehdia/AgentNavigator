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
                        "Your task is to select the best TARGET PAGE nodes for a user's navigation query.\n\n"
                        "Each candidate may contain:\n"
                        "- node_id: unique node identifier\n"
                        "- page_tag: short page/screen label\n"
                        "- page_purpose: what the page/screen itself is mainly for\n"
                        "- screen_type: page type such as settings_list, list_page, modal_menu, dialog, overlay, detail_page\n"
                        "- selected_navigation: current selected section if available\n"
                        "- depth: navigation depth from root; lower means easier to reach\n"
                        "- score: embedding similarity score between the query and the node\n"
                        "- user_intents: possible user goals directly supported by this node\n\n"
                        "Ranking rules:\n"
                        "1. Select the node whose page itself directly satisfies the user query.\n"
                        "2. Prefer page_tag and page_purpose matches over path/menu/waypoint matches.\n"
                        "3. If the query asks for a settings page, prefer candidates with screen_type='settings_list' or page_tag/page_purpose containing that settings destination.\n"
                        "4. Penalize modal_menu, dialog, popup, overlay, and temporary menu screens unless the query explicitly asks to open a menu, dialog, popup, or overflow options.\n"
                        "5. Do not select a menu merely because it contains a menu item that can lead to the desired page. Select the destination page if it is available.\n"
                        "6. Match the specificity of the query: broad query -> broad parent page; specific query -> specific target page.\n"
                        "7. Relevance and target-page correctness are more important than depth or embedding score.\n"
                        "8. Use depth only as a tie-breaker when relevance is similar.\n"
                        "9. Only select node_ids from the provided candidates.\n\n"
                        "Examples:\n"
                        "- For 'navigate to notifications settings page', prefer 'Notifications Settings' with screen_type='settings_list' over 'Notifications Menu' with screen_type='modal_menu'.\n"
                        "- For 'open notifications overflow menu', prefer 'Notifications Menu'.\n"
                        "- For 'go to settings', prefer the main Settings page over specific sub-settings.\n"
                        "- For 'change caption style', prefer the Caption preference/settings page over generic Settings.\n\n"
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
                    "page_tag": c.get("page_tag", ""),
                    "page_purpose": c.get("page_purpose", ""),
                    "screen_type": c.get("screen_type", ""),
                    "selected_navigation": c.get("selected_navigation", ""),
                    "depth": c.get("depth"),
                    "score": c.get("score"),
                    "user_intents": c.get("user_intents", []),
                }
            )
        prompt = (
            "User query:\n"
            f"{user_query}\n\n"
            "Target-page guidance:\n"
            "- Select the page that directly satisfies the user goal.\n"
            "- Do not select an intermediate menu, overflow menu, dialog, or overlay if the actual destination page is available.\n"
            "- For a query asking for a settings page, prefer an actual settings page/list over a menu containing a Settings item.\n"
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