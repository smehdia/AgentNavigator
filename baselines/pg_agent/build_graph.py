# NOTE: reference file from the PG-Agent graph-build + serving pipeline.
# Requires the upstream PG-Agent checkout ($PG_AGENT_ROOT) and a served VLM
# (we used Qwen2.5-VL-72B). See baselines/pg_agent/README.md before running.
#!/usr/bin/env python3
"""Parameterized PG-Agent Odyssey page-graph construction.

Upstream PG-Agent's `document_construction/odyssey_document/main.py` is a
single hardcoded script.  This wrapper keeps the same construction algorithm
shape while making the dataset root, sampling policy, image URL base, model
endpoint, and output path explicit.

It expects a dataset produced by `scripts/pg_agent_export_odyssey_dataset.py`.
Run it on the GPU host that serves Qwen2.5-VL-72B-Instruct and has bge-m3
embeddings available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import base64
import mimetypes
import os
import random
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.parse import urlparse

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.env_utils import configure_huggingface_cache_defaults, load_first_env_file


DEFAULT_PG_AGENT_ROOT = Path("PG-Agent")
DEFAULT_DATASET_ROOT = Path("pg_agent_odyssey_dataset")


def resolve_embedding_model_name(model_name: str) -> str:
    if model_name == "bge-m3" and not Path(model_name).exists():
        return "BAAI/bge-m3"
    return model_name


@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict[str, Any]
    id: str | None = None


class SentenceTransformerEmbeddings:
    """Small local substitute for LangChain HuggingFaceEmbeddings.

    PG-Agent uses LangChain's FAISS wrapper over HuggingFace embeddings.  This
    wrapper keeps the same inputs while avoiding a hard dependency on LangChain
    in adapter-only deployments.
    """

    def __init__(self, model_name: str, device: str):
        configure_huggingface_cache_defaults()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(resolve_embedding_model_name(model_name), device=device)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text], convert_to_numpy=True)[0], dtype=np.float32)


def load_upstream_prompts(pg_agent_root: Path) -> Any:
    prompts_path = pg_agent_root / "document_construction/odyssey_document/prompts.py"
    if not prompts_path.exists():
        raise FileNotFoundError(f"PG-Agent Odyssey prompts not found: {prompts_path}")
    spec = importlib.util.spec_from_file_location("pg_agent_odyssey_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import PG-Agent prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authorization_headers(api_key_env: str | None = None) -> dict[str, str]:
    if not api_key_env:
        return {}
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def is_loopback_image_url(image_url: str) -> bool:
    parsed = urlparse(image_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def inline_loopback_image_url(image_url: str, timeout: float = 30.0) -> str:
    if image_url.startswith("data:") or not is_loopback_image_url(image_url):
        return image_url
    response = requests.get(image_url, timeout=timeout)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip()
    if not content_type or content_type == "application/octet-stream":
        content_type = mimetypes.guess_type(urlparse(image_url).path)[0] or "image/png"
    encoded = base64.b64encode(response.content).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def image_reference_for_payload(image_url: str) -> str:
    if image_url.startswith("data:"):
        return image_url
    return quote(image_url, safe="/:")


class OpenAICompatibleVisionClient:
    def __init__(self, endpoint: str, model: str, timeout: float = 300.0, api_key_env: str | None = None):
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.api_key_env = api_key_env

    def chat(self, image_urls: Iterable[str], query: str) -> str:
        content: list[dict[str, Any]] = []
        for image_url in image_urls:
            inlined = inline_loopback_image_url(image_url)
            content.append({"type": "image_url", "image_url": {"url": image_reference_for_payload(inlined)}})
        content.append({"type": "text", "text": query})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            "temperature": 0,
        }
        headers = {"Content-Type": "application/json"}
        headers.update(authorization_headers(self.api_key_env))
        response = requests.post(
            self.endpoint,
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]


def image_url(base_url: str, screenshot_rel: str) -> str:
    return f"{base_url.rstrip('/')}/{screenshot_rel}"


def action_summary(
    step: dict[str, Any],
    *,
    dataset_root: Path,
    img_url: str,
    prompts: Any,
    client: OpenAICompatibleVisionClient,
) -> str:
    action = step["action"]
    info = step.get("info")
    if action not in {"CLICK", "TEXT", "SCROLL", "LONG_PRESS"}:
        raise ValueError(f"unsupported graph-construction action: {action}")

    if action in {"CLICK", "LONG_PRESS"}:
        if info == "KEY_HOME":
            return "press home to go to the home screen"
        if info == "KEY_BACK":
            return "press back to go to the previous screen"
        if info == "KEY_APPSELECT":
            return "go to the previous App"
        if isinstance(info, list):
            screenshot_path = dataset_root / "data/screenshots" / step["screenshot"]
            with Image.open(screenshot_path) as img:
                width, height = img.size
            point = info[0]
            bbox = f"[{int(point[0] / 1000 * width)}, {int(point[1] / 1000 * height)}]"
            summary = client.chat([img_url], prompts.click_action_summary.format(bbox=bbox))
            return summary[:-1] if summary.endswith(".") else summary
        raise ValueError(f"unknown click action info: {info}")

    if action == "SCROLL":
        start = np.array(info[0])
        end = np.array(info[1])
        delta = end - start
        delta_abs = np.abs(delta)
        left_right = "left" if delta[0] < 0 else "right"
        up_down = "up" if delta[1] < 0 else "down"
        return f"scroll {left_right if delta_abs[0] > delta_abs[1] else up_down}"

    return f"type {info}"


def parse_index_response(text: str) -> int | None:
    match = re.search(r"###\s*Index:\s*([^\n]+)", text)
    if match is None:
        raise ValueError(f"repeat-check response missing '### Index:': {text[:200]}")
    raw = match.group(1).strip().strip("<>").strip("'\"")
    if raw == "None":
        return None
    return int(raw) - 1


def parse_yes_no_response(text: str) -> bool:
    match = re.search(r"###\s*Conclusion:\s*([^\n]+)", text)
    if match is None:
        raise ValueError(f"yes/no response missing '### Conclusion:': {text[:200]}")
    raw = match.group(1).strip().strip("<>").strip("'\"")
    if raw not in {"Yes", "No"}:
        raise ValueError(f"expected Yes/No conclusion, got: {raw}")
    return raw == "Yes"


def make_embedding_model(model_name: str, device: str) -> Any:
    configure_huggingface_cache_defaults()
    resolved_model_name = resolve_embedding_model_name(model_name)
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=resolved_model_name, model_kwargs={"device": device})
    except ImportError:
        return SentenceTransformerEmbeddings(resolved_model_name, device)


def make_document(page_content: str, metadata: dict[str, Any]) -> Any:
    try:
        from langchain.schema import Document

        return Document(page_content=page_content, metadata=metadata)
    except ImportError:
        doc_id = metadata.get("index")
        return SimpleDocument(
            page_content=page_content,
            metadata=metadata,
            id=None if doc_id is None else str(doc_id),
        )


def _embedding_model_l2_search(documents: list[Any], query: str, embedding_model: Any, k: int) -> list[Any]:
    if not documents:
        return []
    page_contents = [document.page_content for document in documents]
    doc_vectors = np.asarray(embedding_model.embed_documents(page_contents), dtype=np.float32)
    query_vector = np.asarray(embedding_model.embed_query(query), dtype=np.float32)
    distances = np.sum((doc_vectors - query_vector) ** 2, axis=1)
    top_indices = np.argsort(distances)[:k]
    return [documents[int(idx)] for idx in top_indices]


def similarity_search(documents: list[Any], query: str, embedding_model: Any, k: int = 4) -> list[Any]:
    try:
        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.from_documents(documents, embedding_model)
        return vectorstore.similarity_search(query, k=k)
    except ImportError:
        return _embedding_model_l2_search(documents, query, embedding_model, k)


def check_repeat_item(
    domain: str,
    img_url: str,
    page_summary: str,
    search_document: dict[str, list[Any]],
    embedding_model: Any,
    prompts: Any,
    client: OpenAICompatibleVisionClient,
) -> tuple[str | None, int | None]:
    if not search_document[domain]:
        return None, None

    search_res = similarity_search(search_document[domain], page_summary, embedding_model)
    old_description = ""
    for idx, res in enumerate(search_res):
        old_description += f"{idx + 1}. {res.page_content}\n"

    check_repeat_prompt = prompts.check_repeat.format(old_description=old_description)
    sample_index = parse_index_response(client.chat([img_url], check_repeat_prompt))
    if sample_index is None:
        return None, None

    old_img_url = search_res[sample_index].metadata["img_url"]
    if not parse_yes_no_response(client.chat([old_img_url, img_url], prompts.check_repeat_2)):
        return None, None

    repeat_index = search_res[sample_index].metadata["index"]
    new_summary = search_res[sample_index].page_content
    return new_summary, repeat_index


def create_new_item(
    domain: str,
    img_url: str,
    screenshot_rel: str,
    knowledge_library: dict[str, dict[int, dict[str, Any]]],
    search_document: dict[str, list[Any]],
    embedding_model: Any,
    prompts: Any,
    client: OpenAICompatibleVisionClient,
) -> dict[str, Any]:
    page_summary = client.chat([img_url], prompts.page_summary)
    new_summary, repeat_index = check_repeat_item(
        domain, img_url, page_summary, search_document, embedding_model, prompts, client
    )
    if repeat_index is None:
        knowledge_item = {
            "index": len(knowledge_library[domain]),
            "page_summary": page_summary,
            "original_image": [],
            "next_page_list": [{"actions": [], "page_index": None}],
        }
        knowledge_library[domain][knowledge_item["index"]] = knowledge_item
        search_document[domain].append(
            make_document(page_summary, {"index": knowledge_item["index"], "img_url": img_url})
        )
    else:
        knowledge_library[domain][repeat_index]["page_summary"] = new_summary
        search_document[domain][repeat_index].page_content = new_summary
        knowledge_item = knowledge_library[domain][repeat_index]

    knowledge_item["original_image"].append(screenshot_rel)
    return knowledge_item


def get_item(
    domain: str,
    img_url: str,
    screenshot_rel: str,
    last_img_url: str | None,
    last_action_summary: str | None,
    last_page_idx: int | None,
    knowledge_library: dict[str, dict[int, dict[str, Any]]],
    search_document: dict[str, list[Any]],
    embedding_model: Any,
    prompts: Any,
    client: OpenAICompatibleVisionClient,
) -> tuple[dict[str, Any], bool]:
    if last_page_idx is None:
        return (
            create_new_item(
                domain,
                img_url,
                screenshot_rel,
                knowledge_library,
                search_document,
                embedding_model,
                prompts,
                client,
            ),
            True,
        )

    assert last_img_url is not None
    assert last_action_summary is not None
    redirection_prompt = prompts.redirection_judge.format(action=last_action_summary)
    if parse_yes_no_response(client.chat([last_img_url, img_url], redirection_prompt)):
        return (
            create_new_item(
                domain,
                img_url,
                screenshot_rel,
                knowledge_library,
                search_document,
                embedding_model,
                prompts,
                client,
            ),
            True,
        )

    knowledge_item = knowledge_library[domain][last_page_idx]
    knowledge_item["original_image"].append(screenshot_rel)
    return knowledge_item, False


def select_episode_ids(
    train_ids: list[str],
    *,
    sample_fraction: float,
    limit: int | None,
    seed: int,
) -> list[str]:
    if sample_fraction <= 0 or sample_fraction > 1:
        raise ValueError("--sample-fraction must be in (0, 1]")
    selected = list(train_ids)
    if sample_fraction < 1:
        count = max(1, int(len(selected) * sample_fraction))
        rng = random.Random(seed)
        selected = rng.sample(selected, count)
    if limit is not None:
        selected = selected[:limit]
    return selected


def filter_episode_ids_by_apps(train_ids: list[str], apps: list[str] | None) -> list[str]:
    if not apps:
        return list(train_ids)
    app_filter = set(apps)
    return [episode_id for episode_id in train_ids if episode_id.split("__", 1)[0] in app_filter]


def normalize_knowledge_library(raw: dict[str, Any]) -> dict[str, dict[int, dict[str, Any]]]:
    return {
        domain: {int(idx): item for idx, item in pages.items()}
        for domain, pages in raw.items()
    }


def search_document_from_library(
    knowledge_library: dict[str, dict[int, dict[str, Any]]],
    image_base_url: str,
) -> dict[str, list[Any]]:
    search_document: dict[str, list[Any]] = {}
    for domain, pages in knowledge_library.items():
        search_document[domain] = []
        for idx in sorted(pages):
            item = pages[idx]
            original_images = item.get("original_image") or []
            img_url = image_url(image_base_url, original_images[0]) if original_images else ""
            search_document[domain].append(
                make_document(item["page_summary"], {"index": item["index"], "img_url": img_url})
            )
    return search_document


def default_progress_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".progress.json")


def load_progress(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {"completed_episode_ids": []}
    return json.loads(progress_path.read_text())


def write_build_state(
    *,
    output_path: Path,
    progress_path: Path,
    knowledge_library: dict[str, dict[int, dict[str, Any]]],
    completed_episode_ids: list[str],
    selected_episode_ids: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(knowledge_library, ensure_ascii=False, indent=2))
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(
        json.dumps(
            {
                "schema": "agentnavigator_pg_agent_graph_build_progress_v1",
                "output": str(output_path),
                "episodes_selected": len(selected_episode_ids),
                "episodes_completed": len(completed_episode_ids),
                "completed_episode_ids": completed_episode_ids,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def plan_graph_build(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    splits_path = dataset_root / "data/splits/splits_random_split.json"
    splits = json.loads(splits_path.read_text())
    candidate_episode_ids = filter_episode_ids_by_apps(splits["train"], args.apps)
    selected_episode_ids = select_episode_ids(
        candidate_episode_ids,
        sample_fraction=args.sample_fraction,
        limit=args.limit,
        seed=args.seed,
    )
    app_filter = set(args.apps or [])
    domains: dict[str, int] = {}
    steps: dict[str, int] = {}
    skipped_by_filter = 0
    missing_screenshots: list[dict[str, str]] = []

    for episode_id in selected_episode_ids:
        episode = json.loads((dataset_root / "data/annotations" / episode_id).read_text())
        domain = episode["task_info"]["category"]
        if app_filter and domain not in app_filter:
            skipped_by_filter += 1
            continue
        domains[domain] = domains.get(domain, 0) + 1
        steps[domain] = steps.get(domain, 0) + len(episode["steps"])
        for step in episode["steps"]:
            screenshot_path = dataset_root / "data/screenshots" / step["screenshot"]
            if not screenshot_path.exists():
                missing_screenshots.append({"episode": episode_id, "screenshot": step["screenshot"]})

    return {
        "mode": "dry_run" if args.dry_run else "build",
        "output": str(args.output),
        "dataset_root": str(dataset_root),
        "splits_path": str(splits_path),
        "episodes_available": len(splits["train"]),
        "episodes_after_app_filter": len(candidate_episode_ids),
        "episodes_selected": len(selected_episode_ids),
        "episodes_after_filter": sum(domains.values()),
        "skipped_by_filter": skipped_by_filter,
        "domains": domains,
        "steps": steps,
        "missing_screenshots": missing_screenshots,
        "sample_fraction": args.sample_fraction,
        "limit": args.limit,
        "seed": args.seed,
        "apps": args.apps or [],
        "image_base_url": args.image_base_url,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
    }


def build_graph(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    prompts = load_upstream_prompts(Path(args.pg_agent_root))
    client = OpenAICompatibleVisionClient(
        args.endpoint,
        args.model,
        timeout=args.timeout,
        api_key_env=args.api_key_env,
    )
    embedding_model = make_embedding_model(args.embedding_model, args.embedding_device)
    output_path = Path(args.output)
    progress_path = args.progress_out or default_progress_path(output_path)

    splits_path = dataset_root / "data/splits/splits_random_split.json"
    splits = json.loads(splits_path.read_text())
    candidate_episode_ids = filter_episode_ids_by_apps(splits["train"], args.apps)
    selected_episode_ids = select_episode_ids(
        candidate_episode_ids,
        sample_fraction=args.sample_fraction,
        limit=args.limit,
        seed=args.seed,
    )

    app_filter = set(args.apps or [])
    if args.resume and output_path.exists():
        knowledge_library = normalize_knowledge_library(json.loads(output_path.read_text()))
        search_document = search_document_from_library(knowledge_library, args.image_base_url)
        progress = load_progress(progress_path)
        completed_episode_ids = list(progress.get("completed_episode_ids") or [])
    else:
        knowledge_library: dict[str, dict[int, dict[str, Any]]] = {}
        search_document: dict[str, list[Any]] = {}
        completed_episode_ids: list[str] = []
    completed_episode_set = set(completed_episode_ids)
    episodes_attempted = 0

    for episode_id in tqdm(selected_episode_ids, desc="PG-Agent graph episodes"):
        if episode_id in completed_episode_set:
            continue
        episodes_attempted += 1
        episode = json.loads((dataset_root / "data/annotations" / episode_id).read_text())
        domain = episode["task_info"]["category"]
        if app_filter and domain not in app_filter:
            continue
        knowledge_library.setdefault(domain, {})
        search_document.setdefault(domain, [])

        goal = episode["task_info"]["instruction"]
        steps = episode["steps"]
        last_page_idx: int | None = None
        last_img_url: str | None = None
        last_action_summary: str | None = None

        for idx, step in enumerate(steps):
            screenshot_rel = step["screenshot"]
            img_url = image_url(args.image_base_url, screenshot_rel)
            if last_page_idx is not None:
                last_action_summary = action_summary(
                    steps[idx - 1],
                    dataset_root=dataset_root,
                    img_url=last_img_url or "",
                    prompts=prompts,
                    client=client,
                )

            knowledge_item, redirection_flag = get_item(
                domain,
                img_url,
                screenshot_rel,
                last_img_url,
                last_action_summary,
                last_page_idx,
                knowledge_library,
                search_document,
                embedding_model,
                prompts,
                client,
            )

            if last_page_idx is not None:
                edge = knowledge_library[domain][last_page_idx]["next_page_list"][-1]
                edge["actions"].append(last_action_summary)
                edge["goal"] = goal
                if redirection_flag:
                    edge["page_index"] = knowledge_item["index"]
                    knowledge_library[domain][last_page_idx]["next_page_list"].append(
                        {"actions": [], "page_index": None}
                    )

            last_page_idx = knowledge_item["index"]
            last_img_url = img_url

        completed_episode_ids.append(episode_id)
        completed_episode_set.add(episode_id)
        write_build_state(
            output_path=output_path,
            progress_path=progress_path,
            knowledge_library=knowledge_library,
            completed_episode_ids=completed_episode_ids,
            selected_episode_ids=selected_episode_ids,
        )

    return {
        "output": str(args.output),
        "progress_out": str(progress_path),
        "dataset_root": str(dataset_root),
        "episodes_selected": len(selected_episode_ids),
        "episodes_completed": len(completed_episode_ids),
        "episodes_attempted": episodes_attempted,
        "resume": args.resume,
        "domains": {domain: len(pages) for domain, pages in knowledge_library.items()},
        "sample_fraction": args.sample_fraction,
        "limit": args.limit,
        "image_base_url": args.image_base_url,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "embedding_device": args.embedding_device,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--pg-agent-root", type=Path, default=DEFAULT_PG_AGENT_ROOT)
    parser.add_argument("--output", default="pg_agent_odyssey_dataset/odyssey_library.json")
    parser.add_argument("--image-base-url", default="http://localhost:6668")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model", default="Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--api-key-env", default="PG_AGENT_API_KEY")
    parser.add_argument("--env-file", type=Path, help="Optional .env file to load before reading --api-key-env")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--sample-fraction", type=float, default=1.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--apps", nargs="*", help="Optional app/category filter, e.g. clock_8.5")
    parser.add_argument("--dry-run", action="store_true", help="Validate selected episodes without model calls")
    parser.add_argument("--plan-out", type=Path, help="Optional JSON path for the dry-run/build summary")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing output/progress file")
    parser.add_argument("--progress-out", type=Path, help="Progress JSON path; defaults to <output>.progress.json")
    args = parser.parse_args()

    load_first_env_file(args.env_file, api_key_env=args.api_key_env, load_config=True)
    summary = plan_graph_build(args) if args.dry_run else build_graph(args)
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
