"""Small .env loading helpers for experiment scripts.

The loader intentionally avoids overriding existing environment variables so a
manually exported secret always wins over a checked local env file.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_ENV_FILE_CANDIDATES = [
    Path(".env"),
    Path(".env"),
    Path("APP NAVIGATION/AgentNavigator/.env"),
    Path("GUI-Explorer/.env"),
]

DEFAULT_CONFIG_FILE_CANDIDATES = [
    Path("configs/config_global.yaml"),
    Path("agents/pg_agent_experiment/AgentNavigator/configs/config_global.yaml"),
    Path("agents/AgentNavigator/configs/config_global.yaml"),
    Path("AgentNavigator/configs/config_global.yaml"),
    Path("APP NAVIGATION/AgentNavigator/configs/config_global.yaml"),
]

DEFAULT_REMOTE_HF_CACHE_ROOT = Path("hf-cache")


def load_env_file(path: Path) -> bool:
    if not path.exists():
        return False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def load_config_global_api_key(
    api_key_env: str | None = "DASHSCOPE_API_KEY",
    candidates: list[Path] | None = None,
) -> Path | None:
    if api_key_env != "DASHSCOPE_API_KEY" or os.environ.get(api_key_env):
        return None
    try:
        import yaml
    except Exception:
        return None
    for candidate in candidates or DEFAULT_CONFIG_FILE_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            cfg = yaml.safe_load(candidate.read_text()) or {}
        except Exception:
            continue
        key = ((cfg.get("default") or {}).get("vlm") or {}).get("api_key")
        if not key:
            key = (cfg.get("vlm") or {}).get("api_key")
        if key:
            os.environ[api_key_env] = str(key)
            return candidate
    return None


def load_first_env_file(
    path: Path | None = None,
    *,
    api_key_env: str | None = None,
    load_config: bool = False,
) -> Path | None:
    candidates = [path] if path else DEFAULT_ENV_FILE_CANDIDATES
    loaded = None
    for candidate in candidates:
        if candidate and load_env_file(candidate):
            loaded = candidate
            break
    if load_config:
        load_config_global_api_key(api_key_env)
    return loaded


def configure_huggingface_cache_defaults(cache_root: Path | None = None) -> Path:
    root = cache_root or DEFAULT_REMOTE_HF_CACHE_ROOT
    os.environ.setdefault("HF_HOME", str(root))
    os.environ.setdefault("HF_HUB_CACHE", str(root / "hub"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(root))
    return root
