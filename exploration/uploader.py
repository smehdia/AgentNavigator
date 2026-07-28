#!/usr/bin/env python3
"""Upload explored app artifacts to the Hugging Face dataset repo.

For each app directory under explored_apps/:
  1. If the app is not on the Hub (under android/<app> or harmony/<app>),
     upload the entire local directory.
  2. If the app is already on the Hub, upload any missing post-process JSON
     files (edge/node level info, path/user intents, navigation plans).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for var in [
    "http_proxy",
    "https_proxy",
    "ftp_proxy",
    "socks_proxy",
    "socks5_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "FTP_PROXY",
    "SOCKS_PROXY",
    "SOCKS5_PROXY",
    "ALL_PROXY",
    "all_proxy",
]:
    os.environ.pop(var, None)

from huggingface_hub import HfApi

REPO_ID = "smehdia/app_navigation"
REPO_TYPE = "dataset"
AGENT_DATA_FILE = "agent_data.json"
GRAPH_FILE = "graph.json"
META_INFO_FILE = "meta_info.json"
PLATFORMS = ("android", "harmony")
POST_PROCESS_JSON_FILES = (
    "edge_level_information.json",
    "node_level_information.json",
    "path_intents.json",
    "user_intents.json",
    "node_navigation_plans.json",
)


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def explored_apps_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return script_dir() / "explored_apps"


def list_local_apps(root: Path, only: list[str] | None = None) -> list[str]:
    apps = sorted(
        p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    if only:
        unknown = sorted(set(only) - set(apps))
        if unknown:
            raise ValueError(f"unknown local app(s): {', '.join(unknown)}")
        apps = [name for name in apps if name in set(only)]
    return apps


def count_graph_nodes(app_dir: Path) -> int | None:
    graph_path = app_dir / GRAPH_FILE
    if not graph_path.is_file():
        return None
    with graph_path.open(encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError(f"{graph_path}: expected a 'nodes' list")
    return len(nodes)


def count_agent_data_nodes(app_dir: Path) -> int | None:
    agent_data_path = app_dir / AGENT_DATA_FILE
    if not agent_data_path.is_file():
        return None
    with agent_data_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{agent_data_path}: expected a JSON object")
    return len(data)


def print_agent_data_stats(root: Path, app_names: list[str]) -> int:
    """Print per-app agent_data coverage; return non-zero if any app is incomplete."""
    incomplete = 0

    for app_name in app_names:
        app_dir = root / app_name
        try:
            total_nodes = count_graph_nodes(app_dir)
            saved_nodes = count_agent_data_nodes(app_dir)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"{app_name}: error reading stats ({exc})")
            incomplete += 1
            continue

        if total_nodes is None:
            print(f"{app_name}: {GRAPH_FILE} missing")
            incomplete += 1
            continue

        if saved_nodes is None:
            print(f"{app_name}: 0 out of {total_nodes} are present in {AGENT_DATA_FILE}")
            incomplete += 1
            continue

        print(
            f"{app_name}: {saved_nodes} out of {total_nodes} "
            f"are present in {AGENT_DATA_FILE}"
        )
        if saved_nodes < total_nodes:
            incomplete += 1

    return 1 if incomplete else 0


def resolve_platform(app_dir: Path, app_name: str) -> str:
    """Return 'android' or 'harmony' for a local app directory."""
    meta_path = app_dir / META_INFO_FILE
    if meta_path.is_file():
        try:
            with meta_path.open(encoding="utf-8") as f:
                meta = json.load(f)
            os_name = (
                meta.get("configs", {})
                .get("driver", {})
                .get("os_name")
            )
            if isinstance(os_name, str) and os_name.lower() in PLATFORMS:
                return os_name.lower()
        except (json.JSONDecodeError, OSError):
            pass

    if app_name.endswith("_harmony"):
        return "harmony"
    return "android"


def list_hf_app_paths(api: HfApi, revision: str) -> dict[str, str]:
    """Map app folder name -> repo path prefix (e.g. android/amazon)."""
    mapping: dict[str, str] = {}
    for platform in PLATFORMS:
        try:
            items = api.list_repo_tree(
                REPO_ID,
                repo_type=REPO_TYPE,
                revision=revision,
                path_in_repo=platform,
                recursive=False,
            )
        except Exception as exc:
            # Platform folder may be missing on some revisions.
            print(f"warning: could not list {platform}/ on Hub ({exc})", file=sys.stderr)
            continue
        for item in items:
            app_name = item.path.rsplit("/", 1)[-1]
            mapping[app_name] = item.path
    return mapping


def remote_has_file(api: HfApi, repo_path: str, filename: str, revision: str) -> bool:
    return api.file_exists(
        REPO_ID,
        f"{repo_path}/{filename}",
        repo_type=REPO_TYPE,
        revision=revision,
    )


def upload_file(
    api: HfApi,
    local_path: Path,
    path_in_repo: str,
    revision: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would upload {local_path} -> {path_in_repo}")
        return

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        commit_message=f"Add {path_in_repo}",
    )
    print(f"uploaded {local_path} -> {path_in_repo}")


def upload_app_directory(
    api: HfApi,
    local_dir: Path,
    repo_path: str,
    revision: str,
    dry_run: bool,
) -> None:
    if dry_run:
        print(f"[dry-run] would upload folder {local_dir} -> {repo_path}/")
        return

    api.upload_folder(
        folder_path=str(local_dir),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        commit_message=f"Add {repo_path}/",
    )
    print(f"uploaded folder {local_dir} -> {repo_path}/")


def upload_missing_post_process_json(
    api: HfApi,
    app_dir: Path,
    repo_path: str,
    revision: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Upload missing post-process JSON files. Returns (uploaded, skipped)."""
    uploaded = 0
    skipped = 0

    for filename in POST_PROCESS_JSON_FILES:
        local_path = app_dir / filename
        if not local_path.is_file():
            print(f"skip {repo_path}/{filename}: local file missing")
            skipped += 1
            continue

        if remote_has_file(api, repo_path, filename, revision):
            print(f"skip {repo_path}/{filename}: already on Hub")
            skipped += 1
            continue

        upload_file(
            api,
            local_path,
            f"{repo_path}/{filename}",
            revision,
            dry_run,
        )
        uploaded += 1

    return uploaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upload explored apps to "
            f"https://huggingface.co/datasets/{REPO_ID}: full directory if missing, "
            "otherwise any missing post-process JSON files."
        )
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Explored apps root (default: exploration/explored_apps).",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub git revision to read/write (default: main).",
    )
    parser.add_argument(
        "--app",
        action="append",
        default=[],
        metavar="NAME",
        help="Only process this app folder (repeatable).",
    )
    parser.add_argument(
        "--get-stat",
        action="store_true",
        help=(
            "Print how many graph nodes have entries in agent_data.json "
            "(reads graph.json and agent_data.json locally; no upload)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without uploading.",
    )
    args = parser.parse_args()

    root = explored_apps_root(args.root)
    if not root.is_dir():
        print(f"error: explored apps root not found: {root}", file=sys.stderr)
        return 1

    try:
        local_apps = list_local_apps(root, args.app or None)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.get_stat:
        return print_agent_data_stats(root, local_apps)

    api = HfApi()
    hf_apps = list_hf_app_paths(api, args.revision)

    uploaded = 0
    skipped = 0

    for app_name in local_apps:
        app_dir = root / app_name
        repo_path = hf_apps.get(app_name)

        if repo_path is None:
            platform = resolve_platform(app_dir, app_name)
            repo_path = f"{platform}/{app_name}"
            print(f"{app_name}: not on Hub; uploading directory -> {repo_path}/")
            upload_app_directory(api, app_dir, repo_path, args.revision, args.dry_run)
            uploaded += 1
            continue

        print(f"{app_name}: on Hub at {repo_path}; checking post-process JSON files")
        file_uploaded, file_skipped = upload_missing_post_process_json(
            api,
            app_dir,
            repo_path,
            args.revision,
            args.dry_run,
        )
        uploaded += file_uploaded
        skipped += file_skipped

    action = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone: {action} {uploaded}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
