#!/usr/bin/env python3
"""Upload complete local explored apps that are missing from the Hub dataset.

Looks at apps under local explored_apps/. For each app that:
  1. Is complete locally (has the post-process JSON files), and
  2. Is not already present under android/ or harmony/ on
     https://huggingface.co/datasets/smehdia/app_navigation,

uploads the entire local app directory to the proper platform folder
(android/<app> or harmony/<app>), resolved from meta_info.json
(configs.driver.os_name) or the `_harmony` name suffix.
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
PLATFORMS = ("android", "harmony")
META_INFO_FILE = "meta_info.json"
POST_PROCESS_JSON_FILES = (
    "user_intents.json",
    "edge_level_information.json",
    "node_level_information.json",
    "path_intents.json",
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


def app_is_complete(app_dir: Path) -> tuple[bool, list[str]]:
    """Return (complete, missing_files) for post-process JSON coverage."""
    missing = [
        filename
        for filename in POST_PROCESS_JSON_FILES
        if not (app_dir / filename).is_file()
    ]
    return (not missing, missing)


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
            print(f"warning: could not list {platform}/ on Hub ({exc})", file=sys.stderr)
            continue
        for item in items:
            app_name = item.path.rsplit("/", 1)[-1]
            mapping[app_name] = item.path
    return mapping


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upload complete local explored apps that are missing from "
            f"https://huggingface.co/datasets/{REPO_ID} under android/ or harmony/."
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
        help="Only process this local app folder name (repeatable).",
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

    api = HfApi()
    hf_apps = list_hf_app_paths(api, args.revision)

    uploaded = 0
    skipped_on_hub = 0
    skipped_incomplete = 0

    for app_name in local_apps:
        app_dir = root / app_name

        if app_name in hf_apps:
            print(f"skip {app_name}: already on Hub at {hf_apps[app_name]}")
            skipped_on_hub += 1
            continue

        complete, missing = app_is_complete(app_dir)
        if not complete:
            print(
                f"skip {app_name}: incomplete "
                f"(missing {', '.join(missing)})"
            )
            skipped_incomplete += 1
            continue

        platform = resolve_platform(app_dir, app_name)
        repo_path = f"{platform}/{app_name}"
        print(f"{app_name}: complete and not on Hub; uploading -> {repo_path}/")
        upload_app_directory(api, app_dir, repo_path, args.revision, args.dry_run)
        uploaded += 1

    action = "would upload" if args.dry_run else "uploaded"
    print(
        f"\nDone: {action} {uploaded} app(s), "
        f"already on Hub {skipped_on_hub}, incomplete {skipped_incomplete}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
