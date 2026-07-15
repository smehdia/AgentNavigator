#!/usr/bin/env python3
"""Download explored Android app artifacts from Hugging Face.

Lists applications under ``android/`` in the ``smehdia/app_navigation`` dataset
and downloads any files that are not already present locally under
``explored_apps/<app>/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Hugging Face uses requests, which picks up system proxy env vars and can fail.
for _proxy_var in (
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
):
    os.environ.pop(_proxy_var, None)

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree
from huggingface_hub.hf_api import RepoFile

REPO_ID = "smehdia/app_navigation"
HF_ANDROID_PREFIX = "android/"
DEFAULT_LOCAL_ROOT = Path(__file__).resolve().parent / "explored_apps"


def list_android_apps(api: HfApi, revision: str) -> list[str]:
    entries = list(
        list_repo_tree(
            REPO_ID,
            repo_type="dataset",
            revision=revision,
            path_in_repo="android",
            recursive=False,
        )
    )
    apps: list[str] = []
    for entry in entries:
        name = Path(entry.path).name
        if name and name != "android":
            apps.append(name)
    return sorted(apps)


def list_local_apps(local_root: Path) -> list[str]:
    if not local_root.is_dir():
        return []
    return sorted(
        p.name for p in local_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def list_remote_files(
    revision: str,
    apps: list[str] | None = None,
    json_only: bool = False,
) -> list[str]:
    if apps:
        paths: list[str] = []
        for app in apps:
            prefix = f"{HF_ANDROID_PREFIX}{app}"
            for entry in list_repo_tree(
                REPO_ID,
                repo_type="dataset",
                revision=revision,
                path_in_repo=prefix,
                recursive=True,
            ):
                if isinstance(entry, RepoFile):
                    paths.append(entry.path)
        return _filter_remote_files(sorted(paths), json_only=json_only)

    paths = []
    for entry in list_repo_tree(
        REPO_ID,
        repo_type="dataset",
        revision=revision,
        path_in_repo="android",
        recursive=True,
    ):
        if isinstance(entry, RepoFile):
            paths.append(entry.path)
    return _filter_remote_files(sorted(paths), json_only=json_only)


def _filter_remote_files(paths: list[str], json_only: bool) -> list[str]:
    if not json_only:
        return paths
    return [path for path in paths if path.lower().endswith(".json")]


def remote_to_local_path(remote_path: str, local_root: Path) -> Path:
    if not remote_path.startswith(HF_ANDROID_PREFIX):
        raise ValueError(f"Unexpected remote path: {remote_path}")
    return local_root / remote_path[len(HF_ANDROID_PREFIX) :]


def download_missing_files(
    local_root: Path,
    revision: str,
    apps: list[str] | None = None,
    token: str | None = None,
    dry_run: bool = False,
    json_only: bool = False,
    force: bool = False,
) -> tuple[int, int, int]:
    remote_files = list_remote_files(
        revision=revision, apps=apps, json_only=json_only
    )
    downloaded = 0
    skipped = 0
    failed = 0

    for remote_path in remote_files:
        local_path = remote_to_local_path(remote_path, local_root)
        exists = local_path.exists()
        if exists and not force:
            skipped += 1
            continue

        if dry_run:
            action = "overwrite" if exists else "download"
            print(f"[dry-run] would {action} {remote_path} -> {local_path}")
            downloaded += 1
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            cached_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                revision=revision,
                filename=remote_path,
                token=token,
                force_download=force,
            )
            shutil.copy2(cached_path, local_path)
            downloaded += 1
            action = "overwrote" if exists else "downloaded"
            print(f"{action} {remote_path} -> {local_path}")
        except Exception as exc:
            failed += 1
            print(f"failed {remote_path}: {exc}", file=sys.stderr)

    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download Android explored-app artifacts from Hugging Face "
            f"({REPO_ID}) into explored_apps/. By default skips files that already exist."
        )
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=DEFAULT_LOCAL_ROOT,
        help=f"Local destination root (default: {DEFAULT_LOCAL_ROOT})",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset git revision/branch (default: main)",
    )
    parser.add_argument(
        "--apps",
        nargs="+",
        metavar="APP",
        help="Only sync specific app folder names (e.g. clock youtube)",
    )
    parser.add_argument(
        "--list-apps",
        action="store_true",
        help="List available Android app folders on Hugging Face and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print files that would be downloaded without downloading",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only download .json files",
    )
    parser.add_argument(
        "--local-apps-only",
        action="store_true",
        help="Only sync apps that already exist under --local-root",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite local files if they already exist",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    if args.list_apps:
        for app in list_android_apps(api, args.revision):
            print(app)
        return 0

    args.local_root.mkdir(parents=True, exist_ok=True)

    apps = args.apps
    if args.local_apps_only:
        local_apps = list_local_apps(args.local_root.resolve())
        if apps:
            apps = sorted(set(apps) & set(local_apps))
        else:
            apps = local_apps
        if not apps:
            print("No local app folders found to sync.", file=sys.stderr)
            return 1
        print("Syncing local apps:", ", ".join(apps))

    if apps:
        available = set(list_android_apps(api, args.revision))
        unknown = sorted(set(apps) - available)
        if unknown:
            print(
                "Skipping app(s) not on Hugging Face android/: "
                + ", ".join(unknown),
                file=sys.stderr,
            )
            apps = sorted(set(apps) & available)
            if not apps:
                print("No matching apps remain on Hugging Face.", file=sys.stderr)
                return 1

    downloaded, skipped, failed = download_missing_files(
        local_root=args.local_root.resolve(),
        revision=args.revision,
        apps=apps,
        token=token,
        dry_run=args.dry_run,
        json_only=args.json_only,
        force=args.force,
    )

    if args.dry_run:
        action = "would sync"
    elif args.force:
        action = "synced"
    else:
        action = "downloaded"
    print(
        f"Done: {downloaded} {action}, {skipped} skipped, {failed} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
