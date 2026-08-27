#!/usr/bin/env python3
"""Upload missing post-process JSON files to the Hugging Face dataset.

Looks at apps already present under android/ and harmony/ in
https://huggingface.co/datasets/smehdia/app_navigation. For each Hub app that
also exists under local explored_apps/, uploads any of these files that are
missing on the Hub (when present locally):

  - user_intents.json
  - edge_level_information.json
  - node_level_information.json
  - path_intents.json
  - node_navigation_plans.json
"""

from __future__ import annotations

import argparse
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


def upload_missing_json_for_app(
    api: HfApi,
    app_name: str,
    app_dir: Path,
    repo_path: str,
    revision: str,
    dry_run: bool,
) -> tuple[int, int]:
    """Upload missing JSON files for one Hub app. Returns (uploaded, skipped)."""
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
            "For each app already on "
            f"https://huggingface.co/datasets/{REPO_ID}, upload missing "
            "post-process JSON files from local explored_apps/."
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
        help="Only process this Hub app folder name (repeatable).",
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

    api = HfApi()
    hf_apps = list_hf_app_paths(api, args.revision)
    if not hf_apps:
        print("error: no apps found on Hub under android/ or harmony/", file=sys.stderr)
        return 1

    only = set(args.app) if args.app else None
    if only:
        unknown = sorted(only - set(hf_apps))
        if unknown:
            print(
                f"error: app(s) not on Hub: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 1
        hf_apps = {name: path for name, path in hf_apps.items() if name in only}

    uploaded = 0
    skipped = 0
    no_local = 0

    for app_name in sorted(hf_apps):
        repo_path = hf_apps[app_name]
        app_dir = root / app_name
        if not app_dir.is_dir():
            print(f"skip {repo_path}: no local explored_apps/{app_name}")
            no_local += 1
            continue

        print(f"{app_name}: checking {repo_path}")
        file_uploaded, file_skipped = upload_missing_json_for_app(
            api,
            app_name,
            app_dir,
            repo_path,
            args.revision,
            args.dry_run,
        )
        uploaded += file_uploaded
        skipped += file_skipped

    action = "would upload" if args.dry_run else "uploaded"
    print(
        f"\nDone: {action} {uploaded}, skipped {skipped}, "
        f"no local copy {no_local}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
