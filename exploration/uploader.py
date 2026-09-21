#!/usr/bin/env python3
"""Upload explored app artifacts to the Hugging Face dataset repo.

For each app directory under explored_apps/:
  1. If the app is not on the Hub (under android/<app> or harmony/<app>),
     upload the entire local directory (unless --existing-only).
  2. If the app is already on the Hub, upload local post-process JSON,
     localizer artifacts, debug_paths/, and screenshots/. Matching remote
     files are replaced. Use --skip-screenshots to omit the heavy folder.
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

from huggingface_hub import CommitOperationDelete, HfApi

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
LOCALIZER_FILES = (
    "siglip_smolvlm_features.pt",
    "ood_classifier.joblib",
)
EXISTING_APP_FILES = POST_PROCESS_JSON_FILES + LOCALIZER_FILES
SCREENSHOTS_DIR = "screenshots"
DEBUG_PATHS_DIR = "debug_paths"


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


def remote_has_debug_paths(api: HfApi, repo_path: str, revision: str) -> bool:
    try:
        items = list(
            api.list_repo_tree(
                REPO_ID,
                repo_type=REPO_TYPE,
                revision=revision,
                path_in_repo=f"{repo_path}/{DEBUG_PATHS_DIR}",
                recursive=False,
            )
        )
    except Exception:
        return False
    return bool(items)


def delete_remote_debug_paths(
    api: HfApi,
    hf_apps: dict[str, str],
    revision: str,
    dry_run: bool,
    only: list[str] | None = None,
) -> int:
    """Delete debug_paths/ folders from Hub apps. Returns 0 on success."""
    targets = sorted(hf_apps.items())
    if only:
        unknown = sorted(set(only) - set(hf_apps))
        if unknown:
            print(f"error: unknown Hub app(s): {', '.join(unknown)}", file=sys.stderr)
            return 1
        only_set = set(only)
        targets = [(name, path) for name, path in targets if name in only_set]

    to_delete: list[str] = []
    for app_name, repo_path in targets:
        folder = f"{repo_path}/{DEBUG_PATHS_DIR}"
        if remote_has_debug_paths(api, repo_path, revision):
            print(f"{app_name}: will delete {folder}/")
            to_delete.append(folder)
        else:
            print(f"{app_name}: no {DEBUG_PATHS_DIR}/ on Hub")

    print(f"\n{len(to_delete)} debug_paths folder(s) to delete")
    if dry_run or not to_delete:
        return 0

    operations = [
        CommitOperationDelete(path_in_repo=folder, is_folder=True) for folder in to_delete
    ]
    # Hub commit size limits: delete in batches of folders.
    batch_size = 40
    for start in range(0, len(operations), batch_size):
        batch = operations[start : start + batch_size]
        first = to_delete[start]
        last = to_delete[min(start + batch_size, len(to_delete)) - 1]
        print(f"deleting batch {start + 1}-{start + len(batch)}: {first} .. {last}")
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            revision=revision,
            operations=batch,
            commit_message=f"Remove debug_paths folders ({start + 1}-{start + len(batch)})",
        )
    print("Done: deleted remote debug_paths folders")
    return 0


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
        commit_message=f"Upload {path_in_repo}",
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
        commit_message=f"Upload {repo_path}/",
    )
    print(f"uploaded folder {local_dir} -> {repo_path}/")


def upload_existing_app_files(
    api: HfApi,
    app_dir: Path,
    repo_path: str,
    revision: str,
    dry_run: bool,
    skip_screenshots: bool = False,
    debug_paths_only: bool = False,
) -> tuple[int, int]:
    """Upload localizer artifacts, post-process JSON, debug_paths/, and screenshots/.

    Local files overwrite the same path on the Hub. Returns (uploaded, skipped).
    """
    uploaded = 0
    skipped = 0
    allow_patterns: list[str] = []

    if debug_paths_only:
        skip_screenshots = True
    else:
        for filename in EXISTING_APP_FILES:
            local_path = app_dir / filename
            if not local_path.is_file():
                print(f"skip {repo_path}/{filename}: local file missing")
                skipped += 1
                continue
            allow_patterns.append(filename)

    debug_dir = app_dir / DEBUG_PATHS_DIR
    if not debug_dir.is_dir() or not any(debug_dir.iterdir()):
        print(f"skip {repo_path}/{DEBUG_PATHS_DIR}/: local folder missing or empty")
        skipped += 1
    else:
        allow_patterns.append(f"{DEBUG_PATHS_DIR}/**")

    if skip_screenshots:
        print(f"skip {repo_path}/{SCREENSHOTS_DIR}/: --skip-screenshots")
        skipped += 1
    else:
        screenshots_dir = app_dir / SCREENSHOTS_DIR
        if not screenshots_dir.is_dir() or not any(screenshots_dir.iterdir()):
            print(f"skip {repo_path}/{SCREENSHOTS_DIR}/: local folder missing or empty")
            skipped += 1
        else:
            allow_patterns.append(f"{SCREENSHOTS_DIR}/**")

    if not allow_patterns:
        return uploaded, skipped

    if dry_run:
        print(f"[dry-run] would upload {app_dir} -> {repo_path}/ ({', '.join(allow_patterns)})")
        return 1, skipped

    api.upload_folder(
        folder_path=str(app_dir),
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        allow_patterns=allow_patterns,
        commit_message=(
            f"Update {repo_path}/ debug_paths"
            if debug_paths_only
            else f"Update {repo_path}/ (JSON, localizer, debug_paths)"
        ),
    )
    print(f"uploaded {app_dir} -> {repo_path}/ ({', '.join(allow_patterns)})")
    return 1, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Upload explored apps to "
            f"https://huggingface.co/datasets/{REPO_ID}: full directory if missing, "
            "otherwise local post-process JSON, localizer artifacts, debug_paths, "
            "and screenshots (replacing matching Hub files)."
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
        "--existing-only",
        action="store_true",
        help="Only update apps already on the Hub; do not upload new app directories.",
    )
    parser.add_argument(
        "--skip-screenshots",
        action="store_true",
        help="When updating existing Hub apps, do not upload screenshots/.",
    )
    parser.add_argument(
        "--debug-paths-only",
        action="store_true",
        help="When updating existing Hub apps, upload only debug_paths/.",
    )
    parser.add_argument(
        "--delete-debug-paths",
        action="store_true",
        help="Delete debug_paths/ folders from Hub apps (no upload).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without uploading or deleting.",
    )
    args = parser.parse_args()

    api = HfApi()
    hf_apps = list_hf_app_paths(api, args.revision)

    if args.delete_debug_paths:
        return delete_remote_debug_paths(
            api,
            hf_apps,
            args.revision,
            args.dry_run,
            args.app or None,
        )

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

    uploaded = 0
    skipped = 0

    for app_name in local_apps:
        app_dir = root / app_name
        repo_path = hf_apps.get(app_name)

        if repo_path is None:
            if args.existing_only:
                print(f"{app_name}: not on Hub; skipping (--existing-only)")
                skipped += 1
                continue
            platform = resolve_platform(app_dir, app_name)
            repo_path = f"{platform}/{app_name}"
            print(f"{app_name}: not on Hub; uploading directory -> {repo_path}/")
            upload_app_directory(api, app_dir, repo_path, args.revision, args.dry_run)
            uploaded += 1
            continue

        print(
            f"{app_name}: on Hub at {repo_path}; "
            "uploading local files (replace if already present)"
        )
        try:
            file_uploaded, file_skipped = upload_existing_app_files(
                api,
                app_dir,
                repo_path,
                args.revision,
                args.dry_run,
                skip_screenshots=args.skip_screenshots,
                debug_paths_only=args.debug_paths_only,
            )
        except Exception as exc:
            print(f"{app_name}: FAILED ({exc})", file=sys.stderr)
            skipped += 1
            continue
        uploaded += file_uploaded
        skipped += file_skipped

    action = "would upload" if args.dry_run else "uploaded"
    print(f"\nDone: {action} {uploaded}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
