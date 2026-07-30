#!/usr/bin/env python3
"""Download explored app artifacts from the Hugging Face dataset repo.

For each app on the Hub under android/<app> or harmony/<app>:
  1. If the local app is missing or incomplete (e.g. no screenshots/),
     download any missing files from the Hub (resumable).
  2. If the app looks complete locally, skip it.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Hugging Face uses requests/httpx, which pick up system proxy env vars and can fail.
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

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

REPO_ID = "smehdia/app_navigation"
REPO_TYPE = "dataset"
PLATFORMS = ("android", "harmony")
DEFAULT_MAX_WORKERS = 2
DEFAULT_RETRIES = 5
DEFAULT_RETRY_WAIT = 5


def script_dir() -> Path:
    return Path(__file__).resolve().parent


def explored_apps_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return script_dir() / "explored_apps"


def list_local_apps(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {
        p.name
        for p in root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    }


def local_app_complete(local_dir: Path) -> bool:
    """True if the app folder looks usable (has screenshots/)."""
    shots = local_dir / "screenshots"
    if not shots.is_dir():
        return False
    return any(shots.iterdir())


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


def list_remote_app_files(
    api: HfApi,
    repo_path: str,
    revision: str,
    skip_debug: bool,
) -> list[str]:
    """List every file under an app folder (paginated tree listing)."""
    paths: list[str] = []
    for entry in api.list_repo_tree(
        REPO_ID,
        repo_type=REPO_TYPE,
        revision=revision,
        path_in_repo=repo_path,
        recursive=True,
    ):
        if not isinstance(entry, RepoFile):
            continue
        if skip_debug and "/debug_paths/" in entry.path:
            continue
        paths.append(entry.path)
    return sorted(paths)


def remote_to_local(remote_path: str, repo_path: str, local_dir: Path) -> Path:
    prefix = repo_path.rstrip("/") + "/"
    if not remote_path.startswith(prefix):
        raise ValueError(f"unexpected remote path {remote_path!r} for {repo_path}")
    return local_dir / remote_path[len(prefix) :]


def download_one_file(
    remote_path: str,
    local_path: Path,
    revision: str,
    token: str | None,
    dry_run: bool,
    retries: int,
    retry_wait: float,
) -> str:
    """Download a single remote file. Returns 'download'|'skip'|'dry-run'."""
    if local_path.is_file():
        return "skip"

    if dry_run:
        print(f"[dry-run] would download {remote_path} -> {local_path}")
        return "dry-run"

    local_path.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                revision=revision,
                filename=remote_path,
                token=token,
            )
            shutil.copy2(cached, local_path)
            return "download"
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(retry_wait * attempt)
    assert last_exc is not None
    raise last_exc


def download_app_directory(
    api: HfApi,
    local_dir: Path,
    repo_path: str,
    revision: str,
    dry_run: bool,
    token: str | None,
    max_workers: int,
    retries: int,
    retry_wait: float,
    skip_debug: bool,
) -> tuple[int, int]:
    """Download missing files for one app. Returns (downloaded, skipped)."""
    remote_files = list_remote_app_files(api, repo_path, revision, skip_debug)
    if not remote_files:
        raise FileNotFoundError(f"no files listed under {repo_path}/ on Hub")

    missing = [
        path
        for path in remote_files
        if not remote_to_local(path, repo_path, local_dir).is_file()
    ]
    print(
        f"{repo_path}: {len(remote_files)} remote files, "
        f"{len(missing)} missing locally"
    )
    if dry_run:
        for path in missing:
            print(f"[dry-run] would download {path}")
        return len(missing), len(remote_files) - len(missing)

    if not missing:
        return 0, len(remote_files)

    downloaded = 0
    skipped = len(remote_files) - len(missing)
    errors: list[str] = []

    def _worker(remote_path: str) -> tuple[str, str]:
        local_path = remote_to_local(remote_path, repo_path, local_dir)
        status = download_one_file(
            remote_path,
            local_path,
            revision,
            token,
            dry_run=False,
            retries=retries,
            retry_wait=retry_wait,
        )
        return remote_path, status

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_worker, path): path for path in missing}
        done = 0
        for fut in as_completed(futures):
            remote_path = futures[fut]
            done += 1
            try:
                _, status = fut.result()
                if status == "download":
                    downloaded += 1
                else:
                    skipped += 1
                if done % 25 == 0 or done == len(missing):
                    print(f"  progress {done}/{len(missing)} ({downloaded} new)")
            except Exception as exc:
                errors.append(f"{remote_path}: {exc}")
                print(f"  failed {remote_path}: {exc}", file=sys.stderr)

    if errors:
        raise RuntimeError(
            f"{len(errors)} file(s) failed under {repo_path}/ "
            f"(re-run to resume; e.g. {errors[0]})"
        )

    print(f"downloaded {repo_path}/ -> {local_dir}")
    return downloaded, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download explored apps from "
            f"https://huggingface.co/datasets/{REPO_ID}: missing files when "
            "the local app folder is absent or incomplete."
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
        help="Hub git revision to read (default: main).",
    )
    parser.add_argument(
        "--app",
        action="append",
        default=[],
        metavar="NAME",
        help="Only process this app folder (repeatable).",
    )
    parser.add_argument(
        "--list-apps",
        action="store_true",
        help="List available app folders on the Hub and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without downloading.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Parallel download threads (default: {DEFAULT_MAX_WORKERS}).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per file on network failure (default: {DEFAULT_RETRIES}).",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=DEFAULT_RETRY_WAIT,
        help=f"Base seconds between retries; grows per attempt (default: {DEFAULT_RETRY_WAIT}).",
    )
    parser.add_argument(
        "--skip-debug",
        action="store_true",
        help="Skip debug_paths/ (large debug screenshots).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-check and fill missing files even if screenshots/ already exists.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)
    hf_apps = list_hf_app_paths(api, args.revision)

    if args.list_apps:
        for app_name in sorted(hf_apps):
            print(f"{app_name}\t{hf_apps[app_name]}")
        return 0

    root = explored_apps_root(args.root)
    root.mkdir(parents=True, exist_ok=True)
    local_apps = list_local_apps(root)

    remote_apps = sorted(hf_apps)
    if args.app:
        requested = set(args.app)
        unknown = sorted(requested - set(hf_apps))
        if unknown:
            print(
                f"error: app(s) not on Hub: {', '.join(unknown)}",
                file=sys.stderr,
            )
            return 1
        remote_apps = [name for name in remote_apps if name in requested]

    downloaded_apps = 0
    skipped_apps = 0
    failed = 0
    files_downloaded = 0
    files_skipped = 0

    for app_name in remote_apps:
        repo_path = hf_apps[app_name]
        local_dir = root / app_name

        if (
            app_name in local_apps
            and local_app_complete(local_dir)
            and not args.force
        ):
            print(f"{app_name}: already local (has screenshots/) at {local_dir}; skipping")
            skipped_apps += 1
            continue

        if app_name in local_apps and not local_app_complete(local_dir):
            print(
                f"{app_name}: incomplete local folder (missing screenshots/); "
                f"resuming {repo_path}/ -> {local_dir}"
            )
        elif app_name in local_apps and args.force:
            print(f"{app_name}: --force; syncing missing files from {repo_path}/")
        else:
            print(f"{app_name}: not local; downloading {repo_path}/ -> {local_dir}")

        try:
            n_dl, n_skip = download_app_directory(
                api,
                local_dir,
                repo_path,
                args.revision,
                args.dry_run,
                token,
                args.max_workers,
                args.retries,
                args.retry_wait,
                args.skip_debug,
            )
        except Exception as exc:
            print(f"error: failed to download {app_name}: {exc}", file=sys.stderr)
            failed += 1
            continue
        files_downloaded += n_dl
        files_skipped += n_skip
        downloaded_apps += 1

    action = "would download" if args.dry_run else "downloaded"
    print(
        f"\nDone: {action} {downloaded_apps} app(s) "
        f"({files_downloaded} files new, {files_skipped} files already present), "
        f"skipped {skipped_apps} complete app(s), failed {failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
