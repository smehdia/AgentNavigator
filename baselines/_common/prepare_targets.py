#!/usr/bin/env python3
"""Generate chosen_target.json files for auto-explored graphs.

Two stages:
1. Dedup: case-insensitive deduplication of navigation memory in the pkl.
2. Localize: for each task, run 2-stage retrieval (embedding + Qwen rerank)
   to find the best target node, then write chosen_target.json.

Usage:
    python baselines/_common/prepare_targets.py \
        --pkl  <explored_graph>/node_information_dict_pure_visual_with_embeddings.pkl \
        --tasks-root <navigation_tasks>/<app> \
        --out-dir results/auto_targets/<app>

    # Batch over the example app list (edit AUTO_APPS for your paths):
    python baselines/_common/prepare_targets.py --batch
"""

import argparse
import json
import os
import pickle
import sys

# <repo>/baselines/_common/prepare_targets.py -> repo root is three levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

for var in [
    "http_proxy", "https_proxy", "ftp_proxy", "socks_proxy",
    "HTTP_PROXY", "HTTPS_PROXY", "FTP_PROXY", "SOCKS_PROXY",
]:
    os.environ.pop(var, None)


# Example batch config for `--batch`. These are placeholders — edit the PKL and
# task-root paths for your own explored graphs / task dirs, or skip --batch and pass
# --pkl/--tasks-root/--out-dir per app. (Your task dirs are typically version-suffixed,
# e.g. <app>_<version>; replace the generic names below accordingly.)
AUTO_APPS = {
    app: {
        "pkl": f"explored_graphs/{app}/node_information_dict_pure_visual_with_embeddings.pkl",
        "tasks_root": f"navigation_tasks/{app}",
    }
    for app in ("amazon", "clock", "google_maps", "airbnb", "youtube")
}


def discover_prompt_from_meta(task_dir: str) -> str | None:
    meta_path = os.path.join(task_dir, "meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path) as f:
            data = json.load(f)
    except Exception:
        return None
    for key in ("retrieval_prompt", "prompt", "task_description"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def dedup_pkl(pkl_path: str, dry_run: bool = False) -> dict:
    """Apply case-insensitive dedup to navigation memory. Returns stats."""
    with open(pkl_path, "rb") as f:
        nodes = pickle.load(f)

    total_wpt_removed = 0
    total_hint_removed = 0
    nodes_fixed = 0

    for key, node in nodes.items():
        nm = node.get("navigation_memory")
        if not nm:
            continue

        wpts = nm.get("relevant_waypoint_sequence", [])
        hints = nm.get("transition_hints", [])

        seen = set()
        dedup_wpts = []
        for w in wpts:
            if w.lower() not in seen:
                dedup_wpts.append(w)
                seen.add(w.lower())

        seen_h = set()
        dedup_hints = []
        for h in hints:
            if h.lower() not in seen_h:
                dedup_hints.append(h)
                seen_h.add(h.lower())

        wpt_removed = len(wpts) - len(dedup_wpts)
        hint_removed = len(hints) - len(dedup_hints)

        if wpt_removed > 0 or hint_removed > 0:
            nodes_fixed += 1
            total_wpt_removed += wpt_removed
            total_hint_removed += hint_removed
            if not dry_run:
                nm["relevant_waypoint_sequence"] = dedup_wpts
                nm["transition_hints"] = dedup_hints

    if not dry_run and (total_wpt_removed > 0 or total_hint_removed > 0):
        with open(pkl_path, "wb") as f:
            pickle.dump(nodes, f)

    return {
        "total_nodes": len(nodes),
        "nodes_fixed": nodes_fixed,
        "wpt_removed": total_wpt_removed,
        "hint_removed": total_hint_removed,
    }


def discover_prompt_from_dir(task_dir: str) -> str | None:
    meta_prompt = discover_prompt_from_meta(task_dir)
    if meta_prompt:
        return meta_prompt
    pj = os.path.join(task_dir, "prompts.json")
    if os.path.isfile(pj):
        with open(pj) as f:
            data = json.load(f)
        if isinstance(data, dict):
            arr = data.get("prompts") or data.get("tasks") or []
        elif isinstance(data, list):
            arr = data
        else:
            arr = []
        if arr:
            item = arr[0]
            if isinstance(item, str):
                return item.strip() or None
            if isinstance(item, dict):
                for k in ("prompt", "text", "instruction", "query", "task"):
                    v = item.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    return None


def load_apps_from_subset_manifest(manifest_path: str) -> dict[str, dict[str, str]]:
    with open(manifest_path) as f:
        manifest = json.load(f)
    apps = {}
    for app_entry in manifest.get("apps", []):
        app = app_entry["app"]
        tasks_root = app_entry["tasks_root"]
        graph_pkl = app_entry["auto_bfs_graph_pkl"]
        apps[app] = {
            "pkl": graph_pkl,
            "tasks_root": tasks_root,
            "out_dir": tasks_root,
        }
    if not apps:
        raise ValueError(f"No app entries found in subset manifest: {manifest_path}")
    return apps


def generate_chosen_targets(pkl_path: str, tasks_root: str, out_dir: str,
                            screenshots_dir: str | None = None):
    """Run 2-stage retrieval and write chosen_target.json per task."""
    from node_retrieval import stage1_retrieval, qwen_select_best_node

    with open(pkl_path, "rb") as f:
        nodes = pickle.load(f)

    print(f"  Loaded {len(nodes)} nodes from {pkl_path}")

    task_dirs = sorted([
        d for d in os.listdir(tasks_root)
        if os.path.isdir(os.path.join(tasks_root, d))
    ])

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for task_name in task_dirs:
        task_dir = os.path.join(tasks_root, task_name)
        prompt = discover_prompt_from_dir(task_dir)
        if not prompt:
            print(f"    {task_name}: no prompt found, skipping")
            continue

        print(f"    {task_name}: '{prompt[:60]}...' ", end="", flush=True)

        candidates = stage1_retrieval(prompt, nodes)
        if not candidates:
            print("-> no candidates")
            continue

        candidates_for_llm = []
        for i, c in enumerate(candidates):
            candidates_for_llm.append({
                "index": i,
                "visual_page_summary": str(c.get("visual_page_summary", "")).strip(),
                "intent_from_current_screen": str(c.get("intent_from_current_screen", "")).strip(),
                "summary_score": c.get("summary_score"),
                "intent_score": c.get("intent_score"),
                "matched_sources": c.get("matched_sources"),
            })

        try:
            result = qwen_select_best_node(prompt, candidates_for_llm)
        except Exception as e:
            print(f"-> qwen rerank failed ({e}), using cosine top-1")
            result = None

        if result and "best_index" in result:
            best = candidates[result["best_index"]]
            confidence = result.get("confidence", "?")
        else:
            best = candidates[0]
            confidence = "cosine-fallback"

        hid, nid = best["hid"], best["nid"]
        nm = best.get("navigation_memory") or {}
        score = best.get("best_embedding_score", 0)

        source_screenshot = None
        if screenshots_dir:
            for ext in (".jpg", ".png"):
                candidate_path = os.path.join(screenshots_dir, f"{hid}_{nid}{ext}")
                if os.path.isfile(candidate_path):
                    source_screenshot = candidate_path
                    break

        chosen = {
            "task_id": task_name,
            "prompt": prompt,
            "hid": hid,
            "nid": nid,
            "embedding_score": score,
            "summary_score": best.get("summary_score"),
            "intent_score": best.get("intent_score"),
            "matched_sources": best.get("matched_sources"),
            "visual_page_summary": best.get("visual_page_summary", ""),
            "intent_from_current_screen": best.get("intent_from_current_screen", ""),
            "navigation_memory": nm,
            "rerank_confidence": confidence,
        }
        if source_screenshot:
            chosen["source_screenshot"] = source_screenshot

        task_out = os.path.join(out_dir, task_name)
        os.makedirs(task_out, exist_ok=True)
        with open(os.path.join(task_out, "chosen_target.json"), "w") as f:
            json.dump(chosen, f, indent=2)

        print(f"-> hid={str(hid)[:20]}... score={score:.3f} conf={confidence}")
        results.append(chosen)

    return results


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pkl", help="Path to processed pkl")
    p.add_argument("--tasks-root", help="Directory containing task subdirs with prompts.json")
    p.add_argument("--out-dir", help="Output directory for chosen_target.json files")
    p.add_argument("--batch", action="store_true",
                   help="Process all 5 auto apps")
    p.add_argument("--subset-manifest",
                   help="Subset manifest generated by generate_spa_bench_subset.py; writes chosen_target.json in place")
    p.add_argument("--dedup-only", action="store_true",
                   help="Only run dedup, skip target generation")
    p.add_argument("--skip-dedup", action="store_true",
                   help="Skip dedup, only generate targets")
    args = p.parse_args()

    if args.batch and args.subset_manifest:
        p.error("Use either --batch or --subset-manifest, not both")
        return

    if args.batch:
        apps = AUTO_APPS
    elif args.subset_manifest:
        apps = load_apps_from_subset_manifest(args.subset_manifest)
    elif args.pkl and args.tasks_root and args.out_dir:
        apps = {"custom": {"pkl": args.pkl, "tasks_root": args.tasks_root, "out_dir": args.out_dir}}
    else:
        p.error("Provide --batch, --subset-manifest, or --pkl + --tasks-root + --out-dir")
        return

    import dashscope
    dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"
    # Key comes from the environment only (DASHSCOPE_API_KEY); no config-file fallback.
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DASHSCOPE_API_KEY not found in environment")
        sys.exit(2)

    for app_name, info in apps.items():
        pkl_path = os.path.join(REPO_ROOT, info["pkl"]) if not os.path.isabs(info["pkl"]) else info["pkl"]
        tasks_root = os.path.join(REPO_ROOT, info["tasks_root"]) if not os.path.isabs(info["tasks_root"]) else info["tasks_root"]
        if args.batch:
            out_dir = os.path.join(REPO_ROOT, "results", "v2_auto", app_name)
        elif args.subset_manifest:
            out_dir = tasks_root
        else:
            out_dir = args.out_dir if not os.path.isabs(args.out_dir) else args.out_dir
        screenshots_dir = os.path.join(os.path.dirname(pkl_path), "screenshots")

        print(f"\n{'='*60}")
        print(f"App: {app_name}")
        print(f"{'='*60}")

        if not args.skip_dedup:
            print(f"  Dedup: {pkl_path}")
            stats = dedup_pkl(pkl_path)
            print(f"    {stats['total_nodes']} nodes, {stats['nodes_fixed']} fixed, "
                  f"{stats['wpt_removed']} waypoints removed, {stats['hint_removed']} hints removed")

        if args.dedup_only:
            continue

        print(f"  Generating targets: {tasks_root} -> {out_dir}")
        generate_chosen_targets(pkl_path, tasks_root, out_dir, screenshots_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
