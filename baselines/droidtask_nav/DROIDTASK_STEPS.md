# DroidTask Benchmark — pipeline checklist

A reproducible path for running our agent on the **DroidTask** benchmark (KG-RAG's
home benchmark) and judging it. KG-RAG itself is not runnable from a fresh clone
(see README.md) — we compare against its published numbers.

## Phase 0: Infrastructure

- Install the 12 Simple-Mobile-Tools (SMT) APKs on your target device/emulator.
  Verify each package/activity in `droidtask_apps.example.json` against the APKs you install.
- Obtain the DroidTask ground truth (task YAMLs + per-state screenshots) and the
  AutoDroid `tasks.csv` from the AutoDroid / DroidTask upstream
  (https://github.com/MobileLLM/AutoDroid). Place GT under `$DROIDTASK_ROOT`.
- Seed any device state some tasks assume (sample media files, a test contact, etc.):
  push files via `adb push`, create contacts via an `am` intent. Gallery / music /
  messenger tasks need pre-existing content.

## Phase 1: Task data pipeline

- `convert.py` turns DroidTask completion tasks into **navigation-only** tasks (v2 LLM policy — the paper's converter)
  (stops at the first stable screen before user-specific text entry / side effects).
  - Inputs: the DroidTask collector tree + GT screenshots (`--source-root`, `--gt-root`).
  - Output: a parallel `DroidTask_nav_*` tree with `prompts.json` + `meta.json`
    (incl. `gt_nav_screenshot`, `gt_nav_state_str`).
  - Rows flagged `needs_manual_review=true` should be checked before reporting.

## Phase 2: Graph post-processing

- Post-process each SMT graph into the pure-visual node dict with embeddings:
  `explored_graphs/smt_<app>_<graph_source>/node_information_dict_pure_visual_with_embeddings.pkl`.
- Note the cross-device caveat: if a graph was explored on a different device than
  the one you run on, screen layouts may differ slightly. Prefer exploring on the
  same device you benchmark on, or document the mismatch.

## Phase 3: Target retrieval

- For each task, retrieve + select the best graph node and write `chosen_target.json`
  per task dir (`run_benchmark.py --prepare-targets` drives the target-prep helper).

## Phase 4: Baseline

- Run baseline on all 12 apps and judge with the GPT judge (`strict_judge.py`).
- Compare against KG-RAG's published numbers (e.g. ~70.5% with a Qwen2-VL-72B backbone;
  75.8% SR with GPT-4 — see README.md).

## Phase 5: Analysis

- Per-app breakdown, KG-RAG vs ours, failure analysis on misses.
