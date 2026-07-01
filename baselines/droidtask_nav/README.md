# KG-RAG / DroidTask baseline

## What this is (read first)

**KG-RAG itself is not runnable end-to-end from a fresh clone.** Its upstream depends
on private, lab-internal Qwen + graph-search services and ships no English-app configs,
so we cannot reproduce its binary here. Instead this directory lets you:

1. Run **our** agent on **DroidTask** — KG-RAG's *home* benchmark — and judge it, and
2. compare against KG-RAG's **published** numbers.

KG-RAG published numbers to compare against (DroidTask):
- **75.8% SR** (GPT-4 backbone), 84.55 DA, 4.10 avg steps;
- **70.5% SR** (Qwen2-VL-72B backbone).

> ⚠️ **Apples-to-oranges caveat:** DroidTask is a **text / DOM** benchmark (agents act on
> serialized view hierarchies). Our agent is **pixel-based**. State this whenever you
> report the comparison.

## Files

| File | Role |
|------|------|
| `run_benchmark.py` | Harness: materialize tasks + chosen targets, then drive the generic nav runner. |
| `convert.py` | Convert DroidTask *completion* tasks → *navigation-only* tasks (v2 LLM policy — the converter used for the paper's results; stops before side effects). |
| `strict_judge.py` | Strict GPT-4o vision judge (final screen vs GT screen). |
| `completion_judge.py` | Completion-first GPT judge (GT screen advisory only). |
| `droidtask_apps.example.json` | Example app metadata for the 12 SMT apps — **verify packages/activities**. |
| `DROIDTASK_STEPS.md` | Full pipeline checklist (phases 0–6). |

## Prerequisites

- Python deps: `openai` (judges), plus the main repo's runtime for the nav runner.
- `adb` + a running device/emulator with the 12 SMT apps installed.
- The **generic nav runner** `run_nav_benchmark.py` and `prepare_auto_targets.py` from the
  vendored under `baselines/_common/` (override `NAV_BENCHMARK_RUNNER` / `PREPARE_TARGETS_SCRIPT` only if yours live elsewhere).
- DroidTask dataset (GT YAMLs + screenshots) + the 12 APKs from the AutoDroid/DroidTask upstream:
  https://github.com/MobileLLM/AutoDroid

## Required environment (`baselines/.env`)

| Var | Meaning |
|-----|---------|
| `DROIDTASK_ROOT` | Root of the DroidTask GT dataset (screenshots resolve against this). |
| `DROIDTASK_NAV_TASKS_ROOT` | Converted DroidTask-Nav tree (output of `convert.py`). |
| `DROIDTASK_APPS_JSON` | App metadata json (defaults to `droidtask_apps.example.json`). |
| `EXPLORED_GRAPHS_ROOT` | Dir holding `smt_<slug>_<graph_source>/…` graphs. |
| `ANDROID_DEVICE_SERIAL` | adb serial of the target device. |
| `OPENAI_API_KEY` | Required by the judges. |
| `NAV_BENCHMARK_RUNNER`, `PREPARE_TARGETS_SCRIPT` | Default to the vendored `baselines/_common/` copies; override only if yours live elsewhere. |

## Steps (from a fresh clone)

```bash
# 0. config
cp baselines/.env.example baselines/.env   # fill in DROIDTASK_ROOT, DROIDTASK_COLLECTOR_ROOT, OPENAI_API_KEY, device
set -a; source baselines/.env; set +a

# 1. convert completion tasks -> navigation tasks (v2 LLM policy — the paper's converter)
python baselines/droidtask_nav/convert.py \
    --source-root "$DROIDTASK_COLLECTOR_ROOT" --gt-root "$DROIDTASK_ROOT" \
    --out-root "$DROIDTASK_NAV_TASKS_ROOT" --model gpt-4o
# review rows flagged needs_manual_review=true

# 2. run the benchmark (our agent) + prepare targets
python baselines/droidtask_nav/run_benchmark.py \
    --all-apps --graph-source "$ANDROID_DEVICE_SERIAL" \
    --prepare-targets --run-benchmark --backbone qwen_vl

# 3. judge
python baselines/droidtask_nav/strict_judge.py --tasks-root "$DROIDTASK_NAV_TASKS_ROOT" --runs-root results/droidtask_nav_smoke/runs
```

See `DROIDTASK_STEPS.md` for the full phase-by-phase checklist.

## Known unmet dependencies (a developer must supply)

- **KG-RAG upstream** — not runnable here (internal services); compare to published numbers only.
- **DroidTask dataset + APKs** — download from the AutoDroid upstream; not vendored.
- **Generic nav runner** (`run_nav_benchmark.py`, `prepare_auto_targets.py`) — part of the main
  TAG-Nav pipeline; point the env vars at them.
- **Explored SMT graphs** — post-processed `…_pure_visual_with_embeddings.pkl` per app (Phase 2).
