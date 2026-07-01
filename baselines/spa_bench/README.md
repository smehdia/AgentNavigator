# SPA-Bench baseline

[SPA-Bench](https://github.com/ai-agents-2030/SPA-Bench) (ICLR 2025) is **not a memory
method** — it is a benchmark/harness. We use it two ways:

1. As a **task source**: its single-app English tasks are converted into navigation-only
   goals our agent can be scored on.
2. As an **evaluation protocol**: its success-detection (here, the GPT-4o VLM judge, with
   PaddleOCR key-component matching bypassed) labels each run success/fail.

It is used to show our navigation graph lifts an agent, not as a competing approach.

## What's here

| File | Role |
|------|------|
| `pilot_common.py` | shared constants: pilot task ids, retrieval prompts, per-app config |
| `generate_subset.py` | materialize the fixed Level-3 / 7-app pilot subset from the SPA-Bench CSV |
| `convert_to_nav.py` | convert SPA-Bench task-completion goals → navigation-only goals (Qwen-VL-Max or GPT-4o) |
| `run_pilot.py` | run each pilot spec through the project's nav benchmark on a device; export zero-indexed screenshots |
| `export_session.py` | materialize a SPA-Bench-compatible results session (uses upstream `framework.utils`) |
| `eval.py` | GPT-4o VLM judge over an exported session (skips OCR) |
| `summarize.py` | aggregate no-graph vs auto-BFS into a summary table |

## Prerequisites

- Python 3.10+ with `pandas`, `requests` (and `dashscope` if you use the DashScope backend).
- `adb` on `PATH`; an Android emulator/device running with the target apps installed.
- The project's own **navigation benchmark runner** (the script that executes one nav task
  on device and writes `step_*.png` + `summary.json`). `run_pilot.py` shells out to it.
- API keys as environment variables (never commit them).

## Setup from a fresh clone

```bash
# 1. Clone this repo, then the upstream SPA-Bench into baselines/external/
cd baselines/external
git clone --recurse-submodules https://github.com/ai-agents-2030/SPA-Bench.git SPA-Bench
cd ../..

# 2. Configure
cp baselines/.env.example baselines/.env      # then edit values
# Required for this baseline:
#   SPA_BENCH_ROOT=baselines/external/SPA-Bench
#   OPENAI_API_KEY=...           (GPT-4o judge + optional openai conversion backend)
#   DASHSCOPE_API_KEY=...        (only if --backend dashscope for conversion)
#   NAV_BENCHMARK_SCRIPT=...     (path to the project's nav benchmark runner)
#   ANDROID_DEVICE_SERIAL=emulator-5554
set -a; . baselines/.env; set +a   # export them into your shell
```

## Required environment

| Variable | Used by | Notes |
|----------|---------|-------|
| `SPA_BENCH_ROOT` | convert, export | upstream SPA-Bench checkout (dataset CSV + `framework.utils`) |
| `OPENAI_API_KEY` | eval, convert(openai) | GPT-4o judge / conversion |
| `DASHSCOPE_API_KEY` | convert(dashscope) | Qwen-VL-Max conversion backend |
| `NAV_BENCHMARK_SCRIPT` | run_pilot | the project's per-task nav runner (or pass `--nav-benchmark-script`) |
| `ANDROID_DEVICE_SERIAL` | run_pilot, export | adb serial (or pass `--device`) |

## Pipeline (end to end)

```bash
# A. Build the fixed pilot subset (downloads the SPA-Bench CSV if no --csv-path)
python baselines/spa_bench/generate_subset.py \
    --out-root spa_bench_subsets/single_app_eng_level3_7apps

# B. (optional) Convert full task-completion goals into navigation-only goals
python baselines/spa_bench/convert_to_nav.py \
    --apps clock calendar chrome airbnb amazon google_maps youtube \
    --out-dir spa_bench_subsets/nav_converted --backend openai

# C. Run a pilot arm on device (no_graph then auto_bfs), per run-spec JSON
python baselines/spa_bench/run_pilot.py \
    --spec-path <arm_spec.json> --device "$ANDROID_DEVICE_SERIAL" \
    --out-root results/spa_bench_pilot --backbone ui_tars

# D. Materialize a SPA-Bench-compatible session from the pilot outputs
python baselines/spa_bench/export_session.py \
    --subset-manifest spa_bench_subsets/single_app_eng_level3_7apps/subset_manifest.json \
    --spec-path <arm_spec.json> \
    --pilot-arm-root results/spa_bench_pilot/no_graph \
    --out-dir results/spa_bench/eval_no_graph

# E. Judge with GPT-4o (OCR bypassed)
python baselines/spa_bench/eval.py \
    --result-dir results/spa_bench/eval_no_graph --agent TAGNavAgent

# F. Summarize no-graph vs auto-BFS
python baselines/spa_bench/summarize.py \
    --subset-manifest .../subset_manifest.json \
    --no-graph-results results/spa_bench/eval_no_graph/results.csv \
    --auto-bfs-results results/spa_bench/eval_auto_bfs/results.csv \
    --out-json results/spa_bench/summary.json --out-md results/spa_bench/summary.md
```

## Hard dependencies to supply

- **Upstream SPA-Bench checkout** (`SPA_BENCH_ROOT`) — needed for the task CSV and, for
  `export_session.py`, the `framework.utils` module that writes the results CSV.
- **The project's nav benchmark runner** (`NAV_BENCHMARK_SCRIPT`) — `run_pilot.py` is only a
  wrapper around it; it does not itself drive the agent.
- **`OPENAI_API_KEY`** — the GPT-4o judge in `eval.py` returns an error result without it.
- **Run-spec JSON** for each arm (`no_graph`, `auto_bfs`) describing per-task memory mode /
  graph pickle / tasks root. Generate these to match your exported graphs.
