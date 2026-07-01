# Running Baseline Models — Developer Handoff

This directory lets a new developer reproduce our **baseline benchmarks** — the
neighboring GUI-agent models we compared against — on the same navigation tasks our
own agent runs on. It is self-contained: every path, host, and key is read from
`baselines/.env`, so **no machine-specific or personal paths are baked into the code.**

> How our own method differs from these baselines (the conceptual story) is in the
> slide deck / `BASELINE_COMPARISON.md`. This directory is the *practical* "how to
> actually run them" half.

## The shared pattern

Each baseline keeps its **upstream policy/agent external** (cloned under
`external/`, never vendored) while a thin **`run_benchmark.py` adapter** in our repo
handles the parts that must match our setup:

```
your task data ──▶ adapter (run_benchmark.py) ──▶ Android device (adb)
                      │  • resets / launches the app per task
                      │  • captures screenshots, converts actions
                      │  • calls the upstream model's policy
                      │  • writes canonical trajectories + results.json
                      ▼
                 judge (strict_judge.py / eval.py) ──▶ success rate
```

## Baselines in this directory

| Baseline | Dir | Runnable from a fresh clone? |
|----------|-----|------------------------------|
| GUI-Explorer (ACL'25) | [`gui_explorer/`](gui_explorer/README.md) | Yes — needs its RAG retrieval server + shipped KB |
| PG-Agent | [`pg_agent/`](pg_agent/README.md) | Yes — needs a served VLM + a built page graph |
| AppAgent (CHI'25) | [`appagent/`](appagent/README.md) | Yes — needs pre-generated element docs |
| SPA-Bench-Nav (ICLR'25 tasks) | [`spa_bench/`](spa_bench/README.md) | Yes — SPA-Bench tasks → navigation, run our agent, GPT-4o judge |
| DroidTask-Nav | [`droidtask_nav/`](droidtask_nav/README.md) | Yes — our agent on DroidTask. KG-RAG itself was **not run** (private services); we cite its published numbers. |

Plus our **internal no-memory baseline** (not in this folder — it's part of the agent
itself): run the inference pipeline with navigation memory disabled
(`use_memory_for_navigation: false`), so every task runs goal-only. That is the
apples-to-apples "memory vs. no-memory" floor.

## Fresh-clone setup (shared by all baselines)

```bash
# 1. Clone this repo and install Python deps
git clone https://github.com/smehdia/AgentNavigator.git
cd AgentNavigator
pip install -r baselines/requirements.txt   # + any per-baseline / upstream extras (see each README)

# 2. Configure paths/keys/hosts (no personal data lives in the scripts)
cp baselines/.env.example baselines/.env
$EDITOR baselines/.env                       # fill in roots, keys, device serial

# 3. Clone the upstream baseline model repos
#    (see baselines/external/README.md for the exact git clone commands)

# 4. Have an Android emulator/device running with the target apps installed
adb devices                                  # confirm your ANDROID_DEVICE_SERIAL

# 5. Provide your navigation task data and point TAGNAV_TASKS_ROOT at it
#    (one subdirectory per app; same format the main inference benchmark uses)
```

Load `.env` before running any adapter, e.g.:

```bash
set -a && source baselines/.env && set +a
python baselines/gui_explorer/run_benchmark.py --app clock --out-dir runs/gui_explorer/clock
```

Then follow the per-baseline `README.md` for the exact commands and any extra services
(retrieval server, policy server, dataset download, element docs).

## Required environment variables

All defined in [`.env.example`](.env.example). Summary:

| Var | Used by | Purpose |
|-----|---------|---------|
| `GUI_EXPLORER_ROOT` / `PG_AGENT_ROOT` / `APPAGENT_ROOT` / `SPA_BENCH_ROOT` | each adapter | upstream model checkout under `external/` |
| `TAGNAV_TASKS_ROOT` | all | root of your per-app navigation task folders |
| `ANDROID_DEVICE_SERIAL` | all | `adb` serial of the running device |
| `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | judges / VLM policies | model API access (never commit real keys) |
| `GUI_EXPLORER_RAG_URL` | gui_explorer | retrieval server endpoint |
| `PG_AGENT_SERVER_URL` | pg_agent | policy server endpoint |

## Notes

- `baselines/.env` and everything in `baselines/external/` are git-ignored — secrets and
  large upstream clones stay local.
- The adapters orchestrate external models; they are **not** zero-dependency. Each
  README lists the hard external dependencies (servers, datasets, docs) honestly.
- KG-RAG's upstream is intentionally not vendored or wrapped for execution — see its
  README for why and for what we actually ran (our agent on DroidTask).
