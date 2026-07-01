# PG-Agent baseline

PG-Agent (arXiv 2509.03536) builds a **page graph** from task episodes and retrieves
page-level guidelines to steer a navigator. We keep it as an **external baseline**: the
page-graph/policy stays in the upstream checkout (or a served endpoint), and our adapter
here drives the emulator, converts actions, and logs trajectories.

## Memory system (the page graph)

- **Where it comes from.** A **page graph** constructed offline from **demonstration
  trajectories**. Upstream builds it from third-party human-demo datasets
  (AITW / Mind2Web / GUI-Odyssey, ~10% subsample); for our benchmark we build it from
  **our own task episodes** with their construction pipeline (`build_graph.py`).
- **Format (how it's built).** Each demonstration is replayed step-by-step —
  **screenshot → action → next screenshot**. Every screen becomes a **node** = a VLM
  one-sentence *page summary* (clustered/deduplicated, so one node = one screen
  archetype); each action that changes pages becomes an **edge** =
  `{summarized action, destination node, the demo's goal label}`. The graph is thus
  *trajectories distilled into page-summary nodes + goal-labeled action edges*.
- **Retrieval at inference.** The current screen is embedded and matched to similar
  page nodes; their outgoing edges (and reachable goals) are surfaced as up to ~10
  natural-language **reference hints** injected into the planner/executor prompt.
  Memory flow: **screenshot → page-summary match → outgoing action/goal hints.**

## Files

| File | Role | Status |
|------|------|--------|
| `run_benchmark.py` | **Per-step adapter** — emulator reset, screenshot + local image server, action conversion, trajectory logging. Loads the PG-Agent policy via `--pg-server-url`, `--pg-entrypoint`, or `--pg-command-template`. | turnkey |
| `strict_judge.py` | GPT/Qwen strict-SR judge over the run output. | turnkey |
| `run_server.py` | Launches the policy/graph HTTP server (`odyssey_server.py`). | reference¹ |
| `odyssey_server.py` | The page-graph + VLM policy server queried per step. | reference¹ |
| `build_graph.py` | Constructs the page graph from task episodes. | reference¹ |
| `env_utils.py` | Small env/config loader used by the server. | helper |

¹ *reference* = data-scrubbed but coupled to the upstream PG-Agent pipeline and a served VLM;
expect to adjust imports/paths for your environment (see "Reproducing the policy server").

## Dependency note (shared)

`run_benchmark.py` imports the shared adb driver + app registry from **`baselines/_common/nav_core.py`**
(`APP_REGISTRY`, `AdbDriver`, `discover_prompt`, `execute_action`). That module must exist — it is
shared by all baselines.

## Required environment (`baselines/.env`)

| Var | Used for |
|-----|----------|
| `PG_AGENT_ROOT` | upstream PG-Agent checkout (`baselines/external/PG-Agent`) |
| `PG_AGENT_SERVER_URL` | policy endpoint, e.g. `http://127.0.0.1:18000/next_action` (or use `--pg-entrypoint`) |
| `TAGNAV_TASKS_ROOT` | root holding your per-app navigation task folders |
| `ANDROID_DEVICE_SERIAL` | adb serial of the running emulator/device |
| `DASHSCOPE_API_KEY` | for `strict_judge.py` |

## Steps from a fresh clone

```bash
# 1. deps (Python 3.10+): pip install opencv-python requests dashscope pyyaml
# 2. upstream model
cd baselines/external && git clone https://github.com/<pg-agent-upstream>/PG-Agent.git PG-Agent && cd ../..
cp baselines/.env.example baselines/.env   # then edit: set PG_AGENT_ROOT, TAGNAV_TASKS_ROOT, device, keys
set -a && . baselines/.env && set +a
# 3. emulator running, target app installed, `adb devices` shows $ANDROID_DEVICE_SERIAL
```

### Run the benchmark (policy served separately)

```bash
# point the adapter at a running PG-Agent policy endpoint:
python baselines/pg_agent/run_benchmark.py \
    --app clock --device "$ANDROID_DEVICE_SERIAL" --max-steps 15 \
    --tasks-root "$TAGNAV_TASKS_ROOT" \
    --out-dir results/pg_agent/clock \
    --pg-server-url "$PG_AGENT_SERVER_URL"
```

Alternatives to `--pg-server-url`: `--pg-entrypoint package.module:function` (in-process call into
your PG-Agent checkout) or `--pg-command-template "python ... --request {request_json}"`.

### Judge

```bash
python baselines/pg_agent/strict_judge.py --root results/pg_agent/clock --out results/pg_agent/clock/verdicts.json
```

## Reproducing the policy server (reference)

PG-Agent needs (a) a **served VLM** — we used **Qwen2.5-VL-72B** behind an OpenAI-compatible endpoint —
and (b) a **page graph** built from labeled task episodes. Build the graph (`build_graph.py`) from your
episodes, then serve it (`run_server.py` → `odyssey_server.py`) and point `PG_AGENT_SERVER_URL` at it.
Our setup **replaced PG-Agent's multi-agent planner with a single Qwen2.5-VL-72B navigator**; the
upstream `document_construction/` + `workflow/` pipeline is the faithful path if you want their full
4-agent loop.

## Hard dependencies a developer must supply

- A running **PG-Agent policy** (served endpoint or importable entrypoint) — not vendored.
- A **served VLM** (Qwen2.5-VL-72B or equivalent) for the policy/graph.
- **Labeled task episodes** to construct the page graph.
- An emulator/device with the target app installed, and `baselines/_common/nav_core.py`.
