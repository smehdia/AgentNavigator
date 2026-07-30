# Inference

Run a natural-language navigation goal on a **physical device** using an explored app graph. Inference **retrieves** a target screen from exploration artifacts, then drives the UI agent step-by-step. Each step uses an **OOD classifier** (on/off-graph) and a **SigLIP+SmolVLM gallery localizer** to suggest next-hop transition hints to the agent.

**Prerequisites:** Complete [exploration](../exploration/README.md), **post-processing**, and **localizer training** (`train_localizer.py`) for the same app so `logs.root` contains `graph.json`, `user_intents.json`, `node_level_information.json`, `edge_level_information.json`, `screenshots/`, `ood_classifier.joblib`, and `siglip_smolvlm_features.pt`.

**Recommended for demos:** use the [Inference GUI](#inference-gui-demo-wizard) (`run_gui.sh`) — a browser wizard that walks through config, device checks, retrieval, and on-device navigation. For scripted or batch runs, use the [CLI](#quick-start-cli) (`inference.py`).

---

## Inference GUI (demo wizard)

A **React + FastAPI** web UI for live demos: pick an app config, verify agent and device, preview the phone, then run the full **retrieval → candidate pick → navigation** loop from a chat panel.

The GUI reuses the same retrieval and navigation logic as [`inference.py`](inference.py), factored into [`gui_demo/pipeline.py`](gui_demo/pipeline.py). The CLI script does not depend on `gui_demo/`.

### Layout

```text
inference/
├── run_gui.sh              # Recommended launcher (build frontend + start server)
└── gui_demo/
    ├── inference_gui.py    # FastAPI backend (REST + WebSockets + static UI)
    ├── pipeline.py         # load_artifacts, build_rerank_candidates, run_retrieval, run_navigation_loop
    ├── requirements-gui.txt
    └── web/                # React + Vite frontend
        ├── src/            # Source (committed)
        └── dist/           # Production build (gitignored; built by run_gui.sh)
```

### What you need before running

| Requirement | Notes |
|-------------|--------|
| **Exploration artifacts** | Under `logs.root`: `graph.json`, `user_intents.json`, `node_level_information.json`, `edge_level_information.json`, `screenshots/{node_id}.jpg` |
| **Localizer artifacts** | `ood_classifier.joblib`, `siglip_smolvlm_features.pt` (from `exploration/train_localizer.py`; gallery `dim=1344`) |
| **Python env** | Same env as CLI inference (FlagEmbedding, dynaconf, dashscope, OpenCV, scikit-learn, transformers, etc.) |
| **GUI extras** | `pip install -r gui_demo/requirements-gui.txt` (FastAPI, uvicorn, PyYAML) |
| **Node.js 18+** | For `npm install` / `npm run build` in `gui_demo/web/` |
| **ADB device** | Emulator or physical device visible in `adb devices` |
| **Agent server** | MAI-UI or UI-TARS at the `agent.url` in config |

Pick or edit a config under `inference/configs/` (e.g. `outlook_android.yaml`). Set `driver.device_id`, `logs.root` (path to explored app output), and API keys before the demo.

### One-time setup

From the `inference/` directory with your Python env active:

```bash
cd inference

# GUI Python packages
pip install -r gui_demo/requirements-gui.txt

# Frontend dependencies (unset proxy if npm hangs)
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
cd gui_demo/web && npm install && cd ../..
```

You only need `npm install` once (or after `package.json` changes). **`run_gui.sh` runs `npm run build` automatically** on every start so `web/dist/` is up to date.

### Run (recommended)

```bash
cd inference
bash run_gui.sh
```

Then open **[http://localhost:8765](http://localhost:8765)**.

`run_gui.sh`:

1. `cd`s to `inference/` (so imports resolve).
2. Unsets common proxy env vars (avoids npm/curl issues).
3. Runs `npm run build` in `gui_demo/web/`.
4. Starts `uvicorn gui_demo.inference_gui:app --host 0.0.0.0 --port 8765`.

### Run (manual / development)

**Production (single port)** — serve built React from `web/dist/`:

```bash
cd inference
cd gui_demo/web && npm run build && cd ../..
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
uvicorn gui_demo.inference_gui:app --host 0.0.0.0 --port 8765
```

**Development (hot-reload frontend, two terminals):**

```bash
# Terminal 1 — API
cd inference
uvicorn gui_demo.inference_gui:app --reload --port 8765

# Terminal 2 — Vite (proxies /api and /ws to :8765)
cd inference/gui_demo/web
npm run dev
```

Open [http://localhost:5173](http://localhost:5173) for dev, or [http://localhost:8765](http://localhost:8765) after a production build.

### Wizard: what to do (5 steps)

Complete each step in order; later steps stay locked until the previous one succeeds.

| Step | UI action | What happens on the backend |
|------|-----------|---------------------------|
| **1. Configure** | Pick a config from the dropdown (all `configs/*.yaml`), edit fields if needed, click **Apply Config** | `POST /api/config/select` or `PUT /api/config` loads YAML into server session; relative `logs.root` is resolved from `inference/` |
| **2. Checks** | Click **Run Checks** | `POST /api/checks/run` — curls agent `GET /v1/models`; runs `driver.check_device()` and reports foreground package |
| **3. Device** | Click **Connect to Device** | `POST /api/device/connect` — builds driver, launches app; `WS /ws/device` streams live JPEG screenshots (~1.5s interval) |
| **4. Load** | Click **Load Resources** | `GET /api/resources/load/stream` (SSE) loads graph, intents, navigation plans, BGE-M3, VLM, and agent |
| **5. Navigate** | Chat panel: prompt → pick candidate → **Execute** | `POST /api/retrieve` → `POST /api/select-node` → `WS /ws/execution` streams agent steps |

**Typical demo flow**

1. Start `bash run_gui.sh` and open the browser.
2. **Configure** — select e.g. `outlook_android`, confirm `logs.root` points to your explored app folder, **Apply**.
3. **Checks** — ensure agent and device both pass (fix `device_id` / agent URL if not).
4. **Device** — connect; confirm live preview shows the app.
5. **Load** — wait for all progress items (first BGE-M3 load can take ~30s).
6. **Navigate** — type a goal (e.g. *Navigate to the Feedback to Microsoft page.*), **Send**.
7. If `use_memory_for_navigation` is on: review **candidate cards** (exploration screenshots + scores), tap the best match.
8. Read the **navigation memory** message, then click **Execute**.
9. Watch **step stream** (thought + annotated screenshot per action). Use **Stop Navigation** to halt after the current step.
10. Send another prompt to run again without reloading resources.

If `use_memory_for_navigation` is **off**, steps 7–8 are skipped: **Execute** is offered right after the prompt and the agent relies on the screenshot + goal only.

On wide screens, a live device preview panel sits beside the chat during execution.

### How it works (logic)

```mermaid
flowchart TB
  subgraph ui["React UI (gui_demo/web)"]
    W1[Configure]
    W2[Checks]
    W3[Device preview WS]
    W4[Load SSE]
    W5[Chat: retrieve / execute WS]
  end

  subgraph api["FastAPI (inference_gui.py)"]
    S[InferenceSession]
    W1 --> S
    W2 --> S
    W3 --> S
    W4 --> S
    W5 --> S
  end

  subgraph pipe["pipeline.py — same ideas as inference.py"]
    LA[load_artifacts]
    RR[run_retrieval]
    FN[format_navigation_plan]
    NL[run_navigation_loop]
  end

  W4 --> LA
  W5 --> RR --> FN --> NL
  NL --> Agent[MAI-UI / UI-TARS]
  NL --> Driver[Android driver]
```

**Session state** — one in-memory `InferenceSession` holds config, driver, agent, loaded artifacts (`RetrievalContext`), retrieval candidates, selected `node_id`, and execution flags. Changing config clears runtime state.

**Load resources** (`load_artifacts` + localizer load in `inference_gui.py` / CLI):

1. `graph.json` → NetworkX graph + per-node **depth** (shortest hop from nearest root).
2. `user_intents.json` → `user_intents`, `embedding`, `embedding_text` (BGE-M3 index from post-process Stage 6).
3. `node_level_information.json` → per-node page descriptions used in VLM reranking.
4. `edge_level_information.json` → per-edge action descriptions for live next-hop hints.
5. `ood_classifier.joblib` + `siglip_smolvlm_features.pt` → OOD SVM and SigLIP+SmolVLM gallery for live localization.
6. BGE-M3 model loaded for query encoding at retrieval time.
7. `VLM` client for stage-2 reranking.
8. SigLIP + SmolVLM encoders (`Encoders`) for per-step screenshot features.
9. Agent client connected to `agent.url`.

**Retrieve** (when `use_memory_for_navigation: true`):

1. **Stage 1** — encode the user query with BGE-M3; cosine-search against precomputed node embeddings in `user_intents.json`; keep top `top_k_in_first_stage_retrieval`.
2. **Stage 2** — `build_rerank_candidates` enriches hits with `page_description`, `user_intents`, and navigation plans; `VLM.rerank_candidates` reranks; keep top `top_k_retrieval_in_stage_2`.
3. Return candidate cards with exploration screenshots (`GET /api/screenshots/{node_id}`).
4. User / `pick_candidate_index` selects **`selected_node_id`** (navigation target).

**Execute** (`run_navigation_loop` / `execute_single_task`):

1. `driver.reset_to_start_page()` (back ×5, force-stop, relaunch, optional scroll-up / `reset_instruction`).
2. Each step:
   - Screenshot → **OOD** (SigLIP cosine profile vs gallery → SVM on/off-graph).
   - **Off-graph** → `format_navigation_plan(goal, [])` (goal + global instructions only).
   - **On-graph** → concat SigLIP+SmolVLM → top-3 gallery matches → shortest-path **next hop** toward `selected_node_id` → up to 3 **transition hints** from `edge_level_information.json` → `format_navigation_plan(goal, hints)`.
   - If `selected_node_id` is already in the top-3 → finish early.
   - `agent.step(prompt, screenshot)` → execute action (no extra `driver.wait`).
3. GUI streams steps over `WS /ws/execution` including **localization** (on/off-graph, top-3, next hops, hints).

This mirrors CLI `inference.py` except candidate selection is **interactive** (tap a card) instead of `pick_candidate_index` or Gradio.

### API surface (reference)

| Endpoint | Purpose |
|----------|---------|
| `GET /api/configs` | List available YAML configs |
| `POST /api/config/select` | Load config by id (e.g. `outlook_android`) |
| `PUT /api/config` | Apply edited config JSON |
| `POST /api/checks/run` | Agent + device health |
| `POST /api/device/connect` | Launch app on device |
| `WS /ws/device` | Live device screenshot stream |
| `GET /api/resources/load/stream` | SSE progress while loading artifacts |
| `POST /api/retrieve` | Run embedding search, build rerank candidates, VLM rerank |
| `POST /api/select-node` | Set selected target `node_id` |
| `POST /api/navigation-memory` | Preview formatted navigation prompt |
| `POST /api/execute/start` | Arm execution (then open execution WebSocket) |
| `POST /api/execute/stop` | Request cooperative stop |
| `WS /ws/execution` | Stream navigation steps |

### GUI troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'gui_demo'` | Run from `inference/`, not repo root. Use `bash run_gui.sh`. |
| `npm run build` fails or hangs | Unset proxy env vars; run `npm install` in `gui_demo/web/` first. |
| Blank page at `:8765` | Build failed or `web/dist/` missing — check `run_gui.sh` output. |
| Agent check fails | Confirm `agent.url`; test `curl -sSf http://HOST:8089/v1/models`. |
| Device check fails | Run `adb devices`; set `driver.device_id` in config. |
| Load fails on localizer | Ensure `ood_classifier.joblib` and `siglip_smolvlm_features.pt` (`dim=1344`) exist under `logs.root`; re-run `train_localizer.py`. |
| Load fails on graph/intents | Verify `logs.root` and that exploration + post-process completed. |
| No candidate cards | Set `use_memory_for_navigation: true`; ensure `embedding` fields exist in `user_intents.json`. |
| Stop does not interrupt instantly | Stop is cooperative — finishes the current agent step, then halts. |

---

## Quick start (CLI)

For interactive demos, prefer the [GUI](#inference-gui-demo-wizard) (`bash run_gui.sh`). Use the CLI for scripted runs, batch evaluation, or Gradio candidate picking (`pick_candidate_index: -1`).

From the `inference/` directory:

```bash
cd inference
python inference.py --config configs/clock_android.yaml
```

If `query` is empty in the config, the script prompts:

```text
Enter your query:
```

### What happens at runtime

1. Load the explored graph and node intents from `logs.root`.
2. If `use_memory_for_navigation` is `true`, run two-stage retrieval, then select a target node (automatically via `pick_candidate_index`, or interactively via Gradio when `pick_candidate_index: -1`).
3. Format navigation memory (or an empty instruction) for the agent.
4. Reset the device to a known start screen (`driver.reset_to_start_page()` — optional scroll-up unless `skip_scroll_up_on_reset`).
5. Loop: agent observes screenshot → proposes action → driver executes until `finish` or `agent.max_steps`.
6. Optionally save screenshots, annotated screenshots, `actions.json`, and `prompt.json` under `output_dir`.

---

## YAML config files

Configs live in `inference/configs/`. Each file has a top-level `default:` block loaded by [Dynaconf](https://www.dynaconf.com/).

| Config | App | Platform |
|--------|-----|----------|
| `airbnb_android.yaml` | Airbnb | Android |
| `alibaba_harmony.yaml` | Alibaba | HarmonyOS |
| `amazon_android.yaml` | Amazon | Android |
| `clock_android.yaml` | Clock | Android |
| `ebay_android.yaml` | eBay | Android |
| `google_maps_android.yaml` | Google Maps | Android |
| `linkedin_android.yaml` | LinkedIn | Android |
| `outlook_android.yaml` | Outlook | Android (includes batch-mode example) |
| `target_android.yaml` | Target | Android |
| `yelp_android.yaml` | Yelp | Android |
| `youtube_android.yaml` | YouTube | Android |
| `zhixing_train_harmony.yaml` | Zhixing Train Tickets | HarmonyOS |

Copy an existing config and adjust `device_id`, API keys, `logs.root`, and paths for your setup.

### `app`

Metadata about the target application (used by drivers/agents where needed).

| Field | Description |
|-------|-------------|
| `name` | Human-readable app name |
| `description` | Short description of what the app does |

### `driver`

Controls the device connection and app launch. Built by `Driver.factory.build_driver`.

| Field | Description |
|-------|-------------|
| `device_id` | ADB / device identifier |
| `os_name` | `"android"` or `"harmony"` |
| `appPackage` | App package name |
| `appActivity` | Launch activity |
| `skip_exploration_for_no_app_package` | *(optional)* Skip work when the foreground app is not the target package |
| `reset_instruction` | *(optional)* Natural-language instruction the agent runs after relaunch to reach a consistent tab/screen (e.g. *"Go Alarm tab..."*). If omitted, reset stops after relaunch + scroll. |
| `skip_scroll_up_on_reset` | *(optional, default `false`)* When `true`, skip the centered scroll-up swipe after relaunch in `reset_to_start_page()`. Use when the start screen is not a scrollable feed and scroll-up would move away from the anchor (e.g. fixed todo list, tab bar). |
| `use_launcher_intent` | *(optional, Android)* Launch via launcher intent |

**Harmony swipe:** `HarmonyDriver.swipe()` accepts the same `duration_ms` argument as Android, but converts it to Harmony’s `swipeVelocityPps_` (px/s) before calling `uitest uiInput swipe`. Passing duration directly to `uitest` would be misread as velocity and produce very slow scrolls.

### `vlm`

API settings for the VLM used in **stage-2 reranking** (`VLM.rerank_candidates`).

| Field | Description |
|-------|-------------|
| `alibaba_api_key` | DashScope / Alibaba API key |
| `yibu_api_key` | Yibu API key (when `use_yibu_api` is true) |
| `use_yibu_api` | Route VLM calls through Yibu instead of Alibaba |

### `agent`

The on-device navigation model (MAI-UI or UI-TARS). Built by `Agents.factory.build_agent`.

| Field | Description |
|-------|-------------|
| `url` | OpenAI-compatible API base URL for the agent server |
| `model_name` | `"mai_ui"` or `"ui_tars"` |
| `verbose_print` | Log agent internals |
| `settings.history_n` | Number of past steps kept in agent memory |
| `settings.resize_factor` | Screenshot resize factor sent to the agent |
| `max_steps` | Maximum on-device navigation steps per task (default: 10) |

#### MAI-UI server (vLLM)

When `model_name` is `mai_ui`, serve **HuggingFace** [`Tongyi-MAI/MAI-UI-8B`](https://huggingface.co/Tongyi-MAI/MAI-UI-8B) behind an OpenAI-compatible endpoint.

**vLLM version:** use **vLLM &lt; 0.2** (the official MAI-UI stack pins **`vllm==0.11.0`**). **vLLM 0.21+** has been observed to return plausible reasoning but **wrong grounding coordinates** (taps land on the wrong UI element). Pin compatible deps on the server, e.g. `transformers==4.57.6` (&lt; 5.0), `tokenizers` 0.22.x, `numpy≤2.2`.

**Smoke test (strongly encouraged):** before a full `inference.py` run, verify grounding on a single device screenshot:

```python
from dynaconf import Dynaconf
from Agents.factory import build_agent
from Driver.factory import build_driver

configs = Dynaconf(settings_files=["configs/amazon_android.yaml"]).default
agent = build_agent(
    model_name=configs.agent.model_name,
    url=configs.agent.url,
    agent_settings=configs.agent.settings,
)
driver = build_driver(settings=configs.driver, agent=agent)

screenshot = driver.take_screenshot()
(parsed, sent_w, sent_h), meta = agent.grounding_action(
    "click on the Haul chip button",  # pick a visible, unambiguous target on this screen
    screenshot,
)
print(parsed.thought, parsed.orig_coords)

# Optional: tap on device and confirm the UI responds as expected
driver.execute_action(parsed)
```

If coordinates look wrong (e.g. Y lands on a different row than the described element), fix the agent server stack (vLLM version and pins above) before relying on navigation.

### `logs`

| Field | Description |
|-------|-------------|
| `root` | Path to exploration output. Must contain `graph.json`, `user_intents.json`, `node_level_information.json`, `node_navigation_plans.json`, and `screenshots/{node_id}.jpg`. |
| `resume_from_checkpoint` | Used by exploration; ignored by inference |

### Inference-specific fields

| Field | Default | Description |
|-------|---------|-------------|
| `query` | — | User goal in natural language (single-task mode). If empty and `batch_mode` is `false`, the script prompts interactively. Ignored when `batch_mode` is `true`. |
| `use_memory_for_navigation` | — | **`true`**: run embedding + VLM retrieval, then pick a **target** `selected_node_id` for shortest-path hints. **`false`**: skip retrieval; still runs OOD/localizer when artifacts exist, but next-hop hints need a selected target. |
| `top_k_in_first_stage_retrieval` | — | Number of nodes returned by embedding search (stage 1). Typical: 15–30. |
| `top_k_retrieval_in_stage_2` | — | Number of nodes kept after VLM rerank (stage 2). These are the candidates you can choose from via `pick_candidate_index` or Gradio. Typical: 3–5. |
| `pick_candidate_index` | `-1` | Which stage-2 candidate to navigate with. See [pick_candidate_index](#pick_candidate_index) below. |
| `batch_mode` | `false` | **`true`**: run every task under `input_dir` instead of a single `query`. Requires `input_dir` and `output_dir`. |
| `input_dir` | — | Root directory of a benchmark dataset. Each immediate subdirectory must contain a `prompts.json` with a `prompts` list; the **first** prompt is used as the task goal. |
| `output_dir` | — | Where to write run artifacts (screenshots, `actions.json`, `prompt.json`). Required in batch mode; optional in single-task mode. |

#### `pick_candidate_index`

After stage-2 VLM reranking, candidates are ordered best-first (index `0` = top VLM pick, `1` = second best, and so on). This field chooses which candidate becomes **`selected_node_id`** (destination for shortest-path next-hop hints during on-device localization).

| Value | Behavior |
|-------|----------|
| `0` | Use the best VLM-ranked candidate (most common for automated runs). |
| `1`, `2`, … | Use the 2nd, 3rd, … best candidate. Must be **&lt; `top_k_retrieval_in_stage_2`**. |
| `-1` | Open the Gradio picker (`pick_candidate`) so you can confirm the target page manually. |

**When to change it:** If the top-ranked page is wrong but a lower-ranked candidate looks correct in exploration screenshots, retry with `pick_candidate_index: 1` (or higher) without re-running retrieval.

**Batch evaluation tip:** To save trajectories for all stage-2 candidates, run batch mode once per index (`0` … `top_k_retrieval_in_stage_2 - 1`). Each run writes to a separate folder (see [Batch mode](#batch-mode)).

### Example (minimal)

```yaml
default:
  app:
    name: "Clock"
    description: "Alarms, timers, stopwatch, and world clocks."

  driver:
    device_id: "YOUR_DEVICE_ID"
    os_name: "android"
    appPackage: "com.google.android.deskclock"
    appActivity: "com.android.deskclock.DeskClock"
    reset_instruction: "Go Alarm tab. (if Alarm tab is highlighted return Finish)"

  vlm:
    alibaba_api_key: "YOUR_KEY"
    yibu_api_key: "YOUR_KEY"
    use_yibu_api: true

  agent:
    url: "http://YOUR_AGENT_HOST:8089/v1"
    model_name: "mai_ui"
    settings:
      history_n: 3
      resize_factor: 0.75

  logs:
    root: "../exploration/explored_apps/clock"

  use_memory_for_navigation: true
  query: "Navigate to the Timer section."
  top_k_in_first_stage_retrieval: 15
  top_k_retrieval_in_stage_2: 5
  pick_candidate_index: 0

  # Optional: batch evaluation over a dataset
  # batch_mode: true
  # input_dir: "/path/to/dataset/"
  # output_dir: "./results"
```

### Example (batch mode)

```yaml
default:
  # ... app, driver, vlm, agent, logs as above ...

  batch_mode: true
  input_dir: "/path/to/AgentNavigatorDataset/DATA/outlook_5.2604.1/"
  output_dir: "./results"

  use_memory_for_navigation: true
  top_k_in_first_stage_retrieval: 30
  top_k_retrieval_in_stage_2: 5
  pick_candidate_index: 0   # run again with 1, 2, ... to save all top-k trajectories
```

---

## Batch mode

When `batch_mode: true`, inference iterates over every task subdirectory under `input_dir`, reads the first prompt from each `prompts.json`, and runs the full retrieval + navigation pipeline for that goal.

### Input layout

```text
input_dir/
  run_001_outlook/
    prompts.json          # { "prompts": ["Navigate to ..."] }
  run_002_outlook/
    prompts.json
  ...
```

### Output layout

Each run is saved under `{output_dir}/{task_name}_{pick_candidate_index}/`:

```text
results/
  run_001_outlook_0/
    0.jpg                 # raw screenshots (one per step)
    1.jpg
    ...
    annotated_screenshots/
      0.jpg               # same frames with click/swipe overlays
      ...
    actions.json          # [{ "type", "coordinate", "thought" }, ...]
    prompt.json           # { "query": "..." }
  run_001_outlook_1/      # same task, 2nd-best stage-2 candidate
  run_002_outlook_0/
  ...
```

The `{pick_candidate_index}` suffix records which stage-2 rerank slot was used for that trajectory.

### Saving all top-k stage-2 trajectories

Stage 2 returns up to `top_k_retrieval_in_stage_2` ranked target pages. A single batch run navigates with **one** of them (controlled by `pick_candidate_index`). To benchmark or compare every reranked candidate:

1. Set `batch_mode: true`, `input_dir`, and `output_dir`.
2. Run inference with `pick_candidate_index: 0`, then `1`, … up to `top_k_retrieval_in_stage_2 - 1`.
3. Each run produces a full trajectory per task under `{task_name}_{index}/`.

This lets you compare whether the 1st, 2nd, or 3rd VLM pick actually reaches the goal on device, without re-running stage-1 embedding search.

---

## Retrieval and navigation hierarchy

Inference splits **finding the right target screen** (retrieval) from **getting there on the device** (localized navigation).

```mermaid
flowchart LR
    Q[User query]
    subgraph artifacts [logs.root artifacts]
        UI[user_intents.json]
        NL[node_level_information.json]
        EL[edge_level_information.json]
        FEAT[siglip_smolvlm_features.pt]
        OOD[ood_classifier.joblib]
        G[graph.json]
    end
    subgraph retrieval [Retrieval — if use_memory_for_navigation]
        S1[Stage 1: BGE-M3 embedding search]
        BR[build_rerank_candidates]
        S2[Stage 2: VLM rerank]
        PICK[pick_candidate_index or Gradio]
    end
    subgraph nav [Per-step navigation]
        LOC[OOD + gallery top-3]
        HOP[shortest-path next hop]
        FM[format_navigation_plan]
        LOOP[agent.step]
    end
    UI --> S1
    NL --> BR
    G --> HOP
    EL --> HOP
    FEAT --> LOC
    OOD --> LOC
    Q --> S1
    S1 --> BR --> S2 --> PICK
    PICK -->|selected_node_id| HOP
    LOC --> HOP --> FM --> LOOP
    Q --> FM
    LOOP -->|screenshot + hints| AGENT[MAI-UI / UI-TARS]
    AGENT -->|action| DRV[Android / Harmony driver]
    DRV -->|execute| DEV[Device]
```

### Stage 1 — Embedding retrieval

**Function:** `retrieve_nodes_from_user_intents_embeddings`

- Loads precomputed `embedding` vectors from **`user_intents.json`** (post-process Stage 6).
- Encodes the user query with **BGE-M3** (`BAAI/bge-m3`).
- Scores all nodes by cosine similarity and returns the top `top_k_in_first_stage_retrieval` hits.

**Query text:** `build_retrieval_query()` returns the user goal string as-is. Node indexing uses `embedding_text` built from **page description + user intents** during post-processing.

### Build rerank payload

**Function:** `build_rerank_candidates`

Joins stage-1 hits with:

| Source file | Fields used |
|-------------|-------------|
| `node_level_information.json` | `high_level`, `medium_level`, `low_level` → `page_description` |
| `user_intents.json` | `user_intents` |
| `node_navigation_plans.json` | plan list per node → `ui_navigation_memory` (waypoints + hints for **VLM reranker** context; live agent prompts use OOD/localizer hints instead) |

Returns `{ "task_prompt": ..., "candidates": [...] }` for the VLM reranker.

### Node depth

**Function:** `get_node_depths`

Each graph node gets a **depth**: shortest path length from the nearest root (`is_root: true`). Depth is attached after reranking for display and Gradio labels; the VLM reranker also sees page descriptions and user intents.

### Stage 2 — VLM rerank

**Function:** `VLM.rerank_candidates`

- Input: user query + compact candidate list (`page_description`, `user_intents`, `retrieval_score`, `ui_navigation_memory`).
- Output: ordered `top_k_node_ids` and per-node `reasoning`.
- Keeps the top `top_k_retrieval_in_stage_2` candidates for `pick_candidate_index` or Gradio.

### Candidate selection — `pick_candidate_index` or Gradio

**Automatic (`pick_candidate_index` ≥ 0):** Uses the candidate at that index in the stage-2 list (no UI).

**Interactive (`pick_candidate_index: -1`):** **Function:** `pick_candidate`

- Opens a local Gradio UI at `http://127.0.0.1:7860`.
- Shows screenshots and metadata (score, depth, page purpose, VLM reasoning).
- User selects a candidate and clicks **Confirm and continue**.

Requires `pip install gradio`.

### Navigation prompt formatting + live localization

**Functions:** `format_navigation_plan`, OOD/`ood_features`, gallery cosine match (CLI `execute_single_task`; GUI `localize_screenshot`)

Live prompts are **not** a static dump of `node_navigation_plans.json`. Each step rebuilds the agent instruction from localization:

| Case | Agent prompt |
|------|----------------|
| Off-graph (`ood_label=0`) | Final goal (+ `in the current application`) + `[Global usage instruction]` |
| On-graph | Same + up to **3** `[Hint i]` entries (`High` / `Low`) for next hops from top-3 matches |
| Top-3 already contains `selected_node_id` | Finish early (no agent call) |

**`format_navigation_plan(final_goal, transition_hints)`**

- Appends ` in the current application` to the goal when missing.
- Renders optional next-step hints (high-level intent + low-level visual grounding).
- Always ends with `[Global usage instruction]` (advance toward goal, do not mix disagreeing plans, finish when done, don’t leave the app).

**Per-step localization pipeline**

```text
screenshot
  → letterbox to target_hw
  → SigLIP z  → 16-d cosine profile vs gallery SigLIP  → OOD SVM
  if on-graph:
      → SmolVLM vision  → concat with z  → cosine top-3 vs gallery
      → for each match: shortest_path → next hop → edge_level_information[src|dst]
      → transition_hints (≤3)
  → format_navigation_plan(goal, hints) → agent.step
```

| Artifact | Role |
|----------|------|
| `ood_classifier.joblib` | SVM on cosine-profile features (`prepare_data_for_ood` / `train_ood_classifier`) |
| `siglip_smolvlm_features.pt` | Gallery `gallery_z` (`dim=1344`), `node_ids`, `target_hw` |
| `edge_level_information.json` | `low_level_action_description` / `high_level_action_description` per edge (list of alternatives; first entry used) |
| `graph.json` | Shortest path from localized node → `selected_node_id` |

Train these with [`exploration/train_localizer.py`](../exploration/README.md#screenshot-localizer-train_localizerpy).

### On-device navigation loop

After `driver.reset_to_start_page()`:

```text
for up to agent.max_steps:
    screenshot = driver.take_screenshot()
    ood / localize → transition_hints (maybe empty)
    prompt = format_navigation_plan(goal, transition_hints)
    if selected node in top-3 → finish
    action = agent.step(prompt, screenshot)
    if action is finish → stop
    else driver.execute_action(action)   # no fixed sleep/wait
```

Console prints per-step timing: `shot=…s loc=…s agent=…s act=…s total=…s`.

The agent (`mai_ui` or `ui_tars`) is a separate server; inference only sends screenshots and the formatted prompt.

### GUI demo parity

[`gui_demo/pipeline.py`](gui_demo/pipeline.py) mirrors CLI localization (`localize_screenshot`, `format_navigation_plan`, `run_navigation_loop`). The React wizard shows **localization** (on/off-graph, top-3, next hops, hints) beside each live step screenshot via [`gui_demo/inference_gui.py`](gui_demo/inference_gui.py).


---

## Required artifacts under `logs.root`

| File / folder | Source | Used for |
|---------------|--------|----------|
| `graph.json` | Exploration export | Graph structure, depths, shortest-path next hops |
| `user_intents.json` | Post-process Stages 4 + 6 | `user_intents`, `embedding`, `embedding_text` |
| `node_level_information.json` | Post-process Stage 1 | Page descriptions for reranking |
| `edge_level_information.json` | Post-process Stage 2 | Live next-hop `high_level` / `low_level` hints |
| `node_navigation_plans.json` | Post-process Stage 5 | Optional: still used when building VLM rerank candidates |
| `ood_classifier.joblib` | `train_localizer.py` | On/off-graph OOD SVM |
| `siglip_smolvlm_features.pt` | `train_localizer.py` | Gallery for localization (`dim=1344`) |
| `screenshots/{node_id}.jpg` | Exploration | Gradio / GUI candidate cards, debugging |

If retrieval artifacts are missing, candidate pick may fail. If localizer artifacts are missing or `gallery_z` has the wrong dim, on-device navigation fails at load / first step.

---

## Dependencies

Run from `inference/` with the same Python environment as exploration. Key packages used by `inference.py`:

- `dynaconf` — config loading
- `FlagEmbedding` — BGE-M3 (stage 1)
- `gradio` — candidate picker
- `networkx` — graph depths / shortest paths
- `dashscope` — VLM rerank (via `VLM.py`)
- `scikit-learn` / `joblib` — OOD classifier
- `transformers` / `torch` — SigLIP + SmolVLM encoders

Device control and agents are provided under `Driver/` and `Agents/` in this directory.

**GUI (`gui_demo/`)** additionally requires packages in [`gui_demo/requirements-gui.txt`](gui_demo/requirements-gui.txt) (`fastapi`, `uvicorn`, `pyyaml`) and a Node.js build of `gui_demo/web/`. See [Inference GUI](#inference-gui-demo-wizard).

---

## Troubleshooting

| Issue | Check |
|-------|--------|
| `No root node found with is_root=True` | `graph.json` must mark at least one node with `is_root: true` |
| Gradio opens but no images | `screenshots/{node_id}.jpg` paths under `logs.root` |
| Agent does nothing / wrong screen | `reset_instruction`, `appPackage` / `appActivity`, agent `url` reachable |
| Agent reasoning OK but taps miss UI | MAI-UI server likely on **vLLM ≥ 0.2** (e.g. 0.21.x) — use **vllm==0.11.0**; run the [grounding smoke test](#mai-ui-server-vllm) |
| Retrieval picks wrong page | Increase `top_k_in_first_stage_retrieval`, try a different `pick_candidate_index`, or use Gradio (`-1`) |
| Wrong page but 2nd candidate looks right | Re-run with `pick_candidate_index: 1` (or higher) |
| `feature dim mismatch` / localizer load error | Re-run `train_localizer.py` so `siglip_smolvlm_features.pt` has `dim=1344` |
| Agent gets no hints / always off-graph | Check OOD model + gallery under `logs.root`; confirm screenshots match `target_hw` |
| Batch mode: `input_dir is required` | Set `input_dir` to the dataset root |
| Batch mode: empty output | Set `output_dir`; check task folders contain `prompts.json` |
| Skip retrieval entirely | Set `use_memory_for_navigation: false` |

---

## Related docs

- [Exploration README](../exploration/README.md) — how `user_intents.json`, `node_level_information.json`, and `node_navigation_plans.json` are produced
- GUI launcher: [`run_gui.sh`](run_gui.sh) · backend: [`gui_demo/inference_gui.py`](gui_demo/inference_gui.py) · pipeline: [`gui_demo/pipeline.py`](gui_demo/pipeline.py)
- Config templates: `inference/configs/*.yaml`
