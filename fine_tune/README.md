# Fine-tuning

Scripts in this directory turn post-processed exploration artifacts into a deployable MAI-UI model:

```
post_process → prepare_data.py → train_lora.py → merge_model_for_deployment.py → vLLM
```

`prepare_data.py` converts exploration artifacts into MAI-UI SFT chat samples (JSONL). Each row is a multimodal conversation: system prompt, user instruction, screenshot, and assistant response (`<thinking>` + `<tool_call>`).

## Usage

```bash
python fine_tune/prepare_data.py \
  --root exploration/explored_apps/amazon \
  --grounding-ratio 1.0 \
  --seed 0
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--root` | *(required)* | Explored app directory (see inputs below) |
| `--output` | `<root>/training_data.jsonl` | Output JSONL path |
| `--grounding-ratio` | `1.0` | Fraction of navigation samples to use as grounding target (0.0–1.0), capped at number of edge actions |
| `--seed` | `0` | Random seed for grounding text/edge selection |

## Input layout (`--root`)

All paths are relative to the explored app root (typically `exploration/explored_apps/<app>/`):

| File / dir | Required | Used for |
|------------|----------|----------|
| `agent_data.json` | yes | Navigation samples |
| `user_intents.json` | yes | Terminate samples |
| `screenshots/` | yes | Screenshot for every sample |
| `graph.json` | grounding only | Edge actions (`type`, `boundingBox`, `description`) |
| `edge_level_information.json` | grounding only | Low/high-level action descriptions per edge |

These files are produced by `exploration/post_process.py` (Stages 2, 4, and 7).

## Output format

Each line in `training_data.jsonl` is a JSON object:

```json
{
  "messages": [
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {"role": "user", "content": [{"type": "text", "text": "<user instruction>"}]},
    {"role": "user", "content": [{"type": "image", "image": "/abs/path/to/screenshot.jpg"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "<thinking>...</thinking>\n<tool_call>...</tool_call>"}]}
  ],
  "metadata": {
    "app_graph_path": "<root>",
    "target_node": "...",
    "source_node": "...",
    "next_node": "...",
    "edge_key": "source|target",
    "user_intent": "...",
    "sample_type": "navigation | grounding | terminate"
  }
}
```

The system prompt is the verbatim MAI-UI system prompt from `inference/Agents/MAI_UI/prompt.py`.

---

## Algorithm overview

```
┌─────────────────────────────────────────────────────────────┐
│  1. Navigation  ← agent_data.json                           │
│  2. Terminate   ← user_intents.json                         │
│  3. Grounding   ← graph.json + edge_level_information.json  │
│                 (optional, controlled by --grounding-ratio) │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    training_data.jsonl
```

Processing order:

1. **Navigation** — one sample per `(target_node, hop, user_instruction)` in `agent_data.json`
2. **Terminate** — one sample per `(target_node, user_intent)` in `user_intents.json`
3. **Grounding** — subsample edge actions up to `min(navigation × ratio, num_edges)`

Navigation and terminate samples are deduplicated. Grounding samples are not deduplicated against navigation/terminate.

---

## Sample type 1: Navigation (planning)

**Source:** `agent_data.json` (post-process Stage 7)

**Purpose:** Teach the model to take the next step along a path toward a destination screen, given a high-level user goal.

### `agent_data.json` layout

```json
{
  "target_node_id": {
    "hop_source|target_node_id": [
      {
        "user_instruction": "Navigate to the account dashboard page",
        "agent_output": "<thinking>...</thinking>\n<tool_call>\n{\"name\": \"mobile_use\", \"arguments\": {...}}\n</tool_call>"
      }
    ]
  }
}
```

- Keyed by **destination node**, then by **hop** `source_on_path|destination_node`.
- Each entry pairs a destination user intent with a full MAI-UI assistant response for that hop.

### Construction

For each entry in `agent_data.json`:

| Field | Source |
|-------|--------|
| **User instruction** | `user_instruction` (destination screen user intent) |
| **Screenshot** | `screenshots/{source_node}.jpg` (current hop screen) |
| **Thinking** | Parsed from `agent_output` — VLM-generated during post-processing |
| **Tool call** | Parsed from `agent_output` — action + **already normalized** coordinates `[0–999]` |

The script validates `agent_output` contains a parseable `<tool_call>` with a JSON `arguments` object. Invalid entries are skipped.

### Thought construction (navigation)

Thoughts are **not generated** by `prepare_data.py`. They come directly from the VLM output stored in `agent_data.json` (Stage 7 `get_agent_thought`). Example:

```
<thinking>
Tapping the You tab is the necessary first step to access the account dashboard.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [298, 907]}}
</tool_call>
```

---

## Sample type 2: Terminate

**Source:** `user_intents.json` (post-process Stage 4)

**Purpose:** Teach the model to recognize when the current screen already satisfies the user's goal and emit `terminate`.

### Construction

For each `user_intent` under each node in `user_intents.json`:

| Field | Source |
|-------|--------|
| **User instruction** | `user_intent` string |
| **Screenshot** | `screenshots/{target_node}.jpg` (destination screen) |
| **Thinking** | Fixed template (see below) |
| **Tool call** | Fixed: `{"action": "terminate", "status": "success"}` |

`source_node`, `next_node`, and `target_node` are all set to the same node. `edge_key` is `"terminate"`.

### Thought construction (terminate)

Fixed string — not VLM-generated:

```
<thinking>
The screen already matches the user's goal, so I will terminate with success.
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
```

---

## Sample type 3: Grounding

**Source:** `graph.json` + `edge_level_information.json` (post-process Stage 2)

**Purpose:** Teach the model to locate and interact with a specific UI element from a natural-language description, independent of a navigation goal.

### Data sources per edge action

| Data | File | Field |
|------|------|-------|
| Action type, bbox, short description | `graph.json` | `type`, `boundingBox`, `description` |
| Low-level grounding text | `edge_level_information.json` | `low_level_action_description` |
| High-level functional text | `edge_level_information.json` | `high_level_action_description` |
| Action ID link | `edge_level_information.json` | `action_ids[0]` → graph edge `key` |

Graph `boundingBox` values are in **raw screen pixels**. Coordinates in the tool call are normalized to `[0–999]` at prepare time.

### Candidate collection

For each entry in `edge_level_information.json`:

1. Parse `edge_key` as `source_node|next_node`
2. Match `action_ids[0]` to the corresponding action in `graph.json` (by `key`)
3. Build a **text pool** from up to three non-empty strings:
   - `low_level_action_description`
   - `high_level_action_description`
   - graph edge `description`
4. Normalize bbox center: `x = round(cx / width × 999)`, `y = round(cy / height × 999)`
5. Map action type → MAI-UI arguments:
   - `scroll` → `{"action": "swipe", "direction": "down", "coordinate": [...]}`
   - `swipe` → `{"action": "swipe", "direction": "left", "coordinate": [...]}`
   - other → `{"action": "click", "coordinate": [...]}`
6. If no bbox but action is scroll/swipe, fallback coordinate is `[499, 499]`

Each valid edge action becomes one **candidate**.

### Subsampling (`--grounding-ratio`)

```
target = min(round(navigation_count × grounding_ratio), num_candidates)
```

- `grounding_ratio = 0.0` → no grounding samples
- `grounding_ratio = 1.0` → up to one grounding sample per edge action, capped at navigation count
- If `navigation × ratio > num_edges`, only `num_edges` samples are created (no duplicate edges)

Candidates are selected **without replacement** via `rng.sample`. For each selected candidate, instruction and thought are drawn **independently** from the text pool.

### Thought construction (grounding)

| Field | Source |
|-------|--------|
| **User instruction** | `rng.choice(text_pool)` |
| **Thinking** | `rng.choice(text_pool)` — independent draw, can differ from instruction |
| **Tool call** | Built from graph edge type + normalized coordinate |

The selected thought text is placed verbatim inside `<thinking>`:

```
<thinking>
{thought}
</thinking>
<tool_call>
{"name":"mobile_use","arguments":{...}}
</tool_call>
```

### Example combinations

Instruction and thought can come from different pool members:

| Instruction (user) | Thought (assistant) |
|--------------------|---------------------|
| low_level: *Tap the Home icon…* | high_level: *Navigate to the home page.* |
| graph desc: *Home icon in bottom navigation bar.* | high_level: *Navigate to the home page.* |
| high_level: *Open the shopping cart.* | high_level: *Open the shopping cart.* |

---

## Coordinate normalization

Graph bboxes are pixel coordinates `[x1, y1, x2, y2]`. Normalization uses the source screenshot dimensions:

```
cx = (x1 + x2) / 2
cy = (y1 + y2) / 2
x_norm = clamp(round(cx / width  × 999), 0, 999)
y_norm = clamp(round(cy / height × 999), 0, 999)
```

Navigation samples already have normalized coordinates in `agent_data.json` (normalized during post-process Stage 7). Only grounding samples are normalized in `prepare_data.py`.

---

## Summary: where thoughts come from

| Sample type | User instruction | Thinking | Tool call / coordinates |
|-------------|------------------|----------|-------------------------|
| **Navigation** | `user_instruction` from `agent_data.json` | VLM output in `agent_data.json` | VLM output in `agent_data.json` (pre-normalized) |
| **Terminate** | `user_intent` from `user_intents.json` | Fixed template | Fixed `terminate` action |
| **Grounding** | Random from text pool | Random from text pool (independent) | Built from `graph.json` edge (normalized here) |

**Text pool (grounding):** `{low_level_action_description, high_level_action_description, graph edge description}`

---

## Prerequisites

Run exploration and post-processing first:

```bash
# 1. Explore app graph
cd exploration && CONFIG=configs/amazon_android.yaml ./run_explore.sh

# 2. Post-process (produces agent_data.json, user_intents.json, graph.json, etc.)
cd exploration && CONFIG=configs/amazon_android.yaml ./run_post_process.sh

# 3. Prepare fine-tuning data
python fine_tune/prepare_data.py --root exploration/explored_apps/amazon

# 4. Train QLoRA adapter
python fine_tune/train_lora.py \
  --data exploration/explored_apps/amazon/training_data.jsonl \
  --output_dir ./mai-ui-qlora

# 5. Merge base model + adapter for vLLM
python fine_tune/merge_model_for_deployment.py \
  --base_model fine_tune/MAI-UI-2B \
  --adapter_path ./mai-ui-qlora \
  --output_dir ./mai-ui-merged
```

---

## Merge for deployment (`merge_model_for_deployment.py`)

`train_lora.py` saves a **LoRA adapter** (not a full model). Before serving with vLLM, merge the adapter into the base weights:

```bash
python fine_tune/merge_model_for_deployment.py \
  --base_model fine_tune/MAI-UI-2B \
  --adapter_path ./mai-ui-qlora \
  --output_dir ./mai-ui-merged
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--base_model` | `fine_tune/MAI-UI-2B` | Base model path or HuggingFace id |
| `--adapter_path` | *(required)* | LoRA output directory from `train_lora.py` |
| `--output_dir` | *(required)* | Where to write the merged full-precision checkpoint |
| `--dtype` | `bf16` | Merged weight dtype: `bf16`, `fp16`, or `fp32` |

The script loads the base `Qwen3VLForConditionalGeneration` in full precision, applies the LoRA weights with `merge_and_unload()`, and writes a HuggingFace checkpoint (weights + processor) suitable for vLLM.

Serve the merged model:

```bash
vllm serve ./mai-ui-merged --dtype bfloat16
```

Point `agent.url` in your inference config at this vLLM endpoint (see `inference/README.md` for MAI-UI server notes).
