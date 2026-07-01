# AppAgent baseline

Runs the **AppAgent** (CHI 2025) deployment policy on our navigation benchmark
tasks. AppAgent keeps a memory of **per-element natural-language docs** (one doc
file per widget, keyed by resource-id) and, at inference, pastes the docs for the
widgets visible on the current screen into its prompt. Our adapter
(`run_benchmark.py`) drives the device, feeds our task prompts, keeps AppAgent's
action space and prompts, and strips its interactive CLI.

## Prerequisites

- Python 3.10+, `adb` on `PATH`, and a running Android emulator/device with the
  target app **installed and logged in**.
- An upstream AppAgent checkout (see below) and its Python deps installed.
- AppAgent's model + API keys configured in the upstream `config.yaml`
  (this adapter reads them from there; it stores no keys).
- Navigation task folders under `TAGNAV_TASKS_ROOT` — one subdir per app named by
  `APP_REGISTRY[app]["tasks_dir"]`; each task is a folder with
  `prompts.json` → `{"prompts": ["<goal>"]}`.

## 1. Clone the upstream model

```bash
cd baselines/external
git clone https://github.com/TencentQQGYLab/AppAgent.git AppAgent
cd AppAgent && pip install -r requirements.txt    # per upstream README
```

## 2. Configure

```bash
cp baselines/.env.example baselines/.env
# edit baselines/.env:  APPAGENT_ROOT, TAGNAV_TASKS_ROOT, ANDROID_DEVICE_SERIAL
set -a; source baselines/.env; set +a
```

Set AppAgent's model + keys in `baselines/external/AppAgent/config.yaml`
(`MODEL: OpenAI|Qwen`, `OPENAI_API_KEY` / `DASHSCOPE_API_KEY`, etc.) per its README.

## 3. (Required for the docs-memory variant) generate AppAgent's element docs

AppAgent's memory is the per-element docs under
`baselines/external/AppAgent/apps/<app>/auto_docs/` (autonomous exploration) or
`demo_docs/` (human demonstration). **These are not produced by this adapter** —
generate them first with AppAgent's own tooling (`scripts/self_explorer.py` for
autonomous, or `scripts/step_recorder.py` + `scripts/document_generation.py` for
demos), per the upstream README. Without them, run with `--docs none` (AppAgent
runs doc-free — a valid but weaker baseline; note which mode you used).

## 4. Run

```bash
python baselines/appagent/run_benchmark.py \
    --app amazon \
    --device "$ANDROID_DEVICE_SERIAL" \
    --max-steps 15 \
    --docs auto \
    --out-dir results/appagent/amazon
```

Per task you get a `summary.json` (+ step log, labeled screenshots, final
`agent_screenshot.png`); the run writes `all_summaries.json` and prints the
`finished_action` rate. Use `--task-id <id>` (repeatable) or `--limit N` to scope.

## 5. Judge

`finished_action` is self-reported and over-counts success — judge against ground
truth. Reuse the shared review workflow (final screenshot vs. GT), the same way
the other baselines are judged; there is no AppAgent-specific judge.

## Required environment

| Var | Meaning |
|-----|---------|
| `APPAGENT_ROOT` | Path to the upstream AppAgent checkout (abs, or relative to repo root) |
| `TAGNAV_TASKS_ROOT` | Root of per-app navigation task folders (or `--tasks-root`) |
| `ANDROID_DEVICE_SERIAL` | adb serial of the target device (or `--device`) |

## Known prerequisite / gotcha

- **Element docs (`auto_docs`/`demo_docs`) must be pre-generated** with AppAgent's
  own tooling for the `--docs auto|demo` memory variant; this adapter only consumes
  them. `--docs none` runs without memory.
