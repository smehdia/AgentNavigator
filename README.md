# AgentNavigator

Explore a mobile app into a navigation graph, then run a natural-language goal on a device. Work happens in two directories:

| Directory | Role |
|-----------|------|
| [`exploration/`](exploration/README.md) | Grow the graph, post-process it, train the screenshot localizer |
| [`inference/`](inference/README.md) | Retrieve a target screen and drive the UI agent step by step |

Configs share a filename (`exploration/configs/<app>.yaml` and `inference/configs/<app>.yaml`). Exploration writes under `exploration/explored_apps/<app>/`. Inference reads that folder via `logs.root`.

## Before you start

- A phone or emulator on `adb devices` (Android) or `hdc list targets` (HarmonyOS).
- API keys in the YAML (`vlm.yibu_api_key` / `vlm.alibaba_api_key`).
- The MAI-UI server running. Download [`smehdia/maiui_8b_llamacpp_quantized`](https://huggingface.co/smehdia/maiui_8b_llamacpp_quantized/tree/main), then:

```bash
huggingface-cli download smehdia/maiui_8b_llamacpp_quantized --local-dir maiui_8b_llamacpp_quantized
cd maiui_8b_llamacpp_quantized
chmod +x start_server.sh
CUDA_VISIBLE_DEVICES=0 ./start_server.sh
curl http://localhost:8080/v1/models
```

Point `agent.url` at `http://localhost:8080/v1` and set `agent.model_name: mai_ui` in both configs. Linux x86_64, NVIDIA GPU, CUDA 12+, about 11 GB VRAM.

## 1. Check the app on the device

From `exploration/`, confirm launch, reset, and foreground package before a long run. `reset_instruction` should land on a stable start screen and return finished.

```bash
cd exploration
python check_driver.py --config configs/clock_android.yaml
```

Set `driver.device_id`, `appPackage`, and `appActivity`. On Android, set `use_launcher_intent: true` when `am start -n` fails (YouTube, Outlook). On HarmonyOS, `appActivity` is the entry ability from `bm dump`, and set `appModule` when the config has one.

## 2. Explore

```bash
cd exploration
bash run_explore.sh configs/clock_android.yaml
```

This writes `graph.json`, `screenshots/`, and `app_graph.pkl` under `logs.root` (for example `explored_apps/clock/`).

## 3. Post-process

Same config, after the graph exists:

```bash
cd exploration
bash run_post_process.sh configs/clock_android.yaml
```

This adds the files inference retrieves from: `node_level_information.json`, `edge_level_information.json`, `user_intents.json` (with embeddings), and `node_navigation_plans.json`.

## 4. Train the localizer

Sibling apps under `explored_apps/` supply off-graph screenshots. The target app needs its own `screenshots/` and `graph.json`.

```bash
cd exploration
python train_localizer.py --app_dir ./explored_apps/clock --root_dir ./explored_apps
```

Outputs `ood_classifier.joblib` and `siglip_smolvlm_features.pt` (`dim=1344`) in that app folder.

## 5. Navigate

Set `logs.root` in the inference config to the explored folder (for example `../exploration/explored_apps/clock`). Set `query` to the goal, or leave it empty and type it when asked.

GUI (demo wizard):

```bash
cd inference
pip install -r gui_demo/requirements-gui.txt
cd gui_demo/web && npm install && cd ../..
bash run_gui.sh
```

Open [http://localhost:8765](http://localhost:8765). Apply the config, run checks, connect the device, load resources, then send a goal.

CLI:

```bash
cd inference
python inference.py --config configs/clock_android.yaml
```

Inference retrieves a target screen (embeddings, then VLM rerank), localizes the live screenshot, and sends next-hop hints to MAI-UI until the goal is done or `agent.max_steps` is reached.

Details: [exploration](exploration/README.md), [inference](inference/README.md).
