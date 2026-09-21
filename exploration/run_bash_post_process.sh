#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Use app_navigation env with deps installed; ignore ~/.local packages.
export PYTHONNOUSERSITE=1
if [ -x "/home/mehdi/anaconda3/envs/app_navigation/bin/python" ]; then
  PYTHON="/home/mehdi/anaconda3/envs/app_navigation/bin/python"
elif [ -x "${CONDA_PREFIX:-}/bin/python" ] && [[ "${CONDA_DEFAULT_ENV:-}" == "app_navigation" ]]; then
  PYTHON="${CONDA_PREFIX}/bin/python"
elif [ -x "/home/mehdi/miniforge3/envs/app_navigation/bin/python" ]; then
  PYTHON="/home/mehdi/miniforge3/envs/app_navigation/bin/python"
else
  PYTHON="python3"
fi
echo "Using Python: $PYTHON"

ROOT_DIR="explored_apps"

# Post-process when graph.json exists but any of these artifacts is missing:
#   user_intents.json, edge_level_information.json, node_level_information.json
needs_post_process() {
  local root="$1"
  [ -d "$root" ] || return 1
  [ -f "$root/graph.json" ] || return 1
  [ ! -f "$root/user_intents.json" ] \
    || [ ! -f "$root/edge_level_information.json" ] \
    || [ ! -f "$root/node_level_information.json" ]
}

config_root() {
  "$PYTHON" -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$1'], merge_enabled=True).default; print(c.logs.root)"
}

run_post_process() {
  local cfg="$1"
  echo "[post_process] START $cfg"
  if "$PYTHON" post_process.py --config "$cfg"; then
    echo "[post_process] SUCCESS $cfg"
  else
    echo "[post_process] FAILED $cfg (continuing)" >&2
  fi
}

# Train when screenshots exist but either localizer artifact is missing:
#   siglip_smolvlm_features.pt, ood_classifier.joblib
needs_train() {
  local app_dir="$1"
  [ -d "$app_dir/screenshots" ] || return 1
  [ -n "$(ls -A "$app_dir/screenshots" 2>/dev/null)" ] || return 1
  [ ! -f "$app_dir/siglip_smolvlm_features.pt" ] \
    || [ ! -f "$app_dir/ood_classifier.joblib" ]
}

run_train() {
  local app_dir="$1"
  echo "[train_localizer] START $app_dir"
  if "$PYTHON" train_localizer.py --app_dir "$app_dir" --root_dir "$ROOT_DIR"; then
    echo "[train_localizer] SUCCESS $app_dir"
  else
    echo "[train_localizer] FAILED $app_dir (continuing)" >&2
  fi
}

shopt -s nullglob

post_cfgs=()
for CONFIG in configs/*.yaml; do
  ROOT=$(config_root "$CONFIG") || continue
  if needs_post_process "$ROOT"; then
    post_cfgs+=("$CONFIG")
  fi
done

echo "Post-process ${#post_cfgs[@]} configs whose explored app is missing user_intents / edge / node JSONs (continue on failure)"
for CONFIG in "${post_cfgs[@]}"; do
  run_post_process "$CONFIG"
done

train_apps=()
for app_dir in "$ROOT_DIR"/*/; do
  app_dir="${app_dir%/}"
  if needs_train "$app_dir"; then
    train_apps+=("$app_dir")
  fi
done

echo "Train localizer for ${#train_apps[@]} apps missing siglip_smolvlm_features.pt or ood_classifier.joblib (continue on failure)"
for app_dir in "${train_apps[@]}"; do
  run_train "$app_dir"
done
