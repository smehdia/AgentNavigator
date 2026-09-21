#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Run post_process when the explored app root is missing any of:
#   user_intents.json, edge_level_information.json, node_level_information.json
# Requires graph.json so there is something to process.

needs_post_process() {
  local root="$1"
  [ -d "$root" ] || return 1
  [ -f "$root/graph.json" ] || return 1
  [ ! -f "$root/user_intents.json" ] \
    || [ ! -f "$root/edge_level_information.json" ] \
    || [ ! -f "$root/node_level_information.json" ]
}

config_root() {
  python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$1'], merge_enabled=True).default; print(c.logs.root)"
}

run_one() {
  local CONFIG="$1"
  local ROOT
  ROOT=$(config_root "$CONFIG")

  if ! needs_post_process "$ROOT"; then
    echo "Skipping $CONFIG (already has user_intents / edge_level_information / node_level_information, or no graph.json)"
    return 0
  fi

  echo "Post-processing $CONFIG -> $ROOT"
  mkdir -p "$ROOT"
  python -u post_process.py --config "$CONFIG" 2>&1 | tee "$ROOT/post_process.log"
}

CONFIG_ARG="${1:-${CONFIG:-}}"

if [ -n "$CONFIG_ARG" ]; then
  run_one "$CONFIG_ARG"
else
  echo "No config given; post-processing explored apps missing user_intents / edge / node JSONs"
  for CONFIG in configs/*.yaml; do
    ROOT=$(config_root "$CONFIG")
    if needs_post_process "$ROOT"; then
      run_one "$CONFIG"
    fi
  done
fi
