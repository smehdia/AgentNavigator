#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

for CONFIG in configs/calendar_harmony.yaml configs/dropbox_harmony.yaml configs/game_center_harmony.yaml confings/notepad_harmony.yaml configs/weather_harmony.yaml; do
  ROOT=$(python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$CONFIG'], merge_enabled=True).default; print(c.logs.root)")
  mkdir -p "$ROOT"
  python -u explore.py --config "$CONFIG" 2>&1 | tee "$ROOT/explore.log"
done