#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

for CONFIG in configs/etsy_android.yaml configs/nike_android.yaml configs/ticktick_android.yaml configs/google_playbooks_android.yaml configs/zoho_meeting_android.yaml configs/deepl_android.yaml configs/likee_android.yaml; do
  ROOT=$(python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$CONFIG'], merge_enabled=True).default; print(c.logs.root)")
  mkdir -p "$ROOT"
  python -u explore.py --config "$CONFIG" 2>&1 | tee "$ROOT/explore.log"
done