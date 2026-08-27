#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

CONFIG="${1:-${CONFIG:-}}"
if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_post_process.sh <config.yaml>" >&2
  echo "   or: CONFIG=configs/foo.yaml bash run_post_process.sh" >&2
  exit 1
fi

ROOT=$(python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$CONFIG'], merge_enabled=True).default; print(c.logs.root)")
mkdir -p "$ROOT"
python -u post_process.py --config "$CONFIG" 2>&1 | tee "$ROOT/post_process.log"
