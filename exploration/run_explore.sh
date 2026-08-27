#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export FORCE_COLOR=1
CONFIG="${1:-${CONFIG:-}}"
if [ -z "$CONFIG" ]; then
  echo "Usage: bash run_explore.sh <config.yaml>" >&2
  echo "   or: CONFIG=configs/foo.yaml bash run_explore.sh" >&2
  exit 1
fi

ROOT=$(python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$CONFIG'], merge_enabled=True).default; print(c.logs.root)")
mkdir -p "$ROOT"
python -u explore.py --config "$CONFIG" 2>&1 | tee "$ROOT/explore.log"
