#!/usr/bin/env bash
export FORCE_COLOR=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
  configs/sound_cloud_android.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "=== Exploring: $config ==="
  CONFIG="$config" ./run_explore.sh
done
