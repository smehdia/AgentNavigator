#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIGS=(
  # configs/alibaba_harmony.yaml
  # configs/calendar_android.yaml
  # configs/chrome_android.yaml
  # configs/clock_android.yaml
  # configs/ebay_android.yaml
  # configs/google_maps_android.yaml
  # configs/linkedin_android.yaml
  # configs/outlook_android.yaml
  # configs/target_android.yaml
  # configs/youtube_android.yaml
  # configs/airbnb_android.yaml
  # configs/tiktok_android.yaml
  # configs/uber_android.yaml
  # configs/uber_eats_android.yaml
  # configs/walmart_android.yaml
  # configs/starbucks_android.yaml
  # configs/target_android.yaml
  # configs/settings_android.yaml
  # configs/spotify_android.yaml
  configs/temu_android.yaml
)

for config in "${CONFIGS[@]}"; do
  echo "=== Post-processing: $config ==="
  CONFIG="$config" ./run_post_process.sh
done
