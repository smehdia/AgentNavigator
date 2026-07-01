#!/usr/bin/env bash
# Run the GUI-Explorer benchmark across the app subset.
#
# Assumes the GUI-Explorer retrieval/embedding server is already running and
# reachable (GUI_EXPLORER_RAG_URL), and that GUI_EXPLORER_ROOT, TAGNAV_TASKS_ROOT,
# OPENAI_API_KEY and the device serial are configured in baselines/.env (or the
# environment). See baselines/gui_explorer/README.md.
#
# No-cost check:
#   DRY_RUN=1 bash baselines/gui_explorer/run_all_apps.sh
#
# Released-KB app subset:
#   APPS="airbnb amazon calendar chrome clock ebay google_maps instagram settings tiktok youtube" \
#   bash baselines/gui_explorer/run_all_apps.sh
#
# Include LinkedIn only to evaluate the missing-KB case (no KB entry upstream):
#   APPS="... linkedin ..." bash baselines/gui_explorer/run_all_apps.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load baselines/.env if present (does not override already-set vars).
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

PYTHON="${PYTHON:-python}"
SCRIPT="$SCRIPT_DIR/run_benchmark.py"
DEVICE="${ANDROID_DEVICE_SERIAL:-emulator-5554}"
OUT_BASE="${OUT_BASE:-results/gui_explorer}"
MAX_STEPS="${MAX_STEPS:-15}"
DRY_RUN="${DRY_RUN:-0}"

# Default excludes LinkedIn because the released GUI-Explorer KB has no
# com.linkedin.android entry. Add it explicitly via APPS=... if needed.
APPS="${APPS:-airbnb amazon calendar chrome clock ebay google_maps instagram settings tiktok youtube}"

echo "=== GUI-Explorer Benchmark (all apps) ==="
echo "Started: $(date)"
echo "Device: $DEVICE"
echo "Output: $OUT_BASE"
echo "Max steps: $MAX_STEPS"
echo "Apps: $APPS"
echo "Dry run: $DRY_RUN"
echo ""

mkdir -p "$OUT_BASE"

for app in $APPS; do
    echo "--- $app --- $(date)"
    cmd=(
        "$PYTHON" "$SCRIPT"
        --app "$app" \
        --device "$DEVICE" \
        --max-steps "$MAX_STEPS" \
        --out-dir "$OUT_BASE/$app"
    )
    printf 'Command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    if [ "$DRY_RUN" = "1" ]; then
        echo "--- $app dry-run only ---"
        echo ""
        continue
    fi
    "${cmd[@]}" 2>&1 | tee "$OUT_BASE/${app}.log"
    echo "--- $app done --- $(date)"
    echo ""
done

echo "=== All done: $(date) ==="

echo ""
echo "=== RESULTS SUMMARY ==="
"$PYTHON" "$SCRIPT_DIR/summarize.py" --roots "$OUT_BASE" --apps $APPS --include-missing
