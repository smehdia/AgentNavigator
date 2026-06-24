#!/usr/bin/env bash
# Start the inference GUI. Must be run from repo or inference dir.
set -e
cd "$(dirname "$0")"

# Driver temp screenshots (e.g. __agentnav_*.jpeg) are written in this directory.

usage() {
  cat <<'EOF'
Usage: run_gui.sh [options] [uvicorn args...]

Options:
  --save-trajectory-mp4 [DIR]
                        After each navigation run, save an MP4 of the trajectory
                        with the action drawn on each frame. Default DIR:
                        ./trajectory_videos
  -h, --help            Show this help

Environment (when --save-trajectory-mp4 is set):
  AGENTNAV_TRAJECTORY_MP4_WIDTH   Frame width in pixels (default: 1920)
  AGENTNAV_TRAJECTORY_MP4_FPS      Playback FPS (default: 1)

Remaining arguments are passed to uvicorn (e.g. --reload --port 9000).
EOF
}

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy \
      ALL_PROXY all_proxy SOCKS_PROXY SOCKS5_PROXY \
      socks_proxy socks5_proxy FTP_PROXY ftp_proxy 2>/dev/null || true

SAVE_TRAJECTORY_MP4_DIR=""
UVICORN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-trajectory-mp4)
      if [[ -n "${2:-}" && "${2:0:1}" != "-" ]]; then
        SAVE_TRAJECTORY_MP4_DIR="$2"
        shift 2
      else
        SAVE_TRAJECTORY_MP4_DIR="./trajectory_videos"
        shift
      fi
      ;;
    --save-trajectory-mp4=*)
      SAVE_TRAJECTORY_MP4_DIR="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      UVICORN_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$SAVE_TRAJECTORY_MP4_DIR" ]]; then
  mkdir -p "$SAVE_TRAJECTORY_MP4_DIR"
  SAVE_TRAJECTORY_MP4_DIR="$(cd "$SAVE_TRAJECTORY_MP4_DIR" && pwd)"
  export AGENTNAV_SAVE_TRAJECTORY_MP4_DIR="$SAVE_TRAJECTORY_MP4_DIR"
  export AGENTNAV_TRAJECTORY_MP4_FPS="${AGENTNAV_TRAJECTORY_MP4_FPS:-1}"
  export AGENTNAV_TRAJECTORY_MP4_WIDTH="${AGENTNAV_TRAJECTORY_MP4_WIDTH:-1920}"
  echo "Trajectory MP4 export enabled → $SAVE_TRAJECTORY_MP4_DIR (${AGENTNAV_TRAJECTORY_MP4_FPS} fps, ${AGENTNAV_TRAJECTORY_MP4_WIDTH}px wide)"
fi

WEB_DIR="gui_demo/web"
echo "Building frontend ($WEB_DIR)..."
(cd "$WEB_DIR" && npm run build)

exec uvicorn gui_demo.inference_gui:app --host 0.0.0.0 --port 8765 "${UVICORN_ARGS[@]}"
