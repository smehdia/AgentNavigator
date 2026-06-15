#!/usr/bin/env bash
# Start the inference GUI. Must be run from repo or inference dir.
set -e
cd "$(dirname "$0")"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy \
      ALL_PROXY all_proxy SOCKS_PROXY SOCKS5_PROXY \
      socks_proxy socks5_proxy FTP_PROXY ftp_proxy 2>/dev/null || true

WEB_DIR="gui_demo/web"
echo "Building frontend ($WEB_DIR)..."
(cd "$WEB_DIR" && npm run build)

exec uvicorn gui_demo.inference_gui:app --host 0.0.0.0 --port 8765 "$@"
