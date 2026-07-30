#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy \
      ALL_PROXY all_proxy SOCKS_PROXY SOCKS5_PROXY \
      socks_proxy socks5_proxy FTP_PROXY ftp_proxy 2>/dev/null || true

python3 -u downloader.py "$@"
