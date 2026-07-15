#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

python3 post_process.py --config configs/vmall_harmony.yaml
python3 post_process.py --config configs/baidu_harmony.yaml
python3 post_process.py --config configs/didi_harmony.yaml
python3 post_process.py --config configs/books_harmony.yaml
python3 post_process.py --config configs/wallet_harmony.yaml
python3 post_process.py --config configs/vlc_harmony.yaml

