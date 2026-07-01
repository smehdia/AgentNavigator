#!/usr/bin/env python3
"""Shared constants and helpers for the SPA-Bench pilot subset."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from urllib.request import urlopen


SPA_BENCH_SINGLE_APP_ENG_URL = (
    "https://raw.githubusercontent.com/ai-agents-2030/SPA-Bench/main/data/single-app-ENG.csv"
)

PILOT_TASK_IDS = [
    "airbnb_2",
    "amazon_2",
    "calendar_2",
    "chrome_2",
    "clock_2",
    "clock_5",
    "google_maps_2",
    "google_maps_5",
    "youtube_2",
    "youtube_5",
]

RETRIEVAL_PROMPTS = {
    "airbnb_2": "Navigate to the wishlist page in Airbnb after saving a stay listing.",
    "amazon_2": "Navigate to the Amazon search results page for goggles with filters available.",
    "calendar_2": "Navigate to the event creation/edit screen in Calendar.",
    "chrome_2": "Navigate to the Reading List page in Chrome bookmarks.",
    "clock_2": "Navigate to the alarm creation or alarm edit screen in Clock.",
    "clock_5": "Navigate to the Clock settings page for world clock style and screen saver style.",
    "google_maps_2": "Navigate to nearby hotel search results with filter controls in Google Maps.",
    "google_maps_5": "Navigate to the directions planning screen with stop and destination controls in Google Maps.",
    "youtube_2": "Navigate to the YouTube subscriptions page with sorting controls.",
    "youtube_5": "Navigate to YouTube video search results for LeBron James with duration filters.",
}

# Example mapping from SPA-Bench app -> your exported task dir + auto-BFS graph pickle.
# Edit `tasks_dir` / `graph_pkl` to match the navigation data you exported for each app.
APP_CONFIG = {
    "airbnb": {
        "tasks_dir": "airbnb_26.06",
        "graph_pkl": "explored_graphs/airbnb/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "amazon": {
        "tasks_dir": "amazon_32.3.0.100",
        "graph_pkl": "explored_graphs/amazon/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "calendar": {
        "tasks_dir": "calendar_2026.05.0-864040481-release",
        "graph_pkl": "explored_graphs/calendar/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "chrome": {
        "tasks_dir": "chrome_144.0.7559.132",
        "graph_pkl": "explored_graphs/chrome/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "clock": {
        "tasks_dir": "clock_8.5",
        "graph_pkl": "explored_graphs/clock/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "google_maps": {
        "tasks_dir": "google_maps_26.06.01.863982022",
        "graph_pkl": "explored_graphs/google_maps/node_information_dict_pure_visual_with_embeddings.pkl",
    },
    "youtube": {
        "tasks_dir": "youtube_21.05.264",
        "graph_pkl": "explored_graphs/youtube/node_information_dict_pure_visual_with_embeddings.pkl",
    },
}


def load_spa_csv_rows(csv_path: str | Path | None = None) -> list[dict[str, str]]:
    """Load SPA-Bench single-app ENG rows from a local file or the canonical URL."""
    if csv_path is None:
        text = urlopen(SPA_BENCH_SINGLE_APP_ENG_URL).read().decode("utf-8")
    else:
        text = Path(csv_path).read_text(encoding="utf-8")
    return list(csv.DictReader(text.splitlines()))


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
