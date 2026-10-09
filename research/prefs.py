"""Settings picked in the UI's menu bar (research mode), kept across reloads and restarts.

This is a single-user app on localhost, so the choice is one per PC rather than per tab.
"""
from __future__ import annotations

import json
import threading

from .config import ROOT

PATH = ROOT / ".cache" / "prefs.json"
MODES = ("basic", "parallel")
_lock = threading.Lock()


def load() -> dict:
    try:
        data = json.loads(PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"mode": data.get("mode") if data.get("mode") in MODES else "basic"}


def save(**changes) -> dict:
    with _lock:
        data = load() | {k: v for k, v in changes.items() if v is not None}
        if data["mode"] not in MODES:
            raise ValueError(f"unknown mode: {data['mode']}")
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(data), encoding="utf-8")
        return data


def mode() -> str:
    return load()["mode"]
