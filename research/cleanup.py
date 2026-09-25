"""Keeps the caches from growing without bound. Limits come from `storage` in models.yaml.

- .cache/files  read files: least recently used removed once the total passes files_max_mb
- .cache/docs   fetched pages: removed after docs_days
- traces/       logs: removed after traces_days
- .cache/tmp    scratch space of the PDF reader: emptied when older than a day
The local models in .cache/hf are left alone (they don't grow and are costly to re-download).
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

from . import config
from .config import ROOT

FILES = ROOT / ".cache" / "files"
DOCS = ROOT / ".cache" / "docs"
TMP = ROOT / ".cache" / "tmp"
TRACES = ROOT / "traces"
DAY = 86400


def prune(files_max_mb: float | None = None, docs_days: float | None = None,
          traces_days: float | None = None) -> str:
    """Apply the limits; arguments override models.yaml. Returns a one-line summary."""
    cfg = config.load().get("storage") or {}
    files_max = (files_max_mb if files_max_mb is not None else cfg.get("files_max_mb", 2000)) * 1024 * 1024
    docs_age = (docs_days if docs_days is not None else cfg.get("docs_days", 30)) * DAY
    traces_age = (traces_days if traces_days is not None else cfg.get("traces_days", 30)) * DAY

    freed = {"files": _prune_files(files_max), "docs": _older_than(DOCS, "*.json", docs_age),
             "traces": _older_than(TRACES, "*.md", traces_age), "tmp": _older_than(TMP, "*", DAY)}
    return " · ".join(f"{k} {_mb(v)}" for k, v in freed.items()) + " 정리됨"


def usage() -> str:
    parts = {"files": FILES, "docs": DOCS, "traces": TRACES, "models": ROOT / ".cache" / "hf"}
    return " · ".join(f"{k} {_mb(_size(p))}" for k, p in parts.items())


def _prune_files(limit: int) -> int:
    if not FILES.exists():
        return 0
    freed = 0
    entries = []
    for folder in FILES.iterdir():
        if not folder.is_dir():
            continue
        done = folder / "result.json"
        if not done.exists() and time.time() - folder.stat().st_mtime > DAY:
            freed += _remove(folder)  # a read that crashed and never finished
        elif done.exists():
            entries.append((done.stat().st_mtime, _size(folder), folder))  # mtime = last use
    total = sum(size for _, size, _ in entries)
    for _, size, folder in sorted(entries):  # oldest use first
        if total <= limit:
            break
        freed += _remove(folder)
        total -= size
    return freed


def _older_than(folder: Path, pattern: str, age: float) -> int:
    if not folder.exists():
        return 0
    cutoff = time.time() - age
    return sum(_remove(p) for p in folder.glob(pattern) if p.stat().st_mtime < cutoff)


def _remove(path: Path) -> int:
    size = _size(path)
    try:
        shutil.rmtree(path) if path.is_dir() else path.unlink()
    except OSError:  # in use (e.g. a trace being written): skip, try next time
        return 0
    return size


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f}MB"
