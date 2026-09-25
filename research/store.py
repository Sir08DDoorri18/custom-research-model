"""On-disk cache of fetched documents, so a source is downloaded once per machine.

Every fetched page/PDF gets a short doc_id that tools pass around, which keeps
long texts out of the conversation until they are actually needed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "docs"
MAX_AGE = 30 * 24 * 3600  # re-fetch a page after 30 days


def doc_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def _path(did: str) -> Path:
    return CACHE / f"{did}.json"


def get(url_or_id: str) -> dict | None:
    did = url_or_id if _looks_like_id(url_or_id) else doc_id(url_or_id)
    p = _path(did)
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - rec.get("fetched_at", 0) > MAX_AGE:
        return None
    return rec


def put(url: str, text: str, title: str = "", via: str = "") -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    rec = {"doc_id": doc_id(url), "url": url, "title": title, "via": via,
           "chars": len(text), "fetched_at": time.time(), "text": text}
    _path(rec["doc_id"]).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    return rec


def _looks_like_id(s: str) -> bool:
    return len(s) == 12 and all(c in "0123456789abcdef" for c in s)
