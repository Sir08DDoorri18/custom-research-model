"""Split documents into passages and rank them with BM25 (keyword overlap, no model needed)."""
from __future__ import annotations

import math
import re
from collections import Counter

_WORD = re.compile(r"[\w가-힣]+")
_STOP = set("the a an of and or in on to for with by from is are was were be been as at that this these those "
            "it its we our their which using use used can may also than into between".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def chunk(text: str, size: int = 1200, overlap: int = 200) -> list[str]:
    """Passages of about `size` characters, cut at sentence ends where possible."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start + size // 2, end), text.rfind("\n", start + size // 2, end))
            if cut > start:
                end = cut + 1
        piece = text[start:end].strip()
        if len(piece) > 80:
            out.append(piece)
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)
        space = text.find(" ", nxt, end)  # start the next passage on a word boundary
        start = space + 1 if space != -1 else nxt
    return out


def bm25(query: str, passages: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    docs = [tokens(p) for p in passages]
    if not docs:
        return []
    avg = sum(map(len, docs)) / len(docs) or 1
    df = Counter(w for d in docs for w in set(d))
    n = len(docs)
    q = set(tokens(query))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for w in q:
            if w in tf:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores
