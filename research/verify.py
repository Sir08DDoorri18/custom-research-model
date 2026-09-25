"""Check that a quote an agent claims to have read actually appears in the source.

This is plain string matching, not an LLM judgement: an agent can't talk its way
past it. A hallucinated quote or URL fails here and gets dropped.
"""
from __future__ import annotations

import re
import threading
import unicodedata

_TRANS = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(_TRANS).lower()
    s = re.sub(r"[^\w\s%.\-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def quote_score(quote: str, source_text: str) -> float:
    """1.0 = verbatim match; otherwise the share of the quote's word n-grams found in the source."""
    q, t = normalize(quote), normalize(source_text)
    if not q:
        return 0.0
    if q in t:
        return 1.0
    words = q.split()
    n = 4 if len(words) >= 8 else 2
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return sum(g in t for g in grams) / len(grams)


def status_for(score: float | None) -> str:
    if score is None:
        return "unchecked"   # page couldn't be fetched (paywall, bot block, unreadable PDF)
    if score >= 0.9:
        return "verified"
    if score >= 0.6:
        return "partial"     # paraphrased or reformatted; kept but marked
    return "not_found"       # quote isn't in the page -> treated as hallucinated


class NLI:
    """Local entailment model: does the source passage support the sentence?

    Optional (pip install -r requirements-nli.txt). A small encoder that reads premise and
    sentence together; it fails in different ways than the chat models, which is the point.
    """
    _instance: "NLI | None" = None
    _failed = False
    _lock = threading.Lock()

    def __init__(self, model_name: str):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.name = model_name

    @classmethod
    def get(cls) -> "NLI | None":
        """Shared instance, or None if the optional dependencies/model aren't available."""
        with cls._lock:  # the UI preloads it while the first answer may already need it
            if cls._instance is None and not cls._failed:
                from .config import nli_model
                try:
                    name = nli_model()
                    cls._instance = cls(name) if name else None
                    cls._failed = cls._instance is None
                except Exception:
                    cls._failed = True
        return cls._instance

    def probs(self, pairs: list[tuple[str, str]], batch: int = 8) -> list[dict[str, float]]:
        """Label probabilities for each (premise, hypothesis) pair, run in batches."""
        import torch
        out = []
        for i in range(0, len(pairs), batch):
            chunk = pairs[i:i + batch]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt")
            with torch.no_grad():
                rows = torch.softmax(self.model(**enc).logits, dim=-1).tolist()
            out += [{self.labels[j]: v for j, v in enumerate(row)} for row in rows]
        return out

    def verdicts(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float, dict]]:
        """Per pair: ("supported" | "unsupported" | "unclear", confidence, raw probabilities)."""
        return [self._label(p) for p in self.probs(pairs)]

    @staticmethod
    def _label(p: dict[str, float]) -> tuple[str, float, dict]:
        ent = p.get("entailment", 0.0)
        if ent >= 0.7:
            return "supported", ent, p
        if ent <= 0.15:  # contradicted, or the passage simply doesn't say it
            return "unsupported", 1 - ent, p
        return "unclear", max(p.values()), p
