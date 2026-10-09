"""What the answer window shows (public/elements/답변창.jsx), in every mode.

The answer is split into lines and sentences exactly as the checker split it (judge.sentences),
so each sentence can carry its verdict: the window underlines it and shows the reason and the
cited evidence when it is clicked. Sources cited in the answer come below it, and a second tab
lists all evidence collected in the conversation so far.
"""
from __future__ import annotations

import re

from . import judge
from .engine import QUOTE_BADGE, Evidence, Session
from .sources import LABELS

TEXT_LIMIT = 600   # characters of a quote or summary shown per evidence item


def build(question: str, answer: str, report: "judge.Report | None", session: Session, mode: str) -> dict:
    claims = report.claims if report else []
    at = {spot: c.i for c in claims for spot in c.spots}
    uncited = set(report.uncited_spots) if report else set()
    cited = list(dict.fromkeys(i for c in claims for i in c.ids))
    return {
        "question": question[:300],
        "mode": mode,
        "summary": report.summary() if report else "",
        "counts": dict(report.counts) if report else {},
        "uncited": len(uncited),
        "checkers": report.checkers if report else [],
        "lines": _lines(answer, at, uncited),
        "claims": {str(c.i): _claim(c) for c in claims},
        "sources": [_evidence(session.evidence[i]) for i in cited if i in session.evidence],
        "all": [_evidence(e) for e in list(session.evidence.values())[-80:]],
        "total": len(session.evidence),
    }


def _lines(answer: str, at: dict, uncited: set) -> list[dict]:
    out, in_code = [], False
    for ln, line in enumerate(answer.splitlines()):
        raw = line.strip()
        if raw.startswith("```"):
            in_code = not in_code
            continue
        if in_code or raw.startswith("|"):
            out.append({"kind": "code" if in_code else "table", "segs": [{"t": line}]})
        elif not raw:
            out.append({"kind": "blank", "segs": []})
        elif raw.startswith("#"):
            level = len(raw) - len(raw.lstrip("#"))
            out.append({"kind": f"h{min(level, 3)}", "segs": [{"t": raw.lstrip("#").strip()}]})
        else:
            kind = _list_kind(raw)
            segs = []
            for sn, sent in enumerate(judge.sentences(raw)):  # same split as the checker: indices match
                if kind != "p" and sn == 0:  # the marker is drawn separately
                    sent = re.sub(r"^([-*•]|\d+[.)])(\s+|$)", "", sent)
                    if not sent:  # "1." alone: the splitter cut it off as a sentence of its own
                        continue
                seg = {"t": sent}
                if (ln, sn) in at:
                    seg["c"] = at[(ln, sn)]
                elif (ln, sn) in uncited:
                    seg["u"] = True
                segs.append(seg)
            out.append({"kind": kind, "segs": segs, "n": _number(raw) if kind == "ol" else None})
    return out


def _list_kind(raw: str) -> str:
    if re.match(r"^[-*•]\s+", raw):
        return "li"
    if re.match(r"^\d+[.)]\s+", raw):
        return "ol"
    return "p"


def _number(raw: str) -> str:
    m = re.match(r"^(\d+)", raw)
    return m.group(1) if m else ""


def _claim(c: "judge.Claim") -> dict:
    return {"final": c.final, "label": judge.LABEL[c.final], "reason": c.reason, "sentence": c.sentence,
            "ids": c.ids, "bad": c.bad_ids, "nli": c.nli,
            "first": list(c.first) if c.first else None,
            "votes": [list(v) for v in c.votes]}


def _evidence(e: Evidence) -> dict:
    d = e.doc
    kind = "초록" if e.abstract_only and d.kind in ("paper", "preprint") else LABELS[d.kind]
    return {"id": e.id, "title": d.label, "url": d.url, "kind": kind, "page": e.page,
            "badge": "" if e.quote_status == "figure" else QUOTE_BADGE[e.quote_status],
            "status": e.quote_status, "doi": d.doi or "", "retracted": d.retracted,
            "summary": _cut(e.summary), "quote": _cut(e.quote), "relevance": e.relevance}


def _cut(text: str) -> str:
    text = text or ""
    return text if len(text) <= TEXT_LIMIT else text[:TEXT_LIMIT] + "…"
