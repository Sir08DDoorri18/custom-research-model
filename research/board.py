"""What the parallel-mode panel shows, built from the session's spans.

Spans carry a lane (see trace.Span): "" for the chat model, "r:<key>" for each researcher,
"compare" and "verify". The board keeps every span it is told about and turns them into one
props dict for the ResearchPanel element (public/elements/ResearchPanel.jsx). Spans are read
at render time, so a step's output appears as soon as the worker thread sets it.
"""
from __future__ import annotations

import time

from .trace import Span

PHASES = [("plan", "계획"), ("research", "조사"), ("compare", "비교"), ("answer", "답변"), ("verify", "검증")]
TEXT_LIMIT = 1500   # characters per input/reasoning/output shown; the full text is in traces/


class Board:
    def __init__(self, brain: str):
        self.brain = brain
        self.spans: dict[str, Span] = {}
        self.started = time.time()
        self.finished: float | None = None
        self.dirty = True

    def on_start(self, span: Span) -> None:
        self.spans[span.id] = span
        self.dirty = True

    def on_end(self, span: Span) -> None:
        self.spans.setdefault(span.id, span)
        self.dirty = True

    def finish(self) -> None:
        self.finished = time.time()
        self.dirty = True

    # ---------- props ----------

    def props(self) -> dict:
        spans = list(self.spans.values())
        roots = [s for s in spans if s.lane.startswith("r:") and s.data.get("role")]
        lane_root = {s.lane: s for s in roots}
        compare = next((s for s in spans if s.lane == "compare" and s.name == "비교"), None)
        verify = next((s for s in spans if s.lane == "verify" and s.name == "답변 검증"), None)
        brain = [s for s in spans if s.lane == ""]
        return {
            "phase": self._phase(roots, compare, verify),
            "phases": [{"key": k, "label": v} for k, v in PHASES],
            "seconds": round((self.finished or time.time()) - self.started),
            "brain": {"label": self.brain, "status": "ok" if self.finished else "run",
                      "steps": [_step(s, 0) for s in brain[-10:]], "hidden": max(0, len(brain) - 10)},
            "researchers": [self._researcher(r, [s for s in spans if s.lane == r.lane and s is not lane_root[r.lane]])
                            for r in roots],
            "compare": self._compare(compare, spans),
            "verify": self._verify(verify, spans),
        }

    def _phase(self, roots: list[Span], compare: Span | None, verify: Span | None) -> str:
        if self.finished:
            return "done"
        if verify:
            return "verify"
        if compare and not compare.ended:
            return "compare"
        if any(not r.ended for r in roots):
            return "research"
        return "answer" if roots else "plan"

    def _researcher(self, root: Span, children: list[Span]) -> dict:
        d = root.data
        model = d.get("model") or next((s.model for s in reversed(children) if s.model and s.kind == "llm"), "")
        return {"key": d.get("role"), "label": d.get("label"), "model": model.split("/")[-1],
                "status": _status(root, d.get("error")), "seconds": round(root.seconds),
                "brief": _cut(root.input, 400),
                "steps": [_step(s, s.depth - root.depth - 1) for s in children],
                "claims": d.get("claims") or [], "conclusion": d.get("conclusion") or "",
                "gaps": d.get("gaps") or "", "error": d.get("error") or "",
                "read": d.get("read") or [], "evidence": d.get("evidence") or []}

    def _compare(self, root: Span | None, spans: list[Span]) -> dict | None:
        if not root:
            return None
        return {"status": _status(root), "seconds": round(root.seconds),
                "rows": root.data.get("rows") or [], "note": root.data.get("note") or "",
                "steps": [_step(s, 0) for s in spans if s.lane == "compare" and s is not root]}

    def _verify(self, root: Span | None, spans: list[Span]) -> dict | None:
        if not root:
            return None
        return {"status": _status(root), "seconds": round(root.seconds),
                "summary": root.output.split("\n", 1)[0] if root.ended else "",
                "steps": [_step(s, s.depth - root.depth - 1) for s in spans if s.lane == "verify" and s is not root]}


def _status(s: Span, error: str | None = None) -> str:
    if not s.ended:
        return "run"
    return "err" if error or "ERROR:" in s.output else "ok"


def _step(s: Span, depth: int) -> dict:
    return {"id": s.id, "name": s.name, "kind": s.kind, "model": s.model.split("/")[-1],
            "status": _status(s), "seconds": round(s.seconds, 1), "depth": max(0, depth),
            "input": _cut(s.input), "reasoning": _cut(s.reasoning), "output": _cut(s.output)}


def _cut(text: str, n: int = TEXT_LIMIT) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n… ({len(text) - n}자 더, traces/에 전체)"
