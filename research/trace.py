"""Records every step (tool call, model call, verdict) as a tree of spans.

A listener (the web UI or the terminal) is told when a span starts and ends, so it can
show progress live. Every finished span is also appended to traces/<session>.md with
full inputs, reasoning and outputs.
"""
from __future__ import annotations

import contextvars
import datetime as dt
import itertools
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol

from .config import ROOT

TRACE_DIR = ROOT / "traces"
_ids = itertools.count(1)
_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("span", default=None)


@dataclass
class Span:
    name: str
    kind: str                      # "tool" | "llm" | "run"
    id: str = field(default_factory=lambda: f"s{next(_ids)}")
    parent_id: str | None = None
    input: str = ""
    output: str = ""
    reasoning: str = ""
    model: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    depth: int = 0

    @property
    def seconds(self) -> float:
        return (self.ended or time.time()) - self.started


class Listener(Protocol):
    def on_start(self, span: Span) -> None: ...
    def on_end(self, span: Span) -> None: ...


class Tracer:
    def __init__(self, session_id: str, listener: Listener | None = None):
        self.listener = listener
        self.root: Span | None = None      # parent for spans started outside any span (e.g. SDK tool calls)
        self._lock = threading.Lock()
        TRACE_DIR.mkdir(exist_ok=True)
        self.path = TRACE_DIR / f"{dt.datetime.now():%Y%m%d-%H%M%S}-{session_id}.md"

    @contextmanager
    def span(self, name: str, kind: str = "tool", input: str = "", parent: Span | None = None):
        parent = parent or _current.get() or self.root
        s = Span(name=name, kind=kind, input=input,
                 parent_id=parent.id if parent else None, depth=parent.depth + 1 if parent else 0)
        token = _current.set(s)
        if self.listener:
            self.listener.on_start(s)
        try:
            yield s
        except Exception as e:
            s.output = (s.output + f"\n\nERROR: {e}").strip()
            raise
        finally:
            s.ended = time.time()
            _current.reset(token)
            if self.listener:
                self.listener.on_end(s)
            self._write(s)

    def note(self, name: str, output: str, kind: str = "llm", reasoning: str = "", model: str = "") -> None:
        """A span that is already finished (e.g. a thinking block that arrived whole)."""
        with self.span(name, kind) as s:
            s.output, s.reasoning, s.model = output, reasoning, model

    def _write(self, s: Span) -> None:
        head = "#" * min(2 + s.depth, 6)
        parts = [f"{head} [{s.kind}] {s.name}" + (f" · {s.model}" if s.model else "") + f" · {s.seconds:.1f}s"]
        if s.input:
            parts.append(f"**input**\n\n```\n{s.input}\n```")
        if s.reasoning:
            parts.append(f"**reasoning**\n\n```\n{s.reasoning}\n```")
        if s.output:
            parts.append(f"**output**\n\n{s.output}")
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write("\n\n".join(parts) + "\n\n")


_tracer: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar("tracer", default=None)


def use(tracer: Tracer) -> None:
    """Make `tracer` the one that module-level span() writes to in this context."""
    _tracer.set(tracer)


@contextmanager
def span(name: str, kind: str = "tool", input: str = ""):
    tracer = _tracer.get()
    if tracer is None:  # tracing off (e.g. eval runs): hand back a throwaway span
        yield Span(name=name, kind=kind, input=input)
        return
    with tracer.span(name, kind, input) as s:
        yield s


def pmap(fn, items, workers: int = 4) -> list:
    """Thread-pool map that keeps the current span/tracer in each worker."""
    from concurrent.futures import ThreadPoolExecutor
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max(1, min(workers, len(items)))) as pool:
        futures = [pool.submit(contextvars.copy_context().run, fn, it) for it in items]
        return [f.result() for f in futures]
