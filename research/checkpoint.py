"""Progress of each conversation's latest question, saved as it goes, so a question that was
cut off (PC asleep, server stopped, connection lost) can be continued instead of redone.

One small JSON file per conversation in .cache/checkpoints: the question, the mode, whether it
finished, and the session's documents, evidence and step results (Session.snapshot). The file
is rewritten after every step that finds something. When a saved conversation is reopened its
evidence is restored, so E-ids cited earlier still check out, and an unfinished question can be
continued from what it had already found.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import threading

from .config import ROOT
from .engine import Session

DIR = ROOT / ".cache" / "checkpoints"
_lock = threading.Lock()


def _path(thread_id: str):
    safe = "".join(c for c in thread_id if c.isalnum() or c == "-")
    return DIR / f"{safe}.json"


def save(thread_id: str | None, session: Session, question: str, mode: str, status: str) -> None:
    """status: "running" while the question is being worked on, "done" once answered."""
    if not thread_id:
        return
    data = {"question": question, "mode": mode, "status": status,
            "updated": dt.datetime.now().isoformat(timespec="seconds"), **session.snapshot()}
    path = _path(thread_id)
    with _lock:
        DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)  # never leave a half-written file behind


def load(thread_id: str | None) -> dict | None:
    if not thread_id:
        return None
    try:
        return json.loads(_path(thread_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def prune(keep: set[str]) -> None:
    """Drop checkpoints of conversations that no longer exist."""
    if not DIR.exists():
        return
    names = {_path(t).name for t in keep}
    for path in DIR.glob("*.json"):
        if path.name not in names:
            path.unlink(missing_ok=True)


def found_something(data: dict) -> bool:
    return bool(data.get("progress") or data.get("evidence"))


def summary(data: dict) -> str:
    """What the cut-off question had found, as markdown for the side window."""
    lines = [f"**질문** {data.get('question', '')}",
             f"_{'병렬' if data.get('mode') == 'parallel' else '기본'} 조사 · 마지막 저장 {data.get('updated', '')} · "
             f"문서 {len(data.get('docs') or [])} · 근거 {len(data.get('evidence') or [])}_"]
    for step in data.get("progress") or []:
        lines.append(f"### {step['label']}\n\n{step['text']}")
    return "\n\n".join(lines)


def continuation(data: dict, request: str) -> str:
    """The message that hands the cut-off question back to the chat model."""
    steps = "\n\n".join(f"## {s['label']}\n{s['text']}" for s in data.get("progress") or [])
    done = [s["label"].split("(")[-1].rstrip(")") for s in data.get("progress") or []
            if s["label"].startswith("조사원 끝남")]
    parallel_note = (f"\nResearchers that already finished: {', '.join(done)}. If you dispatch again, brief only "
                     "the others." if done else "")
    read = sorted({d for s in data.get("progress") or [] if s["label"].startswith("자료 읽기")
                   for d in re.findall(r"D\d+", s["label"])}, key=lambda d: int(d[1:]))
    if read:
        parallel_note += f"\nAlready read (their evidence is below): {', '.join(read)}. Do not read them again."
    return (f"[이어서 하기] The question below was cut off before it was answered. Its work so far is listed "
            f"here; the documents (D#) and evidence (E#) still exist and can be cited as usual. Do not search "
            f"again for what is already covered: look up only what is missing, then answer the question as "
            f"usual, without mentioning that it was cut off or continued."
            f"{parallel_note}\n\n[원래 질문]\n{data.get('question', '')}\n\n[지금까지 찾은 것]\n{steps or '(없음)'}"
            + (f"\n\n[사용자 추가 요청]\n{request}" if request.strip() else ""))
