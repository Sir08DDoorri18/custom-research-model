"""Parallel research mode: several models research one question at the same time, each from its
own angle (papers, official sources, news, counter-evidence), and their findings are compared.

A researcher follows a fixed loop instead of driving tools freely, because the free models are
unreliable at calling tools themselves:

  search queries (its model) -> search (code) -> pick documents (its model)
  -> read (rcs model + verbatim quote check, as in basic mode) -> findings (its model)

Findings are claims tied to evidence ids (E#). The comparison then groups the claims of all
researchers and counts independent sources, so two researchers who read the same page count
as one confirmation, not two. Every step is a span in the researcher's own lane, which is what
the UI panel draws side by side.
"""
from __future__ import annotations

import contextvars
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import config, llm, prompts
from .config import Researcher
from .engine import Doc, Evidence, Session, _norm_title
from .trace import pmap

TIMEOUT = 300        # seconds for all researchers together; late ones are reported, not waited for
STEP_TIMEOUT = 60    # per model call inside a researcher; a slow model is benched and the next one tried
STATUS = {"agreed": "합의", "same_source": "같은 출처", "disputed": "충돌", "single": "단독"}
ENGINES = {"papers": "paper databases (OpenAlex, Semantic Scholar, arXiv)",
           "web": "a web search engine",
           "both": "a paper database and a web search engine (each query goes to both)"}


@dataclass
class Finding:
    id: str                        # c1, c2 ... across all researchers
    researcher: Researcher
    claim: str
    evidence: list[str]
    confidence: str


@dataclass
class Result:
    researcher: Researcher
    model: str = ""
    read: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    claims: list[dict] = field(default_factory=list)
    conclusion: str = ""
    gaps: str = ""
    error: str = ""


@dataclass
class Group:
    status: str
    statement: str
    support: list[Finding]
    oppose: list[Finding]
    sources: int                   # independent sources behind the supporting claims
    note: str = ""


def turn_text(question: str) -> str:
    """The user's message as sent to the chat model in parallel mode."""
    roles = ", ".join(f"{r.key} ({r.label}: {prompts.ROLE_FOCUS.get(r.key, r.label)})" for r in config.researchers())
    return prompts.PARALLEL_TURN.format(roles=roles, question=question)


def dispatch(session: Session, question: str, briefs: dict[str, str] | None = None) -> str:
    """The dispatch_researchers tool: run the researchers, compare, and report to the chat model."""
    if session.mode != "parallel":
        return "병렬 조사 모드가 꺼져 있어요. search_papers / search_web / read_sources를 쓰세요."
    briefs = {k: str(v) for k, v in (briefs or {}).items()}
    roster = config.researchers()
    chosen = [r for r in roster if r.key in briefs] or roster
    ready = [r for r in chosen if any(llm.usable(m) for m in r.models)]
    skipped = [r.label for r in chosen if r not in ready]
    brief_lines = "\n".join(f"- {r.label}: {briefs.get(r.key) or '(지시 없음)'}" for r in chosen)
    with session._span(f"병렬 조사 · {', '.join(r.label for r in ready) or '없음'}",
                       input=f"{question}\n\n{brief_lines}") as s:
        if not ready:
            s.output = "쓸 수 있는 조사원 모델이 없어요 (.env 키 확인, models.yaml의 parallel)."
            return s.output
        started = time.time()
        executor = ThreadPoolExecutor(len(ready))
        futures = {executor.submit(contextvars.copy_context().run, _research, session, r, question,
                                   briefs.get(r.key, "")): r for r in ready}
        done, _ = wait(futures, timeout=TIMEOUT)
        executor.shutdown(wait=False, cancel_futures=True)
        results = [f.result() if f in done else Result(r, error=f"{TIMEOUT}초 안에 끝나지 않음")
                   for f, r in futures.items()]
        groups, note = compare(session, question, results)
        s.output = _report(session, results, groups, note, skipped, time.time() - started)
        return s.output


# ---------- one researcher ----------

def _research(session: Session, r: Researcher, question: str, brief: str) -> Result:
    res = Result(r)
    focus = prompts.ROLE_FOCUS.get(r.key, r.label)
    with session.tracer.span(f"조사원 · {r.label}", "tool", input=brief or question, lane=f"r:{r.key}") as s:
        s.data = {"label": r.label, "role": r.key}
        try:
            _steps(session, r, question, brief, focus, res)
        except Exception as e:  # a failed researcher is reported; the others carry on
            res.error = f"{type(e).__name__}: {str(e)[:200]}"
        s.data.update(model=res.model, conclusion=res.conclusion, gaps=res.gaps, error=res.error,
                      claims=res.claims, read=res.read, evidence=[e.id for e in res.evidence])
        s.output = _fmt_result(res) if not res.error else f"실패: {res.error}"
    return res


def _ask(r: Researcher, res: Result, prompt: str, label: str):
    reply = llm.ask_any(list(r.models), [{"role": "user", "content": prompt}], label, r.label,
                        timeout=STEP_TIMEOUT)
    res.model = str(reply.model)
    return reply


def _steps(session: Session, r: Researcher, question: str, brief: str, focus: str, res: Result) -> None:
    cap = int(config.parallel().get("max_docs", 4))

    # 1. search queries
    reply = _ask(r, res, prompts.QUERIES.format(focus=focus, brief=brief or "-", question=question,
                                                n=2, engine=ENGINES[r.search]), "검색어 정하기")
    queries = [q for q in _json_field(reply.text, "queries", list) if isinstance(q, str) and q.strip()][:3]
    queries = queries or [brief or question]

    # 2. search (code), all queries at once
    finders = [f for f, kinds in ((session.find_papers, ("papers", "both")), (session.find_web, ("web", "both")))
               if r.search in kinds]
    found = pmap(lambda job: job[0](job[1], 6)[0], [(f, q) for q in queries for f in finders], workers=4)
    docs = [d for d in {d.id: d for batch in found for d in batch}.values() if d.kind != "blog"]  # blogs are never evidence
    docs.sort(key=lambda d: d.kind not in r.prefer)               # stable: preferred kinds first
    if not docs:
        res.gaps = "검색 결과 없음"
        return

    # 3. pick what to read
    ids = [d.id for d in docs]
    if len(docs) > cap:
        reply = _ask(r, res, prompts.PICK.format(focus=focus, question=question, n=cap,
                                                 docs=session._list_docs(docs[:20])), "읽을 문서 고르기")
        picked = [i.strip().upper() for i in _json_field(reply.text, "read", list) if isinstance(i, str)]
        ids = [i for i in dict.fromkeys(picked) if i in ids][:cap] or ids[:cap]
    res.read = ids

    # 4. read (rcs model + quote check, shared with basic mode)
    res.evidence, _ = session.read(f"{question}\n{brief}".strip(), ids, " ".join(queries))
    if not res.evidence:
        res.gaps = "읽은 문서에서 관련 근거를 찾지 못함"
        return

    # 5. findings
    own = {e.id for e in res.evidence}
    reply = _ask(r, res, prompts.FINDINGS.format(focus=focus, question=question, language=config.language(),
                                                 evidence="\n".join(session._fmt_evidence(e) for e in res.evidence)),
                 "결론 내기")
    try:
        data = llm.extract_json(reply.text)
    except llm.LLMError:
        data = {}
    data = data if isinstance(data, dict) else {}
    dropped = 0
    for c in data.get("claims") or []:
        if not isinstance(c, dict) or not str(c.get("claim", "")).strip():
            continue
        evs = [e for e in (_eid(x) for x in c.get("evidence") or []) if e in own]
        if not evs:  # a claim must rest on this researcher's own evidence
            dropped += 1
            continue
        conf = str(c.get("confidence", "medium")).lower()
        res.claims.append({"claim": str(c["claim"]).strip(), "evidence": list(dict.fromkeys(evs)),
                           "confidence": conf if conf in ("high", "medium", "low") else "medium"})
    res.conclusion = _text(data.get("conclusion"))
    res.gaps = _text(data.get("gaps")) + (f" (근거 번호가 없어 뺀 주장 {dropped}개)" if dropped else "")


# ---------- comparison ----------

def compare(session: Session, question: str, results: list[Result]) -> tuple[list[Group], str]:
    findings = [Finding(f"c{n}", res.researcher, c["claim"], c["evidence"], c["confidence"])
                for n, (res, c) in enumerate(((res, c) for res in results for c in res.claims), 1)]
    if not findings:
        return [], "비교할 주장이 없어요"
    by_id = {f.id: f for f in findings}
    lines = "\n".join(f"{f.id} [{f.researcher.label}] {f.claim} (evidence: {', '.join(f.evidence)})"
                      for f in findings)
    with session.tracer.span("비교", "tool", input=lines, lane="compare") as s:
        groups, note = [], ""
        refs = config.compare_models() or config.role("first")
        if len(findings) > 1:
            try:
                reply = llm.ask_any(refs, [{"role": "user", "content": prompts.COMPARE.format(
                    question=question, claims=lines, language=config.language())}], "주장 묶기", "비교",
                    timeout=STEP_TIMEOUT)
                rows = llm.extract_json(reply.text)
            except llm.LLMError as e:
                rows, note = [], f"비교 모델 없음, 주장을 묶지 않았어요 ({str(e)[:80]})"
            used: set[str] = set()
            for row in rows if isinstance(rows, list) else []:
                if not isinstance(row, dict):
                    continue
                sup = [by_id[i] for i in _ids(row.get("support")) if i in by_id and i not in used]
                used |= {f.id for f in sup}
                opp = [by_id[i] for i in _ids(row.get("oppose")) if i in by_id and i not in used]
                used |= {f.id for f in opp}
                if sup:
                    groups.append(_group(session, str(row.get("statement") or sup[0].claim), sup, opp,
                                         str(row.get("note") or "")))
                elif opp:  # an opposition with nothing to oppose is a claim of its own
                    groups.append(_group(session, opp[0].claim, opp, []))
            findings = [f for f in findings if f.id not in used]
        groups += [_group(session, f.claim, [f], []) for f in findings]
        order = {"disputed": 0, "agreed": 1, "same_source": 2, "single": 3}
        groups.sort(key=lambda g: (order[g.status], -g.sources))
        s.data = {"rows": [_row(g) for g in groups], "note": note}
        s.output = "\n".join(_fmt_group(g) for g in groups) + (f"\n\n_{note}_" if note else "")
        return groups, note


def _group(session: Session, statement: str, support: list[Finding], oppose: list[Finding], note: str = "") -> Group:
    who = {f.researcher.key for f in support}
    srcs = {source_key(session.evidence[e].doc) for f in support for e in f.evidence if e in session.evidence}
    status = ("disputed" if oppose else "agreed" if len(who) >= 2 and len(srcs) >= 2
              else "same_source" if len(who) >= 2 else "single")
    return Group(status, statement, support, oppose, len(srcs), note)


def source_key(doc: Doc) -> str:
    """What counts as one independent source. A paper is one source whether it was found as
    the journal version or the arXiv preprint, so papers go by title; pages by their URL."""
    if doc.kind in ("paper", "preprint"):
        return "paper:" + (_norm_title(doc.title) or (doc.doi or "").lower())
    if doc.doi:
        return "doi:" + doc.doi.lower()
    if doc.kind in ("file", "figure"):
        return "file:" + doc.title.split(" p.")[0]
    u = urlparse(doc.url)
    return (u.hostname or "").removeprefix("www.") + u.path.rstrip("/") if u.hostname else doc.id


# ---------- formatting ----------

def _row(g: Group) -> dict:
    return {"status": g.status, "label": STATUS[g.status], "statement": g.statement, "note": g.note,
            "support": sorted({f.researcher.label for f in g.support}),
            "oppose": sorted({f.researcher.label for f in g.oppose}),
            "sources": g.sources,
            "evidence": list(dict.fromkeys(e for f in g.support + g.oppose for e in f.evidence))}


def _fmt_group(g: Group) -> str:
    sup = ", ".join(sorted({f.researcher.label for f in g.support}))
    ev = ", ".join(dict.fromkeys(e for f in g.support for e in f.evidence))
    out = f"[{STATUS[g.status]}] {g.statement} — 지지: {sup} · 독립 출처 {g.sources} · [{ev}]"
    if g.oppose:
        opp_ev = ", ".join(dict.fromkeys(e for f in g.oppose for e in f.evidence))
        out += f" / 반대: {', '.join(sorted({f.researcher.label for f in g.oppose}))} [{opp_ev}]"
    return out + (f" · {g.note}" if g.note else "")


def _fmt_result(res: Result) -> str:
    lines = [f"결론: {res.conclusion or '-'}"]
    lines += [f"- {c['claim']} [{', '.join(c['evidence'])}] (확신 {c['confidence']})" for c in res.claims]
    if res.gaps:
        lines.append(f"부족: {res.gaps}")
    return "\n".join(lines)


def _report(session: Session, results: list[Result], groups: list[Group], note: str,
            skipped: list[str], seconds: float) -> str:
    out = [f"## 조사원 {len(results)}명 결과 ({seconds:.0f}초)"]
    for res in results:
        model = res.model.split("/")[-1] if res.model else "-"
        if res.error:
            out.append(f"### {res.researcher.label} · {model}\n실패: {res.error}")
        else:
            read = f"읽은 문서 {', '.join(res.read)}" if res.read else "읽은 문서 없음"
            out.append(f"### {res.researcher.label} · {model} · {read}\n{_fmt_result(res)}")
    if skipped:
        out.append("_모델이 없어 빠진 조사원: " + ", ".join(skipped) + "_")
    out.append("## 비교\n" + ("\n".join(_fmt_group(g) for g in groups) or "비교할 주장이 없어요")
               + (f"\n_{note}_" if note else ""))
    cited = dict.fromkeys(e for res in results for c in res.claims for e in c["evidence"])
    if cited:
        out.append("## 근거\n" + "\n".join(session._fmt_evidence(session.evidence[e])
                                           for e in cited if e in session.evidence))
    return "\n\n".join(out)


def _json_field(text: str, key: str, kind: type):
    try:
        data = llm.extract_json(text)
    except llm.LLMError:
        return kind()
    value = data.get(key) if isinstance(data, dict) else data if kind is list else None
    return value if isinstance(value, kind) else kind()


def _text(value) -> str:
    """Models sometimes return a list or a dict of lists where a sentence was asked for."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return " / ".join(t for t in (_text(v) for v in value) if t)
    return str(value or "").strip()


def _eid(x) -> str:
    m = re.search(r"\d+", str(x))
    return f"E{m.group(0)}" if m else ""


def _ids(value) -> list[str]:
    items = value if isinstance(value, list) else re.split(r"[,\s]+", str(value or ""))
    return [f"c{m.group(0)}" for m in (re.search(r"\d+", str(i)) for i in items) if m]
