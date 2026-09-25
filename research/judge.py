"""Checks a finished answer sentence by sentence against the evidence it cites.

Order of checks, cheapest first:
  1. local NLI model          (free, on this PC)
  2. first-pass judge (Groq)  (every cited sentence)
  3. jury of three models     (only sentences where 1 and 2 disagree or are unsure)
Sentences that end without a majority are marked weak rather than forced either way.
"""
from __future__ import annotations

import contextvars
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from . import config, llm, prompts, trace
from .engine import QUOTE_BADGE, Evidence, Session
from .sources import LABELS
from .verify import NLI

NUMBER = re.compile(r"(?<![\w\-.])\d")  # a quantity, not a digit inside a name like LK-99 or Cu3O7
CITE = re.compile(r"\[(E\d+(?:\s*[,，/]\s*E?\d+)*)\]")
BADGE = {"supported": "[ok]", "partial": "[~]", "unsupported": "[x]", "unchecked": "[?]"}
LABEL = {"supported": "근거 있음", "partial": "일부만 근거", "unsupported": "근거 없음", "unchecked": "검증 못 함"}


@dataclass
class Claim:
    i: int
    sentence: str
    ids: list[str]
    bad_ids: list[str] = field(default_factory=list)
    nli: str | None = None
    first: tuple[str, str, str] | None = None            # (verdict, reason, model)
    votes: list[tuple[str, str, str]] = field(default_factory=list)
    final: str = "unchecked"
    reason: str = ""


@dataclass
class Report:
    claims: list[Claim]
    uncited: list[str]
    checkers: list[str]

    @property
    def counts(self) -> Counter:
        return Counter(c.final for c in self.claims)

    def summary(self) -> str:
        c = self.counts
        names = {"supported": "ok", "partial": "일부", "unsupported": "없음", "unchecked": "미확인"}
        return " · ".join(f"{names[k]} {c.get(k, 0)}" for k in names)

    def problems(self) -> list[Claim]:
        return [c for c in self.claims if c.final in ("partial", "unsupported")]


def split_claims(answer: str, session: Session) -> tuple[list[Claim], list[str]]:
    """Cited sentences become claims. Uncited sentences just before a cited one in the same
    paragraph are checked together with it (people often cite once at the end)."""
    claims, uncited = [], []
    has_citations = bool(CITE.search(answer))
    for line in answer.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "|", "```")):
            continue
        pending: list[str] = []
        # "...입니다. [E1] 다음 문장" -> "...입니다 [E1]. 다음 문장", so the citation stays with its sentence
        line = re.sub(r"([.!?。])\s*((?:\[E\d+[^\]]*\]\s*)+)", lambda m: f" {m.group(2).strip()}{m.group(1)} ", line)
        for sent in re.split(r"(?<=[.!?。])\s+(?=\S)", line):
            ids = [("E" + x.lstrip("E")) for m in CITE.finditer(sent) for x in re.split(r"\s*[,，/]\s*", m.group(1))]
            text = _clean(sent)
            if len(text) < 8:
                continue
            if ids:
                good = [i for i in dict.fromkeys(ids) if i in session.evidence]
                bad = [i for i in ids if i not in session.evidence]
                claims.append(Claim(len(claims) + 1, " ".join(pending + [text]), good, bad))
                pending = []
            else:
                pending.append(text)
        if has_citations:  # numbers with no source, in an answer that otherwise cites
            uncited += [t for t in pending if NUMBER.search(t)]
    return claims, uncited


def _clean(sentence: str) -> str:
    text = CITE.sub("", sentence).replace("**", "").replace("__", "")
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip(" -*•")


def premise(evs: list[Evidence], limit: int = 1800) -> str:
    per = max(400, limit // max(1, len(evs)))
    return "\n\n".join(f"[{e.id}] {e.doc.title}\n{_around(e.passage, e.quote, per)}" for e in evs)


def _around(passage: str, quote: str, n: int) -> str:
    """Up to n characters of the passage, centred on the quote when there is one."""
    if len(passage) <= n:
        return passage
    at = passage.find(quote[:60]) if quote else -1
    start = max(0, min(at - n // 3, len(passage) - n)) if at >= 0 else 0
    return passage[start:start + n]


def check(session: Session, answer: str) -> Report:
    trace.use(session.tracer)
    with session.tracer.span("답변 검증", "tool", input=answer) as root:
        claims, uncited = split_claims(answer, session)
        checkable = [c for c in claims if c.ids]
        for c in claims:
            if not c.ids:
                c.final, c.reason = "unsupported", f"없는 근거 번호 {', '.join(c.bad_ids)}"
        checkers = []

        nli = NLI.get() if checkable else None
        if nli:
            checkers.append("NLI")
            with trace.span("② 뜻 비교 (로컬 NLI)", "llm") as s:
                s.model = nli.name
                # One pair per (sentence, cited source), all scored in batches. A sentence citing
                # several sources is supported if any one of them entails it.
                owners = [c for c in checkable for _ in c.ids]
                pairs = [(premise([session.evidence[i]]), c.sentence) for c in checkable for i in c.ids]
                best: dict[int, tuple] = {}
                for c, v in zip(owners, nli.verdicts(pairs)):
                    if c.i not in best or v[2].get("entailment", 0) > best[c.i][2].get("entailment", 0):
                        best[c.i] = v
                rows = []
                for c in checkable:
                    c.nli = best[c.i][0]
                    rows.append(f"{c.i}. {c.nli} (entail {best[c.i][2].get('entailment', 0):.2f}) — {c.sentence[:80]}")
                s.output = "\n".join(rows)

        if checkable and llm.role_available("first"):
            try:
                ref, _ = _judge_batch("first", checkable, session, "③ 1차 채점")
                checkers.append(f"1차 {ref.split('/')[-1]}")
            except llm.NoModelAvailable:
                pass

        disputed = []
        for c in checkable:
            first = c.first[0] if c.first else None
            if c.nli in ("supported", "unsupported") and first == c.nli:
                c.final, c.reason = first, c.first[1]
            elif c.nli is None and first == "supported":
                c.final, c.reason = first, c.first[1]
            else:
                disputed.append(c)

        if disputed:
            members = _run_jury(disputed, session)
            if members:
                checkers.append("심사단 " + ", ".join(m.split("/")[-1] for m in members))
            for c in disputed:
                c.final, c.reason = _decide(c)

        report = Report(claims, uncited, checkers)
        root.output = report.summary() + "\n\n" + "\n".join(
            f"{BADGE[c.final]} {c.sentence[:100]} — {c.reason}" for c in claims)
        return report


JURY_TIMEOUT = 75  # seconds per juror call; a slow juror is replaced rather than waited for


def _judge_batch(role_or_ref, claims: list[Claim], session: Session, label: str, size: int = 5,
                 timeout: float | None = None) -> tuple[str, list[tuple[Claim, tuple]]]:
    """Judge claims in batches with a role (fallback chain) or one specific model.

    For a role, verdicts are stored on the claims right away as first-pass verdicts. For a
    specific model (a juror), they are returned instead, so the caller decides whose votes count.
    Returns (model used, [(claim, (verdict, reason, model))]).
    """
    used, results = "", []
    for n, start in enumerate(range(0, len(claims), size), 1):
        batch = claims[start:start + size]
        items = "\n\n".join(f"[{k}] Sentence: {c.sentence}\nPassages:\n{premise([session.evidence[i] for i in c.ids])}"
                            for k, c in enumerate(batch, 1))
        msgs = [{"role": "user", "content": prompts.JUDGE.format(items=items)}]
        name = f"{label} {n}" if len(claims) > size else label
        if isinstance(role_or_ref, str):
            reply = llm.ask(role_or_ref, msgs, name)
        else:
            reply = llm.call(role_or_ref, msgs, name, timeout=timeout)
        used = str(reply.model)
        try:
            rows = llm.extract_json(reply.text)
        except llm.LLMError:
            rows = []
        for row in rows if isinstance(rows, list) else []:
            try:
                c = batch[int(row["i"]) - 1]
            except (KeyError, ValueError, IndexError, TypeError):
                continue
            verdict = str(row.get("verdict", "")).lower()
            if verdict not in ("supported", "partial", "unsupported"):
                continue
            entry = (verdict, str(row.get("reason", "")), used)
            if isinstance(role_or_ref, str):
                c.first = entry  # first-pass verdicts are kept even if a later batch fails
            else:
                results.append((c, entry))
    return used, results


def _run_jury(claims: list[Claim], session: Session) -> list[str]:
    """Three members from different families vote.

    The members and one spare start together, each with a short time limit, so a slow or
    dead member costs no extra waiting: the first three votes back are used. More spares are
    tried only if fewer than three votes came back.
    """
    members, fallback = config.jury()
    pool = [m for m in fallback if llm.usable(m)]
    chosen = [m for m in members if llm.usable(m)]
    while len(chosen) < 3 and pool:
        chosen.append(pool.pop(0))
    if not chosen:
        return []
    wave = chosen + pool[:1]
    pool = pool[1:]

    def vote(ref):
        try:
            _, results = _judge_batch(ref, claims, session, f"심사 · {ref.model}", timeout=JURY_TIMEOUT)
            return ref, (results or None)  # no parsable verdicts counts as no vote
        except llm.LLMError:
            return ref, None

    with trace.span("④ 심사단", "tool", input=f"{len(claims)}개 문장") as s:
        answered = []
        executor = ThreadPoolExecutor(len(wave))
        futures = [executor.submit(contextvars.copy_context().run, vote, ref) for ref in wave]
        for f in as_completed(futures):
            if f.result()[1]:
                answered.append(f.result())
            if len(answered) == 3:
                break  # three votes are enough; don't wait for the slowest juror
        executor.shutdown(wait=False, cancel_futures=True)
        while len(answered) < 3 and pool:
            o = vote(pool.pop(0))
            if o[1]:
                answered.append(o)
        kept = answered[:3]
        for _, results in kept:
            for c, entry in results:
                c.votes.append(entry)
        done = [str(ref) for ref, _ in kept]
        s.output = "참여: " + (", ".join(done) or "없음")
        return done


def _decide(c: Claim) -> tuple[str, str]:
    votes = [v for v, _, _ in c.votes]
    if len(votes) >= 2:
        top, n = Counter(votes).most_common(1)[0]
        if n >= 2:
            reason = next(r for v, r, _ in c.votes if v == top)
            return top, f"{reason} ({n}/{len(votes)})"
        return "partial", "심사단 의견이 갈림: " + " / ".join(f"{m.split('/')[-1]} {v}" for v, _, m in c.votes)
    # not enough jurors: fall back on whatever single verdict exists
    if c.first:
        return c.first[0], c.first[1] + " (심사단 없음)"
    if c.votes:
        return c.votes[0][0], c.votes[0][1] + " (심사단 1명)"
    # The small local model alone is not trusted to call a sentence wrong.
    if c.nli == "supported":
        return "supported", "로컬 NLI만 확인 (채점 모델 없음)"
    if c.nli == "unsupported":
        return "partial", "로컬 NLI가 근거에서 확인하지 못함, 채점 모델 없이 판단 보류"
    return "unchecked", "사용할 수 있는 채점 모델이 없어요 (.env 키 확인)"


def render(report: Report, session: Session, answer: str) -> str:
    """Verification summary + source list appended under the answer."""
    out = []
    if report.claims:
        out.append(f"**검증** {report.summary()}  \n_{' · '.join(report.checkers) or '채점 모델 없음'}_")
        for c in report.claims:
            if c.final != "supported":
                out.append(f"- {BADGE[c.final]} {LABEL[c.final]}: “{c.sentence[:120]}” — {c.reason}")
    if report.uncited:
        out.append("- [?] 출처 없는 수치: " + " / ".join(f"“{u[:80]}”" for u in report.uncited[:5]))
    cited = list(dict.fromkeys(i for c in report.claims for i in c.ids))
    if cited:
        out.append("**출처**")
        for i in cited:
            e = session.evidence[i]
            kind = "초록" if e.abstract_only and e.doc.kind in ("paper", "preprint") else LABELS[e.doc.kind]
            doi = f" · doi:{e.doc.doi}" if e.doc.doi else ""
            retracted = " · 철회된 논문" if e.doc.retracted else ""
            title = f"[{e.doc.label}]({e.doc.url})" if e.doc.url else e.doc.label  # user files have no URL
            page = f" p.{e.page}" if e.page else ""
            badge = "" if e.quote_status == "figure" else f"{QUOTE_BADGE[e.quote_status]} · "  # kind already says 그림해석
            out.append(f"- **[{i}]** {badge}{kind} · {title}{page}{doi}{retracted}")
    return "\n\n".join(out)

