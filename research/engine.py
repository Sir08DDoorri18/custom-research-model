"""The research tools the chat model uses, and the per-conversation evidence they collect.

Search tools return documents (D1, D2 ...) as leads. read_sources() reads those documents,
has a cheap model pick out relevant passages with verbatim quotes (RCS), checks each quote
against the passage, and registers what survives as evidence (E1, E2 ...). Only evidence ids
may be cited in an answer.
"""
from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import config, files, llm, prompts, rank, sources, trace
from .trace import Listener, Tracer, pmap
from .verify import quote_score, status_for


@dataclass
class Doc:
    id: str
    title: str
    url: str
    kind: str                      # paper | preprint | institution | news | web | blog | file | figure
    year: int | None = None
    venue: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    arxiv_id: str | None = None
    pdf_url: str | None = None
    abstract: str = ""
    snippet: str = ""
    cited_by: int | None = None
    retracted: bool = False
    text: str = ""                 # full text we already have (user files, figure descriptions)
    image: str | None = None       # figure image on disk
    pages: int | None = None

    @property
    def label(self) -> str:
        bits = [str(b) for b in (self.year, self.venue) if b]
        return f"{self.title} ({', '.join(bits)})" if bits else self.title


@dataclass
class Evidence:
    id: str
    doc: Doc
    passage: str
    summary: str
    quote: str
    relevance: float
    quote_status: str              # verified | partial | mismatch | excerpt | figure
    abstract_only: bool = False
    page: int | None = None


QUOTE_BADGE = {"verified": "원문일치", "partial": "부분일치", "mismatch": "요약만", "excerpt": "발췌", "figure": "그림해석"}


class Session:
    def __init__(self, listener: Listener | None = None, session_id: str | None = None):
        self.id = session_id or uuid.uuid4().hex[:8]
        self.tracer = Tracer(self.id, listener)
        self.docs: dict[str, Doc] = {}
        self.evidence: dict[str, Evidence] = {}
        self._doc_keys: dict[str, str] = {}
        self._passage_keys: dict[tuple[str, str], str] = {}

    def _span(self, name: str, input: str = ""):
        trace.use(self.tracer)  # model calls made inside this tool are traced into this session
        return self.tracer.span(name, "tool", input)

    # ---------- tools ----------

    def search_papers(self, query: str, limit: int = 8) -> str:
        with self._span(f"논문 검색 · {query}", input=query) as s:
            finders = [("OpenAlex", sources.openalex_search), ("Semantic Scholar", sources.s2_search),
                       ("arXiv", sources.arxiv_search)]
            results = pmap(lambda f: _safe(f[0], f[1], query, limit), finders, workers=3)
            notes = [f"{name}: {len(found)}건" + (f" (실패: {err})" if err else "")
                     for (name, _), (found, err) in zip(finders, results)]
            # Interleave the APIs' own relevance rankings instead of sorting by citations,
            # which would push famous but off-topic papers to the top.
            lists = [found for found, _ in results]
            papers = [lst[i] for i in range(max(map(len, lists), default=0)) for lst in lists if i < len(lst)]
            docs = self._register_papers(papers)[: limit + 4]
            s.output = self._list_docs(docs) + "\n\n_" + " · ".join(notes) + "_"
            return s.output

    def search_web(self, query: str, limit: int = 8) -> str:
        with self._span(f"웹 검색 · {query}", input=query) as s:
            found, err = _safe("web", sources.web_search, query, limit)
            docs = [self._add_doc(title=r["title"], url=r["url"], kind=sources.grade_url(r["url"]), snippet=r["snippet"])
                    for r in found]
            s.output = self._list_docs(docs) if docs else f"결과 없음{f' ({err})' if err else ''}"
            return s.output

    def citation_graph(self, doc_id: str, direction: str = "references", limit: int = 10) -> str:
        with self._span(f"인용 따라가기 · {doc_id} {direction}", input=f"{doc_id} {direction}") as s:
            doc = self.docs.get(doc_id)
            if not doc:
                s.output = f"{doc_id}: 알 수 없는 문서 id"
                return s.output
            ref = f"DOI:{doc.doi}" if doc.doi else f"ARXIV:{doc.arxiv_id}" if doc.arxiv_id else None
            if not ref:
                s.output = f"{doc_id}는 DOI나 arXiv id가 없어 인용 관계를 조회할 수 없어요."
                return s.output
            direction = "citations" if direction.startswith("cit") else "references"
            found, err = _safe("Semantic Scholar", sources.s2_graph, ref, direction, limit)
            docs = self._register_papers(found)
            s.output = self._list_docs(docs) if docs else f"결과 없음{f' ({err})' if err else ''}"
            return s.output

    def read_sources(self, question: str, doc_ids: list[str], focus: str = "") -> str:
        with self._span(f"자료 읽기 · {', '.join(doc_ids)}", input=f"{question}\nfocus: {focus}") as s:
            docs, notes = [], []
            for d in doc_ids:
                doc = self.docs.get(d.strip())
                if not doc:
                    notes.append(f"{d}: 알 수 없는 id")
                elif doc.kind == "blog":
                    notes.append(f"{d}: 블로그는 근거로 쓰지 않아요")
                else:
                    docs.append(doc)
            texts = pmap(self._full_text, docs, workers=6)

            query = f"{question} {focus}"
            scored: list[tuple[float, Doc, str, bool]] = []
            for doc, (text, abstract_only) in zip(docs, texts):
                if not text:
                    notes.append(f"{doc.id}: 본문을 가져오지 못함")
                    continue
                if abstract_only and doc.kind in ("paper", "preprint"):
                    notes.append(f"{doc.id}: 전문을 못 구해 초록만 읽음")
                passages = rank.chunk(text)
                top = sorted(zip(rank.bm25(query, passages), passages), key=lambda x: -x[0])[:4]
                scored += [(sc, doc, p, abstract_only) for sc, p in top]
            candidates = [(d, p, a) for _, d, p, a in sorted(scored, key=lambda x: -x[0])]

            items = self._rcs(question, focus, candidates) if candidates else []
            new = self._register_evidence(items)
            body = "\n".join(self._fmt_evidence(e) for e in new) or "관련 있는 내용을 찾지 못했어요."
            s.output = body + (("\n\n_" + " · ".join(notes) + "_") if notes else "")
            return s.output

    def add_files(self, items: list[tuple[str, str | None]]) -> tuple[str, list[str]]:
        """Read files the user added and register each (and each described figure) as documents.
        items: (path, display name). Returns (summary for the chat model, image paths the chat
        model may look at directly)."""
        names = [name or os.path.basename(path) for path, name in items]
        with self._span(f"파일 읽기 · {', '.join(names)}", input="\n".join(names)) as s:
            results = files.read_many([(path, name) for (path, _), name in zip(items, names)])
            lines, images = [], []
            for name, rf in zip(names, results):
                if isinstance(rf, Exception):
                    lines.append(f"- {name}: 읽지 못함 ({rf})")
                else:
                    images += self._register_file(rf, lines)
            s.output = "\n".join(lines) or "읽을 내용이 없어요."
            return s.output, images

    def _register_file(self, rf: "files.ReadFile", lines: list[str]) -> list[str]:
        """Register one read file and its described figures; append summary lines.
        Returns the paths of attached photos (the chat model may look at those itself)."""
        images = []
        if rf.text.strip():
            doc = self._add_doc(title=rf.name, url="", kind="file", text=rf.text, pages=rf.pages)
            bits = [f"{rf.pages}쪽" if rf.pages else "", f"표 {rf.tables}" if rf.tables else "",
                    f"그림 {len(rf.figures)}" if rf.figures else "", rf.engine]
            lines.append(f"[{doc.id}] {sources.LABELS['file']} · {rf.name} · " + " · ".join(b for b in bits if b))
        figures = files.describe(rf) if rf.figures else []
        for k, fig in enumerate(figures, 1):
            if not fig.description:
                lines.append(f"- {rf.name} 그림 {k}: 해석 모델을 쓸 수 없어 건너뜀")
                continue
            where = f" p.{fig.page}" if fig.page else ""
            title = rf.name if rf.is_image else f"{rf.name}{where} 그림 {k}"
            text = (f"캡션: {fig.caption}\n\n" if fig.caption else "") + fig.description
            doc = self._add_doc(title=title, url="", kind="figure", text=text, image=str(fig.image))
            lines.append(f"[{doc.id}] {sources.LABELS['figure']} · {title}")
            if rf.is_image:
                images.append(str(fig.image))
        if len(rf.figures) > len(figures):
            lines.append(f"- 그림 {len(rf.figures) - len(figures)}개는 개수 제한으로 해석하지 않음")
        return images

    # ---------- reading ----------

    def _full_text(self, doc: Doc) -> tuple[str | None, bool]:
        """(text, abstract_only). Papers: open-access PDF, else abstract. Web: the page."""
        if doc.text:
            return doc.text, False
        if doc.kind in ("paper", "preprint"):
            if doc.pdf_url:
                page = sources.fetch(doc.pdf_url)
                if page and len(page["text"]) > 3000:  # shorter = paywall or error page, not the paper
                    return page["text"], False
            return (doc.abstract or None), True
        page = sources.fetch(doc.url)
        if page:
            if not doc.title and page.get("title"):
                doc.title = page["title"]
            return page["text"], False
        return (doc.snippet or None), True

    def _rcs(self, question: str, focus: str, candidates: list[tuple[Doc, str, bool]]) -> list[dict]:
        """Relevance + summary + verbatim quote for each passage, by the cheap 'rcs' models."""
        if not llm.role_available("rcs"):
            # No reader model: fall back to the best keyword matches, shown as raw excerpts.
            return [{"doc": d, "passage": p, "abstract_only": a, "relevance": 5, "summary": "",
                     "quote": p[:400], "excerpt": True} for d, p, a in candidates[:8]]
        batches = [candidates[i:i + 8] for i in range(0, len(candidates), 8)]

        def read(args):
            n, batch = args
            numbered = "\n\n".join(f"[{i + 1}] (source: {d.title})\n{p}" for i, (d, p, _) in enumerate(batch))
            msg = prompts.RCS.format(question=question, focus=focus or "-", language=config.language(),
                                     passages=numbered)
            try:
                reply = llm.ask("rcs", [{"role": "user", "content": msg}], f"읽기 {n}/{len(batches)}")
                rows = llm.extract_json(reply.text)
            except llm.LLMError:
                return []
            out = []
            for row in rows if isinstance(rows, list) else []:
                try:
                    d, p, a = batch[int(row["i"]) - 1]
                    out.append({"doc": d, "passage": p, "abstract_only": a, "relevance": float(row.get("relevance", 0)),
                                "summary": str(row.get("summary", "")), "quote": str(row.get("quote", "")),
                                "excerpt": False})
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
            return out

        return [r for rows in pmap(read, list(enumerate(batches, 1)), workers=3) for r in rows]

    def _register_evidence(self, items: list[dict], min_relevance: float = 4, cap: int = 10) -> list[Evidence]:
        new = []
        for it in sorted(items, key=lambda r: -r["relevance"]):
            if it["relevance"] < min_relevance or len(new) >= cap:
                continue
            doc, passage, quote = it["doc"], it["passage"], it["quote"].strip()
            if doc.kind == "figure":
                status = "figure"  # a model's reading of an image: nothing to match a quote against
            elif it["excerpt"]:
                status = "excerpt"
            elif quote:
                status = status_for(quote_score(quote, passage))
                status = "mismatch" if status == "not_found" else status
            else:
                status = "mismatch"
            key = (doc.id, passage[:200])
            if key in self._passage_keys:
                new.append(self.evidence[self._passage_keys[key]])
                continue
            eid = f"E{len(self.evidence) + 1}"
            ev = Evidence(eid, doc, passage, it["summary"], quote if status != "mismatch" else "",
                          it["relevance"], status, it["abstract_only"], _page_of(doc.text, passage))
            self.evidence[eid] = ev
            self._passage_keys[key] = eid
            new.append(ev)
        return new

    # ---------- registry & formatting ----------

    def _register_papers(self, papers: list[dict]) -> list[Doc]:
        merged: dict[str, dict] = {}
        for p in papers:
            key = (p.get("doi") or "").lower() or _norm_title(p["title"])
            if key in merged:  # keep the richest record
                base = merged[key]
                for k, v in p.items():
                    if not base.get(k) and v:
                        base[k] = v
            else:
                merged[key] = dict(p)
        return [self._add_doc(**{k: p.get(k) for k in ("title", "url", "kind", "year", "venue", "authors", "doi",
                                                       "arxiv_id", "pdf_url", "abstract", "cited_by", "retracted")})
                for p in merged.values()]

    def _add_doc(self, title: str, url: str, kind: str, **kw) -> Doc:
        key = (kw.get("doi") or "").lower() or url or _norm_title(title)
        if key in self._doc_keys:
            doc = self.docs[self._doc_keys[key]]
            for k, v in kw.items():
                if v and not getattr(doc, k):
                    setattr(doc, k, v)
            return doc
        doc = Doc(id=f"D{len(self.docs) + 1}", title=title or url, url=url or "", kind=kind,
                  **{k: v for k, v in kw.items() if v is not None})
        self.docs[doc.id] = doc
        self._doc_keys[key] = doc.id
        return doc

    def _list_docs(self, docs: list[Doc]) -> str:
        lines = []
        for d in docs:
            head = f"[{d.id}] {sources.LABELS[d.kind]} · {d.label}"
            if d.cited_by is not None:
                head += f" · 인용 {d.cited_by}"
            if d.retracted:
                head += " · RETRACTED (철회됨)"
            if d.kind == "blog":
                head += " · 근거로 사용 불가"
            if d.kind not in ("paper", "preprint"):
                head += f" — {urlparse(d.url).hostname or d.url}"
            text = (d.abstract or d.snippet or "").strip()
            lines.append(head + (f"\n    {text[:280]}{'…' if len(text) > 280 else ''}" if text else ""))
        return "\n".join(lines)

    def _fmt_evidence(self, e: Evidence) -> str:
        where = "초록" if e.abstract_only and e.doc.kind in ("paper", "preprint") else sources.LABELS[e.doc.kind]
        page = f" p.{e.page}" if e.page else ""
        out = f"[{e.id}] 관련성 {e.relevance:g} · {QUOTE_BADGE[e.quote_status]} · {where} · {e.doc.label}{page}"
        if e.doc.retracted:
            out += " · RETRACTED"
        if e.summary:
            out += f"\n    요약: {e.summary}"
        if e.quote:
            out += f"\n    인용: \"{e.quote[:500]}\""
        return out


def _page_of(text: str, passage: str) -> int | None:
    """Page number of a passage in text carrying [p.N] markers (user files), if any."""
    at = text.find(passage[:80]) if text else -1
    if at < 0:
        return None
    marks = re.findall(r"\[p\.(\d+)\]", text[:at + 40])
    return int(marks[-1]) if marks else None


def _norm_title(t: str) -> str:
    return re.sub(r"\W", "", (t or "").lower())[:80]


def _safe(name, fn, *args):
    """Run a search API call; return (results, error message) instead of raising."""
    try:
        return fn(*args), None
    except Exception as e:  # network errors, rate limits, API changes: report and carry on
        return [], f"{type(e).__name__}: {str(e)[:120]}"
