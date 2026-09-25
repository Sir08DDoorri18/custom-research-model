"""Free search APIs (no keys needed), page fetching, and source grading."""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import httpx

from . import store

UA = "custom-research-model/0.2 (personal research tool)"
# httpx, not urllib: arXiv's API answers urllib requests with 406.
_http = httpx.Client(headers={"User-Agent": UA, "Accept-Language": "en,ko;q=0.8"}, follow_redirects=True, timeout=25)


def _get(url: str, headers: dict | None = None, retries: int = 2) -> tuple[bytes, str]:
    for attempt in range(retries + 1):
        r = _http.get(url, headers=headers)
        if r.status_code == 429 and attempt < retries:  # shared free quota (Semantic Scholar): back off
            time.sleep(float(r.headers.get("retry-after") or 2 * (attempt + 1)))
            continue
        r.raise_for_status()
        return r.content, r.headers.get("content-type", "")
    raise RuntimeError("unreachable")


def _json(url: str, headers: dict | None = None):
    return json.loads(_get(url, headers=headers)[0])


# ---------- papers ----------

def openalex_search(query: str, n: int = 8) -> list[dict]:
    q = urllib.parse.urlencode({"search": query, "per-page": n})
    papers = []
    for w in _json(f"https://api.openalex.org/works?{q}").get("results", []):
        abstract = _openalex_abstract(w.get("abstract_inverted_index"))
        if not abstract:
            continue
        loc = w.get("best_oa_location") or {}
        papers.append({
            "title": w.get("title") or "",
            "year": w.get("publication_year"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])[:4]],
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "doi": _bare_doi(w.get("doi")),
            "url": w.get("doi") or w.get("id"),
            "pdf_url": loc.get("pdf_url"),
            "cited_by": w.get("cited_by_count", 0),
            "abstract": abstract,
            "kind": "paper",
            "retracted": bool(w.get("is_retracted")),
        })
    return papers


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {i: word for word, idxs in inv.items() for i in idxs}
    return " ".join(pos[i] for i in sorted(pos))


def arxiv_search(query: str, n: int = 5) -> list[dict]:
    # arXiv answers 406 if the ':' in "all:" is percent-encoded, so keep it literal.
    terms = " AND ".join(f"all:{w}" for w in query.split())
    q = urllib.parse.urlencode({"search_query": terms, "max_results": n}, safe=":", quote_via=urllib.parse.quote)
    root = ET.fromstring(_get(f"https://export.arxiv.org/api/query?{q}")[0])
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for e in root.findall("a:entry", ns):
        abs_url = e.findtext("a:id", "", ns)
        papers.append({
            "title": " ".join(e.findtext("a:title", "", ns).split()),
            "year": int(e.findtext("a:published", "0000", ns)[:4]),
            "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:4],
            "venue": "arXiv (preprint)",
            "doi": None,
            "arxiv_id": re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1]),
            "url": abs_url,
            "pdf_url": abs_url.replace("/abs/", "/pdf/"),
            "cited_by": None,
            "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
            "kind": "preprint",
        })
    return papers


S2 = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = "title,year,venue,citationCount,externalIds,url,abstract,authors,openAccessPdf,publicationTypes"


def _s2_headers() -> dict:
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def s2_search(query: str, n: int = 8) -> list[dict]:
    q = urllib.parse.urlencode({"query": query, "limit": n, "fields": S2_FIELDS})
    return [p for p in map(_s2_paper, _json(f"{S2}/paper/search?{q}", _s2_headers()).get("data") or []) if p]


def s2_graph(paper_ref: str, direction: str = "references", n: int = 10) -> list[dict]:
    """Papers this one cites (references) or papers citing it (citations).
    paper_ref: "DOI:10.x/y", "ARXIV:2101.00001" or a Semantic Scholar id."""
    q = urllib.parse.urlencode({"fields": S2_FIELDS, "limit": n})
    data = _json(f"{S2}/paper/{urllib.parse.quote(paper_ref, safe=':')}/{direction}?{q}", _s2_headers())
    key = "citedPaper" if direction == "references" else "citingPaper"
    return [p for p in (_s2_paper(d.get(key) or {}) for d in data.get("data") or []) if p]


def _s2_paper(p: dict) -> dict | None:
    if not p.get("title"):
        return None
    ext = p.get("externalIds") or {}
    arxiv_only = "ArXiv" in ext and not p.get("venue")
    return {
        "title": p["title"],
        "year": p.get("year"),
        "authors": [a.get("name", "") for a in (p.get("authors") or [])[:4]],
        "venue": p.get("venue") or ("arXiv (preprint)" if arxiv_only else None),
        "doi": ext.get("DOI"),
        "arxiv_id": ext.get("ArXiv"),
        "url": f"https://doi.org/{ext['DOI']}" if ext.get("DOI") else p.get("url"),
        "pdf_url": (p.get("openAccessPdf") or {}).get("url") or None,
        "cited_by": p.get("citationCount"),
        "abstract": p.get("abstract") or "",
        "kind": "preprint" if arxiv_only else "paper",
    }


def _bare_doi(doi: str | None) -> str | None:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", doi) if doi else None


# ---------- web ----------

def web_search(query: str, n: int = 8) -> list[dict]:
    from ddgs import DDGS
    return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
            for r in DDGS().text(query, max_results=n) if r.get("href")]


class _TextExtractor(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "aside"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in ("p", "br", "div", "li", "h1", "h2", "h3", "h4", "tr"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.depth:
            self.parts.append(data)


def fetch(url: str) -> dict | None:
    """Readable text of a page or PDF, cached on disk. Returns {"text", "title"} or None."""
    cached = store.get(url)
    if cached:
        return cached
    try:
        body, ctype = _get(url)
    except (httpx.HTTPError, ValueError, OSError):
        return None
    if "pdf" in ctype or url.lower().endswith(".pdf") or body[:5] == b"%PDF-":
        text, title = _pdf_text(body), ""
    else:
        p = _TextExtractor()
        try:
            p.feed(body.decode("utf-8", errors="replace"))
        except Exception:
            return None
        text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.parts))
        text = re.sub(r"\s*\n\s*", "\n", text).strip()
        title = " ".join(p.title.split())
    if not text or len(text) < 200:
        return None
    return store.put(url, text, title=title, via="fetch")


def _pdf_text(body: bytes) -> str | None:
    """Text of a fetched PDF in reading order (two-column layouts included), first 40 pages.
    PyMuPDF: fast enough for search results; added files get the slower Docling read."""
    try:
        import pymupdf
        with pymupdf.open(stream=body, filetype="pdf") as doc:
            return "\n".join(page.get_text("text", sort=True) for page in doc.pages(0, min(40, doc.page_count)))
    except Exception:
        return None


# ---------- grading ----------

LABELS = {"paper": "논문", "institution": "기관", "preprint": "프리프린트", "news": "언론", "web": "웹", "blog": "블로그",
          "file": "내 파일", "figure": "그림해석"}

_INSTITUTION = re.compile(r"(\.gov|\.go\.kr|\.edu|\.ac\.[a-z]{2}|\.mil|\.int|who\.int|nist\.gov|nasa\.gov|"
                          r"europa\.eu|oecd\.org|kostat|nature\.com|science\.org|aps\.org|ieee\.org|acs\.org|"
                          r"springer\.com|wiley\.com|sciencedirect\.com|iop\.org|pnas\.org|cell\.com)$")
_PREPRINT = re.compile(r"(arxiv\.org|biorxiv\.org|medrxiv\.org|chemrxiv\.org|ssrn\.com|researchgate\.net)$")
_NEWS = re.compile(r"(reuters\.com|apnews\.com|bbc\.co\.uk|bbc\.com|nytimes\.com|theguardian\.com|wsj\.com|"
                   r"ft\.com|bloomberg\.com|economist\.com|newscientist\.com|scientificamerican\.com|"
                   r"quantamagazine\.org|phys\.org|sciencedaily\.com|yna\.co\.kr|chosun\.com|joongang\.co\.kr|"
                   r"donga\.com|hani\.co\.kr|khan\.co\.kr|mk\.co\.kr|hankyung\.com|dongascience\.com|"
                   r"zdnet\.co\.kr|etnews\.com|techcrunch\.com|theverge\.com|arstechnica\.com|wired\.com)$")
_BLOG = re.compile(r"(medium\.com|substack\.com|tistory\.com|blog\.naver\.com|velog\.io|brunch\.co\.kr|"
                   r"wordpress\.com|blogspot\.com|reddit\.com|quora\.com|namu\.wiki|dcinside\.com)$")


def grade_url(url: str) -> str:
    host = (urllib.parse.urlparse(url).hostname or "").lower()
    for kind, rx in (("preprint", _PREPRINT), ("institution", _INSTITUTION), ("news", _NEWS), ("blog", _BLOG)):
        if rx.search(host):
            return kind
    return "web"
