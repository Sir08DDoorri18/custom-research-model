"""Files the user adds: PDFs, documents, tables and pictures.

Text, tables and equations are read on this PC (Docling, with PyMuPDF as a fast fallback).
Figures inside documents and photos are described by an image-reading model (the `vision`
role in models.yaml), which means those images are sent to that provider.
Results are cached by file content, so the same file is only read once.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import config, llm
from .config import ROOT
from .trace import pmap

CACHE = ROOT / ".cache" / "files"
ASCII_HOME = ROOT / ".cache" / "tmp"
TEXT = {".txt", ".md", ".csv", ".tsv", ".json"}
DOCLING = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
SUPPORTED = TEXT | DOCLING | IMAGES


@dataclass
class Figure:
    image: Path
    page: int | None = None
    caption: str = ""
    description: str = ""


@dataclass
class ReadFile:
    name: str
    text: str
    engine: str
    pages: int | None = None
    tables: int = 0
    figures: list[Figure] = field(default_factory=list)

    @property
    def is_image(self) -> bool:
        return Path(self.name).suffix.lower() in IMAGES


def read(path: str | Path, name: str | None = None) -> ReadFile:
    """Text and figures of one file (figures not yet described)."""
    result = read_many([(path, name)])[0]
    if isinstance(result, Exception):
        raise result
    return result


def read_many(items: list[tuple[str | Path, str | None]]) -> list[ReadFile | Exception]:
    """Read several files; documents that need Docling share one child process, so its
    models load once. Each entry is a ReadFile, or the exception that stopped that file."""
    out: list[ReadFile | Exception | None] = [None] * len(items)
    pending: list[tuple[int, Path, Path, str]] = []  # (index, local copy, cache folder, suffix)
    for i, (path, name) in enumerate(items):
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED:
            out[i] = ValueError(f"지원하지 않는 형식이에요: {suffix} (가능: {' '.join(sorted(SUPPORTED))})")
            continue
        data = path.read_bytes()
        folder = CACHE / hashlib.sha256(data).hexdigest()[:16]
        if (folder / "result.json").exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        local = folder / f"input{suffix}"  # ASCII path for the PDF engine
        local.write_bytes(data)
        if suffix in DOCLING:
            pending.append((i, local, folder, suffix))
        else:
            _finish(folder, local, _convert_simple(local, suffix))

    if pending:
        failures = _docling([(local, folder) for _, local, folder, _ in pending])
        for i, local, folder, suffix in pending:
            if not (folder / "partial.json").exists():
                reason = failures.get(str(folder), "unknown error")
                if suffix != ".pdf":
                    out[i] = RuntimeError(f"문서를 읽지 못했어요: {reason}")
                    continue
                _finish(folder, local, _pymupdf(local, reason))
            else:
                _finish(folder, local, json.loads((folder / "partial.json").read_text(encoding="utf-8")))
                (folder / "partial.json").unlink()

    for i, (path, name) in enumerate(items):
        if out[i] is None:
            path = Path(path)
            out[i] = _load(CACHE / hashlib.sha256(path.read_bytes()).hexdigest()[:16], name or path.name)
    return out


def _finish(folder: Path, local: Path, result: dict) -> None:
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    if local.suffix.lower() not in IMAGES:  # the extracted text is all we need; photos are the figure
        local.unlink(missing_ok=True)


def _load(folder: Path, name: str) -> ReadFile:
    cached = folder / "result.json"
    os.utime(cached)  # last use, for cleanup's least-recently-used order
    result = json.loads(cached.read_text(encoding="utf-8"))
    figures = [Figure(folder / f["file"], f.get("page"), f.get("caption", ""), f.get("description", ""))
               for f in result.get("figures", [])]
    return ReadFile(name, result.get("text", ""), result.get("engine", ""), result.get("pages"),
                    result.get("tables", 0), figures)


def _convert_simple(local: Path, suffix: str) -> dict:
    if suffix in TEXT:
        return {"engine": "text", "text": local.read_text(encoding="utf-8", errors="replace")}
    return {"engine": "image", "text": "", "figures": [{"file": local.name, "page": None, "caption": ""}]}


def _docling(jobs: list[tuple[Path, Path]]) -> dict[str, str]:
    """Convert documents in one child process. Returns {folder: error} for those that failed."""
    ASCII_HOME.mkdir(parents=True, exist_ok=True)
    home = str(ASCII_HOME)
    env = {**os.environ, "TEMP": home, "TMP": home, "USERPROFILE": home, "HOME": home, "APPDATA": home,
           "LOCALAPPDATA": home, "HOMEPATH": home[2:], "HOMEDRIVE": home[:2],
           "HF_HOME": str(ROOT / ".cache" / "hf"), "PYTHONIOENCODING": "utf-8"}
    cmd = [sys.executable, "-m", "research.docread"] + [str(p) for job in jobs for p in job]
    if (config.load().get("files") or {}).get("formulas"):
        cmd.append("--formulas")
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=1800 * len(jobs))
        log = (proc.stderr or proc.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        log = f"{type(e).__name__}: {e}"
    errors = {}
    for line in log.splitlines():  # docread prints "FAILED <folder> <reason>" per failed file
        if line.startswith("FAILED "):
            _, folder, reason = line.split(" ", 2)
            errors[folder] = reason
    fallback = log.splitlines()[-1][:300] if log else "docread produced no output"
    return {str(folder): errors.get(str(folder), fallback) for _, folder in jobs}


def _pymupdf(local: Path, reason: str = "") -> dict:
    """Fast fallback: good reading order and tables, no figure extraction."""
    import pymupdf4llm
    chunks = pymupdf4llm.to_markdown(str(local), page_chunks=True)
    text = "\n\n".join(f"[p.{i}]\n{c['text'].strip()}" for i, c in enumerate(chunks, 1) if c["text"].strip())
    return {"engine": "pymupdf" + (f" (docling 실패: {reason[:80]})" if reason else ""), "text": text,
            "pages": len(chunks)}


# ---------- figures ----------

VISION_PROMPT = """This image is {where}.{caption}
Describe it for someone who cannot see it, in {language}:
- what kind of image it is (line chart, bar chart, diagram, table, photo, handwritten notes...)
- for charts: axes with units and ranges, each series, the trend, and key values read off the
  plot (say they are approximate)
- transcribe any text, labels, numbers or equations exactly
Do not guess beyond what is visible."""


def describe(read_file: ReadFile, limit: int = 12) -> list[Figure]:
    """Have the vision model describe each figure (cached next to the figure images)."""
    todo = [f for f in read_file.figures[:limit] if not f.description]

    def one(fig: Figure) -> None:
        where = f"page {fig.page} of {read_file.name}" if fig.page else f"the file {read_file.name}"
        caption = f"\nIts caption in the document: {fig.caption}" if fig.caption else ""
        msg = [{"role": "user", "content": [
            {"type": "text", "text": VISION_PROMPT.format(where=where, caption=caption, language=config.language())},
            {"type": "image_url", "image_url": {"url": data_url(fig.image)}}]}]
        label = f"그림 해석 · {read_file.name}" + (f" p.{fig.page}" if fig.page else "")
        try:
            fig.description = llm.ask("vision", msg, label).text
        except llm.LLMError as e:
            fig.description = ""
            print(f"[files] figure not described: {e}", flush=True)

    pmap(one, todo, workers=3)
    _save_descriptions(read_file)
    return read_file.figures[:limit]


def _save_descriptions(read_file: ReadFile) -> None:
    if not read_file.figures:
        return
    cached = read_file.figures[0].image.parent / "result.json"
    result = json.loads(cached.read_text(encoding="utf-8"))
    by_file = {f.image.name: f.description for f in read_file.figures}
    for f in result.get("figures", []):
        f["description"] = by_file.get(f["file"], f.get("description", ""))
    cached.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def data_url(image: Path, max_side: int = 1568) -> str:
    """PNG data URL, downscaled so the long side is at most max_side pixels."""
    b64, mime = image_b64(image, max_side)
    return f"data:{mime};base64,{b64}"


def image_b64(image: Path, max_side: int = 1568) -> tuple[str, str]:
    from PIL import Image
    with Image.open(image) as im:
        im = im.convert("RGB") if im.mode not in ("RGB", "L") else im
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"
