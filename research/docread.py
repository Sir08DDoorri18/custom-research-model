"""Reads documents with Docling and writes each result next to it. Run by files.py as

    python -m research.docread <input> <output dir> [<input> <output dir> ...] [--formulas]

in its own process, because Docling's PDF engine can't handle non-ASCII characters in the
Windows home/temp paths (e.g. a Korean user name) and files.py points those at an ASCII
folder for this process only. Several files share one process so the models load once.
Output per file: <output dir>/partial.json and fig<N>.png; "FAILED <output dir> <reason>"
on stderr for a file that could not be read.
"""
import json
import logging
import sys
from pathlib import Path

PAGE_BREAK = "\n\n<<<PAGE>>>\n\n"
MAX_FIGURES = 24
MIN_FIGURE_PX = 120  # smaller pictures are logos and icons


def main(jobs: list[tuple[str, str]], formulas: bool) -> None:
    logging.disable(logging.WARNING)
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.generate_picture_images = True   # figures come out as images for the vision model
    opts.images_scale = 2.0
    opts.do_formula_enrichment = formulas  # equations as LaTeX (slower)
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    for inp, out in jobs:
        try:
            convert(converter, inp, out, formulas)
        except Exception as e:  # report and go on with the other files
            print(f"FAILED {out} {type(e).__name__}: {str(e)[:200]}".replace("\n", " "), file=sys.stderr, flush=True)


def convert(converter, inp: str, out: str, formulas: bool) -> None:
    doc = converter.convert(inp).document

    pages = doc.export_to_markdown(page_break_placeholder=PAGE_BREAK).split(PAGE_BREAK)
    text = "\n\n".join(f"[p.{n}]\n{body.strip()}" for n, body in enumerate(pages, 1) if body.strip()) \
        if len(pages) > 1 else pages[0]

    figures = []
    for pic in doc.pictures:
        if len(figures) >= MAX_FIGURES:
            break
        img = pic.get_image(doc)
        if img is None or min(img.size) < MIN_FIGURE_PX:
            continue
        name = f"fig{len(figures) + 1}.png"
        img.save(Path(out) / name)
        figures.append({"file": name, "page": pic.prov[0].page_no if pic.prov else None,
                        "caption": pic.caption_text(doc) or ""})

    result = {"engine": "docling" + (" +formulas" if formulas else ""), "text": text,
              "pages": doc.num_pages(), "tables": len(doc.tables), "figures": figures}
    (Path(out) / "partial.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(list(zip(args[0::2], args[1::2])), "--formulas" in sys.argv)
