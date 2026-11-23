"""Renders a TDD as an actual downloadable PDF — the file a user downloads
from planning_agent's own UI and hands to coding_agent's PDF input.

The same HTML body Confluence gets (src/tdd/render.py), with a print style:
headings, tables and numbered steps in a proportional font, so it reads as
a document (2026-11-02; before, it was the plain text in one monospace
block). Its text still comes out cleanly for coding_agent's extraction.

xhtml2pdf, not PyMuPDF: PyMuPDF is AGPL, which is a real licensing concern
for code that ships as part of a product; xhtml2pdf is BSD-licensed. It
supports a subset of CSS 2: the style below keeps to it.
"""

from __future__ import annotations

import html
import io
import json

_STYLE = """
@page { size: A4; margin: 1.8cm 1.6cm; }
body { font-family: Helvetica; font-size: 9.5pt; color: #1f2328; line-height: 1.35; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; }
h2 { font-size: 12.5pt; margin: 14pt 0 5pt 0; padding-bottom: 2pt;
     border-bottom: 0.6pt solid #b8bec6; }
h3 { font-size: 10.5pt; margin: 8pt 0 3pt 0; }
h4 { font-size: 10pt; margin: 8pt 0 2pt 0; }
p { margin: 3pt 0; }
p.meta { color: #59636e; font-size: 9pt; margin-bottom: 8pt; }
p.note { background-color: #fff4d6; padding: 4pt; border: 0.6pt solid #e0c26a; }
p.foot { color: #59636e; font-size: 8.5pt; margin-top: 14pt; }
ul, ol { margin: 2pt 0 2pt 0; }
li { margin-bottom: 2pt; }
table { width: 100%; margin: 3pt 0 6pt 0; }
th, td { border: 0.5pt solid #c9ced4; padding: 3pt 4pt; vertical-align: top;
         text-align: left; font-size: 9pt; }
th { background-color: #eef1f4; }
table.kv th { width: 16%; }
"""


class PdfRenderError(Exception):
    """Raised when xhtml2pdf fails to produce a PDF at all — callers must
    not treat a failed render as an empty-but-valid file."""


TASK_KEY = "/CodingTask"


def pdf_from_html(body_html: str, title: str = "", task: dict | None = None) -> bytes:
    """An HTML body (src/tdd/render.render_html) as a styled A4 PDF. `task`
    (render.coding_task) goes into the file's metadata under TASK_KEY, as
    JSON: not shown on any page, read by coding_agent instead of the text."""
    from xhtml2pdf import pisa

    page = (
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>{body_html}</body></html>"
    )
    buffer = io.BytesIO()
    result = pisa.CreatePDF(page, dest=buffer, encoding="utf-8")
    if result.err:
        raise PdfRenderError(f"xhtml2pdf reported {result.err} error(s) rendering {title!r}")
    if task is None:
        return buffer.getvalue()
    return _with_metadata(buffer.getvalue(), {TASK_KEY: json.dumps(task, ensure_ascii=False)})


def _with_metadata(pdf: bytes, entries: dict[str, str]) -> bytes:
    from pypdf import PdfWriter

    writer = PdfWriter(clone_from=io.BytesIO(pdf))
    writer.add_metadata(entries)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()
