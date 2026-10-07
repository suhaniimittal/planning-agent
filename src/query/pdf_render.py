"""Renders a TDD as an actual downloadable PDF — the file a user downloads
from planning_agent's own UI and hands, unmodified, to coding_agent's PDF
input. Kept separate from render.py (which only ever produces text/markup,
never bytes) for the same one-module-per-responsibility reason
confluence_api.py is its own file.

Deliberately reuses render_text()'s exact content rather than building a
second, separate document layout: coding_agent's own PDF-extraction step
re-derives the structured TDD from this PDF's text via an LLM call, so
whatever render_text() already produces (section-numbered, unambiguous
plain text) is also the ideal shape for that later extraction — a fancier
visual layout would only risk making that extraction less reliable, not
more.

xhtml2pdf, not PyMuPDF: PyMuPDF is AGPL, which is a real licensing concern
for code that ships as part of a product; xhtml2pdf is BSD-licensed. It
converts HTML to PDF, so the plain text is wrapped in a minimal HTML shell
(a single <pre> block) rather than reformatted — the wrapping is only there
to give xhtml2pdf something to render, not to restyle the content.
"""

from __future__ import annotations

import html
import io
import textwrap

from .models import TechnicalDesignDoc
from .render import render_text

# xhtml2pdf ignores `white-space: pre-wrap`/`word-wrap`, so a <pre> line is
# never wrapped: anything past ~105 characters (9pt monospace on an A4 page)
# is drawn off the right edge — still in the file's text, but invisible to a
# reader. Lines are therefore wrapped here, before xhtml2pdf sees them.
_WRAP_WIDTH = 100


def _split_long(line: str, continuation: str) -> list[str]:
    """Splits a line that's still too long after word wrapping — one
    unbreakable token, almost always a deep file path — right after a "/",
    never inside a name, so a reader (or coding_agent reading this text)
    sees whole path segments. A hard cut only if there's no "/" to use."""
    pieces: list[str] = []
    while len(line) > _WRAP_WIDTH:
        cut = line.rfind("/", len(continuation) + 1, _WRAP_WIDTH) + 1 or _WRAP_WIDTH
        pieces.append(line[:cut])
        line = continuation + line[cut:]
    pieces.append(line)
    return pieces


def _wrap_lines(text: str) -> str:
    """Wraps every line longer than _WRAP_WIDTH. Continuation lines keep the
    original line's indentation plus two spaces, so nested sections still
    read as nested; a single over-long token (e.g. a deep file path) is split
    at a "/" rather than left running off the page."""
    wrapped: list[str] = []
    for line in text.splitlines():
        if len(line) <= _WRAP_WIDTH:
            wrapped.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        continuation = indent + "  "
        for piece in textwrap.wrap(
            line,
            width=_WRAP_WIDTH,
            subsequent_indent=continuation,
            break_long_words=False,
            break_on_hyphens=False,
        ):
            wrapped.extend(_split_long(piece, continuation))
    return "\n".join(wrapped)


class PdfRenderError(Exception):
    """Raised when xhtml2pdf fails to produce a PDF at all — callers must
    not treat a failed render as an empty-but-valid file."""


def _html_shell(text: str) -> str:
    escaped = html.escape(text)
    return (
        "<html><head><style>"
        "body { font-family: monospace; font-size: 9pt; }"
        "pre { white-space: pre-wrap; word-wrap: break-word; }"
        "</style></head><body><pre>" + escaped + "</pre></body></html>"
    )


def render_pdf(tdd: TechnicalDesignDoc) -> bytes:
    from xhtml2pdf import pisa

    text = _wrap_lines(render_text(tdd))
    buffer = io.BytesIO()
    result = pisa.CreatePDF(_html_shell(text), dest=buffer)
    if result.err:
        raise PdfRenderError(f"xhtml2pdf reported {result.err} error(s) rendering {tdd.title!r}")
    return buffer.getvalue()
