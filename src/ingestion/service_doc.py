"""Reads a service's own committed description — `SERVICE.md` at its repo
root — which is what ticket text is matched against when choosing which
services a design involves.

This replaced a generated summary, and the reason is measurable. The old
summaries were written from parsed code, so they described implementation:
ui's said "hooks, styles, components, fixtures, reducers". A ticket says
"Review Deferred Report screen". Those share no vocabulary, and ui ranked
9th of 11 services on a bug in its own form validation. The same ticket
against a committed SERVICE.md scores 0.389 instead of 0.172, and ranks 1st.

A committed file also stops the description drifting: a generated summary
was reworded on every run, which moved its embedding, which silently
reshuffled which services a TDD considered.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

SERVICE_DOC_FILENAME = "SERVICE.md"

# text-embedding-3-small rejects input over 8192 tokens, and embedder._fit
# silently truncates to this many UTF-8 bytes to stay under it. A doc past
# this loses its tail from the embedding with nothing to show for it, so
# ingestion says so out loud rather than letting it pass.
MAX_EMBED_BYTES = 8000


# Headings that introduce a list of things the service does NOT handle.
# Such a section is written to push a service DOWN on tickets about those
# things, and in an embedding it does the opposite: embeddings have no
# representation of negation, so "this service does not send email" and
# "this service sends email" land in nearly the same place. Measured on an
# email ticket, notifications-service's exclusions section was the single
# highest-scoring section in any service's doc (0.353) — the paragraph
# written to disqualify it made it the top match.
_EXCLUSION_HEADING = re.compile(
    r"\s*(what it does not do|what this service does not do"
    r"|not in scope|out of scope|exclusions?|what it doesn't do"
    r"|what this service doesn't do)\b",
    re.I,
)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

# Bumped when the text fed to the embedder is derived differently from the
# same unchanged file. The hash below folds it in, so every service re-embeds
# once on the next run without anyone editing a SERVICE.md.
DOC_EMBED_VERSION = 2


def embedding_text(text: str) -> str:
    """`text` with its exclusions section removed — what actually gets
    embedded.

    The section stays in the doc stored on the Service node, because the
    planner reads that text and can reason about "does not". Only the
    embedding, which cannot, is spared it.

    A section runs from its heading to the next heading at the same level or
    shallower, so a deeper sub-heading inside the exclusions goes with it.
    """
    lines = text.splitlines()
    kept: list[str] = []
    skipping_at: int | None = None
    for line in lines:
        heading = _HEADING.match(line)
        if heading:
            level = len(heading.group(1))
            if skipping_at is not None and level <= skipping_at:
                skipping_at = None
            if skipping_at is None and _EXCLUSION_HEADING.match(heading.group(2)):
                skipping_at = level
                continue
        if skipping_at is None:
            kept.append(line)
    return "\n".join(kept).strip() or text.strip()


def doc_hash(text: str) -> str:
    """Identity of a doc's content, so an edited file is re-embedded and an
    unchanged one is left alone. Keyed on the TEXT, not on the commit: a
    commit that changes only code must not move this service's embedding.

    Hashed over the WHOLE file, not the embedded extract, because the doc
    stored on the node is the whole file: an edit confined to the exclusions
    section changes nothing about the embedding but must still refresh the
    text the planner reads.
    """
    keyed = f"{DOC_EMBED_VERSION}\n{text}"
    return hashlib.sha1(keyed.encode("utf-8")).hexdigest()


def read_local_service_doc(root: Path) -> str | None:
    """`SERVICE.md` from a checkout on disk, or None if it isn't there.

    Used for local mode and for a full GitHub ingest, where the tree is
    already extracted — an incremental GitHub run has no tree and fetches
    the file through the API instead (see ingest.py).
    """
    path = root / SERVICE_DOC_FILENAME
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    return text if text.strip() else None


def oversize_note(service: str, text: str) -> str | None:
    """A warning when only part of `text` will reach the embedding, or None
    when all of it will."""
    size = len(text.encode("utf-8"))
    if size <= MAX_EMBED_BYTES:
        return None
    return (
        f"  [DOC TOO LONG] {service}: {SERVICE_DOC_FILENAME} is {size} bytes, "
        f"only the first {MAX_EMBED_BYTES} are embedded"
    )
