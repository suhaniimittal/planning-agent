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
from pathlib import Path

SERVICE_DOC_FILENAME = "SERVICE.md"

# text-embedding-3-small rejects input over 8192 tokens, and embedder._fit
# silently truncates to this many UTF-8 bytes to stay under it. A doc past
# this loses its tail from the embedding with nothing to show for it, so
# ingestion says so out loud rather than letting it pass.
MAX_EMBED_BYTES = 8000


def doc_hash(text: str) -> str:
    """Identity of a doc's content, so an edited file is re-embedded and an
    unchanged one is left alone. Keyed on the TEXT, not on the commit: a
    commit that changes only code must not move this service's embedding."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


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
