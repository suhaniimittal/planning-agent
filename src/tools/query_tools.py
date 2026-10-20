"""Tool entrypoint for the query-time flow."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

from aetherion_sdk import tool
from common_lib.storage.storage_client import storage

from src.ingestion.confluence_api import ConfluenceApiError, create_page
from src.query.pdf_render import render_pdf
from src.query.query_flow import build_tdd
from src.query.render import render_confluence

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def _publish_to_confluence(tdd) -> str | None:
    """Best-effort: publishes the TDD as a new Confluence page and returns
    its URL, or None if Confluence isn't configured or the call fails for
    any reason. Never raises — a Confluence outage or a missing credential
    must never stop TDD generation from succeeding, same "one failure never
    aborts the run" principle used throughout this pipeline (SummarizerError,
    GitHubApiError, RepoFetchError all follow it too)."""
    space_key = os.environ.get("CONFLUENCE_SPACE_KEY")
    if not space_key:
        return None
    try:
        body_html = render_confluence(tdd)
        return create_page(space_key, tdd.title, body_html)
    except ConfluenceApiError as e:
        print(f"  [CONFLUENCE SKIP] could not publish {tdd.title!r}: {e}")
        return None


def _team_id() -> str:
    """Same bucket-resolution convention the ingestion pipeline used for its
    (now-removed) S3 doc source: TENANT_ID, falling back to
    STORAGE_AGENTS_BUCKET — the bucket name must equal the team_id per the
    storage client's own contract."""
    return os.environ.get("TENANT_ID") or os.environ["STORAGE_AGENTS_BUCKET"]


def _slug(text: str, max_len: int = 45) -> str:
    """A filename-safe slug, cut at a word boundary.

    Cutting at a fixed character count left names ending mid-word
    ("...silently-fails-wit"), so the truncation backs up to the last
    hyphen instead when one is close enough to the limit."""
    slug = _NON_SLUG_CHARS.sub("-", text.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    if len(slug) <= max_len:
        return slug or "technical-design"
    cut = slug.rfind("-", max_len // 2, max_len)
    return slug[: cut if cut > 0 else max_len].rstrip("-") or "technical-design"


def _stamp(generated_at: str | None) -> str:
    """The run's own generation time as a compact, sortable filename part.

    Taken from the TDD's `generated_at` rather than the clock, so the
    filename and the "Generated:" line inside the PDF always name the same
    run — the whole point of the stamp is telling two runs of one ticket
    apart, which a filename that disagreed with the document would defeat.
    Falls back to now() only for a TDD that was never stamped."""
    try:
        when = datetime.fromisoformat(generated_at) if generated_at else datetime.now(UTC)
    except ValueError:
        when = datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y%m%d-%H%M%S")


@tool()
async def generate_technical_design(
    issue_text: str,
    top_k: int | None = None,
    hops: int | None = None,
    service_names: list[str] | None = None,
    mode: str | None = None,
) -> dict:
    """Search the service graph for services relevant to a requirement/bug
    report and produce a technical design doc: structured data plus a real
    downloadable PDF — the file a user downloads and hands, unmodified, to
    coding_agent's own PDF input. Uploaded via the platform's storage
    client, with its S3 key returned so the calling agent can surface it as
    a structured_results download link (this is the platform's only file-
    download mechanism, but nothing about it is ever exposed to the user —
    they just see a "Download TDD" button). Also published as a new
    Confluence page when CONFLUENCE_SPACE_KEY is configured — best-effort,
    never blocks the PDF output if it fails or isn't configured.

    `service_names`: the services this run covers; None considers every
    service in the graph."""
    # Only what the caller actually chose is forwarded, so build_tdd's own
    # signature stays the single definition of what top_k and hops default
    # to — nothing here restates the numbers.
    chosen = {
        name: value for name, value in (("top_k", top_k), ("hops", hops)) if value is not None
    }
    tdd = await build_tdd(issue_text, service_names=service_names, mode=mode, **chosen)
    pdf_bytes = render_pdf(tdd)

    storage.init_client()
    # Named from the TDD's own title, not the raw issue text: the title is
    # written to describe the change, while the ticket text opens with
    # whatever the reporter typed first and truncated to something like
    # "deferred-order-submit-silently-fails-wit".
    #
    # The timestamp suffix gives every run its own name and its own storage
    # key. Without it, re-running a ticket overwrote its previous PDF at the
    # same key, and a browser downloading the result still appended "(2)",
    # "(3)" to avoid a local filename collision — so a stale download and a
    # fresh one looked identical and only the "Generated:" line inside the
    # PDF could tell them apart. The cost is accepted: one ticket run twice
    # now leaves two files.
    file_name = f"technical_designs/{_slug(tdd.title)}-{_stamp(tdd.generated_at)}.pdf"
    _, s3_key = storage.store_object(_team_id(), file_name, pdf_bytes, "application/pdf")

    _publish_to_confluence(tdd)

    return {"s3_key": s3_key}
