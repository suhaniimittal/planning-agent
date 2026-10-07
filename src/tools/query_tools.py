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


def _slug(text: str, max_len: int = 40) -> str:
    slug = _NON_SLUG_CHARS.sub("-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "issue"


@tool()
async def generate_technical_design(issue_text: str, top_k: int = 5, hops: int = 2) -> dict:
    """Search the service graph for services relevant to a requirement/bug
    report and produce a technical design doc: structured data plus a real
    downloadable PDF — the file a user downloads and hands, unmodified, to
    coding_agent's own PDF input. Uploaded via the platform's storage
    client, with its S3 key returned so the calling agent can surface it as
    a structured_results download link (this is the platform's only file-
    download mechanism, but nothing about it is ever exposed to the user —
    they just see a "Download TDD" button). Also published as a new
    Confluence page when CONFLUENCE_SPACE_KEY is configured — best-effort,
    never blocks the PDF output if it fails or isn't configured."""
    tdd = await build_tdd(issue_text, top_k=top_k, hops=hops)
    pdf_bytes = render_pdf(tdd)

    storage.init_client()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    file_name = f"technical_designs/{timestamp}_{_slug(issue_text)}.pdf"
    _, s3_key = storage.store_object(_team_id(), file_name, pdf_bytes, "application/pdf")

    _publish_to_confluence(tdd)

    return {"s3_key": s3_key}
