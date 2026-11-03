"""Tool entrypoint for the TDD flow without stored code (2026-11-02).

One activity, start to end: the input checked, the attachments read from
storage, the ticket's memory before, the flow (src/tdd/flow.py), the memory
after, then the document as a PDF in storage and, when CONFLUENCE_SPACE_KEY
is set, a Confluence page. It returns the TDD's text and the work order —
never a file's text: an activity's result is kept in Temporal's history.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from aetherion_sdk import tool
from common_lib.storage.storage_client import RetrievalMode, storage

from src.ingestion.confluence_api import ConfluenceApiError, create_page
from src.memory import hooks
from src.query.pdf_render import pdf_from_html
from src.tdd import attachments as card_attachments
from src.tdd import config, flow, inputs, render

_NON_SLUG = re.compile(r"[^a-z0-9]+")


@tool()
async def plan_tdd(planning_input: str, attachments: list[dict] | None = None) -> dict:
    """A plain-English TDD and a work order for one Jira card.

    `planning_input`: the input as JSON text — card, services, questions,
    optional issue_key (src/tdd/inputs.py). `attachments`: uploaded files,
    [{key, name?, description?}], each joined to the card file that describes
    it (attachments.pair). An input that is not valid raises
    inputs.InputError, listing every problem."""
    plan = inputs.parse(planning_input)
    plan.card.files, pairing = card_attachments.pair(plan.card.files, attachments or [])
    for note in pairing:
        print(f"  [TDD] attachments: {note}")
    settings = config.load()
    storage.init_client()
    bucket = _team_id()
    loaded = await asyncio.to_thread(
        card_attachments.load, plan.card.files, lambda key: _read_upload(bucket, key)
    )
    await card_attachments.describe(loaded)
    story = _story_text(plan)
    services = [s.service for s in plan.services]
    memory = await hooks.memory_before(plan.memory_key, story, services)
    doc = await flow.run(plan, settings, loaded, memory["text"] if memory else "")
    doc.warnings += pairing
    if memory:
        await hooks.memory_after(plan.memory_key, story, memory["checked"], doc)

    text = render.render_text(doc)
    html = render.render_html(doc)
    pdf = pdf_from_html(html, doc.title, render.coding_task(doc, render.issue(plan, loaded)))
    file_name = f"technical_designs/{_slug(doc.title)}-{_stamp(doc.generated_at)}.pdf"
    _, s3_key = storage.store_object(bucket, file_name, pdf, "application/pdf")
    return {
        "s3_key": s3_key,
        "title": doc.title,
        "status": doc.status,
        "draft_reason": doc.draft_reason,
        "high_level": doc.high_level,
        "text": text,
        "confluence_html": html,
        "confluence_url": _publish(doc.title, html),
        "decisions": [d.model_dump() for d in doc.decisions],
        "work_order": doc.work_order.model_dump(),
        "warnings": doc.warnings,
        # What the TDD was based on: for Orchestration and the log, not the document.
        "evidence": {"files_read": doc.files_read, "not_read": doc.not_read},
    }


def _story_text(plan: inputs.PlanInput) -> str:
    """What the memory compares between versions: the card and its answers.
    An answer that changes is a requirement that changed."""
    card = plan.card
    lines = [card.title, card.description]
    lines += [f"{q.question} {' / '.join(q.answers)}" for q in plan.questions if q.answered]
    return "\n".join(line for line in lines if line)


def _publish(title: str, body_html: str) -> str | None:
    """Best effort: a Confluence outage or a missing credential never fails a TDD."""
    space_key = os.environ.get("CONFLUENCE_SPACE_KEY")
    if not space_key:
        return None
    try:
        return create_page(space_key, title, body_html)
    except ConfluenceApiError as error:
        print(f"  [CONFLUENCE SKIP] could not publish {title!r}: {error}")
        return None


def _team_id() -> str:
    # TENANT_ID, else STORAGE_AGENTS_BUCKET: the bucket name must equal the
    # team id, per the storage client's own contract.
    return os.environ.get("TENANT_ID") or os.environ["STORAGE_AGENTS_BUCKET"]


def _read_upload(bucket: str, key: str) -> bytes:
    """An uploaded file's bytes, from storage. Running locally (APP_ENV=local),
    a key storage does not have is also tried as a path on this machine: the
    test form uploads to the hosted platform's storage, which a local worker
    cannot see."""
    try:
        return storage.retrieve(bucket, key, RetrievalMode.FULL_OBJECT)
    except Exception:
        local = Path(key).expanduser()
        if os.environ.get("APP_ENV", "").strip().lower() == "local" and local.is_file():
            return local.read_bytes()
        raise


def _slug(text: str, max_len: int = 45) -> str:
    slug = re.sub("-+", "-", _NON_SLUG.sub("-", text.lower())).strip("-")
    if len(slug) > max_len:
        cut = slug.rfind("-", max_len // 2, max_len)
        slug = slug[: cut if cut > 0 else max_len].rstrip("-")
    return slug or "technical-design"


def _stamp(generated_at: str) -> str:
    try:
        when = datetime.fromisoformat(generated_at)
    except ValueError:
        when = datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y%m%d-%H%M%S")
