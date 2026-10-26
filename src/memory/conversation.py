"""The conversation about a TDD, kept in Memsy (2026-10-26).

Neo4j keeps what was decided (facts, versions, decisions). Memsy keeps the
conversation: the questions and cross-questions people ask about a TDD, the
answers, and a note for each published version, so a later question is
answered knowing what was already discussed, even on another day.

- actor_id   = "tdd:<STORY>"  — everything about one story's TDD
- session_id = "page:<page id>" for a TDD page's discussion, "story:<STORY>"
  for questions asked on the Jira story, "versions:<STORY>" for version notes
- team_id    = the Jira project, so a project's conversations can be searched

Memsy is optional and never blocks an answer: without MEMSY_API_KEY nothing
is sent; a slow or failing Memsy is logged and the answer goes on without it.
Only Planning talks to Memsy. Code is never sent, only the conversation.
"""

from __future__ import annotations

import json
import logging
import os

from src.ingestion.secret_store import secret

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 8.0
RECALL_LIMIT = 5
MAX_CONTENT = 7_500  # Memsy's limit is 8,000 characters per event


def actor_of(issue_key: str) -> str:
    return f"tdd:{issue_key.strip().upper()}"


def session_of(issue_key: str, page_id: str = "") -> str:
    return f"page:{page_id}" if page_id else f"story:{issue_key.strip().upper()}"


def _client():
    """An async Memsy client, or None when Memsy is not configured."""
    key = secret("MEMSY_API_KEY")
    base = os.environ.get("MEMSY_BASE_URL", "").strip()
    if not key or not base:
        return None
    from memsy import AsyncMemsyClient

    return AsyncMemsyClient(base_url=base, api_key=key, timeout=TIMEOUT_SECONDS, max_retries=1)


def _event(issue_key: str, session: str, kind: str, content: str, metadata: dict):
    from memsy import EventPayload

    return EventPayload(
        actor_id=actor_of(issue_key),
        session_id=session,
        kind=kind,
        content=content[:MAX_CONTENT],
        team_id=issue_key.split("-")[0].upper(),
        metadata=json.dumps({"story_key": issue_key.upper(), **metadata})[:4000],
    )


async def recall(issue_key: str, question: str, limit: int = RECALL_LIMIT) -> list[str]:
    """What was said before about this TDD, most relevant first; [] when none
    or when Memsy is not available."""
    client = _client()
    if client is None or not issue_key or not question.strip():
        return []
    try:
        found = await client.search(question, actor_id=actor_of(issue_key), limit=limit)
        return [str(r.content).strip() for r in (found.results or []) if str(r.content).strip()]
    except Exception as error:  # noqa: BLE001 - recall is a help, never a requirement
        logger.warning("Memsy recall skipped for %s: %s", issue_key, error)
        return []
    finally:
        await _close(client)


async def remember_exchange(
    issue_key: str,
    page_id: str,
    question: str,
    answer: str,
    kind: str = "question",
    verdict: str = "",
) -> bool:
    """The question and the answer, as one turn of the TDD's conversation."""
    client = _client()
    if client is None or not issue_key or not question.strip():
        return False
    session = session_of(issue_key, page_id)
    meta = {"kind": kind, "verdict": verdict, "page_id": page_id}
    events = [_event(issue_key, session, "user_message", question, meta)]
    if answer.strip():
        events.append(_event(issue_key, session, "assistant_message", answer, meta))
    return await _ingest(client, issue_key, events)


async def remember_version(
    issue_key: str, version: str, page_id: str, decisions: list[dict], superseded: list[str]
) -> bool:
    """A note that a TDD version was published, with the decisions it carries."""
    client = _client()
    if client is None or not issue_key or not version:
        return False
    agreed = "; ".join(
        f"{d.get('title') or d.get('id')}: {d.get('choice') or ''}".strip(": ")
        for d in decisions
        if d.get("id")
    )
    text = f"{issue_key.upper()} TDD {version} was published."
    if agreed:
        text += f" Decisions in force: {agreed}."
    if superseded:
        text += f" Replaced decisions: {', '.join(superseded)}."
    event = _event(
        issue_key,
        f"versions:{issue_key.strip().upper()}",
        "app_event",
        text,
        {"tdd_version": version, "page_id": page_id},
    )
    return await _ingest(client, issue_key, [event])


async def _ingest(client, issue_key: str, events: list) -> bool:
    try:
        await client.ingest(events)
        return True
    except Exception as error:  # noqa: BLE001 - the answer is posted either way
        logger.warning("Memsy could not save the conversation for %s: %s", issue_key, error)
        return False
    finally:
        await _close(client)


async def _close(client) -> None:
    close = getattr(client, "aclose", None) or getattr(client, "close", None)
    if close is None:
        return
    try:
        result = close()
        if hasattr(result, "__await__"):
            await result
    except Exception:  # noqa: BLE001 - closing never matters to the caller
        pass
