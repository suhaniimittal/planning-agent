"""Tool entrypoints for Planning's memory (2026-10-26).

All the I/O of the agent's "facts" and "record" modes: Memsy reads and
writes, and the model calls. The agent only routes to these.
"""

from __future__ import annotations

from aetherion_sdk import tool

from src.memory import conversation, operations


@tool()
async def story_facts(issue_key: str, story_text: str) -> dict:
    """Is this story the same requirement its TDD answered? Stores nothing in
    the memory."""
    return await operations.check_story(issue_key, story_text)


@tool()
async def record_tdd(payload: dict) -> dict:
    """A published TDD version: its facts and decisions, remembered — and a note
    in the TDD's Memsy conversation (2026-10-26)."""
    stored = operations.remember(payload)
    await conversation.remember_version(
        str(payload.get("issue_key") or ""),
        str(payload.get("tdd_version") or ""),
        str(payload.get("page_id") or ""),
        list(payload.get("decisions") or []),
        list(payload.get("superseded") or []),
    )
    return stored
