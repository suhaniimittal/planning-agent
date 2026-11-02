"""The memory around a TDD run (2026-10-27; moved here 2026-11-02).

Before planning: what the ticket's memory says. After: the TDD saved as the
ticket's next version. Called by src/tools/tdd_tools.py around the TDD flow.

Memory never stops a TDD: a Memsy that cannot be read or written is logged,
and the run goes on without it.
"""

from __future__ import annotations

import asyncio

from src.memory import conversation, operations, record


def change_line(change) -> str:
    if not isinstance(change, dict):
        return str(change)
    what = change.get("slot") or change.get("value") or "requirement"
    if change.get("old") or change.get("new"):
        return f"{what}: {change.get('old') or '(none)'} -> {change.get('new') or '(none)'}"
    return f"{what}: {change.get('why') or change}"


async def memory_before(
    issue_key: str, story_text: str, services: list[str] | None
) -> dict | None:
    """The ticket's memory before planning (2026-10-27): whether its story
    changed since its last version, its agreed decisions and its project's on
    the same services, and its own proposed decisions. {"checked", "text"},
    or None when Memsy cannot be read — memory never stops a TDD."""
    key = issue_key.upper()
    try:
        checked = await operations.check_story(key, story_text)
        agreed = await asyncio.to_thread(record.decisions_in_force, key, services)
        earlier = await asyncio.to_thread(record.proposed_decisions, key)
    except Exception as error:  # noqa: BLE001 - planned without memory instead
        print(f"  [MEMORY] {key}: not available ({type(error).__name__}: {error}); "
              "planning without it")
        return None
    current = checked.get("current") or {}
    changes = [change_line(c) for c in checked.get("changes") or []]
    print(f"  [MEMORY] {key}: story {checked.get('verdict')}"
          + (f" since version {current['tdd_version']}" if current else "")
          + (f" ({'; '.join(changes)})" if changes else "")
          + f"; {len(agreed)} agreed and {len(earlier)} proposed decisions")
    parts = []
    if current:
        parts.append(f"This ticket's TDD was last written as version {current['tdd_version']}.")
    if checked.get("verdict") in ("changed", "unsure") and (changes or checked.get("reason")):
        parts.append(
            "The ticket has changed since then: " + ("; ".join(changes) or checked["reason"])
            + ". Design for the ticket as it reads now."
        )
    block = record.decisions_block(agreed, key, earlier)
    if block:
        parts.append(block)
    return {"checked": checked, "text": "\n\n".join(parts)}


async def memory_after(issue_key: str, story_text: str, checked: dict, tdd) -> None:
    """The generated TDD saved to the ticket's memory as its next version,
    with its decisions proposed (2026-10-27). ``tdd`` is anything with a
    ``title`` and DesignDecision ``decisions``. A failure is logged; the TDD
    is delivered either way."""
    key = issue_key.upper()
    decisions = [d.model_dump() for d in tdd.decisions]
    try:
        saved = await asyncio.to_thread(
            operations.remember_generated, key, story_text, checked, decisions
        )
        await conversation.remember_version(
            key, saved["version"], "", decisions, [], generated=tdd.title
        )
    except Exception as error:  # noqa: BLE001 - the TDD is delivered either way
        print(f"  [MEMORY] {key}: could not save this version ({type(error).__name__}: {error})")
        return
    print(f"  [MEMORY] {key}: saved as version {saved['version']} "
          f"({saved['decisions']} decisions, proposed)")
