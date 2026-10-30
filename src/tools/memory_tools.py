"""Tool entrypoints for Planning's memory and bug investigation (2026-10-26).

The agent's "facts", "record" and "investigate" modes: the model calls and
the Neo4j reads. The memory is given by Orchestration and handed back to it;
Planning stores nothing (2026-10-30). And the agreed TDD input as the
planner's text (``handoff_text``).
"""

from __future__ import annotations

import asyncio

from aetherion_sdk import tool

from src.memory import legacy, operations
from src.query import handoff
from src.query.investigate import investigate


@tool()
async def story_facts(story_text: str, snapshot: dict) -> dict:
    """Is this story the same requirement its TDD answered, against the story's
    snapshot Orchestration gives? The first check removes the old Neo4j memory
    nodes (legacy)."""
    await asyncio.to_thread(legacy.drop_once)
    return await operations.check_story(story_text, dict(snapshot or {}))


@tool()
async def record_tdd(payload: dict) -> dict:
    """A published TDD version: its facts and decisions in the story's new
    snapshot, handed back for Orchestration to keep (2026-10-30)."""
    return operations.remember(payload)


@tool()
async def investigate_bug(report_text: str, repos: str = "") -> dict:
    """A bug's likely root cause, affected files and fix, from the code graph —
    only the services named in ``repos`` ("name: org/repo" lines) when given."""
    return await investigate(report_text, service_names(repos))


@tool()
async def handoff_text(payload: dict) -> dict:
    """The agreed TDD input (card, services, questions) as the planner's text,
    and the ranked services' repositories (2026-10-29)."""
    return {"issue_text": handoff.issue_text(payload), "repos": handoff.repos_text(payload)}


def service_names(repos: str) -> list[str]:
    """The service names in a repos text: "bank: org/bank" -> "bank"."""
    names = []
    for line in (repos or "").replace(",", "\n").splitlines():
        line = line.split("#", 1)[0].strip()
        name, sep, repo = line.partition(":")
        if not sep:
            name, sep, repo = line.partition("=")
        if sep and name.strip() and repo.strip():
            names.append(name.strip())
    return names
