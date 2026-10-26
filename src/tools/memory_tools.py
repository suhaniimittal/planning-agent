"""Tool entrypoints for Planning's memory and bug investigation (2026-10-26).

All the I/O of the agent's "facts", "record" and "investigate" modes: Neo4j
reads and writes, and the model calls. The agent only routes to these.
"""

from __future__ import annotations

from aetherion_sdk import tool

from src.memory import operations
from src.query.investigate import investigate


@tool()
async def story_facts(issue_key: str, story_text: str) -> dict:
    """Is this story the same requirement its TDD answered? Stores nothing."""
    return await operations.check_story(issue_key, story_text)


@tool()
async def record_tdd(payload: dict) -> dict:
    """A published TDD version: its facts and decisions, remembered."""
    return operations.remember(payload)


@tool()
async def investigate_bug(report_text: str, repos: str = "") -> dict:
    """A bug's likely root cause, affected files and fix, from the code graph —
    only the services named in ``repos`` ("name: org/repo" lines) when given."""
    return await investigate(report_text, service_names(repos))


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
