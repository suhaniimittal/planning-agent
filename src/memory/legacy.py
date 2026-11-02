"""Planning's old memory nodes in Neo4j, removed (2026-10-26).

Since 0.0.32 the memory (facts, TDD versions, decisions) lives in Memsy and
Neo4j keeps only the code. The Requirement, Fact, TddVersion and Decision
nodes left from 0.0.29-0.0.31 are read by nothing, so the first memory check
a worker makes deletes them, once; on a clean graph it only counts zeros.
Stories rebuild their memory in Memsy on their next "create TDD".
"""

from __future__ import annotations

import logging

from src.ingestion.graph_writer import get_driver

logger = logging.getLogger(__name__)

LABELS = ("Requirement", "Fact", "TddVersion", "Decision")
INDEXES = (
    "requirement_story_idx",
    "fact_story_idx",
    "tdd_version_story_idx",
    "decision_uid_idx",
    "decision_project_idx",
)

_done = False


def count() -> dict[str, int]:
    with get_driver().session() as session:
        return {
            label: session.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            for label in LABELS
        }


def drop() -> dict[str, int]:
    """Deletes the old memory nodes and their indexes; how many there were."""
    found = count()
    with get_driver().session() as session:
        for label, n in found.items():
            if n:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")
        for name in INDEXES:
            session.run(f"DROP INDEX {name} IF EXISTS")
    return found


def drop_once() -> None:
    """drop(), the first time a worker asks; never stops the memory check."""
    global _done
    if _done:
        return
    try:
        found = drop()
        _done = True
        if any(found.values()):
            logger.info("Removed Planning's old Neo4j memory nodes: %s", found)
    except Exception as error:  # noqa: BLE001 - cleanup is retried on the next check
        logger.warning("Old Neo4j memory nodes not removed yet: %s", error)
