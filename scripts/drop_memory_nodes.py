"""Remove Planning's old memory nodes from Neo4j (2026-10-26).

Since Planning 0.0.32 the memory (facts, TDD versions, decisions) lives in
Memsy and Neo4j keeps only the code. This deletes the Requirement, Fact,
TddVersion and Decision nodes and their indexes left from 0.0.29-0.0.31.
Stories rebuild their memory in Memsy on their next "create TDD".

Usage (from planning-agent/, with NEO4J_* in .env or a port-forward):
    .venv/bin/python scripts/drop_memory_nodes.py            # count only
    .venv/bin/python scripts/drop_memory_nodes.py --apply    # delete
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.ingestion.graph_writer import get_driver  # noqa: E402

LABELS = ("Requirement", "Fact", "TddVersion", "Decision")
INDEXES = (
    "requirement_story_idx",
    "fact_story_idx",
    "tdd_version_story_idx",
    "decision_uid_idx",
    "decision_project_idx",
)


def main(apply: bool) -> None:
    with get_driver().session() as session:
        for label in LABELS:
            count = session.run(f"MATCH (n:{label}) RETURN count(n) AS n").single()["n"]
            print(f"{label}: {count} node(s)")
            if apply and count:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")
        if apply:
            for name in INDEXES:
                session.run(f"DROP INDEX {name} IF EXISTS")
            print("Deleted; Neo4j now holds only the code graph.")
        else:
            print("Nothing changed. Run again with --apply to delete.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv[1:])
