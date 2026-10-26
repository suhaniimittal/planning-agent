"""Remove Planning's old memory nodes from Neo4j (2026-10-26).

Since Planning 0.0.32 the memory (facts, TDD versions, decisions) lives in
Memsy and Neo4j keeps only the code. The published agent removes the old
Requirement, Fact, TddVersion and Decision nodes itself on its first memory
check (src/memory/legacy.py); this runs the same cleanup by hand.

Usage (from planning-agent/, with NEO4J_* in .env or a port-forward):
    .venv/bin/python scripts/drop_memory_nodes.py            # count only
    .venv/bin/python scripts/drop_memory_nodes.py --apply    # delete
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.memory import legacy  # noqa: E402


def main(apply: bool) -> None:
    found = legacy.drop() if apply else legacy.count()
    for label, n in found.items():
        print(f"{label}: {n} node(s)")
    if apply:
        print("Deleted; Neo4j now holds only the code graph.")
    else:
        print("Nothing changed. Run again with --apply to delete.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv[1:])
