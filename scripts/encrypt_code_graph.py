"""Encrypt the code already stored in Neo4j in plain text (2026-10-22).

Ingestion encrypts everything it writes from now on, but nodes written before
that still hold plain code until their service is re-ingested. This encrypts
them in place — no clone, no re-parse, no new embeddings — and adds the
search hashes (`text_terms`) to each chunk. Safe to run more than once: a
value already encrypted is left as it is.

Usage (from planning-agent/, with NEO4J_* and CODE_GRAPH_KEY in .env):
    .venv/bin/python scripts/encrypt_code_graph.py           # count only, changes nothing
    .venv/bin/python scripts/encrypt_code_graph.py --apply   # encrypt
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.ingestion import code_crypto  # noqa: E402
from src.ingestion.graph_writer import get_driver  # noqa: E402

BATCH = 500

# label -> (the property naming a node within its service, its service property)
_KEYS = {
    "Class": ("qualified_name", "service"),
    "Function": ("qualified_name", "service"),
    "Chunk": ("qualified_name", "service"),
    "Service": ("name", "name"),
}


def _plain_nodes(session, label: str) -> list[dict]:
    key, service = _KEYS[label]
    fields = code_crypto.ENCRYPTED_FIELDS[label]
    plain = " OR ".join(
        f"(n.{f} IS NOT NULL AND n.{f} <> '' AND NOT n.{f} STARTS WITH $prefix)" for f in fields
    )
    returned = ", ".join(f"n.{f} AS {f}" for f in fields)
    result = session.run(
        f"MATCH (n:{label}) WHERE {plain} "
        f"RETURN n.{service} AS service, n.{key} AS qualified_name, {returned}",
        prefix=code_crypto.PREFIX,
    )
    return [dict(r) for r in result]


def _sealed(label: str, row: dict) -> dict:
    out = code_crypto.encrypt_row(row, label=label, service=row["service"])
    if label == "Chunk" and not code_crypto.is_encrypted(row.get("text")):
        out["text_terms"] = code_crypto.text_terms(row["service"], row.get("text") or "")
    return out


def _write(session, label: str, rows: list[dict]) -> None:
    key, service = _KEYS[label]
    sets = [f"n.{f} = row.{f}" for f in code_crypto.ENCRYPTED_FIELDS[label]]
    if label == "Chunk":
        sets.append("n.text_terms = coalesce(row.text_terms, n.text_terms)")
    match = f"{key}: row.qualified_name"
    if service != key:
        match = f"{service}: row.service, {match}"
    for start in range(0, len(rows), BATCH):
        session.run(
            f"UNWIND $rows AS row MATCH (n:{label} {{{match}}}) " f"SET {', '.join(sets)}",
            rows=rows[start : start + BATCH],
        )


def main(apply: bool) -> None:
    code_crypto.master_key()  # no key: stop before reading anything
    with get_driver().session() as session:
        for label in _KEYS:
            rows = _plain_nodes(session, label)
            print(f"{label}: {len(rows)} node(s) holding plain text")
            if apply and rows:
                _write(session, label, [_sealed(label, r) for r in rows])
                print(f"{label}: encrypted")
    if not apply:
        print("Nothing changed. Run again with --apply to encrypt.")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv[1:])
