"""Planning's memory in Neo4j (2026-10-26): what a story requires, which TDD
version answered it, and the design decisions its versions keep.

The same database as the code graph, encrypted the same way (code_crypto):
a fact's statement and values, a decision's title, choice and reason are
stored as ciphertext; slots, ids, statuses, hashes and versions stay plain so
they can be queried. Only the Planning agent reads or writes this.

    (:Requirement {story_key, project_key, facts_hash, story_text_hash, updated})
        -[:HAS_FACT]->(:Fact {story_key, slot, kind, statement*, values*})
    (:TddVersion {story_key, version, page_id, facts_hash, created})
    (:Decision {uid, id, story_key, project_key, status, source, override_by,
                title*, choice*, reason*, targets})
        -[:APPLIES_TO]->(:Function | :Class | :File)        * = encrypted

A decision is agreed on one story and visible to the project: a later design
in the same project that touches the same service gets it as a constraint.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.ingestion import code_crypto
from src.ingestion.graph_writer import get_driver

AGREED, SUPERSEDED = "agreed", "superseded"


def create_memory_indexes() -> None:
    """Idempotent: safe on every run."""
    with get_driver().session() as session:
        for statement in (
            "CREATE INDEX requirement_story_idx IF NOT EXISTS FOR (r:Requirement) ON (r.story_key)",
            "CREATE INDEX fact_story_idx IF NOT EXISTS FOR (f:Fact) ON (f.story_key)",
            "CREATE INDEX tdd_version_story_idx IF NOT EXISTS FOR (v:TddVersion) ON (v.story_key)",
            "CREATE INDEX decision_uid_idx IF NOT EXISTS FOR (d:Decision) ON (d.uid)",
            "CREATE INDEX decision_project_idx IF NOT EXISTS FOR (d:Decision) ON (d.project_key)",
        ):
            session.run(statement)


def project_of(story_key: str) -> str:
    return story_key.split("-")[0].upper() if story_key else ""


def decision_uid(story_key: str, decision_id: str) -> str:
    return f"{story_key.upper()}:{decision_id}"


# --- facts --------------------------------------------------------------------------


def _sealed_fact(story_key: str, fact: dict) -> dict:
    row = {
        "slot": fact["slot"],
        "kind": fact.get("kind") or "rule",
        "statement": fact.get("statement") or "",
        "values": json.dumps(list(fact.get("values") or []), ensure_ascii=False),
    }
    return code_crypto.encrypt_row(row, label="Fact", service=story_key, qualified_name=row["slot"])


def _opened_fact(story_key: str, row: dict) -> dict:
    opened = code_crypto.decrypt_row(
        dict(row), label="Fact", service=story_key, qualified_name=row["slot"]
    )
    values = opened.get("values") or "[]"
    try:
        opened["values"] = json.loads(values)
    except (TypeError, ValueError):
        opened["values"] = []
    return {k: opened.get(k) for k in ("slot", "kind", "statement", "values")}


def save_requirement(
    story_key: str, facts: list[dict], facts_hash: str, story_text_hash: str
) -> int:
    """Replaces the story's facts with ``facts``; returns how many were stored."""
    story_key = story_key.upper()
    rows = [_sealed_fact(story_key, f) for f in facts if f.get("slot")]
    with get_driver().session() as session:
        session.run(
            """
            MERGE (r:Requirement {story_key: $story_key})
            SET r.project_key = $project_key, r.facts_hash = $facts_hash,
                r.story_text_hash = $story_text_hash, r.updated = $now
            WITH r
            OPTIONAL MATCH (r)-[:HAS_FACT]->(old:Fact)
            DETACH DELETE old
            """,
            story_key=story_key,
            project_key=project_of(story_key),
            facts_hash=facts_hash,
            story_text_hash=story_text_hash,
            now=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        if rows:
            session.run(
                """
                MATCH (r:Requirement {story_key: $story_key})
                UNWIND $rows AS row
                CREATE (f:Fact {story_key: $story_key})
                SET f += row
                CREATE (r)-[:HAS_FACT]->(f)
                """,
                story_key=story_key,
                rows=rows,
            )
    return len(rows)


def load_requirement(story_key: str) -> dict | None:
    """{facts, facts_hash, story_text_hash} for the story, or None when nothing is stored."""
    story_key = story_key.upper()
    with get_driver().session() as session:
        head = session.run(
            "MATCH (r:Requirement {story_key: $story_key}) "
            "RETURN r.facts_hash AS facts_hash, r.story_text_hash AS story_text_hash",
            story_key=story_key,
        ).single()
        if head is None:
            return None
        rows = session.run(
            """
            MATCH (:Requirement {story_key: $story_key})-[:HAS_FACT]->(f:Fact)
            RETURN f.slot AS slot, f.kind AS kind, f.statement AS statement, f.values AS values
            ORDER BY f.slot
            """,
            story_key=story_key,
        )
        facts = [_opened_fact(story_key, dict(r)) for r in rows]
    return {
        "facts": facts,
        "facts_hash": head["facts_hash"] or "",
        "story_text_hash": head["story_text_hash"] or "",
    }


# --- TDD versions --------------------------------------------------------------------


def save_version(story_key: str, version: str, page_id: str, facts_hash: str) -> None:
    with get_driver().session() as session:
        session.run(
            """
            MERGE (v:TddVersion {story_key: $story_key, version: $version})
            SET v.page_id = $page_id, v.facts_hash = $facts_hash,
                v.created = coalesce(v.created, $now)
            """,
            story_key=story_key.upper(),
            version=version,
            page_id=page_id,
            facts_hash=facts_hash,
            now=datetime.now(UTC).isoformat(timespec="seconds"),
        )


def current_version(story_key: str) -> dict | None:
    """The newest recorded version of the story's TDD: {tdd_version, page_id}."""
    with get_driver().session() as session:
        row = session.run(
            """
            MATCH (v:TddVersion {story_key: $story_key})
            RETURN v.version AS version, v.page_id AS page_id
            ORDER BY v.created DESC, v.version DESC LIMIT 1
            """,
            story_key=story_key.upper(),
        ).single()
    return {"tdd_version": row["version"], "page_id": row["page_id"]} if row else None


# --- decisions -----------------------------------------------------------------------


def _targets(decision: dict) -> list[dict]:
    out = []
    for target in decision.get("applies_to") or []:
        if isinstance(target, dict) and target.get("service") and target.get("file_path"):
            out.append(
                {
                    "service": str(target["service"]),
                    "file_path": str(target["file_path"]),
                    "symbol": str(target.get("symbol") or ""),
                }
            )
    return out


def save_decisions(
    story_key: str,
    decisions: list[dict],
    superseded: list[str] | None = None,
    source: str = "",
) -> int:
    """Stores agreed decisions (by stable id, per story) and marks superseded ones.

    A decision with the same id is updated in place — the newest version's
    wording wins. Links go only to code nodes that exist.
    """
    story_key = story_key.upper()
    project = project_of(story_key)
    rows = []
    for decision in decisions:
        if not decision.get("id") or not decision.get("choice"):
            continue
        uid = decision_uid(story_key, str(decision["id"]))
        sealed = code_crypto.encrypt_row(
            {
                "title": str(decision.get("title") or decision["id"]),
                "choice": str(decision["choice"]),
                "reason": str(decision.get("reason") or ""),
            },
            label="Decision",
            service=project,
            qualified_name=uid,
        )
        rows.append(
            {
                **sealed,
                "uid": uid,
                "id": str(decision["id"]),
                "status": str(decision.get("status") or AGREED),
                "override_by": str(decision.get("override_by") or ""),
                "targets": json.dumps(_targets(decision)),
                "target_rows": _targets(decision),
            }
        )
    with get_driver().session() as session:
        for old in superseded or []:
            session.run(
                "MATCH (d:Decision {uid: $uid}) SET d.status = $status",
                uid=decision_uid(story_key, str(old)),
                status=SUPERSEDED,
            )
        if rows:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Decision {uid: row.uid})
                SET d.id = row.id, d.story_key = $story_key, d.project_key = $project,
                    d.status = row.status, d.override_by = row.override_by,
                    d.title = row.title, d.choice = row.choice, d.reason = row.reason,
                    d.targets = row.targets, d.source = $source, d.updated = $now
                WITH d, row
                OPTIONAL MATCH (d)-[old:APPLIES_TO]->()
                DELETE old
                WITH DISTINCT d, row
                UNWIND row.target_rows AS t
                OPTIONAL MATCH (f:File {service: t.service, file_path: t.file_path})
                OPTIONAL MATCH (s) WHERE (s:Function OR s:Class) AND s.service = t.service
                  AND s.file_path = t.file_path AND t.symbol <> ''
                  AND (s.name = t.symbol OR s.qualified_name ENDS WITH t.symbol)
                FOREACH (_ IN CASE WHEN s IS NULL AND f IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (d)-[:APPLIES_TO]->(f))
                FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (d)-[:APPLIES_TO]->(s))
                """,
                rows=rows,
                story_key=story_key,
                project=project,
                source=source,
                now=datetime.now(UTC).isoformat(timespec="seconds"),
            )
    return len(rows)


def _opened_decision(row: dict) -> dict:
    opened = code_crypto.decrypt_row(
        dict(row), label="Decision", service=row["project_key"], qualified_name=row["uid"]
    )
    try:
        targets = json.loads(row.get("targets") or "[]")
    except ValueError:
        targets = []
    return {
        "id": opened["id"],
        "story_key": opened["story_key"],
        "title": opened["title"],
        "choice": opened["choice"],
        "reason": opened["reason"],
        "status": opened["status"],
        "override_by": opened.get("override_by") or "",
        "applies_to": targets,
    }


def decisions_in_force(story_key: str, services: list[str] | None = None) -> list[dict]:
    """The story's agreed decisions, then the project's agreed decisions on the
    same services (a later story's design must keep them too)."""
    story_key = story_key.upper()
    with get_driver().session() as session:
        rows = session.run(
            """
            MATCH (d:Decision)
            WHERE d.status = $agreed AND (d.story_key = $story_key OR d.project_key = $project)
            RETURN d.uid AS uid, d.id AS id, d.story_key AS story_key,
                   d.project_key AS project_key, d.status AS status,
                   d.override_by AS override_by, d.title AS title, d.choice AS choice,
                   d.reason AS reason, d.targets AS targets
            ORDER BY d.story_key, d.id
            """,
            agreed=AGREED,
            story_key=story_key,
            project=project_of(story_key),
        )
        decisions = [_opened_decision(dict(r)) for r in rows]
    wanted = set(services or [])
    own = [d for d in decisions if d["story_key"] == story_key]
    project = [
        d
        for d in decisions
        if d["story_key"] != story_key
        and (not wanted or any(t.get("service") in wanted for t in d["applies_to"]))
    ]
    return own + project


def decisions_block(decisions: list[dict], story_key: str) -> str:
    """The decisions as the planner reads them."""
    if not decisions:
        return ""
    lines = [
        "DECISIONS IN FORCE — keep each one unless the requirement now contradicts it; "
        "if you change one, say so in overall_reasoning and give the new one a new id. "
        "Return every decision you keep in `decisions`, with the same id."
    ]
    for d in decisions:
        where = (
            "this story" if d["story_key"] == story_key.upper() else f"project, {d['story_key']}"
        )
        files = ", ".join(sorted({t["file_path"] for t in d["applies_to"]})) or "no file named"
        lines.append(
            f"- [{d['id']}] {d['title']}: {d['choice']} (why: {d['reason']}; from {where}; "
            f"applies to {files})"
        )
    return "\n".join(lines)
