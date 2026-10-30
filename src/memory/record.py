"""A story's TDD memory as a snapshot (2026-10-26; kept by Orchestration since
2026-10-30): what a story requires, which TDD version answered it, and the
design decisions its versions keep.

Planning stores nothing. Orchestration reads the story's snapshot from the
memory service, gives it to Planning (modes "facts" and "record"), and keeps
the snapshot Planning hands back. These are the snapshot's rules: pure, no I/O.

Snapshot: {"facts", "facts_hash", "story_text_hash", "story_text",
"versions": [{"version", "page_id", "facts_hash", "created"}],
"decisions": [{"uid", "id", "story_key", "title", "choice", "reason", "status",
"override_by", "source", "applies_to"}]}
"""

from __future__ import annotations

from datetime import UTC, datetime

AGREED, SUPERSEDED = "agreed", "superseded"


def decision_uid(story_key: str, decision_id: str) -> str:
    return f"{story_key.upper()}:{decision_id}"


def agreed(snapshot: dict) -> list[dict]:
    return [d for d in snapshot.get("decisions") or [] if d.get("status") == AGREED]


def current_version(snapshot: dict) -> dict | None:
    """The newest recorded version of the story's TDD: {tdd_version, page_id}."""
    versions = snapshot.get("versions") or []
    if not versions:
        return None
    return {"tdd_version": versions[-1]["version"], "page_id": versions[-1].get("page_id", "")}


def with_facts(
    snapshot: dict, facts: list[dict], facts_hash: str, story_text_hash: str, story_text: str = ""
) -> dict:
    """The requirement the TDD answers; ``story_text`` is what the next check
    compares the story with (2026-10-27)."""
    return {
        **snapshot,
        "facts": facts,
        "facts_hash": facts_hash,
        "story_text_hash": story_text_hash,
        "story_text": story_text or snapshot.get("story_text") or "",
    }


def with_version(snapshot: dict, version: str, page_id: str, facts_hash: str) -> dict:
    versions = [v for v in snapshot.get("versions") or [] if v.get("version") != version]
    now = datetime.now(UTC).isoformat(timespec="seconds")
    versions.append(
        {"version": version, "page_id": page_id, "facts_hash": facts_hash, "created": now}
    )
    return {**snapshot, "versions": versions}


def with_decisions(
    snapshot: dict,
    story_key: str,
    decisions: list[dict],
    superseded: list[str] | None = None,
    source: str = "",
) -> tuple[dict, int]:
    """The snapshot with ``decisions`` agreed (same id = updated in place) and
    ``superseded`` ids retired; (snapshot, how many decisions were stored)."""
    story_key = story_key.upper()
    by_uid = {d["uid"]: dict(d) for d in snapshot.get("decisions") or []}
    for old in superseded or []:
        uid = decision_uid(story_key, str(old))
        if uid in by_uid:
            by_uid[uid]["status"] = SUPERSEDED
    stored = 0
    for decision in decisions:
        if not decision.get("id") or not decision.get("choice"):
            continue
        uid = decision_uid(story_key, str(decision["id"]))
        by_uid[uid] = {
            "uid": uid,
            "id": str(decision["id"]),
            "story_key": story_key,
            "title": str(decision.get("title") or decision["id"]),
            "choice": str(decision["choice"]),
            "reason": str(decision.get("reason") or ""),
            "status": str(decision.get("status") or AGREED),
            "override_by": str(decision.get("override_by") or ""),
            "source": source,
            "applies_to": _targets(decision),
        }
        stored += 1
    return {**snapshot, "decisions": sorted(by_uid.values(), key=lambda d: d["uid"])}, stored


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
