"""Planning's memory in Memsy (2026-10-26): what a story requires, which TDD
version answered it, and the design decisions its versions keep.

Neo4j keeps only the code. Memsy keeps the memory, next to the conversation
about the same TDD (conversation.py):

    actor  tdd:<STORY>       session memory:<STORY>:<seq>:<part>   the story's snapshots
    actor  project:<PROJECT> session memory:project:<PROJECT>:...  the project's agreed decisions

Memsy rewrites the text it stores and cannot update or delete, so the memory is
kept as numbered SNAPSHOTS whose exact data rides in each event's metadata,
which Memsy returns unchanged (``payload.source_metadata``):

- every save writes snapshot N+1, the whole state; reading takes the newest
  complete snapshot, so a newer one replaces an older one without deleting it;
- the data is compressed, then encrypted with CODE_GRAPH_KEY before it leaves
  Planning (company requirements and decisions); a snapshot too big for Memsy's
  4 KB metadata is split into parts, each in its own session, because Memsy
  merges similar events of one session and keeps only one of them;
- a save waits until Memsy has processed it, so the next request reads it.

Snapshot (decrypted): {"facts", "facts_hash", "story_text_hash",
"versions": [{"version", "page_id", "facts_hash", "created"}],
"decisions": [{"uid", "id", "story_key", "title", "choice", "reason", "status",
"override_by", "source", "applies_to"}]}
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import os
import time
import zlib
from datetime import UTC, datetime

from src.ingestion import code_crypto
from src.ingestion.secret_store import secret

logger = logging.getLogger(__name__)

AGREED, SUPERSEDED = "agreed", "superseded"
PART_CHARS = 3_500  # Memsy keeps at most 4,096 characters of metadata per event
LIST_PAGE = 100
LIST_PAGES = 10  # newest 1,000 memories of an actor are searched for its snapshot
WAIT_SECONDS = 30
FORMAT = "zlib-aesgcm-1"  # snapshots of another format are not read


class MemoryUnavailable(RuntimeError):
    """Memsy is not configured or not reachable."""


def project_of(story_key: str) -> str:
    return story_key.split("-")[0].upper() if story_key else ""


def decision_uid(story_key: str, decision_id: str) -> str:
    return f"{story_key.upper()}:{decision_id}"


def _client():
    key = secret("MEMSY_API_KEY")
    base = os.environ.get("MEMSY_BASE_URL", "").strip()
    if not key or not base:
        raise MemoryUnavailable("MEMSY_API_KEY and MEMSY_BASE_URL are not set")
    from memsy import MemsyClient

    return MemsyClient(base_url=base, api_key=key, timeout=15.0, max_retries=2)


# --- snapshots ------------------------------------------------------------------------


def _scope(story_key: str = "", project: str = "") -> tuple[str, str, str]:
    """(actor, session, kind of snapshot) for a story or a project."""
    if project:
        return f"project:{project}", f"memory:project:{project}", "project_memory"
    key = story_key.upper()
    return f"tdd:{key}", f"memory:{key}", "tdd_memory"


def _payload(item) -> dict:
    payload = getattr(item, "payload", None)
    if isinstance(payload, dict):
        return payload
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(str(payload))
            if isinstance(value, dict):
                return value
        except (ValueError, SyntaxError):
            continue
    return {}


def _load(actor: str, kind: str, client=None) -> tuple[int, dict]:
    """(seq, snapshot) of the newest complete snapshot; (0, {}) when none."""
    client = client or _client()
    parts: dict[int, dict[int, str]] = {}
    sizes: dict[int, int] = {}
    for page in range(LIST_PAGES):
        listed = client.memories.list(actor_id=actor, limit=LIST_PAGE, offset=page * LIST_PAGE)
        for item in listed.items or []:
            for source in _payload(item).get("source_metadata") or []:
                meta = source.get("metadata") or {}
                if (meta.get("type"), meta.get("actor"), meta.get("format")) != (
                    kind,
                    actor,
                    FORMAT,
                ):
                    continue
                seq, part = int(meta["seq"]), int(meta["part"])
                parts.setdefault(seq, {})[part] = str(meta["data"])
                sizes[seq] = int(meta["parts"])
        if len(listed.items or []) < LIST_PAGE:
            break
    for seq in sorted(parts, reverse=True):
        if len(parts[seq]) == sizes[seq]:
            sealed = "".join(parts[seq][i] for i in range(sizes[seq]))
            packed = code_crypto.decrypt(
                sealed, label="MemsySnapshot", service=actor, qualified_name=str(seq), field="data"
            )
            return seq, json.loads(zlib.decompress(base64.b64decode(packed)))
    return 0, {}


def _save(actor: str, session: str, kind: str, seq: int, snapshot: dict, summary: str) -> None:
    """Writes snapshot ``seq`` and waits until Memsy has made it readable."""
    from memsy import EventPayload

    client = _client()
    packed = base64.b64encode(zlib.compress(json.dumps(snapshot, sort_keys=True).encode(), 9))
    sealed = code_crypto.encrypt(
        packed.decode(),
        label="MemsySnapshot",
        service=actor,
        qualified_name=str(seq),
        field="data",
    )
    chunks = [sealed[i : i + PART_CHARS] for i in range(0, len(sealed), PART_CHARS)]
    events = [
        EventPayload(
            actor_id=actor,
            session_id=f"{session}:{seq}:{i + 1}",
            kind="app_event",
            content=f"{summary} Memory snapshot {seq}, part {i + 1} of {len(chunks)}.",
            metadata=json.dumps(
                {
                    "type": kind,
                    "actor": actor,
                    "format": FORMAT,
                    "seq": seq,
                    "part": i,
                    "parts": len(chunks),
                    "data": chunk,
                }
            ),
        )
        for i, chunk in enumerate(chunks)
    ]
    ids = client.ingest(events).event_ids
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        status = client.status(ids)
        if status.failed_ids:
            raise MemoryUnavailable(f"Memsy could not process the memory snapshot: {status}")
        if not status.pending_ids and _load(actor, kind, client)[0] >= seq:
            return
        time.sleep(1.5)
    logger.warning("Memsy snapshot %s %s not readable after %ss", actor, seq, WAIT_SECONDS)


# --- the story's memory ------------------------------------------------------------------


def load_snapshot(story_key: str) -> tuple[int, dict]:
    actor, _, kind = _scope(story_key)
    return _load(actor, kind)


def save_snapshot(story_key: str, seq: int, snapshot: dict) -> None:
    actor, session, kind = _scope(story_key)
    latest = (snapshot.get("versions") or [{}])[-1].get("version") or "no version"
    summary = (
        f"{story_key.upper()} TDD {latest}: {len(snapshot.get('facts') or [])} requirement "
        f"facts and {len(agreed(snapshot))} decisions in force."
    )
    _save(actor, session, kind, seq, snapshot, summary)
    _publish_to_project(story_key, snapshot)


def agreed(snapshot: dict) -> list[dict]:
    return [d for d in snapshot.get("decisions") or [] if d.get("status") == AGREED]


def load_requirement(story_key: str) -> dict | None:
    """{facts, facts_hash, story_text_hash} for the story, or None when nothing is stored."""
    _, snap = load_snapshot(story_key)
    if not snap.get("facts") and not snap.get("facts_hash"):
        return None
    return {
        "facts": list(snap.get("facts") or []),
        "facts_hash": snap.get("facts_hash") or "",
        "story_text_hash": snap.get("story_text_hash") or "",
    }


def current_version(story_key: str, snapshot: dict | None = None) -> dict | None:
    """The newest recorded version of the story's TDD: {tdd_version, page_id}."""
    snap = snapshot if snapshot is not None else load_snapshot(story_key)[1]
    versions = snap.get("versions") or []
    if not versions:
        return None
    return {"tdd_version": versions[-1]["version"], "page_id": versions[-1].get("page_id", "")}


def with_facts(snapshot: dict, facts: list[dict], facts_hash: str, story_text_hash: str) -> dict:
    return {
        **snapshot,
        "facts": facts,
        "facts_hash": facts_hash,
        "story_text_hash": story_text_hash,
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


# --- the project's decisions -------------------------------------------------------------


def _publish_to_project(story_key: str, snapshot: dict) -> None:
    """The project's snapshot: every story's agreed decisions, this story's replaced."""
    project = project_of(story_key)
    actor, session, kind = _scope(project=project)
    seq, current = _load(actor, kind)
    keep = [d for d in current.get("decisions") or [] if d["story_key"] != story_key.upper()]
    merged = {"decisions": sorted(keep + agreed(snapshot), key=lambda d: d["uid"])}
    if merged == {"decisions": current.get("decisions") or []}:
        return
    summary = f"Project {project}: {len(merged['decisions'])} agreed design decisions."
    _save(actor, session, kind, seq + 1, merged, summary)


def decisions_in_force(story_key: str, services: list[str] | None = None) -> list[dict]:
    """The story's agreed decisions, then the project's agreed decisions on the
    same services (a later story's design must keep them too)."""
    story_key = story_key.upper()
    own = agreed(load_snapshot(story_key)[1])
    actor, _, kind = _scope(project=project_of(story_key))
    wanted = set(services or [])
    project = [
        d
        for d in _load(actor, kind)[1].get("decisions") or []
        if d["story_key"] != story_key
        and d.get("status") == AGREED
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
