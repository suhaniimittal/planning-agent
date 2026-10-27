"""What Memsy holds for a story (2026-10-27) — read only, nothing is written.

Shows the story's newest memory snapshot (decrypted: facts, TDD versions,
decisions), the project's agreed decisions, and the conversation Memsy keeps
about the TDD (as Memsy rewrote it).

Usage (from planning-agent/, with MEMSY_* and CODE_GRAPH_KEY in .env):
    .venv/bin/python scripts/show_memory.py BTS-36
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.memory import record  # noqa: E402


def _conversation(story_key: str) -> list[tuple[str, str, str]]:
    """(time, session, text) of the story's memories that are not snapshots, oldest first."""
    rows = []
    listed = record._client().memories.list(actor_id=f"tdd:{story_key}", limit=100)
    for item in listed.items or []:
        sources = record._payload(item).get("source_metadata") or []
        metas = [s.get("metadata") or {} for s in sources]
        if any(m.get("type") == "tdd_memory" for m in metas):
            continue
        session = record._payload(item).get("session_id") or ""
        rows.append((str(item.created_at)[:19], str(session), str(item.text or "")))
    return sorted(rows)


def main(story_key: str) -> None:
    story_key = story_key.upper()
    seq, snap = record.load_snapshot(story_key)
    print(f"== {story_key}: memory snapshot {seq or 'none'} (actor tdd:{story_key})")
    if not seq:
        print("   nothing remembered yet: the next create TDD is verdict 'new'")
    for fact in snap.get("facts") or []:
        print(f"   fact  {fact.get('slot')}: {', '.join(fact.get('values') or []) or '-'}")
    for v in snap.get("versions") or []:
        print(f"   version {v['version']}  page {v.get('page_id')}  {v.get('created', '')}")
    for d in snap.get("decisions") or []:
        print(f"   decision [{d['status']}] {d['id']}: {d['choice']}")
    project = record.project_of(story_key)
    _, shared = record._load(*record._scope(project=project)[::2])
    print(f"\n== project {project}: agreed decisions every story on the same service keeps")
    for d in shared.get("decisions") or []:
        services = ", ".join(sorted({t["service"] for t in d["applies_to"]})) or "-"
        print(f"   {d['story_key']} [{d['id']}] {d['choice']}  (services: {services})")
    print("\n== conversation about the TDD (as Memsy rewrote it)")
    for when, session, text in _conversation(story_key):
        print(f"   {when}  {session:<24} {text[:160]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: scripts/show_memory.py <STORY-KEY>")
    main(sys.argv[1])
