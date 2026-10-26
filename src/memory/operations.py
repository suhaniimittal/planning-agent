"""What the agent's memory modes do (2026-10-26) — called from src/tools/memory_tools.py.

``check_story``  mode "facts": is this story the requirement its TDD answered?
``record``       mode "record": a TDD version was published — remember it.
"""

from __future__ import annotations

from src.memory import equivalence, facts, record


async def check_story(issue_key: str, story_text: str, judge=None, extract=None) -> dict:
    """The contract's mode "facts" response, without status/mode. Stores nothing."""
    seq, snapshot = record.load_snapshot(issue_key)
    stored = (
        {
            "facts": list(snapshot.get("facts") or []),
            "facts_hash": snapshot.get("facts_hash") or "",
            "story_text_hash": snapshot.get("story_text_hash") or "",
        }
        if snapshot.get("facts") or snapshot.get("facts_hash")
        else None
    )
    current = record.current_version(issue_key, snapshot)
    text_hash = facts.text_hash(story_text)
    if stored and stored["story_text_hash"] == text_hash:
        return {
            "verdict": "same",
            "facts": stored["facts"],
            "facts_hash": stored["facts_hash"],
            "story_text_hash": text_hash,
            "current": current,
            "changes": [],
            "reason": "The story's text is the same as when its TDD was written.",
        }
    new_facts = await (extract(story_text) if extract else facts.extract_facts(story_text))
    new_hash = facts.facts_hash(new_facts)
    if not stored or not stored["facts"]:
        return {
            "verdict": "new",
            "facts": new_facts,
            "facts_hash": new_hash,
            "story_text_hash": text_hash,
            "current": current,
            "changes": [],
            "reason": "No requirement is remembered for this story yet.",
        }
    found = await equivalence.compare(stored["facts"], new_facts, judge=judge, new_text=story_text)
    return {
        "verdict": found["verdict"],
        "facts": new_facts,
        "facts_hash": new_hash,
        "story_text_hash": text_hash,
        "current": current,
        "changes": found["changes"],
        "reason": found["reason"],
    }


def remember(payload: dict) -> dict:
    """The contract's mode "record": the published version, its facts, its decisions —
    one new memory snapshot in Memsy holding all of it (2026-10-26)."""
    issue_key = str(payload.get("issue_key") or "").upper()
    if not issue_key:
        raise ValueError("record needs the issue_key")
    seq, snapshot = record.load_snapshot(issue_key)
    stored_facts = 0
    story_facts = payload.get("facts")
    facts_hash = str(payload.get("facts_hash") or "")
    if story_facts:
        canonical = facts.canonical(list(story_facts))
        facts_hash = facts_hash or facts.facts_hash(canonical)
        # Computed here from the raw story when it is sent: one definition of
        # "the same text" for mode "facts" and for what is stored.
        story_text = str(payload.get("story_text") or "")
        text_hash = facts.text_hash(story_text) if story_text else ""
        snapshot = record.with_facts(
            snapshot, canonical, facts_hash, text_hash or str(payload.get("story_text_hash") or "")
        )
        stored_facts = len(canonical)
    elif not facts_hash:
        # A revision asked on the TDD page: the story did not change, so its
        # stored facts and hashes stay; the new version answers the same facts.
        facts_hash = snapshot.get("facts_hash") or ""
    version = str(payload.get("tdd_version") or "")
    if version:
        snapshot = record.with_version(
            snapshot, version, str(payload.get("page_id") or ""), facts_hash
        )
    snapshot, stored_decisions = record.with_decisions(
        snapshot,
        issue_key,
        list(payload.get("decisions") or []),
        superseded=[str(s) for s in payload.get("superseded") or []],
        source=str(payload.get("source") or ""),
    )
    record.save_snapshot(issue_key, seq + 1, snapshot)
    return {"facts": stored_facts, "decisions": stored_decisions}
