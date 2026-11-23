"""Is a story still the same requirement as the one its TDD answered? (2026-10-26)

Cheapest check first, and a number is never left to a model's judgement:

1. the SET of values (numbers, amounts, dates, times, quoted messages) is
   compared exactly — any value added or removed is a change. A set, not a
   list per fact: a model files the same values under different facts on
   different wordings, so which fact holds "IST" says nothing about meaning;
2. identical canonical facts — the story was only reworded;
3. otherwise one model call compares the old story text with the new one
   (the old facts when the old text is not stored): "I am not feeling well"
   and "I am feeling sick" are the same; "successful transfers count" and "all
   transfers count" are not. A judge that is not confident says "unsure", and
   the caller keeps the TDD and says so.

Facts are an extraction and miss a detail now and then; the texts are what
people wrote (2026-10-27). So the judge reads the texts, not the new fact list
(a light rewording was called "changed" because one run's facts left out
"is not recorded"), and a value the new facts lack but the new text still
states is not a removed value.
"""

from __future__ import annotations

import json

from src.memory.facts import canonical, facts_hash, parse_json
from src.memory.facts import normalise_value, written_in
from src.query import planner

SAME_CONFIDENCE = 0.8

JUDGE_SYSTEM_PROMPT = """You compare two versions of the same software requirement. The OLD one \
is the story's text (or, when that is not known, a list of facts extracted from it); the NEW \
one is the story's text. Judge only by what the two texts ask for: does the NEW text ask for \
exactly what the OLD one asks for? A detail stated in both, in any words or order, is not a \
change.

- "same": a paraphrase — different words, same behaviour ("I am not feeling well" / "I am \
feeling sick"; "limited to" / "capped at"; reordered, split or merged facts).
- "different": the behaviour, scope, condition, actor or result changed ("successful \
transfers count" / "all transfers count"; "reject" / "warn"), or something was added or \
removed that changes what must be built or tested.
- Be strict: if a developer would build or test anything differently, it is "different".

Return ONLY a JSON object:
{"verdict": "same" | "different", "confidence": 0.0-1.0,
 "changes": [{"slot": "<a fact slot listed below if one fits, else a short name>",
              "why": "what changed, in plain words"}],
 "reason": "one sentence"}"""


def _value_set(facts: list[dict]) -> dict[str, str]:
    """{normalised value: the value as written} across all facts."""
    return {normalise_value(v): v for f in facts for v in f["values"]}


def _slot_of(value: str, facts: list[dict]) -> dict:
    for fact in facts:
        if any(normalise_value(v) == value for v in fact["values"]):
            return fact
    return {"slot": "", "kind": "rule"}


def value_changes(
    old: list[dict], new: list[dict], old_text: str = "", new_text: str = ""
) -> list[dict]:
    """The values added and removed, each with the fact it belongs to. With the
    texts given, a value is added only when the old text did not state it, and
    removed only when the new text no longer states it."""
    before, after = _value_set(old), _value_set(new)
    added = {v for v in set(after) - set(before) if not written_in(after[v], old_text)}
    removed = {v for v in set(before) - set(after) if not written_in(before[v], new_text)}
    changes = []
    for value in sorted(added):
        fact = _slot_of(value, new)
        replaced = sorted(before[v] for v in removed if _slot_of(v, old)["slot"] == fact["slot"])
        if not replaced:
            # The old value may still appear elsewhere (2,00,000 in an acceptance
            # line): the fact under the same slot says what it was.
            same_slot = next((f for f in old if f["slot"] == fact["slot"]), None)
            replaced = list(same_slot["values"]) if same_slot else []
        changes.append(
            {
                "slot": fact["slot"],
                "kind": fact["kind"],
                "old": ", ".join(replaced) or "(none)",
                "new": after[value],
                "why": "a number, amount, date, time or message was added or changed",
            }
        )
    covered = {c["old"] for c in changes}
    for value in sorted(removed):
        if any(before[value] in c for c in covered):
            continue
        fact = _slot_of(value, old)
        changes.append(
            {
                "slot": fact["slot"],
                "kind": fact["kind"],
                "old": before[value],
                "new": "(none)",
                "why": "a number, amount, date, time or message was removed",
            }
        )
    return changes


def _statements(facts: list[dict]) -> list[str]:
    return [f"[{f['slot']}] {f['statement']}" for f in facts]


def _version(name: str, text: str, facts: list[dict]) -> str:
    """One side for the judge: the story's text, or its facts when no text is known."""
    if text.strip():
        return f"{name} STORY TEXT:\n" + text.strip()[:12000]
    return f"{name} version (facts):\n" + json.dumps(
        _statements(facts), ensure_ascii=False, indent=1
    )


async def compare(
    old_facts: list[dict],
    new_facts: list[dict],
    judge=None,
    new_text: str = "",
    old_text: str = "",
) -> dict:
    """{verdict: reworded | changed | unsure, changes, reason}. ``new_text`` is the
    new story as written, ``old_text`` the one its TDD was written from (when
    stored); ``judge`` replaces the model call in tests."""
    old, new = canonical(old_facts), canonical(new_facts)
    changed_values = value_changes(old, new, old_text, new_text)
    if changed_values:
        return {
            "verdict": "changed",
            "changes": changed_values,
            "reason": "A number, amount, date, time or message changed.",
        }
    if facts_hash(old) == facts_hash(new):
        return {
            "verdict": "reworded",
            "changes": [],
            "reason": "The requirement's facts are identical.",
        }

    prompt = (
        _version("OLD", old_text, old)
        + "\n\n"
        + _version("NEW", new_text, new)
        + "\n\nFact slots (for naming changes): "
        + ", ".join(sorted({f["slot"] for f in [*old, *new]}))
    )
    call = judge or planner.call_llm
    raw = await call(JUDGE_SYSTEM_PROMPT, prompt)
    try:
        result = parse_json(raw)
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "unsure", "changes": [], "reason": "The comparison could not be read."}
    verdict = str(result.get("verdict") or "").lower()
    try:
        confidence = float(result.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    reason = str(result.get("reason") or "").strip()
    if verdict == "same" and confidence >= SAME_CONFIDENCE:
        return {
            "verdict": "reworded",
            "changes": [],
            "reason": reason or "Same meaning, new wording.",
        }
    if verdict == "different":
        by_slot = {f["slot"]: f for f in [*old, *new]}
        changes = [
            {
                "slot": str(c.get("slot") or ""),
                "kind": (by_slot.get(str(c.get("slot"))) or {}).get("kind", "rule"),
                "old": (next((f for f in old if f["slot"] == c.get("slot")), {}) or {}).get(
                    "statement", ""
                ),
                "new": (next((f for f in new if f["slot"] == c.get("slot")), {}) or {}).get(
                    "statement", ""
                ),
                "why": str(c.get("why") or ""),
            }
            for c in result.get("changes") or []
            if isinstance(c, dict)
        ]
        return {"verdict": "changed", "changes": changes, "reason": reason}
    return {
        "verdict": "unsure",
        "changes": [],
        "reason": reason or "Not confident the meaning is unchanged.",
    }
