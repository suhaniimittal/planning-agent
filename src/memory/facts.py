"""A story's requirement as canonical facts (2026-10-26).

Two stories mean the same thing when they ask for the same facts, not when
their characters match: "limited to 2,00,000 INR per day" and "a daily cap of
2,00,000 INR" are one requirement. A story is turned into facts in fixed
slots — a rule, a limit, an acceptance criterion — whose numbers, amounts,
dates and times are copied verbatim, so a changed number can be caught
without asking a model whether it matters.
"""

from __future__ import annotations

import hashlib
import json
import re

from src.query import planner

KINDS = ("rule", "limit", "acceptance", "scope", "out_of_scope", "actor")

# Jira and Markdown formatting, punctuation and spacing carry no requirement.
_MARKUP = re.compile(r"[*_`~#>|{}\[\]()\"'“”‘’]+")
_PUNCT = re.compile(r"[^\w\s₹%:/.-]+")
_SPACE = re.compile(r"\s+")
_QUOTES = re.compile(r"[\"'“”‘’`]+")
# Thousands separators inside numbers: 2,00,000 and 200,000 and 200000 agree.
_NUMBER_SEPARATORS = re.compile(r"(?<=\d)[,\s](?=\d)")
_RUPEES = re.compile(r"\brs\b\.?")


def normalise_text(text: str) -> str:
    text = _MARKUP.sub(" ", (text or "").lower())
    text = _PUNCT.sub(" ", text)
    return _SPACE.sub(" ", text).strip(" .-")


def text_hash(text: str) -> str:
    return hashlib.sha256(normalise_text(text).encode("utf-8")).hexdigest()


def _value_words(value: str) -> list[str]:
    value = _NUMBER_SEPARATORS.sub("", str(value or "").lower())
    # "Rs" and "Rs." as a word only: "characters " is not "characteinr" (2026-10-27).
    value = _RUPEES.sub("inr ", value).replace("₹", "inr ")
    # Quotes and end punctuation around a message are not part of it.
    value = _QUOTES.sub(" ", value)
    return [w.strip(" .!?,;:") for w in value.split() if w.strip(" .!?,;:")]


def normalise_value(value: str) -> str:
    # "₹2,00,000" and "2,00,000 INR" name the same amount: word order is ignored.
    return " ".join(sorted(_value_words(value)))


def written_in(value: str, text: str) -> bool:
    """Is ``value`` still written in ``text``? Every word of it appears there
    (2026-10-27): a value the extraction missed on one run is not a removed
    value when the story still states it."""
    words = set(_value_words(text))
    return bool(words) and all(w in words for w in _value_words(value))


def canonical(facts: list[dict]) -> list[dict]:
    """Facts in one order with one spelling: slot, kind, statement, values."""
    out = []
    for fact in facts:
        slot = re.sub(r"[^a-z0-9]+", "_", str(fact.get("slot") or "").lower()).strip("_")
        if not slot:
            continue
        kind = str(fact.get("kind") or "rule").lower()
        out.append(
            {
                "slot": slot,
                "kind": kind if kind in KINDS else "rule",
                "statement": _SPACE.sub(" ", str(fact.get("statement") or "")).strip(),
                "values": sorted(
                    {str(v).strip() for v in fact.get("values") or [] if str(v).strip()}
                ),
            }
        )
    return sorted(out, key=lambda f: (f["kind"], f["slot"]))


def facts_hash(facts: list[dict]) -> str:
    material = [
        [
            f["kind"],
            f["slot"],
            normalise_text(f["statement"]),
            sorted(normalise_value(v) for v in f["values"]),
        ]
        for f in canonical(facts)
    ]
    return hashlib.sha256(json.dumps(material, ensure_ascii=False).encode("utf-8")).hexdigest()


def parse_json(raw: str) -> dict:
    """The model's JSON object, tolerating a code fence or text around it."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the model returned no JSON object")
    return json.loads(text[start : end + 1])


EXTRACT_SYSTEM_PROMPT = """You turn a software requirement (a Jira story or bug report) into its \
canonical facts, so two wordings of the same requirement give the same facts.

Rules:
- One fact per distinct requirement: each rule, limit, acceptance criterion, scope line, \
out-of-scope line, and the actor.
- "slot": a short snake_case name for WHAT the fact is about, independent of wording \
(daily_limit, limit_window, counted_transfers, reject_message, reset_time, actor, \
ac_reject_over_limit). Use the same slot for the same thing whatever the words.
- "kind": rule | limit | acceptance | scope | out_of_scope | actor.
- "statement": the fact in plain, neutral words.
- "values": every number, amount, currency, date, time, time zone, percentage and quoted \
message in the fact, copied EXACTLY as written (keep "2,00,000 INR", "00:00", "IST", the \
message text). Empty list when there are none.
- Each acceptance criterion (Given/When/Then) is its own fact of kind "acceptance".
- Leave out greetings, opinions, process chatter and anything that is not a requirement.
- Never invent a fact that is not stated.

Return ONLY a JSON object: {"facts": [{"slot": "...", "kind": "...", "statement": "...", \
"values": ["..."]}]}"""


async def extract_facts(story_text: str) -> list[dict]:
    raw = await planner.call_llm(EXTRACT_SYSTEM_PROMPT, f"REQUIREMENT:\n{story_text.strip()}")
    return canonical(parse_json(raw).get("facts") or [])
