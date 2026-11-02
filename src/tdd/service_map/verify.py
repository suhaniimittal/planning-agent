"""Keeps only facts someone can point to.

Every fact carries a quote from a file. This checks the quote really is in that
file's text (ignoring spacing). A fact the patterns found always passes, since
its quote was cut from the file. A fact an LLM proposed is dropped when its
quote is not there: the model invented it.
"""

from __future__ import annotations

from .schema import Evidence, ServiceFacts


def norm(text: str) -> str:
    return " ".join(text.split())


def verify(facts: ServiceFacts, texts: dict[str, str]) -> dict[str, int]:
    """Marks each fact's evidence ``verified`` and drops LLM facts that fail.
    Returns {"checked", "verified", "dropped"}."""
    normed = {path: norm(text) for path, text in texts.items()}
    counts = {"checked": 0, "verified": 0, "dropped": 0}

    def holds(evidence: Evidence) -> bool:
        text = normed.get(evidence.file)
        evidence.verified = bool(
            text is not None and evidence.quote and norm(evidence.quote) in text
        )
        return evidence.verified

    def keep(items: list) -> list:
        kept = []
        for item in items:
            counts["checked"] += 1
            if holds(item.evidence):
                counts["verified"] += 1
                kept.append(item)
            elif getattr(item, "source", "rule") == "rule":
                kept.append(item)  # kept, but shown as unverified in the output
            else:
                counts["dropped"] += 1
        return kept

    facts.exposes = keep(facts.exposes)
    facts.calls = keep(facts.calls)
    facts.topics = keep(facts.topics)
    facts.libs = keep(facts.libs)
    return counts
