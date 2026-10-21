"""Turns an issue into exact names to search for, and per-service chunk hits into
the handful of functions worth showing the planner — pure logic, no I/O
(the reads live in graph_reader.search_service_chunks).

Two different jobs use the ticket differently. Picking WHICH services are
relevant always uses one embedding of the whole ticket. Finding code INSIDE
a chosen service uses the ticket's parts as well: a chunk only has to match
one part strongly to rank, so a specific detail buried in a multi-topic
ticket is not averaged away. Measured on a real bug report: the function
holding the reported fault ranked #1 when searched by parts, and fell out
of the top 8 entirely when searched by one whole-ticket embedding.

Why exact identifiers: code names quoted in an issue (`approveDeferredReport`,
`internationalPhoneNumber`) are the strongest possible signal, and meaning-
based search alone can rank the code that contains them below vaguer matches.
"""

from __future__ import annotations

import re

# A sentence ends at .!? followed by whitespace, or at a blank line. NOT at
# every newline: pasted PDF text is hard-wrapped, and splitting there shreds
# one sentence into three fragments that each pass the length filter below
# and each spend a query slot while meaning nothing on their own.
#
# The lookbehinds suppress the split after an abbreviation, which is a period
# that does not end a sentence. Without them "...whitespace (e.g. a value
# like ..." breaks into a fragment ending at "(e.g." — observed on the
# deferred-submit ticket, where it mutilated the sentence carrying the root
# cause. Each lookbehind must be fixed-width, hence one per abbreviation.
# Deliberately NOT included: a generic `(?<![A-Z]\.)` for initials, which
# would also suppress the split after a sentence ending in an acronym
# ("...stays stuck in DEFERRED." ), merging it with the next one.
_ABBREVIATIONS = r"(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\betc\.)(?<!\bvs\.)(?<!\bcf\.)(?<!\bapprox\.)"
_SENTENCE_SPLIT = re.compile(rf"(?<=[.!?]){_ABBREVIATIONS}\s+|\n\s*\n+")
# Shorter than this is a fragment ("Description", a bullet marker) with no
# meaning of its own.
MIN_SENTENCE_CHARS = 40
MAX_QUERY_PARTS = 20
# Document furniture that carries no searchable meaning — page footers,
# generation stamps, bare ticket-id headers. A work-breakdown PDF can open
# with a dozen of these, and taking parts in document order spent 7 of 10
# slots on them before reaching a single requirement.
_BOILERPLATE = re.compile(
    r"^\W*(page\s+\d+\s+of\s+\d+"
    r"|generated\b[^.]*\d{4}"
    r"|[A-Z]+-\d+\s*[—-]"
    r"|(pdf|doc|document)\s+only\b)",
    re.I,
)

# camelCase (approveDeferredReport) and multi-hump PascalCase
# (EmploymentVerificationInput); a single capitalized word is too common.
_IDENTIFIER = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+){2,}\b")
# "Form errors" — a quoted multi-word phrase, usually a UI label or log text
_QUOTED_PHRASE = re.compile(r'"([^"\n]{4,60})"')


def _is_boilerplate(piece: str) -> bool:
    return bool(_BOILERPLATE.match(piece))


def _spread(parts: list[str], limit: int) -> list[str]:
    """`limit` parts sampled evenly across `parts`, keeping their order.

    Taking the first `limit` instead would stop inside the header block of a
    long document and never reach its requirements.
    """
    if len(parts) <= limit:
        return parts
    step = len(parts) / limit
    return [parts[int(i * step)] for i in range(limit)]


#
def issue_queries(issue_text: str) -> list[str]:
    """The whole ticket first, then its individual sentences/lines.

    Part 1 is always the entire ticket — that is the one callers use on its
    own to score services. The rest are only ever used to search code.
    """
    whole = issue_text.strip()
    pieces: list[str] = []
    for piece in _SENTENCE_SPLIT.split(issue_text):
        piece = piece.strip()
        if len(piece) < MIN_SENTENCE_CHARS or piece == whole or _is_boilerplate(piece):
            continue
        if piece not in pieces:
            pieces.append(piece)
    return [whole] + _spread(pieces, MAX_QUERY_PARTS - 1)


def issue_identifiers(issue_text: str) -> list[str]:
    """Code names and quoted phrases from the issue, to be matched exactly."""
    found = set(_IDENTIFIER.findall(issue_text))
    for phrase in _QUOTED_PHRASE.findall(issue_text):
        phrase = phrase.strip().rstrip(" .,;:…")
        if " " in phrase and not phrase.startswith(("+", "{")):
            found.add(phrase)
    return sorted(found)


def rank_functions(
    hits: list[dict],
    max_functions: int = 12,
    keep_within: float = 0.12,
    identifier_bonus: float = 0.08,
    max_identifier_hits: int = 25,
) -> list[dict]:
    """One service's chunk hits -> its most relevant functions.

    A function's score is its best chunk's score, plus `identifier_bonus`
    for each distinct issue identifier its chunks contain — except an
    identifier found in more than `max_identifier_hits` chunks, which is too
    common in this service to mean anything. A function containing an
    issue identifier always gets a place — it's the strongest signal there
    is; the remaining places go to functions within `keep_within` of the
    service's best score (weak matches are noise, not context). At most
    `max_functions` in all.

    Returns [{"function_qualified_name", "text", "score", "identifiers"}],
    best first; `text` is the best-scoring chunk.
    """
    identifier_counts: dict[str, int] = {}
    for hit in hits:
        for word in hit.get("identifiers") or []:
            identifier_counts[word] = identifier_counts.get(word, 0) + 1
    specific = {w for w, n in identifier_counts.items() if n <= max_identifier_hits}

    by_function: dict[str, dict] = {}
    for hit in hits:
        name = hit["function_qualified_name"]
        entry = by_function.setdefault(
            name,
            {
                "function_qualified_name": name,
                "text": hit["text"],
                "best": -1.0,
                "identifiers": set(),
            },
        )
        if hit["score"] > entry["best"]:
            entry["best"] = hit["score"]
            entry["text"] = hit["text"]
        entry["identifiers"].update(w for w in hit.get("identifiers") or [] if w in specific)

    ranked = [
        {
            "function_qualified_name": e["function_qualified_name"],
            "text": e["text"],
            "score": e["best"] + identifier_bonus * len(e["identifiers"]),
            "identifiers": sorted(e["identifiers"]),
        }
        for e in by_function.values()
    ]
    ranked.sort(key=lambda r: -r["score"])
    if not ranked:
        return []
    floor = ranked[0]["score"] - keep_within
    named = [r for r in ranked if r["identifiers"]][:max_functions]
    rest = [r for r in ranked if not r["identifiers"] and r["score"] >= floor]
    kept = (named + rest)[:max_functions]
    kept.sort(key=lambda r: -r["score"])
    return kept


def services_named_in(issue_text: str, service_names: list[str]) -> list[str]:
    """Services the issue mentions by name ("... in candidate-service") —
    the reporter already knows these are involved."""
    lowered = issue_text.lower()
    return [
        name
        for name in service_names
        if re.search(rf"(?<![\w-]){re.escape(name.lower())}(?![\w-])", lowered)
    ]
