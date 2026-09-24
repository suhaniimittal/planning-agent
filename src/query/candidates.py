"""Pure candidate-set merging — no I/O. Combines vector-search hits and
graph-neighbor hits into one deduplicated, ordered list of service names.
"""

from __future__ import annotations

import re


def merge_candidates(vector_hits: list[dict], neighbors: list[dict]) -> list[str]:
    """Union of vector-search + graph-neighbor names, de-duplicated:
    vector hits first (by score desc), then neighbors (by hop distance asc).
    """
    ordered = sorted(vector_hits, key=lambda h: -h["score"]) + sorted(
        neighbors, key=lambda n: n["hop_distance"]
    )
    seen: dict[str, None] = {}
    for item in ordered:
        seen.setdefault(item["name"], None)
    return list(seen.keys())


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
# Splits a camelCase/PascalCase run into subwords, acronym runs kept intact:
# "EmailLogRow" -> ["Email","Log","Row"], "getShippingUri" -> ["get","Shipping","Uri"],
# "HTTPMonitoringInterceptor" -> ["HTTP","Monitoring","Interceptor"].
_CAMEL_SUBWORD_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")


def _normalize(word: str) -> str:
    """A very small, deliberately conservative suffix-stripper — not a real
    stemmer, just enough to bridge "emails"/"email" and "logged"/"log" for
    keyword-overlap purposes. A symbol named "EmailLogRow" splits cleanly
    into "email"/"log"/"row", but an issue report describing the same idea
    in plain English almost always uses inflected forms ("emails",
    "logged", "getting") — without this, the camelCase split alone still
    isn't enough to connect them. Same "cheap text-based signal, not ML"
    spirit as the rest of this ranker, and the same idea as extractors.py's
    own `_singular` helper, generalized here to common verb suffixes too.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    stripped = None
    if len(word) > 4 and word.endswith("ing"):
        stripped = word[:-3]
    elif len(word) > 4 and word.endswith("ed"):
        stripped = word[:-2]
    elif len(word) > 3 and word.endswith("es"):
        return word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if stripped is None:
        return word
    # Undo a doubled final consonant left over from stripping -ed/-ing —
    # "logged" -> "logg" -> "log", "getting" -> "gett" -> "get".
    if len(stripped) > 2 and stripped[-1] == stripped[-2] and stripped[-1] in _CONSONANTS:
        stripped = stripped[:-1]
    return stripped


def _tokenize(text: str) -> set[str]:
    """Alphanumeric runs (split on underscores/dots/whitespace/punctuation),
    lowercased — PLUS each run's camelCase/PascalCase subwords, split out
    separately — PLUS a normalized (de-pluralized/de-tensed) form of every
    token collected so far.

    Symbol names in Java/TypeScript/Kotlin are almost always camelCase or
    PascalCase (`EmailLogRow`, `getShippingUri`), which a plain
    alphanumeric-run split treats as ONE indivisible token — so a symbol
    whose name obviously relates to the issue in spirit ("email", "log")
    could score ZERO keyword overlap and be excluded before ranking even
    starts, purely because nothing glued its subwords together with an
    underscore the way a Python snake_case name would have. Splitting
    camelCase closes part of that gap — but a symbol's subwords are always
    in base form ("email", "log"), while an issue report describing the
    same thing in plain English is almost always inflected ("emails",
    "logged") — so normalization closes the rest of it. Every original,
    un-normalized token is kept alongside its normalized form, so an exact
    mention still matches exactly as before — this only adds recall, never
    removes it.
    """
    tokens: set[str] = set()
    for word in _WORD_RE.findall(text.lower()):
        tokens.add(word)
        tokens.add(_normalize(word))
    for word in _WORD_RE.findall(text):
        subwords = _CAMEL_SUBWORD_RE.findall(word)
        if len(subwords) > 1:
            for subword in subwords:
                lowered = subword.lower()
                tokens.add(lowered)
                tokens.add(_normalize(lowered))
    return tokens


def select_relevant_symbols(
    symbols: list[dict],
    issue_text: str,
    max_per_service: int = 5,
    max_total: int = 40,
) -> list[dict]:
    """Rank Class/Function metadata (from graph_reader.load_code_symbols_for_services)
    by cheap keyword overlap with the issue text, capped per-service then
    globally.

    A service can have far more parsed symbols than fit in one LLM prompt,
    and this pipeline is built around a SINGLE planning call (no room for a
    second LLM-based ranking pass without changing that constraint) — so
    this uses plain token overlap between the issue text and each symbol's
    name/docstring/signature, the same text-based-signal spirit as
    extractors.py, not a new ML component. Ties broken by symbol order
    (stable sort), so results are deterministic given the same input.
    """
    issue_tokens = _tokenize(issue_text)
    if not issue_tokens:
        return []

    by_service: dict[str, list[tuple[int, dict]]] = {}
    for symbol in symbols:
        text = " ".join(
            str(symbol.get(field) or "") for field in ("name", "docstring", "signature")
        )
        overlap = len(issue_tokens & _tokenize(text))
        if overlap > 0:
            by_service.setdefault(symbol["service"], []).append((overlap, symbol))

    shortlisted: list[tuple[int, dict]] = []
    for service_symbols in by_service.values():
        service_symbols.sort(key=lambda pair: -pair[0])
        shortlisted += service_symbols[:max_per_service]

    shortlisted.sort(key=lambda pair: -pair[0])
    return [symbol for _, symbol in shortlisted[:max_total]]


def select_relevant_functions_by_embedding(
    chunk_hits: list[dict],
    max_per_service: int = 5,
    max_total: int = 40,
) -> list[dict]:
    """Roll up chunk-level vector-search hits (graph_reader.vector_search_chunks)
    to one entry per function, keyed by (service, function_qualified_name) —
    a function's score is its BEST-scoring chunk, never an average (a long
    function split into several chunks shouldn't be penalized just for being
    long). Capped per-service then globally, same shape and reasoning as
    select_relevant_symbols above, which this supersedes as the primary
    relevance signal — real semantic similarity instead of literal keyword
    overlap.
    """
    best_per_function: dict[tuple[str, str], dict] = {}
    for hit in chunk_hits:
        key = (hit["service"], hit["function_qualified_name"])
        if key not in best_per_function or hit["score"] > best_per_function[key]["score"]:
            best_per_function[key] = hit

    by_service: dict[str, list[dict]] = {}
    for hit in best_per_function.values():
        by_service.setdefault(hit["service"], []).append(hit)

    shortlisted: list[dict] = []
    for service_hits in by_service.values():
        service_hits.sort(key=lambda h: -h["score"])
        shortlisted += service_hits[:max_per_service]

    shortlisted.sort(key=lambda h: -h["score"])
    return shortlisted[:max_total]
