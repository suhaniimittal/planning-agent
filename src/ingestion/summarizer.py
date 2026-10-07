"""Code-derived Service summaries: one batched LLM call per ingestion run,
covering every service, instead of one call per service (30-40 services
would otherwise mean 30-40 calls) and instead of copying a paragraph out of
a RepoDocAI markdown doc.

Each service contributes a lean digest — entry-point file names, its most
substantial classes (by method count, capped), and its top-level function
names. No file counts, method counts, or full source go into the prompt;
those are only used here to decide what makes the cut.

build_digest() takes plain class/function dicts, not a RepoGraph, so the
SAME code works whether the caller just parsed a fresh RepoGraph (first
ingestion — see ingest.py's _class_row/_function_row) or is rebuilding a
digest from a service's CURRENT state already in Neo4j (an incremental run
that only reparsed a couple of files still needs the service's FULL symbol
set for a meaningful digest — see code_graph_writer.load_current_symbols).
Never a repo clone either way.

Same provider-fallback split as embedder.py/planner.py: AI Gateway if
AGENTS_GATEWAY_KEY + AI_GATEWAY_URL are set, else OpenAI directly.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import PurePosixPath

_GATEWAY_PROVIDER, _GATEWAY_MODEL = "anthropic", "claude-sonnet-4-6"
_OPENAI_MODEL = "gpt-4o"

_MAX_CLASSES_PER_SERVICE = 15
_MAX_TOP_LEVEL_FUNCTIONS_PER_SERVICE = 20

# Basenames checked case-insensitively against every parsed file's path.
# Deliberately a lookup, not a guess: a file either matches one of these
# well-known entry-point names for its language or it doesn't end up in
# `entry_points` at all — no fuzzy matching, no "probably this one."
_ENTRY_POINT_BASENAMES: dict[str, set[str]] = {
    "python": {"main.py", "app.py", "manage.py", "wsgi.py", "asgi.py"},
    "java": {"main.java"},  # plus the *Application.java suffix check below
    "kotlin": {"main.kt", "application.kt"},
    "typescript": {"index.ts", "main.ts", "server.ts", "app.ts"},
    "tsx": {"index.tsx", "main.tsx", "app.tsx"},
    "javascript": {"index.js", "main.js", "server.js", "app.js"},
}


class SummarizerError(Exception):
    """Raised when the batched LLM call fails outright or its response can't
    be parsed at all — caller must catch this and skip summary regeneration
    for this run entirely, never let it abort ingestion."""


def _dominant_language(classes: list[dict], functions: list[dict]) -> str | None:
    counts = Counter(s["language"] for s in (*classes, *functions))
    return counts.most_common(1)[0][0] if counts else None


def _find_entry_points(classes: list[dict], functions: list[dict], language: str | None) -> list[str]:
    if language is None:
        return []
    file_paths = {s["file_path"] for s in (*classes, *functions)}
    known_basenames = _ENTRY_POINT_BASENAMES.get(language, set())

    found = []
    for path in sorted(file_paths):
        basename = PurePosixPath(path).name
        if basename.lower() in known_basenames:
            found.append(path)
        elif language == "java" and basename.endswith("Application.java"):
            found.append(path)
    return found


def _top_classes(classes: list[dict], functions: list[dict], limit: int = _MAX_CLASSES_PER_SERVICE) -> list[dict]:
    method_counts = Counter(
        f["parent_class"] for f in functions if f.get("is_method") and f.get("parent_class")
    )
    ranked = sorted(classes, key=lambda c: method_counts.get(c["qualified_name"], 0), reverse=True)
    return [{"name": c["name"], "docstring": c.get("docstring")} for c in ranked[:limit]]


def _top_level_functions(functions: list[dict], limit: int = _MAX_TOP_LEVEL_FUNCTIONS_PER_SERVICE) -> list[str]:
    return [f["name"] for f in functions if not f.get("is_method")][:limit]


def build_digest(service_name: str, classes: list[dict], functions: list[dict]) -> dict:
    """Pure function: a service's current classes/functions (plain dicts,
    each needing at minimum qualified_name/name/file_path/language, plus
    is_method/parent_class for functions and docstring for classes) -> the
    lean JSON-able digest sent to the LLM.

    No file counts or method counts in the output — those only decide which
    classes make `_top_classes`'s cut, they don't help the model describe
    what the service does.
    """
    language = _dominant_language(classes, functions)
    return {
        "service": service_name,
        "language": language,
        "entry_points": _find_entry_points(classes, functions, language),
        "classes": _top_classes(classes, functions),
        "top_level_functions": _top_level_functions(functions),
    }


def _build_prompt(digests: list[dict]) -> tuple[str, str]:
    system_prompt = (
        "You are given structural summaries of several microservices, "
        "extracted directly from their source code (not documentation). "
        "For each service, write a 2-3 sentence description of its main "
        "responsibility, based only on what's evidenced by the class/"
        "function names, docstrings, and entry points given — do not invent "
        "functionality that isn't implied by this data. If a service's "
        "names are too generic to say anything specific, write a short, "
        "honest, generic description rather than guessing at business "
        "meaning.\n\n"
        "Respond with ONLY a JSON array, no markdown fences, no prose "
        'outside the JSON: [{"service": "<name>", "summary": "<2-3 '
        'sentences>"}, ...] — exactly one entry per service given below, in '
        "any order."
    )
    user_prompt = json.dumps(digests, indent=2)
    return system_prompt, user_prompt


def _use_gateway() -> bool:
    return bool(os.environ.get("AGENTS_GATEWAY_KEY") and os.environ.get("AI_GATEWAY_URL"))


async def _call_via_gateway(system_prompt: str, user_prompt: str) -> str:
    from agent_lib.gateway.ai import AiGatewayClient

    async with AiGatewayClient(
        gateway_key=os.environ["AGENTS_GATEWAY_KEY"],
        gateway_url=os.environ["AI_GATEWAY_URL"],
    ) as client:
        reply = await client.chat(
            provider=_GATEWAY_PROVIDER,
            model_name=_GATEWAY_MODEL,
            prompt=user_prompt,
            system_prompt=system_prompt,
            max_tokens=8000,
        )
        return reply["content"]


async def _call_via_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import AsyncOpenAI

    # No response_format here: OpenAI's native JSON mode requires a top-level
    # JSON *object*, but the response here is a JSON *array* (one entry per
    # service) — the system prompt's instruction plus _parse_response's
    # defensive fence-stripping is the mechanism for valid JSON instead, same
    # as the gateway path (which has no response_format option at all).
    client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    resp = await client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=8000,
    )
    return resp.choices[0].message.content


def _parse_response(raw: str, expected_services: set[str]) -> dict[str, str]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SummarizerError(f"not valid JSON: {e}\nRaw: {raw[:500]}") from e

    if not isinstance(data, list):
        raise SummarizerError(f"expected a JSON array, got {type(data).__name__}")

    summaries: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict) or "service" not in entry or "summary" not in entry:
            continue  # one malformed entry never invalidates the rest
        summaries[entry["service"]] = entry["summary"]

    missing = expected_services - summaries.keys()
    if missing:
        print(f"  [SUMMARIZER] no summary returned for: {sorted(missing)}")

    return summaries


async def generate_summaries(digests: list[dict]) -> dict[str, str]:
    """One batched call covering every digest given. Returns
    {service_name: summary_text} — services the model didn't return anything
    usable for are simply absent (logged), never crash the whole run.

    Raises SummarizerError only when the call/response fails outright (e.g.
    provider error, totally unparseable output) — callers should catch this
    and skip summary regeneration for the run entirely rather than let it
    abort ingestion.
    """
    if not digests:
        return {}

    system_prompt, user_prompt = _build_prompt(digests)
    try:
        if _use_gateway():
            raw = await _call_via_gateway(system_prompt, user_prompt)
        else:
            raw = await _call_via_openai(system_prompt, user_prompt)
    except Exception as e:  # noqa: BLE001 - a provider error must never abort ingestion
        raise SummarizerError(f"LLM call failed: {e}") from e

    expected = {d["service"] for d in digests}
    return _parse_response(raw, expected)
