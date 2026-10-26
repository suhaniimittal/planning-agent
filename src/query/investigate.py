"""A bug's likely root cause, found in the code graph (2026-10-26).

Moved here from the Requirement Review agent, which searched GitHub for at
most 30 files: Planning already holds every repository's code, encrypted, and
reads it the same way it does to write a TDD. Nothing is ingested and GitHub
is not called — the graph is refreshed on every merge.

The model may only name files whose code it was shown; any other path is
dropped. "high" confidence needs code that shows the faulty logic. When no code
in the graph is about the bug, it says so in plain words and reasons from the
report alone.
"""

from __future__ import annotations

from src.ingestion.embedder import embed
from src.memory.facts import parse_json

from . import code_search, graph_reader, planner

_MAX_FUNCTIONS = 8
_MAX_REPORT_CHARS = 12_000
_MAX_SOURCE_CHARS = 4_000

SYSTEM_PROMPT = """You investigate a bug report against the code shown, as a senior engineer.

1. Say what is wrong (actual) and what should happen (expected), from the report.
2. Find the most likely ROOT CAUSE in the code shown. Point at the function and the logic.
   When none of the code shown is about this bug, say so and reason from the report alone.
3. List the AFFECTED FILES: only paths that appear in a "Code" label below, exactly as written.
4. Propose a FIX in a few sentences. Do not write the code.

Write for a busy developer, in plain words: "root_cause" at most 2 sentences,
"proposed_fix" at most 3, at most 3 "evidence" items, each "why" one short line.

"root_cause_confidence": "high" only when code shown proves the faulty logic; "medium" when the
code strongly suggests it; "low" otherwise. With no relevant code, never "high".

Return ONLY a JSON object:
{"actual": "...", "expected": "...", "root_cause": "...",
 "root_cause_confidence": "high|medium|low",
 "evidence": ["short quote or fact, with where it comes from"],
 "affected_files": [{"path": "<service>/<file path> exactly as in a Code label", "why": "..."}],
 "proposed_fix": "...", "code_relevant": true or false}"""


async def _code_for(
    report_text: str, only: list[str] | None = None
) -> tuple[list[dict], list[str], list[str]]:
    """(symbols shown with their source, services searched, identifiers searched).

    ``only``: the services of the run's repositories; every service when empty."""
    embedding = await embed(report_text[:8000])
    services = sorted(graph_reader.summary_scores(embedding, list(only) if only else None))
    identifiers = code_search.issue_identifiers(report_text)
    hits: list[dict] = []
    for service in services:
        for hit in graph_reader.search_service_chunks(service, [embedding], identifiers, k=8):
            hits.append({**hit, "service": service})
    hits.sort(key=lambda h: (-len(h.get("identifiers") or []), -(h.get("score") or 0.0)))
    chosen: list[tuple[str, str]] = []
    for hit in hits:
        key = (hit["service"], hit["function_qualified_name"])
        if key not in chosen:
            chosen.append(key)
        if len(chosen) >= _MAX_FUNCTIONS:
            break
    symbols = [s for s in graph_reader.load_symbols(chosen) if s.get("source_snippet")]
    return symbols, services, identifiers


def _code_label(symbol: dict) -> str:
    return f"{symbol['service']}/{symbol['file_path']}"


def build_user_prompt(report_text: str, symbols: list[dict]) -> str:
    parts = ["BUG REPORT:", report_text[:_MAX_REPORT_CHARS]]
    if symbols:
        parts += ["", "CODE (from the code graph):"]
        for s in symbols:
            parts += [
                f"--- Code {_code_label(s)} :: {s['qualified_name']}",
                str(s["source_snippet"])[:_MAX_SOURCE_CHARS],
            ]
    else:
        parts += ["", "CODE: none of the code in the graph matched this report."]
    return "\n".join(parts)


def ground(raw: dict, symbols: list[dict], services: list[str], identifiers: list[str]) -> dict:
    """Only files the model was shown; confidence capped without relevant code."""
    shown = {_code_label(s): s for s in symbols}
    by_file = {s["file_path"]: _code_label(s) for s in symbols}
    files = []
    for entry in raw.get("affected_files") or []:
        if not isinstance(entry, dict):
            continue
        path = str(entry.get("path") or "").strip()
        path = path if path in shown else by_file.get(path, "")
        if path and path not in [f["path"] for f in files]:
            files.append({"path": path, "why": str(entry.get("why") or "").strip()})
    confidence = str(raw.get("root_cause_confidence") or "low").lower()
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    relevant = bool(files) and raw.get("code_relevant") is not False
    if not relevant and confidence == "high":
        confidence = "medium"
    searched = ", ".join(identifiers[:6]) or "the report's own words"
    where = ", ".join(services) or "none"
    if relevant:
        note = f"Code read from the code graph ({where}); searched for: {searched}."
    else:
        files = []
        note = (
            f"The code for this bug was not found in the repositories Planning has ({where}); "
            f"searched for: {searched}. The cause above comes from the report."
        )
    return {
        "actual": str(raw.get("actual") or "").strip(),
        "expected": str(raw.get("expected") or "").strip(),
        "root_cause": str(raw.get("root_cause") or "").strip(),
        "root_cause_confidence": confidence,
        "evidence": [str(e).strip() for e in raw.get("evidence") or [] if str(e).strip()][:8],
        "affected_files": files[:8],
        "proposed_fix": str(raw.get("proposed_fix") or "").strip(),
        "code_note": note,
    }


async def investigate(report_text: str, only: list[str] | None = None) -> dict:
    symbols, services, identifiers = await _code_for(report_text, only)
    raw = await planner.call_llm(SYSTEM_PROMPT, build_user_prompt(report_text, symbols))
    return ground(parse_json(raw), symbols, services, identifiers)
