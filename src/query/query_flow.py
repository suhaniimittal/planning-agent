"""Orchestrator for the query-time flow: given a plain-text requirement or
bug report, pick the few services it involves (by summary AND by their own
code), find each one's most relevant functions, and produce a structured
technical design doc via a single LLM call.

Mirrors how ingest.py orchestrates over embedder.py/graph_writer.py.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from src.ingestion.embedder import embed_many

from . import code_search, graph_reader, issue_split, planner
from .models import ServiceDependency, TechnicalDesignDoc
from .verify import verify_tdd

# Services beyond top_k whose code is also searched before choosing — so a
# service with a generic summary can still win on its code.
_CODE_EVIDENCE_EXTRA = 3
# A service's rank = this weight x summary match + the rest x best code match.
_SUMMARY_WEIGHT = 0.5
# Functions up to this long are shown whole; longer ones by their best chunk.
_FULL_SOURCE_MAX_CHARS = 1500
# A matched method's whole class is shown when the class is at most this
# long — enough for a typical input/DTO class (the one in the worked example
# is ~7,400 chars), small enough that it cannot crowd out other services.
_FULL_CLASS_MAX_CHARS = 8000


def _print_selection_debug(
    candidate_names: list[str],
    seed_names: list[str],
    selection_log: list[dict],
    snippet_map: dict[tuple[str, str], str],
) -> None:
    """Human-readable answer to "which service/class/function actually went
    into this run" — everything here is a readout of decisions already
    made above, never a new decision of its own."""
    seed_set = set(seed_names)
    print("  [DEBUG] Candidate services:")
    for name in candidate_names:
        tag = "seed" if name in seed_set else "neighbor — summary only"
        print(f"    - {name} ({tag})")

    by_service: dict[str, list[dict]] = {}
    for entry in selection_log:
        by_service.setdefault(entry["service"], []).append(entry)

    print("  [DEBUG] Selected classes/functions per service:")
    if not selection_log:
        print("    (none — every candidate fell back to doc text only)")
    for name in candidate_names:
        entries = by_service.get(name, [])
        if not entries:
            print(f"    {name}: (no classes/functions matched)")
            continue
        print(f"    {name}:")
        for e in entries:
            has_snippet = (name, e["qualified_name"]) in snippet_map
            snippet_tag = "with real code" if has_snippet else "metadata only, no snippet"
            print(f"      - [{e['kind']}] {e['name']} — {e['reason']} ({snippet_tag})")


def _referenced_service_names(tdd: TechnicalDesignDoc) -> list[str]:
    """Every service name the LLM actually touched: services it proposed
    changes for, plus anything it named as a dependency source/target in
    either architecture diagram. Order-preserving, de-duplicated — used to
    scope the final `service_dependencies` verification query."""
    ordered: list[str] = []
    seen: set[str] = set()

    def add(name: str | None) -> None:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    for service_plan in tdd.services:
        add(service_plan.service)
    for component in list(getattr(tdd, "current_architecture_components", None) or []) + list(
        getattr(tdd, "proposed_architecture_components", None) or []
    ):
        add(component.name)
        for dep in component.depends_on:
            add(dep)
    return ordered


def _widen_to_small_classes(
    parent_of: dict[tuple[str, str], tuple[str, str]],
    code_symbols_by_service: dict[str, list[dict]],
    snippet_map: dict[tuple[str, str], str],
) -> None:
    """Show a matched method's WHOLE class, when that class is small enough.

    Search ranks a method by what its own body means, so a sibling that never
    mentions the issue's words is invisible — even when it is the real place
    to change. Seen for real: a bug about a phone value keeping its
    whitespace matched `isPhoneValid` (which only READS the field), while
    `toAvro` (which WRITES it, and held the actual one-line fix) ranked
    nowhere and was hidden behind "(+2 more)" in the file listing. Both sit
    in the same 167-line class. Showing that class costs a few thousand
    characters and puts every sibling in front of the planner.

    Only for classes under `_FULL_CLASS_MAX_CHARS`; a large class would
    crowd out every other service's code. Mutates its arguments in place.
    """
    if not parent_of:
        return
    classes = {pair for pair in parent_of.values()}
    loaded = {
        (c["service"], c["qualified_name"]): c
        for c in graph_reader.load_symbols(sorted(classes))
        if c["kind"] == "Class"
    }
    for (service, method_qname), class_key in parent_of.items():
        klass = loaded.get(class_key)
        if klass is None:
            continue
        source = klass.get("source_snippet") or ""
        if not 0 < len(source) <= _FULL_CLASS_MAX_CHARS:
            continue
        shown = code_symbols_by_service.get(service, [])
        if any(sym["qualified_name"] == klass["qualified_name"] for sym in shown):
            continue  # another matched method already pulled this class in
        method = next((s for s in shown if s["qualified_name"] == method_qname), None)
        entry = {
            **klass,
            "matched_identifiers": (method or {}).get("matched_identifiers") or [],
            "shown_because": method_qname.rsplit("::", 1)[-1],
        }
        shown.append(entry)
        snippet_map[(service, klass["qualified_name"])] = source


async def build_tdd(
    issue_text: str,
    top_k: int = 5,
    hops: int = 2,
    debug_dump_prompts: bool = False,
    service_names: list[str] | None = None,
    mode: str | None = None,
) -> TechnicalDesignDoc:
    """`service_names`: the services this run covers (the Repositories the
    user listed) — only those are ever considered. None considers every
    service in the graph.

    `top_k` is the most services the TDD may involve; `hops` widens only the
    BACKGROUND context (graph-connected services shown by summary, never
    given code to change).

    `debug_dump_prompts=True` writes the exact system/user prompt sent to
    the LLM to llm_system_prompt.txt / llm_user_prompt.txt in the current
    directory — off by default so normal callers (query_tools.py, the
    agent) never get unexpected local file writes."""
    # The whole ticket, then its parts. `vectors[0]` (the whole ticket) is
    # what scores services; all of them together search each service's code,
    # so a detail that only one sentence describes still finds its function.
    # The parts come from an LLM where possible (self-contained, references
    # resolved) and from sentence splitting otherwise — see issue_split.
    detected_mode, queries = await issue_split.issue_queries(issue_text)
    # An explicit mode from the caller wins; otherwise take what the split
    # classified. The prompt itself falls back to bug for anything unknown.
    mode = mode or detected_mode
    vectors = await embed_many(queries)
    embedding = vectors[0]
    identifiers = code_search.issue_identifiers(issue_text)

    # 1. Which services. Summary similarity alone misjudges a service whose
    #    summary is generic — a large UI repo ranked 4th of 6 for a bug in
    #    its own form validation — so the services' own code gets a vote:
    #    every service on the shortlist is searched, and its best function
    #    match counts as much as its summary match. A service the issue
    #    names outright is always included.
    summary = graph_reader.summary_scores(embedding, service_names)
    by_summary = sorted(summary, key=lambda name: -summary[name])
    named = code_search.services_named_in(issue_text, by_summary)
    shortlist = list(dict.fromkeys(named + by_summary[: top_k + _CODE_EVIDENCE_EXTRA]))

    # 2. What code, per service: each service's own top chunks (never a
    #    shared top-k a large service can monopolize), test code excluded.
    searches = await asyncio.gather(
        *(
            asyncio.to_thread(graph_reader.search_service_chunks, service, vectors, identifiers)
            for service in shortlist
        )
    )
    ranked_by_service = {
        service: code_search.rank_functions(hits)
        for service, hits in zip(shortlist, searches, strict=True)
    }
    code_score = {s: (r[0]["score"] if r else 0.0) for s, r in ranked_by_service.items()}
    combined = {
        s: _SUMMARY_WEIGHT * summary[s] + (1 - _SUMMARY_WEIGHT) * code_score[s] for s in shortlist
    }
    selected = list(dict.fromkeys(named + sorted(shortlist, key=lambda s: -combined[s])))
    selected = selected[: max(top_k, len(named))]

    print("  [TDD] services (summary / code -> combined), * = selected:")
    for s in sorted(shortlist, key=lambda s: -combined[s]):
        mark = "*" if s in selected else " "
        note = "  (named in the issue)" if s in named else ""
        print(f"    {mark} {s}: {summary[s]:.3f} / {code_score[s]:.3f} -> {combined[s]:.3f}{note}")

    # Graph-connected services are shown as background only (summary, no
    # code) — they explain dependencies, they aren't candidates for change.
    # The graph traversal supports 1-2 hops; more would pull in most of the graph anyway.
    neighbors = (
        [n["name"] for n in graph_reader.expand_neighbors(selected, hops=min(hops, 2))]
        if hops and hops > 0
        else []
    )
    if service_names is not None:
        neighbors = [n for n in neighbors if n in service_names]
    context_names = selected + [n for n in neighbors if n not in selected]
    order = {name: i for i, name in enumerate(context_names)}
    candidate_docs = sorted(
        graph_reader.load_candidate_docs(context_names),
        key=lambda d: order.get(d["name"], len(order)),
    )
    # Only links between the chosen services: a link to a background
    # service is shown nowhere else, and only cluttered the diagrams.
    known_dependencies = graph_reader.get_edges_among(selected)

    symbols = graph_reader.load_symbols(
        [(s, hit["function_qualified_name"]) for s in selected for hit in ranked_by_service[s]]
    )
    by_key = {(m["service"], m["qualified_name"]): m for m in symbols}
    code_symbols_by_service: dict[str, list[dict]] = {}
    snippet_map: dict[tuple[str, str], str] = {}
    selection_log: list[dict] = []
    parent_of: dict[tuple[str, str], tuple[str, str]] = {}
    for service in selected:
        for hit in ranked_by_service[service]:
            key = (service, hit["function_qualified_name"])
            symbol = by_key.get(key)
            if symbol is None:
                continue  # stale chunk (function renamed/removed since) — never fabricate metadata
            symbol = {**symbol, "matched_identifiers": hit["identifiers"]}
            code_symbols_by_service.setdefault(service, []).append(symbol)
            if symbol.get("parent_class"):
                parent_of[key] = (service, symbol["parent_class"])
            # A short function is shown whole; a long one only by its
            # best-matching chunk — the relevant slice, exactly the point of
            # chunking.
            source = symbol.get("source_snippet") or ""
            snippet_map[key] = source if 0 < len(source) <= _FULL_SOURCE_MAX_CHARS else hit["text"]
            reason = f"score {hit['score']:.3f}"
            if hit["identifiers"]:
                reason += f", contains {', '.join(hit['identifiers'])}"
            selection_log.append(
                {
                    "service": service,
                    "kind": symbol["kind"],
                    "name": symbol["name"],
                    "qualified_name": symbol["qualified_name"],
                    "reason": reason,
                }
            )
    _widen_to_small_classes(parent_of, code_symbols_by_service, snippet_map)

    shown_files = sorted(
        {
            (service, symbol["file_path"])
            for service, symbols_ in code_symbols_by_service.items()
            for symbol in symbols_
            if symbol.get("file_path")
        }
    )
    file_context = graph_reader.load_file_context(shown_files)
    # Paths only, looked up AFTER the files are chosen — test code still never
    # competes in the search (see graph_reader.load_sibling_tests).
    sibling_tests = graph_reader.load_sibling_tests(shown_files)
    # Where a NEW test would go, for the shown files that have none beside
    # them. Only for services whose tests demonstrably sit beside their
    # code — a service with a mirrored test tree gets nothing here rather
    # than a path nobody measured, and the planner asks instead.
    conventions = graph_reader.test_naming_conventions(selected)
    new_test_paths = {}
    for service, file_path in shown_files:
        infix = conventions.get(service)
        if not infix or sibling_tests.get((service, file_path)):
            continue
        suggested = graph_reader.new_test_path(file_path, infix)
        if suggested:
            new_test_paths[(service, file_path)] = suggested

    system_prompt = planner.build_system_prompt(mode)
    user_prompt = planner.build_user_prompt(
        issue_text,
        candidate_docs,
        code_symbols_by_service,
        snippet_map,
        seed_names=set(selected),
        known_dependencies=known_dependencies,
        file_context=file_context,
        sibling_tests=sibling_tests,
        new_test_paths=new_test_paths,
    )

    if debug_dump_prompts:
        with open("llm_system_prompt.txt", "w") as f:
            f.write(system_prompt)
        with open("llm_user_prompt.txt", "w") as f:
            f.write(user_prompt)
        print(
            "  [DEBUG] wrote llm_system_prompt.txt and llm_user_prompt.txt "
            f"({len(system_prompt)} + {len(user_prompt)} chars)"
        )
        _print_selection_debug(context_names, selected, selection_log, snippet_map)

    raw = await planner.call_llm(system_prompt, user_prompt)
    tdd = planner.parse_tdd(raw)

    # A generation timestamp is a fact about this run, not something to ask
    # an LLM to guess — same "compute it, don't trust a guess" principle as
    # service_dependencies/has_real_source below.
    tdd.generated_at = datetime.now(UTC).isoformat()

    # Real graph data, never the LLM's guess — and scoped to only services
    # the LLM actually referenced, not the whole raw candidate pool (which
    # can include unrelated services, e.g. when top_k exceeds the number of
    # real services in the graph). "Referenced" includes both services the
    # LLM proposed changes for AND any service it named as a dependency in
    # its architecture diagram — e.g. if the LLM only flags "orders" but
    # draws "orders -> shipping", "shipping" must still be eligible here, or
    # this field silently misses a dependency the diagram itself shows.
    flagged_names = _referenced_service_names(tdd)
    edges = graph_reader.get_edges_among(flagged_names)
    tdd.service_dependencies = [ServiceDependency(**e) for e in edges]

    # has_real_source is likewise never trusted from the LLM's own output —
    # computed here by checking whether the LLM's named function_or_symbol
    # matches a symbol that actually had a snippet loaded above. The LLM
    # only ever sees a symbol's plain `name` or its qualified_name's
    # "ClassName.name" suffix (see _format_code_symbol), never the full
    # qualified_name with its file_path prefix — so match against both of
    # those forms, not the raw qualified_name string.
    matchable_names_with_snippet: dict[str, set[str]] = {}
    for service, symbols in code_symbols_by_service.items():
        names = set()
        for symbol in symbols:
            if (service, symbol["qualified_name"]) not in snippet_map:
                continue
            names.add(symbol["name"])
            names.add(symbol["qualified_name"].split("::", 1)[-1])
        matchable_names_with_snippet[service] = names

    for service_plan in tdd.services:
        candidates_for_service = matchable_names_with_snippet.get(service_plan.service, set())
        for change in service_plan.changes:
            change.has_real_source = change.function_or_symbol in candidates_for_service

    # Every file/symbol the LLM named, checked against the real code graph —
    # same "verify, don't trust" principle as above, applied to the paths a
    # coding agent will actually open.
    shown_files_by_service = {
        service: {symbol["file_path"] for symbol in symbols if symbol.get("file_path")}
        for service, symbols in code_symbols_by_service.items()
    }
    verify_tdd(tdd, shown_files_by_service)

    return tdd
