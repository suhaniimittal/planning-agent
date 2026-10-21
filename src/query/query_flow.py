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

# A service's rank = this weight x summary match + the rest x best code match.
_SUMMARY_WEIGHT = 0.5
# Functions up to this long are shown whole; longer ones by their best chunk.
_FULL_SOURCE_MAX_CHARS = 1500
# A matched method's whole class is shown when the class is at most this
# long — enough for a typical input/DTO class (the one in the worked example
# is ~7,400 chars), small enough that it cannot crowd out other services.
_FULL_CLASS_MAX_CHARS = 8000

# How much of a service's matched code may be shown WHOLE rather than as a
# chunk, best match first. A budget, not a count of symbols: the thing that
# costs is characters in the prompt, and a count cannot express that when
# one match is 300 characters and the next is 12,000.
#
# A large component is exactly the case the 1,500 rule handles worst. On a
# real ticket the component holding the bug was 12,135 characters, so the
# planner saw one ~700-character chunk of it. A review done with the file
# open then found four things the design had missed — the blur handler
# fires even when nothing was edited, the callback it uses is internal and
# cannot be asserted on, the view dereferences a field with no null check,
# and an input is gated behind a prop — every one of them inside those
# 12,135 characters, already in the graph, and none of them in the chunk.
#
# That component ranked THIRD for its own ticket, behind two matches of 316
# and ~2,000 characters. So "the top two symbols" would not have reached it,
# while a budget does: the small ones barely spend any of it.
_WHOLE_SOURCE_BUDGET_CHARS = 20000


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


# How many terms one symbol is looked up by. A long function mentions
# dozens of names, and each one costs a scan of its service's spec files,
# so the longest are kept: a long identifier is a specific one
# (sendReportRequestedLinkCcEmailAddresses, 39 chars, matches 3 spec files),
# while the short ones are the generic names every file shares.
_MAX_TEST_LOOKUP_TERMS = 12


def _test_lookup_terms(source: str, issue_identifiers: list[str]) -> list[str]:
    """What to search the test files for, to find the tests covering a symbol
    whose own name no test ever mentions."""
    terms = set(code_search.symbol_identifiers(source))
    # A ticket that DOES quote a name or a UI label is a strong signal, but
    # only where the symbol genuinely touches it.
    terms.update(word for word in issue_identifiers if word in source)
    return sorted(sorted(terms, key=len, reverse=True)[:_MAX_TEST_LOOKUP_TERMS])


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
    top_k: int = 8,
    hops: int = 0,
    debug_dump_prompts: bool = False,
    service_names: list[str] | None = None,
    mode: str | None = None,
) -> TechnicalDesignDoc:
    """`service_names`: the services this run covers (the Repositories the
    user listed) — only those are ever considered. None considers every
    service in the graph.

    `top_k` is the most services the TDD may involve. It is a cap on the
    answer, not on the search: every service in the run has its code read
    before any of them are ranked, so this never decides which services
    were allowed to compete.

    `hops` widens only the BACKGROUND context: services reached over CALLS
    edges, shown by summary and never given code to change. It defaults to
    0 — off. Every neighbour becomes its own "no change" block in the
    document, and on a graph this densely connected one hop already reaches
    most of it, so the tail cost it adds is real while the context it buys
    is not. Pass 1 or 2 when a change's blast radius across services is
    the actual question.

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

    # 1. Which services. EVERY service in the run has its code searched —
    #    nothing is eliminated on its summary first. Summary similarity
    #    misjudges a service whose summary is generic: a large UI repo
    #    ranked 4th of 6 for a bug in its own form validation, and was cut
    #    before a line of that code was read. Ranking only afterwards, on
    #    summary AND code together, is what makes that impossible.
    #
    #    This is affordable because `service_names` is the list a person
    #    typed into the form, and because the searches below run in
    #    parallel: measured on a 12-service graph, the extra services are
    #    the small ones (~0.1s each) while the slowest was already being
    #    searched. A run large enough to need a budget should get one
    #    measured in CHUNKS SCANNED, which is what actually costs — not in
    #    services, and not guessed.
    summary = graph_reader.summary_scores(embedding, service_names)
    by_summary = sorted(summary, key=lambda name: -summary[name])
    # A subset of by_summary (services_named_in filters the list it is
    # given), so it needs no separate place in the shortlist — only in the
    # selection below, where it overrides the ranking.
    named = code_search.services_named_in(issue_text, by_summary)
    shortlist = by_summary

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
        whole_source_left = _WHOLE_SOURCE_BUDGET_CHARS
        for hit in ranked_by_service[service]:
            key = (service, hit["function_qualified_name"])
            symbol = by_key.get(key)
            if symbol is None:
                continue  # stale chunk (function renamed/removed since) — never fabricate metadata
            symbol = {**symbol, "matched_identifiers": hit["identifiers"]}
            code_symbols_by_service.setdefault(service, []).append(symbol)
            if symbol.get("parent_class"):
                parent_of[key] = (service, symbol["parent_class"])
            # Whole source while this service's budget lasts, best match
            # first; a chunk once it is spent. A chunk is the right unit for
            # the supporting cast — showing the relevant slice of something
            # peripheral is the point of chunking — and the wrong unit for
            # the thing about to be edited, where the surrounding code is how
            # a reader learns what calls it and what it assumes. Always whole
            # below _FULL_SOURCE_MAX_CHARS, so a long tail of small matches
            # still arrives intact after the budget runs out.
            source = symbol.get("source_snippet") or ""
            if 0 < len(source) <= _FULL_SOURCE_MAX_CHARS:
                snippet_map[key] = source
            elif 0 < len(source) <= whole_source_left:
                snippet_map[key] = source
                whole_source_left -= len(source)
            else:
                snippet_map[key] = hit["text"]
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
    # How this file imports things, so code written for it uses module paths
    # that exist rather than invented ones.
    file_imports = graph_reader.load_file_imports(shown_files)
    # Paths only, looked up AFTER the files are chosen — test code still never
    # competes in the search (see graph_reader.load_sibling_tests).
    sibling_tests = graph_reader.load_sibling_tests(shown_files)
    # Tests that NAME a symbol being changed, wherever they live. The sibling
    # lookup above only sees the same folder, which finds nothing on a project
    # that keeps src/common/ apart from src/apps/ — and a design that believes
    # no test exists has a new one written from scratch instead of the real one
    # edited.
    #
    # Each symbol is looked up by the identifiers its OWN SOURCE mentions,
    # plus any issue identifier that appears there. The name alone finds
    # nothing on a UI project — no test contains "AtsIntegrationSectionView"
    # or "handleCcEmailAddressesBlur", because a component renders through a
    # wrapper and a handler is never named — and the ticket alone finds
    # nothing either, because a real ticket is English prose: "ATS Link CC
    # email addresses changes semi-colon to comma" yields no identifiers at
    # all. What does work is the field the handler reads,
    # sendReportRequestedLinkCcEmailAddresses, which is in its source and in
    # the three specs that cover it.
    tests_by_symbol = graph_reader.tests_naming(
        [
            (service, symbol["name"], _test_lookup_terms(source, identifiers))
            for service, symbols_ in code_symbols_by_service.items()
            for symbol in symbols_
            if symbol.get("name")
            for source in [snippet_map.get((service, symbol["qualified_name"])) or ""]
        ]
    )
    # Their actual text, so the planner copies a fixture that works instead
    # of inventing one. Only test files carry source, so this reads nothing
    # for anything else.
    test_sources = graph_reader.load_test_sources(
        sorted({(service, path_) for (service, _), paths in tests_by_symbol.items() for path_ in paths})
    )
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
        file_imports=file_imports,
        sibling_tests=sibling_tests,
        tests_by_symbol=tests_by_symbol,
        test_sources=test_sources,
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
