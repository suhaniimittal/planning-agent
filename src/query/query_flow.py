"""Orchestrator for the query-time flow: given a plain-text requirement or
bug report, narrow the graph down to a handful of relevant services and
produce a structured technical design doc via a single LLM call.

Mirrors how ingest.py orchestrates over embedder.py/graph_writer.py.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.ingestion.embedder import embed

from . import candidates, graph_reader, planner
from .models import ServiceDependency, TechnicalDesignDoc
from .verify import verify_tdd


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
    for component in (
        list(getattr(tdd, "current_architecture_components", None) or [])
        + list(getattr(tdd, "proposed_architecture_components", None) or [])
    ):
        add(component.name)
        for dep in component.depends_on:
            add(dep)
    return ordered


async def build_tdd(
    issue_text: str,
    top_k: int = 5,
    hops: int = 2,
    debug_dump_prompts: bool = False,
) -> TechnicalDesignDoc:
    """`debug_dump_prompts=True` writes the exact system/user prompt sent to
    the LLM to llm_system_prompt.txt / llm_user_prompt.txt in the current
    directory — off by default so normal callers (query_tools.py, the
    agent) never get unexpected local file writes."""
    embedding = await embed(issue_text)
    vector_hits = graph_reader.vector_search_services(embedding, k=top_k)
    seed_names = [h["name"] for h in vector_hits]
    neighbors = graph_reader.expand_neighbors(seed_names, hops=hops)
    candidate_names = candidates.merge_candidates(vector_hits, neighbors)
    candidate_docs = graph_reader.load_candidate_docs(candidate_names)

    # Real CALLS edges among the candidates — computed now, from the graph,
    # and shown to the model AS INPUT (not just used to overwrite its output
    # afterward, like the final service_dependencies field is). Without
    # this, the model has no confirmed ground truth for cross-service
    # relationships and — appropriately cautious — tends to leave
    # architecture-diagram `depends_on` lists empty rather than guess from
    # prose alone.
    known_dependencies = graph_reader.get_edges_among(candidate_names)

    # Debug-only bookkeeping: which symbol came from which selection path
    # (class keyword match / function embedding hit / function fallback) —
    # never affects what's actually sent to the LLM, just makes it visible
    # afterward when debug_dump_prompts=True.
    selection_log: list[dict] = []

    # Additive: real code symbols alongside the existing doc text, when the
    # code graph exists for a candidate (never required — a service with no
    # code graph still gets planned from its doc alone, same as today).
    raw_symbols = graph_reader.load_code_symbols_for_services(candidate_names)
    raw_classes = [s for s in raw_symbols if s.get("kind") == "Class"]
    raw_functions = [s for s in raw_symbols if s.get("kind") == "Function"]
    metadata_by_key = {(s["service"], s["qualified_name"]): s for s in raw_symbols}

    code_symbols_by_service: dict[str, list[dict]] = {}
    snippet_map: dict[tuple[str, str], str] = {}

    # Classes have no per-symbol embeddings — only functions get chunked and
    # embedded at ingestion time (see the chunking design) — so keyword
    # overlap is still the only relevance signal available for them.
    relevant_classes = candidates.select_relevant_symbols(raw_classes, issue_text)
    # A matched class's full source_snippet already contains every one of
    # its own methods verbatim — tracked here so a method belonging to one
    # of these classes never ALSO gets shown a second time as its own
    # separate Function entry below (real duplication this caused before
    # this check existed: the same method's code appearing twice in one
    # prompt, once buried in its class and once highlighted on its own).
    matched_class_qnames_by_service: dict[str, set[str]] = {}
    for symbol in relevant_classes:
        code_symbols_by_service.setdefault(symbol["service"], []).append(symbol)
        matched_class_qnames_by_service.setdefault(symbol["service"], set()).add(symbol["qualified_name"])
        selection_log.append(
            {
                "service": symbol["service"],
                "kind": "Class",
                "name": symbol["name"],
                "qualified_name": symbol["qualified_name"],
                "reason": "keyword match",
            }
        )
    snippet_map.update(
        graph_reader.load_source_snippets(
            [(s["service"], s["qualified_name"]) for s in relevant_classes]
        )
    )

    def _already_shown_via_its_class(service: str, parent_class: str | None) -> bool:
        return parent_class is not None and parent_class in matched_class_qnames_by_service.get(service, set())

    # Functions: real semantic similarity is the primary signal now, via the
    # chunk embeddings built at ingestion time — reusing the SAME issue
    # embedding already computed above, not a second embed() call.
    chunk_hits = graph_reader.vector_search_chunks(embedding, candidate_names)
    ranked_functions = candidates.select_relevant_functions_by_embedding(chunk_hits)

    services_with_chunk_hits = {hit["service"] for hit in ranked_functions}
    for hit in ranked_functions:
        key = (hit["service"], hit["function_qualified_name"])
        meta = metadata_by_key.get(key)
        if meta is None:
            continue  # stale hit (function renamed/removed since the chunk was embedded) — skip, never fabricate metadata
        if _already_shown_via_its_class(hit["service"], meta.get("parent_class")):
            continue  # this method's code is already fully visible in its class's own snippet above
        code_symbols_by_service.setdefault(hit["service"], []).append(meta)
        # The winning CHUNK's text, not the function's full source_snippet —
        # the relevant slice for a long function, the whole thing for a
        # short one, exactly the point of chunking.
        snippet_map[key] = hit["text"]
        selection_log.append(
            {
                "service": hit["service"],
                "kind": "Function",
                "name": meta["name"],
                "qualified_name": meta["qualified_name"],
                "reason": f"semantic match (score={hit['score']:.3f})",
            }
        )

    # Fallback: any candidate service with zero chunk hits (not yet
    # re-ingested since chunking shipped, or nothing semantically close
    # among what it has) still gets a keyword-based functions shortlist —
    # never silently left with no code symbols just because embeddings
    # haven't (yet) been built for that one service.
    services_needing_fallback = set(candidate_names) - services_with_chunk_hits
    if services_needing_fallback:
        fallback_pool = [s for s in raw_functions if s["service"] in services_needing_fallback]
        fallback_ranked = candidates.select_relevant_symbols(fallback_pool, issue_text)
        fallback_kept = [
            symbol
            for symbol in fallback_ranked
            if not _already_shown_via_its_class(symbol["service"], symbol.get("parent_class"))
        ]
        for symbol in fallback_kept:
            code_symbols_by_service.setdefault(symbol["service"], []).append(symbol)
            selection_log.append(
                {
                    "service": symbol["service"],
                    "kind": "Function",
                    "name": symbol["name"],
                    "qualified_name": symbol["qualified_name"],
                    "reason": "keyword fallback (no chunk embeddings for this service)",
                }
            )
        snippet_map.update(
            graph_reader.load_source_snippets(
                [(s["service"], s["qualified_name"]) for s in fallback_kept]
            )
        )

    system_prompt = planner.build_system_prompt()
    user_prompt = planner.build_user_prompt(
        issue_text,
        candidate_docs,
        code_symbols_by_service,
        snippet_map,
        seed_names=set(seed_names),
        known_dependencies=known_dependencies,
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
        _print_selection_debug(candidate_names, seed_names, selection_log, snippet_map)

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
