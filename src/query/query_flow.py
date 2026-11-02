"""Orchestrator for the query-time flow: given a plain-text requirement or
bug report, pick the services it involves (by summary AND by their own
code), choose the files the planner sees, and produce a structured
technical design doc.

There is one planner: whole files chosen by `context`, the sectioned prompt
from `prompt`, the lookup loop when the model is called directly
(`tool_loop`), and the same checks and verification for every design.

Mirrors how ingest.py orchestrates over embedder.py/graph_writer.py.
"""

from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

from src.ingestion.embedder import embed_many

from . import (
    answer_checks,
    code_search,
    graph_reader,
    issue_split,
    planner,
    prompt,
    ticket_points,
    tool_loop,
)
from .context import Context, build_context, names_a_value
from .models import ServiceDependency, TechnicalDesignDoc
from .verify import verify_tdd

# The planner reads what ingestion pipeline version 4 and later writes: every
# file's text and the links between files. Ingestion rebuilds an older graph
# on its next sync, so an older service here has not been synced since.
_MIN_PIPELINE_VERSION = 4


def _warn_old_graphs(selected: list[str]) -> None:
    versions = graph_reader.pipeline_versions(selected)
    old = sorted(s for s in selected if (versions.get(s) or 0) < _MIN_PIPELINE_VERSION)
    if old:
        print(
            f"  [TDD] not yet ingested at pipeline version {_MIN_PIPELINE_VERSION}: {old}. "
            "Their files and links are missing until they are synced again."
        )


def _new_tests() -> bool:
    """Whether the TDD may ask for new tests (PLANNER_NEW_TESTS, off unless
    set). Off until the coding agent can run a test before pushing it: its
    copy of a repo has no dependencies installed, so a new test's first run
    was CI, where it failed. Existing tests a change breaks are updated
    either way."""
    return os.environ.get("PLANNER_NEW_TESTS", "").strip().lower() in ("1", "on", "true", "yes")


def _print_context_debug(context: Context) -> None:
    print(f"  [TDD] {len(context.files)} files, {context.used:,}/{context.budget:,} chars:")
    for f in context.files:
        print(f"    [{f.tier:9}] {f.status:7} {len(f.text):>7,}  {f.service}/{f.path}  — {f.reason}")
    for t in context.tables:
        print(f"    [table    ] {t['service']}.{t['name']}")
    if context.left_out:
        print(f"    left out ({len(context.left_out)}), best first:")
        for service, path, tier, score in context.left_out[:15]:
            print(f"      {score:.3f} [{tier}] {service}/{path}")


async def _plan(
    issue_text: str,
    mode: str,
    identifiers: list[str],
    selected: list[str],
    candidate_docs: list[dict],
    ranked_by_service: dict[str, list[dict]],
    service_names: list[str] | None,
    debug_dump_prompts: bool,
    vectors: list[list[float]] | None = None,
    images: list | None = None,
    *,
    memory: str = "",
    technology_constraints: list[str] | None = None,
    api_contracts: str = "",
) -> TechnicalDesignDoc:
    """The planner: whole files chosen by context, the sectioned prompt,
    the lookup loop when the model is called directly, then the checks and
    the verification against the graph."""
    _warn_old_graphs(selected)
    # Listed while the context is chosen: neither needs the other.
    points_task = asyncio.create_task(ticket_points.extract(issue_text))
    context = await asyncio.to_thread(
        build_context,
        {s: ranked_by_service[s] for s in selected},
        service_names,
        vectors=vectors,
    )
    # A service reached only through a link (a Kafka handler, a resolver)
    # gets its summary too, and a verdict.
    involved = list(dict.fromkeys(selected + context.services()))
    docs = list(candidate_docs)
    missing = [s for s in involved if s not in {d["name"] for d in docs}]
    if missing:
        docs += graph_reader.load_candidate_docs(missing)
    known_dependencies = graph_reader.get_edges_with_evidence(involved)

    new_tests = _new_tests()
    use_loop = tool_loop.enabled()
    if use_loop and planner._use_gateway():
        print("  [loop] PLANNER_LOOP is on, but the gateway path has no loop yet: one call")
        use_loop = False

    def system(lookups: bool) -> str:
        return prompt.build_system_prompt(
            mode,
            value_trace=names_a_value(issue_text, identifiers),
            stored_data=bool(context.tables),
            test_writing=context.has_tests,
            lookups=(tool_loop.MAX_ROUNDS, tool_loop.MAX_LOOKUPS) if lookups else None,
            new_tests=new_tests,
        )

    system_prompt = system(use_loop)
    points = await points_task
    if points:
        print(f"  [TDD] ticket points ({len(points)}): " + " | ".join(points))
    user_prompt = prompt.build_user_prompt(
        issue_text, docs, context, known_dependencies, points,
        # The gateway takes text only: no images, so no note that they follow.
        image_names=None if planner._use_gateway() else [image.name for image in images or []],
        memory=memory,
        technology_constraints=technology_constraints,
        api_contracts=api_contracts,
    )
    _print_context_debug(context)
    print(f"  [TDD] prompt: {len(system_prompt):,} + {len(user_prompt):,} chars")
    if debug_dump_prompts:
        with open("llm_system_prompt.txt", "w") as f:
            f.write(system_prompt)
        with open("llm_user_prompt.txt", "w") as f:
            f.write(user_prompt)

    raw = None
    if use_loop:
        lookups = tool_loop.Lookups(services=service_names or involved, context=context)
        try:
            result = await tool_loop.run(
                system_prompt, user_prompt, lookups,
                check=answer_checks.check_for(points, new_tests=new_tests),
                recheck=prompt.RECHECK, images=images,
            )
            raw = result.raw
            # What it read counts as shown, for verification.
            context.files += lookups.files
            print(f"  [loop] answered in round {result.rounds} after {lookups.count} lookups "
                  f"({lookups.chars:,} chars), {result.send_backs} sent back, "
                  f"re-checked: {'yes' if result.rechecked else 'no'}")
        except Exception as error:  # the single call is always there to fall back on
            print(f"  [loop] failed ({error!r}); falling back to one call")
            system_prompt = system(False)
    if raw is None:
        # Images only when there are some, so a text-only call keeps call_llm's
        # two-argument form.
        args = (system_prompt, user_prompt, images) if images else (system_prompt, user_prompt)
        raw = await planner.call_llm(*args)
    tdd = planner.parse_tdd(raw)
    tdd.generated_at = datetime.now(UTC).isoformat()
    # Read from the build files, not written by the model (2026-10-15).
    tdd.technology_constraints = list(technology_constraints or [])
    if missing := ticket_points.note_uncovered(tdd, points):
        print(f"  [TDD] ticket points still unanswered: {missing} (added to open questions)")
    tdd.service_dependencies = _links_of_changed_services(
        tdd, graph_reader.get_edges_among(_referenced_service_names(tdd))
    )
    citable = context.symbol_names_by_service()
    for service_plan in tdd.services:
        names = citable.get(service_plan.service, set())
        for change in service_plan.changes:
            change.has_real_source = change.function_or_symbol in names
    shown_files = context.shown_files_by_service()
    verify_tdd(tdd, shown_files)
    tdd.decisions = ground_decisions(tdd.decisions, shown_files)
    return tdd


# A service's rank = this weight x summary match + the rest x best code match.
_SUMMARY_WEIGHT = 0.5

def _links_of_changed_services(
    tdd: TechnicalDesignDoc, edges: list[dict]
) -> list[ServiceDependency]:
    """The links with a changing service at one end. A UI-only fix listed
    all 32 links among the eight services it gave a verdict on, none of
    which it touched."""
    changed = {s.service for s in tdd.services if s.changes}
    return [
        ServiceDependency(**e) for e in edges if e["source"] in changed or e["target"] in changed
    ]


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


async def build_tdd(
    issue_text: str,
    top_k: int = 8,
    hops: int = 0,
    debug_dump_prompts: bool = False,
    service_names: list[str] | None = None,
    technology_constraints: list[str] | None = None,
    api_contracts: str = "",
    mode: str | None = None,
    memory: str = "",
    images: list | None = None,
) -> TechnicalDesignDoc:
    """`images`: screenshots from the ticket (images.TicketImage), shown to
    the planner with its first call. `memory`: what this ticket's earlier
    versions and its project decided (src/memory), shown as its own section.
    `technology_constraints` / `api_contracts`: read from the repositories,
    which the design must fit.

    `service_names`: the services this run covers (the Repositories the
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

    return await _plan(
        issue_text, mode, identifiers, selected, candidate_docs, ranked_by_service,
        service_names, debug_dump_prompts, vectors, images,
        memory=memory,
        technology_constraints=technology_constraints,
        api_contracts=api_contracts,
    )


def ground_decisions(decisions, shown_files_by_service: dict[str, set[str]]):
    """A decision's targets kept only where the file was shown or exists in the graph."""
    for decision in decisions:
        kept = []
        for target in decision.applies_to:
            shown = target.file_path in shown_files_by_service.get(target.service, set())
            if shown or graph_reader.file_exists(target.service, target.file_path):
                kept.append(target)
        decision.applies_to = kept
    return decisions
