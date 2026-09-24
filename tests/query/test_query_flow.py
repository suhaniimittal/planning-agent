import pytest

from src.query import query_flow
from src.query.models import ArchComponent, ServiceDependency
from src.query.planner import TddParseError


class FakeServicePlan:
    def __init__(self, service, changes=()):
        self.service = service
        self.changes = list(changes)


class FakeTdd:
    def __init__(self, services=(), current_architecture_components=(), proposed_architecture_components=()):
        self.services = list(services)
        self.current_architecture_components = list(current_architecture_components)
        self.proposed_architecture_components = list(proposed_architecture_components)


@pytest.mark.asyncio
async def test_build_tdd_calls_pipeline_in_order(monkeypatch):
    calls = []

    async def fake_embed(text):
        calls.append(("embed", text))
        return [0.1, 0.2]

    def fake_vector_search(embedding, k):
        calls.append(("vector_search", embedding, k))
        return [{"name": "orders", "score": 0.9}]

    def fake_expand_neighbors(seed_names, hops):
        calls.append(("expand_neighbors", seed_names, hops))
        return [{"name": "shipping", "hop_distance": 1}]

    def fake_merge_candidates(vector_hits, neighbors):
        calls.append(("merge_candidates", vector_hits, neighbors))
        return ["orders", "shipping"]

    def fake_load_candidate_docs(names):
        calls.append(("load_candidate_docs", names))
        return [{"name": n, "full_doc": "d", "key_symbols": [], "domain": None} for n in names]

    def fake_load_code_symbols(names):
        calls.append(("load_code_symbols_for_services", names))
        return [
            {"service": "orders", "qualified_name": "a.py::fn", "name": "fn", "kind": "Function"}
        ]

    def fake_select_relevant_symbols(symbols, issue_text):
        calls.append(("select_relevant_symbols", symbols, issue_text))
        return []

    def fake_load_source_snippets(pairs):
        calls.append(("load_source_snippets", pairs))
        return {}

    def fake_vector_search_chunks(embedding, service_names):
        calls.append(("vector_search_chunks", embedding, service_names))
        # Only "orders" has a chunk hit — "shipping" must fall back to
        # keyword ranking, exercising both paths in the same call.
        return [{"service": "orders", "function_qualified_name": "a.py::fn", "text": "code", "score": 0.9}]

    def fake_select_relevant_functions_by_embedding(chunk_hits):
        calls.append(("select_relevant_functions_by_embedding", chunk_hits))
        return chunk_hits

    def fake_get_edges_among(names):
        calls.append(("get_edges_among", names))
        return [{"source": "orders", "target": "shipping"}]

    def fake_build_system_prompt():
        return "system"

    def fake_build_user_prompt(
        issue_text,
        docs,
        code_symbols_by_service=None,
        snippets_by_key=None,
        seed_names=None,
        known_dependencies=None,
    ):
        calls.append(("build_user_prompt", issue_text, docs))
        return "user"

    async def fake_call_llm(system_prompt, user_prompt):
        calls.append(("call_llm", system_prompt, user_prompt))
        return "raw"

    def fake_parse_tdd(raw):
        calls.append(("parse_tdd", raw))
        return FakeTdd(services=[FakeServicePlan("orders"), FakeServicePlan("shipping")])

    monkeypatch.setattr(query_flow, "embed", fake_embed)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_services", fake_vector_search)
    monkeypatch.setattr(query_flow.graph_reader, "expand_neighbors", fake_expand_neighbors)
    monkeypatch.setattr(query_flow.candidates, "merge_candidates", fake_merge_candidates)
    monkeypatch.setattr(query_flow.graph_reader, "load_candidate_docs", fake_load_candidate_docs)
    monkeypatch.setattr(
        query_flow.graph_reader, "load_code_symbols_for_services", fake_load_code_symbols
    )
    monkeypatch.setattr(
        query_flow.candidates, "select_relevant_symbols", fake_select_relevant_symbols
    )
    monkeypatch.setattr(query_flow.graph_reader, "load_source_snippets", fake_load_source_snippets)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_chunks", fake_vector_search_chunks)
    monkeypatch.setattr(
        query_flow.candidates,
        "select_relevant_functions_by_embedding",
        fake_select_relevant_functions_by_embedding,
    )
    monkeypatch.setattr(query_flow.graph_reader, "get_edges_among", fake_get_edges_among)
    monkeypatch.setattr(query_flow.planner, "build_system_prompt", fake_build_system_prompt)
    monkeypatch.setattr(query_flow.planner, "build_user_prompt", fake_build_user_prompt)
    monkeypatch.setattr(query_flow.planner, "call_llm", fake_call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", fake_parse_tdd)

    result = await query_flow.build_tdd("issue text", top_k=3, hops=1)

    assert isinstance(result, FakeTdd)
    assert result.service_dependencies == [ServiceDependency(source="orders", target="shipping")]
    step_names = [c[0] for c in calls]
    assert step_names == [
        "embed",
        "vector_search",
        "expand_neighbors",
        "merge_candidates",
        "load_candidate_docs",
        "get_edges_among",  # known_dependencies — shown to the LLM as input
        "load_code_symbols_for_services",
        "select_relevant_symbols",  # classes
        "load_source_snippets",  # classes
        "vector_search_chunks",
        "select_relevant_functions_by_embedding",
        "select_relevant_symbols",  # fallback, for "shipping" (no chunk hit)
        "load_source_snippets",  # fallback
        "build_user_prompt",
        "call_llm",
        "parse_tdd",
        "get_edges_among",  # service_dependencies — computed post-hoc, never trusted from the LLM
    ]
    # merge_candidates output feeds load_candidate_docs input
    load_call = next(c for c in calls if c[0] == "load_candidate_docs")
    assert load_call[1] == ["orders", "shipping"]
    # get_edges_among is called twice: once early (candidate pool, shown to
    # the LLM as known_dependencies) and once late (scoped to only the
    # LLM-flagged services, for the final service_dependencies field)
    edges_calls = [c for c in calls if c[0] == "get_edges_among"]
    assert len(edges_calls) == 2
    assert edges_calls[0][1] == ["orders", "shipping"]
    assert edges_calls[1][1] == ["orders", "shipping"]
    # vector_search_chunks reuses the SAME embedding computed at the top —
    # never a second embed() call
    chunks_call = next(c for c in calls if c[0] == "vector_search_chunks")
    assert chunks_call[1] == [0.1, 0.2]


@pytest.mark.asyncio
async def test_build_tdd_scopes_dependency_diagram_to_flagged_services_only(monkeypatch):
    """Regression test: with top_k larger than the real service count, the raw
    candidate pool can include services unrelated to the issue (e.g. orders/
    shipping showing up for a user-only issue). The dependency diagram must
    only reflect what the LLM actually flagged in `services`, not every
    candidate that happened to be vector-searched in."""
    edges_calls = []

    async def fake_embed(text):
        return [0.1]

    def fake_vector_search(embedding, k):
        return [{"name": "user", "score": 0.9}, {"name": "orders", "score": 0.4}]

    def fake_expand_neighbors(seed_names, hops):
        return []

    def fake_merge_candidates(vector_hits, neighbors):
        return ["user", "orders", "shipping", "flask"]

    def fake_load_candidate_docs(names):
        return [{"name": n, "full_doc": "d", "key_symbols": [], "domain": None} for n in names]

    def fake_load_code_symbols(names):
        return []

    def fake_select_relevant_symbols(symbols, issue_text):
        return []

    def fake_load_source_snippets(pairs):
        return {}

    def fake_vector_search_chunks(embedding, service_names):
        return []

    def fake_select_relevant_functions_by_embedding(chunk_hits):
        return []

    def fake_get_edges_among(names):
        edges_calls.append(names)
        return []

    def fake_build_system_prompt():
        return "system"

    def fake_build_user_prompt(
        issue_text,
        docs,
        code_symbols_by_service=None,
        snippets_by_key=None,
        seed_names=None,
        known_dependencies=None,
    ):
        return "user"

    async def fake_call_llm(system_prompt, user_prompt):
        return "raw"

    def fake_parse_tdd(raw):
        # LLM only flagged "user" as relevant, despite a wider candidate pool
        return FakeTdd(services=[FakeServicePlan("user")])

    monkeypatch.setattr(query_flow, "embed", fake_embed)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_services", fake_vector_search)
    monkeypatch.setattr(query_flow.graph_reader, "expand_neighbors", fake_expand_neighbors)
    monkeypatch.setattr(query_flow.candidates, "merge_candidates", fake_merge_candidates)
    monkeypatch.setattr(query_flow.graph_reader, "load_candidate_docs", fake_load_candidate_docs)
    monkeypatch.setattr(
        query_flow.graph_reader, "load_code_symbols_for_services", fake_load_code_symbols
    )
    monkeypatch.setattr(
        query_flow.candidates, "select_relevant_symbols", fake_select_relevant_symbols
    )
    monkeypatch.setattr(query_flow.graph_reader, "load_source_snippets", fake_load_source_snippets)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_chunks", fake_vector_search_chunks)
    monkeypatch.setattr(
        query_flow.candidates,
        "select_relevant_functions_by_embedding",
        fake_select_relevant_functions_by_embedding,
    )
    monkeypatch.setattr(query_flow.graph_reader, "get_edges_among", fake_get_edges_among)
    monkeypatch.setattr(query_flow.planner, "build_system_prompt", fake_build_system_prompt)
    monkeypatch.setattr(query_flow.planner, "build_user_prompt", fake_build_user_prompt)
    monkeypatch.setattr(query_flow.planner, "call_llm", fake_call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", fake_parse_tdd)

    await query_flow.build_tdd("user login issue", top_k=5, hops=2)

    # get_edges_among is called twice: once early with the full candidate
    # pool (for known_dependencies, shown to the LLM as input) and once
    # late, scoped to ONLY the LLM-flagged services (for the final
    # service_dependencies field) — this test's actual invariant is that
    # the LATE call stays scoped to "user" alone, not the wider pool.
    assert edges_calls == [["user", "orders", "shipping", "flask"], ["user"]]


@pytest.mark.asyncio
async def test_build_tdd_widens_dependency_scope_to_architecture_diagram_targets(monkeypatch):
    """Regression test: the LLM can flag changes for only "orders" while still
    drawing "orders -> shipping" in its architecture diagram. The final
    service_dependencies verification must include "shipping" too, or a real,
    graph-confirmed dependency the diagram itself shows goes missing from
    this field even though it's genuinely there."""
    edges_calls = []

    async def fake_embed(text):
        return [0.1]

    def fake_vector_search(embedding, k):
        return [{"name": "orders", "score": 0.9}]

    def fake_expand_neighbors(seed_names, hops):
        return [{"name": "shipping", "hop_distance": 1}]

    def fake_merge_candidates(vector_hits, neighbors):
        return ["orders", "shipping"]

    def fake_load_candidate_docs(names):
        return [{"name": n, "full_doc": "d", "key_symbols": [], "domain": None} for n in names]

    def fake_load_code_symbols(names):
        return []

    def fake_select_relevant_symbols(symbols, issue_text):
        return []

    def fake_load_source_snippets(pairs):
        return {}

    def fake_vector_search_chunks(embedding, service_names):
        return []

    def fake_select_relevant_functions_by_embedding(chunk_hits):
        return []

    def fake_get_edges_among(names):
        edges_calls.append(names)
        return [{"source": "orders", "target": "shipping"}] if "shipping" in names else []

    def fake_build_system_prompt():
        return "system"

    def fake_build_user_prompt(
        issue_text,
        docs,
        code_symbols_by_service=None,
        snippets_by_key=None,
        seed_names=None,
        known_dependencies=None,
    ):
        return "user"

    async def fake_call_llm(system_prompt, user_prompt):
        return "raw"

    def fake_parse_tdd(raw):
        # LLM only flagged "orders" in services, but its own architecture
        # diagram names "shipping" as a dependency target.
        return FakeTdd(
            services=[FakeServicePlan("orders")],
            current_architecture_components=[
                ArchComponent(name="orders", depends_on=["shipping"]),
                ArchComponent(name="shipping"),
            ],
        )

    monkeypatch.setattr(query_flow, "embed", fake_embed)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_services", fake_vector_search)
    monkeypatch.setattr(query_flow.graph_reader, "expand_neighbors", fake_expand_neighbors)
    monkeypatch.setattr(query_flow.candidates, "merge_candidates", fake_merge_candidates)
    monkeypatch.setattr(query_flow.graph_reader, "load_candidate_docs", fake_load_candidate_docs)
    monkeypatch.setattr(
        query_flow.graph_reader, "load_code_symbols_for_services", fake_load_code_symbols
    )
    monkeypatch.setattr(
        query_flow.candidates, "select_relevant_symbols", fake_select_relevant_symbols
    )
    monkeypatch.setattr(query_flow.graph_reader, "load_source_snippets", fake_load_source_snippets)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_chunks", fake_vector_search_chunks)
    monkeypatch.setattr(
        query_flow.candidates,
        "select_relevant_functions_by_embedding",
        fake_select_relevant_functions_by_embedding,
    )
    monkeypatch.setattr(query_flow.graph_reader, "get_edges_among", fake_get_edges_among)
    monkeypatch.setattr(query_flow.planner, "build_system_prompt", fake_build_system_prompt)
    monkeypatch.setattr(query_flow.planner, "build_user_prompt", fake_build_user_prompt)
    monkeypatch.setattr(query_flow.planner, "call_llm", fake_call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", fake_parse_tdd)

    result = await query_flow.build_tdd("issue text", top_k=3, hops=1)

    # Late get_edges_among call must include "shipping" even though it
    # wasn't in tdd.services, because the diagram named it as a dependency.
    assert edges_calls[-1] == ["orders", "shipping"]
    assert result.service_dependencies == [ServiceDependency(source="orders", target="shipping")]


@pytest.mark.asyncio
async def test_build_tdd_propagates_tdd_parse_error(monkeypatch):
    async def fake_embed(text):
        return [0.1]

    def fake_vector_search(embedding, k):
        return []

    def fake_expand_neighbors(seed_names, hops):
        return []

    def fake_merge_candidates(vector_hits, neighbors):
        return []

    def fake_load_candidate_docs(names):
        return []

    def fake_load_code_symbols(names):
        return []

    def fake_select_relevant_symbols(symbols, issue_text):
        return []

    def fake_load_source_snippets(pairs):
        return {}

    def fake_vector_search_chunks(embedding, service_names):
        return []

    def fake_select_relevant_functions_by_embedding(chunk_hits):
        return []

    def fake_get_edges_among(names):
        return []

    def fake_build_system_prompt():
        return "system"

    def fake_build_user_prompt(
        issue_text,
        docs,
        code_symbols_by_service=None,
        snippets_by_key=None,
        seed_names=None,
        known_dependencies=None,
    ):
        return "user"

    async def fake_call_llm(system_prompt, user_prompt):
        return "not valid json"

    def fake_parse_tdd(raw):
        raise TddParseError("boom")

    monkeypatch.setattr(query_flow, "embed", fake_embed)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_services", fake_vector_search)
    monkeypatch.setattr(query_flow.graph_reader, "expand_neighbors", fake_expand_neighbors)
    monkeypatch.setattr(query_flow.candidates, "merge_candidates", fake_merge_candidates)
    monkeypatch.setattr(query_flow.graph_reader, "load_candidate_docs", fake_load_candidate_docs)
    monkeypatch.setattr(query_flow.graph_reader, "get_edges_among", fake_get_edges_among)
    monkeypatch.setattr(
        query_flow.graph_reader, "load_code_symbols_for_services", fake_load_code_symbols
    )
    monkeypatch.setattr(
        query_flow.candidates, "select_relevant_symbols", fake_select_relevant_symbols
    )
    monkeypatch.setattr(query_flow.graph_reader, "load_source_snippets", fake_load_source_snippets)
    monkeypatch.setattr(query_flow.graph_reader, "vector_search_chunks", fake_vector_search_chunks)
    monkeypatch.setattr(
        query_flow.candidates,
        "select_relevant_functions_by_embedding",
        fake_select_relevant_functions_by_embedding,
    )
    monkeypatch.setattr(query_flow.planner, "build_system_prompt", fake_build_system_prompt)
    monkeypatch.setattr(query_flow.planner, "build_user_prompt", fake_build_user_prompt)
    monkeypatch.setattr(query_flow.planner, "call_llm", fake_call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", fake_parse_tdd)

    with pytest.raises(TddParseError):
        await query_flow.build_tdd("issue text")
