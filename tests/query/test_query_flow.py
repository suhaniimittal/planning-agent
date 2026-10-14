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


def _fn(service, qname, source="def f(): pass", file_path=None):
    return {
        "service": service,
        "qualified_name": qname,
        "name": qname.split("::")[-1].split(".")[-1],
        "file_path": file_path or qname.split("::")[0],
        "line_start": 1,
        "line_end": 2,
        "signature": "f()",
        "docstring": None,
        "language": "python",
        "source_snippet": source,
        "is_method": False,
        "parent_class": None,
        "kind": "Function",
    }


class FakeGraph:
    """Everything build_tdd reads, configurable per test, with every call recorded."""

    def __init__(self):
        self.summary = {}
        self.hits = {}  # service -> chunk hit rows
        self.symbols = {}  # (service, qname) -> metadata row
        self.neighbors = []
        self.edges = []
        self.file_context = {}
        self.calls = []
        self.prompt = {}
        self.tdd = FakeTdd()


@pytest.fixture
def graph(monkeypatch):
    g = FakeGraph()

    async def fake_embed_many(texts):
        g.calls.append(("embed_many", list(texts)))
        return [[float(i)] for i in range(len(texts))]

    def summary_scores(embedding, names):
        g.calls.append(("summary_scores", embedding, names))
        return {n: s for n, s in g.summary.items() if names is None or n in names}

    def search_service_chunks(service, vectors, identifiers):
        g.calls.append(("search_service_chunks", service, identifiers))
        return g.hits.get(service, [])

    def expand_neighbors(seed_names, hops):
        g.calls.append(("expand_neighbors", list(seed_names), hops))
        return [{"name": n, "hop_distance": 1} for n in g.neighbors]

    def load_candidate_docs(names):
        g.calls.append(("load_candidate_docs", list(names)))
        return [{"name": n, "summary": "d"} for n in reversed(names)]  # order must not matter

    def get_edges_among(names):
        g.calls.append(("get_edges_among", list(names)))
        return [e for e in g.edges if e["source"] in names and e["target"] in names]

    def load_symbols(pairs):
        g.calls.append(("load_symbols", list(pairs)))
        return [g.symbols[p] for p in pairs if p in g.symbols]

    def load_file_context(pairs):
        g.calls.append(("load_file_context", list(pairs)))
        return {p: g.file_context.get(p, []) for p in pairs}

    def build_user_prompt(issue_text, docs, code_symbols_by_service=None, snippets_by_key=None, **kwargs):
        g.prompt = {
            "docs": [d["name"] for d in docs],
            "code": code_symbols_by_service,
            "snippets": snippets_by_key,
            **kwargs,
        }
        return "user"

    async def call_llm(system_prompt, user_prompt):
        return "raw"

    monkeypatch.setattr(query_flow, "embed_many", fake_embed_many)
    for name, fn in {
        "summary_scores": summary_scores,
        "search_service_chunks": search_service_chunks,
        "expand_neighbors": expand_neighbors,
        "load_candidate_docs": load_candidate_docs,
        "get_edges_among": get_edges_among,
        "load_symbols": load_symbols,
        "load_file_context": load_file_context,
    }.items():
        monkeypatch.setattr(query_flow.graph_reader, name, fn)
    monkeypatch.setattr(query_flow.planner, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(query_flow.planner, "build_user_prompt", build_user_prompt)
    monkeypatch.setattr(query_flow.planner, "call_llm", call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", lambda raw: g.tdd)
    monkeypatch.setattr(query_flow, "verify_tdd", lambda tdd, shown: None)
    return g


def _hit(qname, score, identifiers=()):
    return {"function_qualified_name": qname, "text": f"chunk of {qname}", "score": score,
            "identifiers": list(identifiers)}


def _searched(g):
    return [c[1] for c in g.calls if c[0] == "search_service_chunks"]


@pytest.mark.asyncio
async def test_code_evidence_can_outrank_a_better_summary_match(graph):
    """The real failure: a UI repo's generic summary ranked it 4th of 6 for a
    bug in its own form validation, so a summary-only top-3 dropped it."""
    graph.summary = {"compliance": 0.65, "reports": 0.64, "candidate": 0.637, "ui": 0.596, "pdf": 0.5}
    graph.hits = {
        "compliance": [_hit("c.kt::a", 0.71)],
        "reports": [_hit("r.kt::a", 0.60)],
        "candidate": [_hit("cand.kt::isPhoneValid", 0.78)],
        "ui": [_hit("form.ts::validationSchema", 0.87)],
    }
    for s, rows in graph.hits.items():
        for h in rows:
            graph.symbols[(s, h["function_qualified_name"])] = _fn(s, h["function_qualified_name"])

    await query_flow.build_tdd("issue", top_k=2, hops=0)

    assert set(graph.prompt["seed_names"]) == {"ui", "candidate"}
    assert set(graph.prompt["code"]) == {"ui", "candidate"}
    # the shortlist that got searched is wider than top_k (2 + 3 extra)
    assert set(_searched(graph)) == {"compliance", "reports", "candidate", "ui", "pdf"}


@pytest.mark.asyncio
async def test_a_service_named_in_the_issue_is_always_selected(graph):
    graph.summary = {"orders": 0.9, "payments": 0.8, "candidate-service": 0.1}
    graph.hits = {"orders": [_hit("o.py::f", 0.9)], "payments": [_hit("p.py::f", 0.9)]}

    await query_flow.build_tdd("the bug is in candidate-service", top_k=1, hops=0)

    assert graph.prompt["seed_names"] == {"candidate-service"}


@pytest.mark.asyncio
async def test_only_the_runs_own_services_are_considered(graph):
    graph.summary = {"ui": 0.9, "leftover-service": 0.95}
    await query_flow.build_tdd("issue", top_k=5, hops=0, service_names=["ui"])

    assert ("summary_scores", [0.0], ["ui"]) in graph.calls
    assert graph.prompt["docs"] == ["ui"]


@pytest.mark.asyncio
async def test_whole_ticket_picks_services_while_its_parts_search_code(graph):
    """The whole ticket scores services; its parts additionally search code,
    so a detail only one sentence describes still reaches its function."""
    graph.summary = {"ui": 0.9}
    issue = (
        "Submit silently fails on the deferred report screen for this candidate.\n"
        "The approveDeferredReport mutation is never sent to the server at all."
    )

    await query_flow.build_tdd(issue, top_k=1, hops=0)

    embedded = next(c[1] for c in graph.calls if c[0] == "embed_many")
    assert embedded[0] == issue.strip()  # part 1 is always the whole ticket
    assert len(embedded) == 3  # whole ticket + its two sentences
    # services are scored on the whole ticket alone, never on a part
    assert next(c for c in graph.calls if c[0] == "summary_scores")[1] == [0.0]
    search = next(c for c in graph.calls if c[0] == "search_service_chunks")
    assert search[2] == ["approveDeferredReport"]


@pytest.mark.asyncio
async def test_a_matched_method_pulls_in_its_whole_small_class(graph):
    """The real failure: the method that READ a field matched the ticket, the
    sibling that WROTE it (holding the actual fix) did not — same small class."""
    graph.summary = {"svc": 0.9}
    graph.hits = {"svc": [_hit("Input.kt::Input.isPhoneValid", 0.9)]}
    method = _fn("svc", "Input.kt::Input.isPhoneValid", source="fun isPhoneValid() = ...")
    method["parent_class"] = "Input.kt::Input"
    graph.symbols[("svc", "Input.kt::Input.isPhoneValid")] = method
    graph.symbols[("svc", "Input.kt::Input")] = {
        **_fn("svc", "Input.kt::Input", source="class Input { isPhoneValid ... toAvro ... }"),
        "kind": "Class",
    }

    await query_flow.build_tdd("phone is wrong", top_k=1, hops=0)

    shown = {s["qualified_name"] for s in graph.prompt["code"]["svc"]}
    assert "Input.kt::Input" in shown  # the whole class came along
    assert graph.prompt["snippets"][("svc", "Input.kt::Input")] == (
        "class Input { isPhoneValid ... toAvro ... }"
    )


@pytest.mark.asyncio
async def test_a_large_class_is_not_pulled_in(graph):
    graph.summary = {"svc": 0.9}
    graph.hits = {"svc": [_hit("Big.kt::Big.method", 0.9)]}
    method = _fn("svc", "Big.kt::Big.method")
    method["parent_class"] = "Big.kt::Big"
    graph.symbols[("svc", "Big.kt::Big.method")] = method
    graph.symbols[("svc", "Big.kt::Big")] = {
        **_fn("svc", "Big.kt::Big", source="x" * 20000),
        "kind": "Class",
    }

    await query_flow.build_tdd("issue", top_k=1, hops=0)

    shown = {s["qualified_name"] for s in graph.prompt["code"]["svc"]}
    assert "Big.kt::Big" not in shown


@pytest.mark.asyncio
async def test_short_functions_are_shown_whole_and_long_ones_by_their_chunk(graph):
    graph.summary = {"ui": 0.9}
    graph.hits = {"ui": [_hit("a.ts::short", 0.9), _hit("b.ts::long", 0.89), _hit("gone.ts::stale", 0.88)]}
    graph.symbols[("ui", "a.ts::short")] = _fn("ui", "a.ts::short", source="short body")
    graph.symbols[("ui", "b.ts::long")] = _fn("ui", "b.ts::long", source="x" * 5000)

    await query_flow.build_tdd("issue", top_k=1, hops=0)

    snippets = graph.prompt["snippets"]
    assert snippets[("ui", "a.ts::short")] == "short body"
    assert snippets[("ui", "b.ts::long")] == "chunk of b.ts::long"
    # a chunk whose function no longer exists is skipped, never fabricated
    assert [s["qualified_name"] for s in graph.prompt["code"]["ui"]] == ["a.ts::short", "b.ts::long"]


@pytest.mark.asyncio
async def test_file_context_is_loaded_for_every_shown_file(graph):
    graph.summary = {"ui": 0.9}
    graph.hits = {"ui": [_hit("a.ts::f", 0.9)]}
    graph.symbols[("ui", "a.ts::f")] = _fn("ui", "a.ts::f")
    graph.file_context[("ui", "a.ts")] = ["f", "g"]

    await query_flow.build_tdd("issue", top_k=1, hops=0)

    assert graph.prompt["file_context"] == {("ui", "a.ts"): ["f", "g"]}


@pytest.mark.asyncio
async def test_graph_neighbors_are_background_context_only(graph):
    graph.summary = {"orders": 0.9, "shipping": 0.1, "billing": 0.1}
    graph.neighbors = ["shipping", "not-in-this-run"]
    graph.edges = [
        {"source": "orders", "target": "shipping"},
        {"source": "shipping", "target": "billing"},
    ]

    await query_flow.build_tdd(
        "issue", top_k=1, hops=1, service_names=["orders", "shipping", "billing"]
    )

    assert graph.prompt["docs"] == ["orders", "shipping"]  # selected first, then context
    assert graph.prompt["seed_names"] == {"orders"}
    # only links between selected services are given as known dependencies
    assert graph.prompt["known_dependencies"] == []


@pytest.mark.asyncio
async def test_known_dependencies_are_links_between_selected_services(graph):
    graph.summary = {"ui": 0.9, "candidate": 0.8, "pdf": 0.1}
    graph.edges = [
        {"source": "ui", "target": "candidate"},
        {"source": "candidate", "target": "pdf"},
    ]

    await query_flow.build_tdd("issue", top_k=2, hops=0)

    assert graph.prompt["known_dependencies"] == [{"source": "ui", "target": "candidate"}]


@pytest.mark.asyncio
async def test_dependency_diagram_is_scoped_to_flagged_services_only(graph):
    """Regression test: the final service_dependencies must only reflect what
    the LLM actually flagged in `services`, not every candidate."""
    graph.summary = {"user": 0.9, "orders": 0.8, "shipping": 0.7}
    graph.tdd = FakeTdd(services=[FakeServicePlan("user")])

    await query_flow.build_tdd("user login issue", top_k=3, hops=0)

    edges_calls = [c[1] for c in graph.calls if c[0] == "get_edges_among"]
    assert edges_calls[-1] == ["user"]


@pytest.mark.asyncio
async def test_dependency_scope_widens_to_architecture_diagram_targets(graph):
    """Regression test: the LLM can flag only "orders" while drawing
    "orders -> shipping"; the verified dependencies must include it."""
    graph.summary = {"orders": 0.9}
    graph.edges = [{"source": "orders", "target": "shipping"}]
    graph.tdd = FakeTdd(
        services=[FakeServicePlan("orders")],
        current_architecture_components=[
            ArchComponent(name="orders", depends_on=["shipping"]),
            ArchComponent(name="shipping"),
        ],
    )

    result = await query_flow.build_tdd("issue", top_k=1, hops=0)

    edges_calls = [c[1] for c in graph.calls if c[0] == "get_edges_among"]
    assert edges_calls[-1] == ["orders", "shipping"]
    assert result.service_dependencies == [ServiceDependency(source="orders", target="shipping")]


@pytest.mark.asyncio
async def test_build_tdd_propagates_tdd_parse_error(graph, monkeypatch):
    def fail(raw):
        raise TddParseError("boom")

    monkeypatch.setattr(query_flow.planner, "parse_tdd", fail)
    with pytest.raises(TddParseError):
        await query_flow.build_tdd("issue text")


@pytest.mark.asyncio
async def test_build_tdd_verifies_changes_against_the_graph(graph, monkeypatch):
    """build_tdd must hand every change to the verifier, along with the files
    whose code was shown to the LLM."""
    from src.query.models import FileChange, ServicePlan, TechnicalDesignDoc

    graph.summary = {"ui": 0.9}
    graph.hits = {"ui": [_hit("src/m/view.tsx::M.view", 0.9)]}
    graph.symbols[("ui", "src/m/view.tsx::M.view")] = _fn("ui", "src/m/view.tsx::M.view")
    change = FileChange(
        file_path="src/m", function_or_symbol="ghost", change_description="d",
        implementation_notes="i", pseudocode_sketch="p", reasoning="r",
    )
    graph.tdd = TechnicalDesignDoc(
        title="t", issue_summary="s", overall_reasoning="r",
        services=[ServicePlan(service="ui", complexity="low", reasoning="r", changes=[change])],
    )
    seen = {}
    monkeypatch.setattr(query_flow, "verify_tdd", lambda t, shown: seen.update(tdd=t, shown=shown))

    result = await query_flow.build_tdd("issue", top_k=1, hops=0)

    assert seen["tdd"] is result
    assert seen["shown"] == {"ui": {"src/m/view.tsx"}}


@pytest.mark.asyncio
async def test_hops_above_two_are_capped(graph):
    graph.summary = {"orders": 0.9}
    await query_flow.build_tdd("issue", top_k=1, hops=5)
    assert ("expand_neighbors", ["orders"], 2) in graph.calls
