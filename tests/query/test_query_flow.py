import pytest

from src.query import query_flow
from src.query.models import (
    ArchComponent,
    DecisionTarget,
    DesignDecision,
    FileChange,
    ServiceDependency,
    ServicePlan,
    TechnicalDesignDoc,
)
from src.query.planner import TddParseError


def _tdd(**fields) -> TechnicalDesignDoc:
    return TechnicalDesignDoc(
        **{"title": "t", "issue_summary": "s", "overall_reasoning": "r", "services": [], **fields}
    )


# --- build_tdd: which services the planner is given ---------------------------------------


class FakeGraph:
    """Everything build_tdd reads before it plans, with every call recorded,
    and what it handed to the planner."""

    def __init__(self):
        self.summary = {}
        self.hits = {}  # service -> chunk hit rows
        self.neighbors = []
        self.calls = []
        self.plan = {}


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

    async def plan(issue_text, mode, identifiers, selected, candidate_docs, ranked_by_service,
                   service_names, debug_dump_prompts, vectors=None, images=None, **kwargs):
        g.plan = {
            "selected": list(selected),
            "docs": [d["name"] for d in candidate_docs],
            "ranked": ranked_by_service,
            "images": images,
            **kwargs,
        }
        return _tdd()

    monkeypatch.setattr(query_flow, "embed_many", fake_embed_many)
    for name, fn in {
        "summary_scores": summary_scores,
        "search_service_chunks": search_service_chunks,
        "expand_neighbors": expand_neighbors,
        "load_candidate_docs": load_candidate_docs,
    }.items():
        monkeypatch.setattr(query_flow.graph_reader, name, fn)
    monkeypatch.setattr(query_flow, "_plan", plan)
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

    await query_flow.build_tdd("issue", top_k=2, hops=0)

    assert set(graph.plan["selected"]) == {"ui", "candidate"}
    # every service in the run was searched, not just the top few by summary
    assert set(_searched(graph)) == {"compliance", "reports", "candidate", "ui", "pdf"}


@pytest.mark.asyncio
async def test_no_service_is_cut_before_its_code_is_searched(graph):
    """With more services than any old shortlist would have held, the WORST
    summary match still gets its code read — and wins on it."""
    graph.summary = {
        "a": 0.70, "b": 0.69, "c": 0.68, "d": 0.67, "e": 0.66,
        "f": 0.65, "g": 0.64, "h": 0.63, "i": 0.62, "ui": 0.10,
    }
    graph.hits = {"ui": [_hit("form.ts::validationSchema", 0.95)]}

    await query_flow.build_tdd("issue", top_k=2, hops=0)

    assert set(_searched(graph)) == set(graph.summary)
    assert "ui" in graph.plan["selected"]


@pytest.mark.asyncio
async def test_a_service_named_in_the_issue_is_always_selected(graph):
    graph.summary = {"orders": 0.9, "payments": 0.8, "candidate-service": 0.1}
    graph.hits = {"orders": [_hit("o.py::f", 0.9)], "payments": [_hit("p.py::f", 0.9)]}

    await query_flow.build_tdd("the bug is in candidate-service", top_k=1, hops=0)

    assert graph.plan["selected"] == ["candidate-service"]


@pytest.mark.asyncio
async def test_only_the_runs_own_services_are_considered(graph):
    graph.summary = {"ui": 0.9, "leftover-service": 0.95}
    await query_flow.build_tdd("issue", top_k=5, hops=0, service_names=["ui"])

    assert ("summary_scores", [0.0], ["ui"]) in graph.calls
    assert graph.plan["docs"] == ["ui"]


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
async def test_graph_neighbors_are_background_context_only(graph):
    graph.summary = {"orders": 0.9, "shipping": 0.1, "billing": 0.1}
    graph.neighbors = ["shipping", "not-in-this-run"]

    await query_flow.build_tdd(
        "issue", top_k=1, hops=1, service_names=["orders", "shipping", "billing"]
    )

    assert graph.plan["docs"] == ["orders", "shipping"]  # selected first, then context
    assert graph.plan["selected"] == ["orders"]


@pytest.mark.asyncio
async def test_hops_above_two_are_capped(graph):
    graph.summary = {"orders": 0.9}
    await query_flow.build_tdd("issue", top_k=1, hops=5)
    assert ("expand_neighbors", ["orders"], 2) in graph.calls


@pytest.mark.asyncio
async def test_memory_constraints_contracts_and_screenshots_reach_the_planner(graph):
    graph.summary = {"orders": 0.9}

    await query_flow.build_tdd(
        "issue", top_k=1, hops=0,
        technology_constraints=["Language: Java 17 (from pom.xml)."],
        api_contracts="openapi: 3.0.0",
        memory="DECISIONS IN FORCE (agreed) ...",
        images=["shot"],
    )

    assert graph.plan["technology_constraints"] == ["Language: Java 17 (from pom.xml)."]
    assert graph.plan["api_contracts"] == "openapi: 3.0.0"
    assert graph.plan["memory"] == "DECISIONS IN FORCE (agreed) ..."
    assert graph.plan["images"] == ["shot"]


# --- _plan: the planner itself, around the model call ----------------------------------------


class FakeContext:
    def __init__(self, shown=None):
        self.files, self.tables, self.left_out = [], [], []
        self.has_tests, self.used, self.budget = False, 0, 240_000
        self._shown = shown or {}

    def services(self):
        return list(self._shown)

    def symbol_names_by_service(self):
        return {}

    def shown_files_by_service(self):
        return self._shown


class Planned:
    def __init__(self):
        self.tdd = _tdd()
        self.shown = {}
        self.user_prompt_kwargs = {}
        self.edges_asked = []
        self.edges = []
        self.verified = None
        self.existing_files = set()


@pytest.fixture
def planned(monkeypatch):
    p = Planned()

    def build_context(ranked, service_names, vectors=None):
        return FakeContext(p.shown)

    async def extract(issue_text):
        return []

    def build_user_prompt(issue_text, docs, context, known_dependencies, points, **kwargs):
        p.user_prompt_kwargs = kwargs
        return "user"

    async def call_llm(system_prompt, user_prompt, *images):
        return "raw"

    def get_edges_among(names):
        p.edges_asked.append(list(names))
        return [e for e in p.edges if e["source"] in names and e["target"] in names]

    monkeypatch.setattr(query_flow, "build_context", build_context)
    monkeypatch.setattr(query_flow.ticket_points, "extract", extract)
    monkeypatch.setattr(query_flow.prompt, "build_user_prompt", build_user_prompt)
    monkeypatch.setattr(query_flow.prompt, "build_system_prompt", lambda *a, **k: "system")
    monkeypatch.setattr(query_flow.tool_loop, "enabled", lambda: False)
    monkeypatch.setattr(query_flow.planner, "_use_gateway", lambda: False)
    monkeypatch.setattr(query_flow.planner, "call_llm", call_llm)
    monkeypatch.setattr(query_flow.planner, "parse_tdd", lambda raw: p.tdd)
    monkeypatch.setattr(query_flow, "verify_tdd", lambda tdd, shown: setattr(p, "verified", shown))
    for name, fn in {
        "pipeline_versions": lambda names: {n: 5 for n in names},
        "load_candidate_docs": lambda names: [],
        "get_edges_with_evidence": lambda names: [],
        "get_edges_among": get_edges_among,
        "file_exists": lambda service, path: (service, path) in p.existing_files,
    }.items():
        monkeypatch.setattr(query_flow.graph_reader, name, fn)
    return p


async def _run_plan(**kwargs):
    return await query_flow._plan(
        "issue", "bug", [], ["orders"], [{"name": "orders", "summary": "d"}], {"orders": []},
        None, False, **kwargs,
    )


@pytest.mark.asyncio
async def test_the_prompt_gets_memory_constraints_and_contracts(planned):
    await _run_plan(memory="m", technology_constraints=["Java 17"], api_contracts="c")

    assert planned.user_prompt_kwargs["memory"] == "m"
    assert planned.user_prompt_kwargs["technology_constraints"] == ["Java 17"]
    assert planned.user_prompt_kwargs["api_contracts"] == "c"


@pytest.mark.asyncio
async def test_technology_constraints_come_from_the_build_files_never_the_model(planned):
    """2026-10-15: the versions are a fact read from the repository; whatever the
    model writes there is replaced."""
    planned.tdd = _tdd(technology_constraints=["Language: Java 22 (invented by the model)"])

    result = await _run_plan(technology_constraints=["Language: Java 17 (from pom.xml)."])

    assert result.technology_constraints == ["Language: Java 17 (from pom.xml)."]


@pytest.mark.asyncio
async def test_dependency_diagram_is_scoped_to_flagged_services_only(planned):
    """The final service_dependencies only reflect what the model actually
    flagged in `services`, not every candidate."""
    planned.tdd = _tdd(services=[ServicePlan(service="user", complexity="low", reasoning="r")])

    await _run_plan()

    assert planned.edges_asked[-1] == ["user"]


@pytest.mark.asyncio
async def test_dependency_scope_widens_to_architecture_diagram_targets(planned):
    """The model can flag only "orders" while drawing "orders -> shipping";
    the verified dependencies must include it."""
    change = FileChange(
        file_path="o.kt", change_description="c", implementation_notes="n",
        pseudocode_sketch="p", reasoning="r",
    )
    planned.edges = [{"source": "orders", "target": "shipping"}]
    planned.tdd = _tdd(
        services=[ServicePlan(service="orders", complexity="low", reasoning="r", changes=[change])],
        current_architecture_components=[
            ArchComponent(name="orders", depends_on=["shipping"]),
            ArchComponent(name="shipping"),
        ],
    )

    result = await _run_plan()

    assert planned.edges_asked[-1] == ["orders", "shipping"]
    assert result.service_dependencies == [ServiceDependency(source="orders", target="shipping")]


@pytest.mark.asyncio
async def test_a_parse_error_is_not_swallowed(planned, monkeypatch):
    def fail(raw):
        raise TddParseError("boom")

    monkeypatch.setattr(query_flow.planner, "parse_tdd", fail)
    with pytest.raises(TddParseError):
        await _run_plan()


@pytest.mark.asyncio
async def test_every_change_is_verified_against_the_files_shown(planned):
    planned.shown = {"ui": {"src/m/view.tsx"}}

    await _run_plan()

    assert planned.verified == {"ui": {"src/m/view.tsx"}}


@pytest.mark.asyncio
async def test_a_decision_keeps_only_targets_that_exist(planned):
    planned.shown = {"ui": {"shown.tsx"}}
    planned.existing_files = {("ui", "in-graph.tsx")}
    planned.tdd = _tdd(decisions=[DesignDecision(
        id="d", title="t", choice="c", reason="r",
        applies_to=[DecisionTarget(service="ui", file_path=f) for f in
                    ("shown.tsx", "in-graph.tsx", "invented.tsx")],
    )])

    result = await _run_plan()

    assert [t.file_path for t in result.decisions[0].applies_to] == ["shown.tsx", "in-graph.tsx"]


def test_the_architecture_lists_only_the_links_of_services_that_change():
    tdd = TechnicalDesignDoc.model_validate({
        "title": "t", "issue_summary": "s", "overall_reasoning": "r",
        "services": [
            {"service": "ui", "complexity": "low", "reasoning": "r", "changes": [
                {"file_path": "a.tsx", "change_description": "c", "implementation_notes": "n",
                 "pseudocode_sketch": "p", "reasoning": "r"}]},
            {"service": "reports-api", "complexity": "low", "reasoning": "no change"},
            {"service": "models", "complexity": "low", "reasoning": "no change"},
        ],
    })
    edges = [{"source": "ui", "target": "reports-api"},
             {"source": "reports-api", "target": "ui"},
             {"source": "reports-api", "target": "models"}]

    links = query_flow._links_of_changed_services(tdd, edges)

    assert [(d.source, d.target) for d in links] == [("ui", "reports-api"), ("reports-api", "ui")]


def test_new_tests_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("PLANNER_NEW_TESTS", raising=False)
    assert query_flow._new_tests() is False
    monkeypatch.setenv("PLANNER_NEW_TESTS", "on")
    assert query_flow._new_tests() is True
