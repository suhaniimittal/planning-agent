import pytest

from src.agent import agent as agent_module


class Recorder:
    def __init__(self, results):
        self.results = results
        self.calls = []

    async def execute(self, name, *args, **kwargs):
        self.calls.append((name, args))
        return self.results[name]


@pytest.fixture
def tools(monkeypatch):
    def install(**results):
        recorder = Recorder(results)
        monkeypatch.setattr(agent_module.toolExecutor, "execute", recorder.execute)
        return recorder

    return install


@pytest.mark.asyncio
async def test_a_requirement_syncs_the_graph_then_writes_the_tdd(tools):
    recorder = tools(
        run_ingestion={"status": "success", "services": ["orders"]},
        generate_technical_design={"s3_key": "technical_designs/x.pdf"},
    )
    out = await agent_module.planning_agent.fn(
        {"issue_text": "Add bulk print", "repos": "orders: org/orders"}
    )
    assert [name for name, _ in recorder.calls] == ["run_ingestion", "generate_technical_design"]
    assert out["status"] == "success"
    assert out["results"][0]["file_key"] == "technical_designs/x.pdf"


@pytest.mark.asyncio
async def test_repos_without_a_requirement_only_refreshes_the_graph(tools):
    recorder = tools(run_ingestion={"status": "success", "services": ["orders"]})
    out = await agent_module.planning_agent.fn({"repos": "orders: org/orders"})
    assert [name for name, _ in recorder.calls] == ["run_ingestion"]
    assert out == {
        "status": "success",
        "mode": "graph_sync",
        "services": ["orders"],
        "results": [],
    }


@pytest.mark.asyncio
async def test_the_orchestrator_can_ask_for_the_document_itself(tools):
    recorder = tools(
        run_ingestion={"status": "success", "services": ["bank"], "repos": {"bank": "o/bank"}},
        generate_technical_design={
            "s3_key": "technical_designs/x.pdf",
            "title": "Daily limit",
            "confluence_html": "<h1>Daily limit</h1>",
            "text": "1. Overview",
        },
    )
    out = await agent_module.planning_agent.fn(
        {"issue_text": "Add a daily limit", "repos": "bank: o/bank", "return_document": True}
    )
    assert recorder.calls[1] == (
        "generate_technical_design",
        # top_k/hops left as None: generate_technical_design owns their defaults;
        # the last value is the optional bug/feature mode.
        ("Add a daily limit", None, None, ["bank"], True, {"bank": "o/bank"}, None),
    )
    assert out["document"] == {
        "title": "Daily limit",
        "confluence_html": "<h1>Daily limit</h1>",
        "text": "1. Overview",
    }


@pytest.mark.asyncio
async def test_a_design_question_is_answered_without_syncing_or_writing(tools):
    recorder = tools(
        answer_design_question={
            "answer": "Because the history already holds every transfer.",
            "change_recommended": False,
            "proposed_change": "",
        }
    )
    out = await agent_module.planning_agent.fn(
        {
            "mode": "discuss",
            "question": "Why not a separate limits table?",
            "tdd_text": "1. Overview …",
            "discussion": ["Q: why a Clock?"],
        }
    )
    assert [name for name, _ in recorder.calls] == ["answer_design_question"]
    assert out["status"] == "success" and out["mode"] == "discuss"
    assert out["answer"].startswith("Because") and out["change_recommended"] is False


@pytest.mark.asyncio
async def test_a_design_question_needs_the_tdd(tools):
    tools()
    out = await agent_module.planning_agent.fn({"mode": "discuss", "question": "Why?"})
    assert out["status"] == "error" and "TDD" in out["message"]
