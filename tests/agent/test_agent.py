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
        # then the optional bug/feature mode, then the story whose decisions are kept.
        ("Add a daily limit", None, None, ["bank"], True, {"bank": "o/bank"}, None, None),
    )
    assert out["document"] == {
        "title": "Daily limit",
        "confluence_html": "<h1>Daily limit</h1>",
        "text": "1. Overview",
        "decisions": [],
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


@pytest.mark.asyncio
async def test_a_facts_check_runs_no_ingestion_and_no_design(tools):
    recorder = tools(story_facts={"verdict": "reworded", "facts": [], "changes": []})
    out = await agent_module.planning_agent.fn(
        {"mode": "facts", "issue_key": "BTS-25", "story_text": "Daily cap of 2,00,000 INR"}
    )
    assert recorder.calls == [("story_facts", ("BTS-25", "Daily cap of 2,00,000 INR"))]
    assert out == {
        "status": "success",
        "mode": "facts",
        "verdict": "reworded",
        "facts": [],
        "changes": [],
    }


@pytest.mark.asyncio
async def test_a_record_passes_the_whole_payload(tools):
    recorder = tools(record_tdd={"facts": 3, "decisions": 1})
    payload = {"mode": "record", "issue_key": "BTS-25", "tdd_version": "v1.4"}
    out = await agent_module.planning_agent.fn(payload)
    assert recorder.calls == [("record_tdd", (payload,))]
    assert out["stored"] == {"facts": 3, "decisions": 1}


@pytest.mark.asyncio
async def test_an_investigation_reads_the_graph_only(tools):
    recorder = tools(investigate_bug={"root_cause": "x", "affected_files": []})
    out = await agent_module.planning_agent.fn(
        {"mode": "investigate", "issue_key": "BTS-31", "report_text": "Submit does nothing"}
    )
    assert [n for n, _ in recorder.calls] == ["investigate_bug"]
    assert out["investigation"]["root_cause"] == "x"


@pytest.mark.asyncio
async def test_a_discussion_carries_the_story_and_the_kind(tools):
    recorder = tools(answer_design_question={"answer": "No", "verdict": "refuse"})
    await agent_module.planning_agent.fn(
        {
            "mode": "discuss",
            "question": "use double",
            "tdd_text": "TDD",
            "issue_key": "BTS-25",
            "kind": "update",
            "page_id": "28180482",
        }
    )
    # the page is the conversation's Memsy session (2026-10-26)
    assert recorder.calls[0][1] == ("use double", "TDD", [], "BTS-25", "update", "28180482")


@pytest.mark.asyncio
async def test_missing_input_is_an_error_not_a_crash(tools):
    tools()
    out = await agent_module.planning_agent.fn({"mode": "facts", "issue_key": "BTS-25"})
    assert out["status"] == "error" and out["error"] == "MissingInput"


# --- the agreed TDD input (2026-10-29) -----------------------------------------------


@pytest.mark.asyncio
async def test_the_agreed_input_becomes_the_text_and_repositories(tools):
    recorder = tools(
        handoff_text={"issue_text": "BTS-60 (Story): Add a remark", "repos": "bank: o/bank@main"},
        run_ingestion={"services": ["bank"], "repos": {"bank": "o/bank@main"}},
        generate_technical_design={"s3_key": "technical_designs/x.pdf"},
    )
    payload = {
        "card": {"key": "BTS-60", "title": "Add a remark"},
        "services": [{"service": "bank", "repo": "o/bank", "branch": "main", "rank": 9}],
        "questions": [],
    }
    out = await agent_module.planning_agent.fn(payload)
    assert [name for name, _ in recorder.calls] == [
        "handoff_text",
        "run_ingestion",
        "generate_technical_design",
    ]
    assert recorder.calls[1] == ("run_ingestion", ("bank: o/bank@main",))
    design = recorder.calls[2][1]
    assert design[0] == "BTS-60 (Story): Add a remark" and design[-1] == "BTS-60"
    assert out["status"] == "success"
