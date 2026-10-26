"""What Orchestration relies on (2026-10-26): page revisions keep the story's facts,
the stored text hash is Planning's own, memory-off answers are unchanged, and an
investigation stays inside the run's repositories."""

import pytest

from src.memory import facts, operations, record
from src.query import design_qa, investigate
from src.tools import memory_tools, query_tools

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "2,00,000 INR a day",
    "values": ["2,00,000 INR"],
}


@pytest.fixture
def store(monkeypatch):
    calls = []
    monkeypatch.setattr(record, "create_memory_indexes", lambda: None)
    monkeypatch.setattr(record, "save_requirement", lambda *a: calls.append(("facts", a)) or 1)
    monkeypatch.setattr(record, "save_version", lambda *a: calls.append(("version", a)))
    monkeypatch.setattr(
        record, "save_decisions", lambda *a, **k: calls.append(("decisions", a, k)) or len(a[1])
    )
    monkeypatch.setattr(
        record,
        "load_requirement",
        lambda k: {"facts": [LIMIT], "facts_hash": "stored-fh", "story_text_hash": "th"},
    )
    return calls


def test_a_page_revision_keeps_the_stored_facts(store):
    out = operations.remember(
        {
            "issue_key": "BTS-25",
            "tdd_version": "v1.5",
            "page_id": "9",
            "decisions": [{"id": "paise", "choice": "c"}],
            "superseded": ["double"],
        }
    )
    assert [c[0] for c in store] == ["version", "decisions"]
    assert store[0][1] == ("BTS-25", "v1.5", "9", "stored-fh")
    assert store[1][2]["superseded"] == ["double"]
    assert out == {"facts": 0, "decisions": 1}


def test_the_stored_text_hash_is_computed_from_the_story(store):
    operations.remember(
        {
            "issue_key": "BTS-25",
            "facts": [LIMIT],
            "facts_hash": "fh",
            "story_text": "*Daily* limit 2,00,000 INR",
            "story_text_hash": "theirs",
        }
    )
    assert store[0][1][3] == facts.text_hash("daily limit 2,00,000 inr")


@pytest.mark.asyncio
async def test_without_a_kind_the_answer_is_exactly_as_before(monkeypatch):
    seen = {}

    async def llm(system, user):
        seen["system"], seen["user"] = system, user
        return '{"answer": "A", "change_recommended": true, "proposed_change": "Use paise"}'

    async def no_code(question, tdd):
        return []

    monkeypatch.setattr(design_qa.planner, "call_llm", llm)
    monkeypatch.setattr(design_qa, "_related_code", no_code)
    monkeypatch.setattr(record, "load_requirement", lambda k: pytest.fail("memory read"))
    fn = getattr(
        query_tools.answer_design_question, "__wrapped__", query_tools.answer_design_question
    )
    out = await fn("why not paise?", "TDD", [], "BTS-25", None)
    assert out == {"answer": "A", "change_recommended": True, "proposed_change": "Use paise"}
    assert seen["system"] == design_qa.LEGACY_SYSTEM_PROMPT
    assert seen["user"].startswith("QUESTION:\nwhy not paise?")


def test_the_run_repositories_name_the_services():
    repos = "bank: AetherionAgentDemo/bank-transfer-service\nui = org/ui # front end\n\n"
    assert memory_tools.service_names(repos) == ["bank", "ui"]
    assert memory_tools.service_names("") == []


@pytest.mark.asyncio
async def test_an_investigation_searches_only_those_services(monkeypatch):
    asked = {}

    async def fake_embed(text):
        return [0.1]

    def scores(embedding, names=None):
        asked["names"] = names
        return {n: 0.0 for n in names or ["bank", "ui"]}

    monkeypatch.setattr(investigate, "embed", fake_embed)
    monkeypatch.setattr(investigate.graph_reader, "summary_scores", scores)
    monkeypatch.setattr(investigate.graph_reader, "search_service_chunks", lambda *a, **k: [])
    monkeypatch.setattr(investigate.graph_reader, "load_symbols", lambda pairs: [])
    _, services, _ = await investigate._code_for("Submit does nothing", ["bank"])
    assert asked["names"] == ["bank"] and services == ["bank"]
