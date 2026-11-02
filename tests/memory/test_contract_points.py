"""What Orchestration relies on (2026-10-26): page revisions keep the story's facts,
the stored text hash is Planning's own, and memory-off answers keep their shape."""

import pytest

from src.memory import facts, operations, record
from src.query import design_qa
from src.tools import query_tools

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "2,00,000 INR a day",
    "values": ["2,00,000 INR"],
}


@pytest.fixture
def store(monkeypatch):
    from tests.memory.fake_memsy import FakeMemsy

    fake = FakeMemsy()
    monkeypatch.setattr(record, "_client", lambda: fake)
    operations.remember(
        {
            "issue_key": "BTS-25",
            "tdd_version": "v1.4",
            "page_id": "9",
            "facts": [LIMIT],
            "facts_hash": "stored-fh",
            "story_text": "story",
        }
    )
    return fake


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
    assert out == {"facts": 0, "decisions": 1}
    assert record.load_requirement("BTS-25")["facts_hash"] == "stored-fh"
    _, snap = record.load_snapshot("BTS-25")
    assert snap["versions"][-1] == {
        **snap["versions"][-1],
        "version": "v1.5",
        "facts_hash": "stored-fh",
    }


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
    assert record.load_requirement("BTS-25")["story_text_hash"] == facts.text_hash(
        "daily limit 2,00,000 inr"
    )


@pytest.mark.asyncio
async def test_without_a_kind_the_answer_is_exactly_as_before(monkeypatch):
    seen = {}

    async def llm(system, user):
        seen["system"], seen["user"] = system, user
        return '{"answer": "A", "change_recommended": true, "proposed_change": "Use paise"}'

    monkeypatch.setattr(design_qa.planner, "call_llm", llm)
    monkeypatch.setattr(record, "load_requirement", lambda k: pytest.fail("memory read"))
    fn = getattr(
        query_tools.answer_design_question, "__wrapped__", query_tools.answer_design_question
    )
    out = await fn("why not paise?", "TDD", [], "BTS-25", None)
    assert out == {"answer": "A", "change_recommended": True, "proposed_change": "Use paise"}
    assert seen["system"] == design_qa.LEGACY_SYSTEM_PROMPT
    assert seen["user"].startswith("QUESTION:\nwhy not paise?")
