"""A generated TDD saved to its ticket's memory (2026-10-27): each run is the
ticket's next version, its decisions proposed — built on by the ticket's
next version, seen by no other ticket until the TDD is published."""

import pytest

from src.memory import hooks, operations, record
from src.query.models import DesignDecision
from src.tdd.models import TddDocument
from tests.memory.fake_memsy import FakeMemsy

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "2,00,000 INR a day",
    "values": ["2,00,000 INR"],
}
ON_GRID = [{"service": "ui", "file_path": "grid/view.tsx", "symbol": "InfiniteScrollGrid"}]
CHECKED = {"facts": [LIMIT], "facts_hash": "fh", "story_text_hash": "th"}


@pytest.fixture
def memsy(monkeypatch):
    fake = FakeMemsy()
    monkeypatch.setattr(record, "_client", lambda: fake)
    return fake


def _decision(id_, choice="title sized by content"):
    return {"id": id_, "title": id_, "choice": choice, "reason": "r", "applies_to": ON_GRID}


def _generate(*decisions, story="Controls shift when a report is selected."):
    return operations.remember_generated("KS-1", story, CHECKED, list(decisions))


def test_each_generated_tdd_is_the_tickets_next_version(memsy):
    assert _generate(_decision("title-width")) == {"version": "1", "decisions": 1}
    assert _generate(_decision("title-width")) == {"version": "2", "decisions": 1}

    _, snapshot = record.load_snapshot("KS-1")
    assert [v["version"] for v in snapshot["versions"]] == ["1", "2"]
    assert snapshot["facts"] == [LIMIT]
    assert snapshot["story_text"] == "Controls shift when a report is selected."


def test_generated_decisions_are_proposed_and_replace_the_last_versions(memsy):
    _generate(_decision("title-width"), _decision("nowrap"))
    _generate(_decision("title-width"), _decision("reserved-slot"))

    proposed = record.proposed_decisions("KS-1")
    assert sorted(d["id"] for d in proposed) == ["reserved-slot", "title-width"]
    assert record.decisions_in_force("KS-1") == []


def test_a_proposed_decision_never_reaches_another_ticket(memsy):
    _generate(_decision("title-width"))

    assert memsy.stored("project:KS") == []
    assert record.decisions_in_force("KS-2", ["ui"]) == []


def test_publishing_agrees_the_decisions_and_shares_them_with_the_project(memsy):
    _generate(_decision("title-width"))
    operations.remember(
        {"issue_key": "KS-1", "tdd_version": "2", "decisions": [_decision("title-width")]}
    )

    assert [d["id"] for d in record.decisions_in_force("KS-1")] == ["title-width"]
    assert record.proposed_decisions("KS-1") == []
    assert [d["id"] for d in record.decisions_in_force("KS-2", ["ui"])] == ["title-width"]


def test_a_generated_tdd_never_turns_an_agreed_decision_back_into_a_proposal(memsy):
    operations.remember({"issue_key": "KS-1", "tdd_version": "1", "decisions": [_decision("slot")]})
    _generate(_decision("slot"))

    assert [d["id"] for d in record.decisions_in_force("KS-1")] == ["slot"]
    assert record.proposed_decisions("KS-1") == []


def test_the_planner_reads_agreed_and_proposed_decisions_apart():
    agreed = [{**_decision("slot"), "story_key": "KS-1", "applies_to": ON_GRID}]
    earlier = [{**_decision("title-width"), "story_key": "KS-1", "applies_to": ON_GRID}]

    block = record.decisions_block(agreed, "KS-1", earlier)

    first, second = block.split("\n\n")
    assert first.startswith("DECISIONS IN FORCE (agreed)") and "[slot]" in first
    assert second.startswith("DECISIONS FROM THIS TICKET'S EARLIER VERSIONS (not yet agreed)")
    assert "[title-width]" in second
    assert record.decisions_block([], "KS-1", []) == ""


# --- the TDD tool around the planner ------------------------------------------------------


@pytest.mark.asyncio
async def test_before_planning_the_memory_says_what_changed_and_what_was_decided(monkeypatch):
    async def check_story(key, text):
        return {
            "verdict": "changed",
            "current": {"tdd_version": "2", "page_id": ""},
            "changes": [{"slot": "daily_limit", "old": "2,00,000", "new": "5,00,000"}],
        }

    monkeypatch.setattr(operations, "check_story", check_story)
    monkeypatch.setattr(record, "decisions_in_force", lambda key, services: [])
    monkeypatch.setattr(
        record, "proposed_decisions",
        lambda key: [{**_decision("title-width"), "story_key": "KS-1", "applies_to": ON_GRID}],
    )

    memory = await hooks.memory_before("ks-1", "story", ["ui"])

    assert "last written as version 2" in memory["text"]
    assert "daily_limit: 2,00,000 -> 5,00,000" in memory["text"]
    assert "[title-width]" in memory["text"]


@pytest.mark.asyncio
async def test_memory_that_cannot_be_read_never_stops_a_tdd(monkeypatch, capsys):
    async def down(key, text):
        raise record.MemoryUnavailable("MEMSY_API_KEY and MEMSY_BASE_URL are not set")

    monkeypatch.setattr(operations, "check_story", down)

    assert await hooks.memory_before("KS-1", "story", ["ui"]) is None
    assert "planning without it" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_after_planning_the_tdd_is_saved_as_the_next_version(memsy, monkeypatch, capsys):
    notes = []

    async def remember_version(*args, **kwargs):
        notes.append((args, kwargs))
        return True

    monkeypatch.setattr(hooks.conversation, "remember_version", remember_version)
    tdd = TddDocument(
        title="Keep the header on one line",
        decisions=[DesignDecision(id="title-width", title="t", choice="c", reason="r")],
    )

    await hooks.memory_after("ks-1", "story", CHECKED, tdd)

    assert [d["id"] for d in record.proposed_decisions("KS-1")] == ["title-width"]
    assert notes[0][0][:2] == ("KS-1", "1")
    assert notes[0][1] == {"generated": "Keep the header on one line"}
    assert "saved as version 1 (1 decisions, proposed)" in capsys.readouterr().out
