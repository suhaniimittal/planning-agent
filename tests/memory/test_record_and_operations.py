"""Planning's TDD memory as a snapshot (2026-10-26): Orchestration gives the
story's snapshot and keeps the one Planning hands back (2026-10-30). Exact
values, the newest version wins, replaced decisions drop out."""

import pytest

from src.memory import operations, record

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "2,00,000 INR a day",
    "values": ["2,00,000 INR"],
}
ON_TRANSFER = [{"service": "bank", "file_path": "TransferService.java", "symbol": "transfer"}]


def _record(snapshot=None, **payload):
    out = operations.remember(
        {"issue_key": "BTS-25", "memory": {"snapshot": snapshot or {}}, **payload}
    )
    return out["snapshot"]


def test_a_published_version_is_handed_back_as_the_new_snapshot():
    given = {"facts": [], "versions": []}
    out = operations.remember(
        {
            "issue_key": "BTS-25",
            "memory": {"snapshot": given},
            "tdd_version": "v1.0",
            "page_id": "9",
            "facts": [LIMIT],
            "facts_hash": "fh",
            "decisions": [{"id": "paise", "title": "Paise", "choice": "long paise"}],
        }
    )
    assert out["facts"] == 1 and out["decisions"] == 1
    assert out["snapshot"]["facts"] == [LIMIT] and out["snapshot"]["facts_hash"] == "fh"
    assert record.current_version(out["snapshot"]) == {"tdd_version": "v1.0", "page_id": "9"}
    assert given == {"facts": [], "versions": []}  # the given snapshot is not changed


def test_the_newest_version_wins_and_replaced_decisions_drop_out():
    first = _record(
        tdd_version="v1.0",
        page_id="9",
        facts=[LIMIT],
        facts_hash="fh",
        decisions=[{"id": "double", "choice": "double", "applies_to": ON_TRANSFER}],
    )
    second = _record(
        first,
        tdd_version="v1.1",
        page_id="9",
        decisions=[{"id": "paise", "choice": "long paise", "applies_to": ON_TRANSFER}],
        superseded=["double"],
    )
    assert record.current_version(second)["tdd_version"] == "v1.1"
    assert [d["id"] for d in record.agreed(second)] == ["paise"]
    assert second["facts"] == [LIMIT]  # a page revision keeps the facts


def test_decisions_in_force_are_written_for_the_planner():
    snapshot = _record(
        tdd_version="v1.0",
        decisions=[
            {"id": "paise", "title": "Paise", "choice": "long paise", "applies_to": ON_TRANSFER}
        ],
    )
    block = record.decisions_block(record.agreed(snapshot), "BTS-41")
    assert "[paise] Paise: long paise" in block and "project, BTS-25" in block


@pytest.mark.asyncio
async def test_an_unchanged_text_is_same_without_extracting():
    snapshot = _record(
        tdd_version="v1.3", page_id="9", facts=[LIMIT], facts_hash="fh", story_text="Limit 2,00,000"
    )

    async def never(text):
        raise AssertionError("extracted")

    found = await operations.check_story("limit 2,00,000", snapshot, extract=never)
    assert found["verdict"] == "same" and found["current"]["tdd_version"] == "v1.3"


@pytest.mark.asyncio
async def test_a_story_with_no_memory_is_new():
    async def extract(text):
        return [LIMIT]

    found = await operations.check_story("anything", {}, extract=extract)
    assert found["verdict"] == "new" and found["facts"][0]["slot"] == "daily_limit"


@pytest.mark.asyncio
async def test_the_story_text_is_remembered_and_compared_next_time():
    text = "BTS-25: transfers are limited to 2,00,000 INR a day."
    snapshot = _record(
        tdd_version="v1.0", page_id="9", facts=[LIMIT], facts_hash="fh", story_text=text
    )
    assert snapshot["story_text"] == text
    prompts = []

    async def judge(system, prompt):
        prompts.append(prompt)
        return '{"verdict": "same", "confidence": 0.9, "changes": [], "reason": "r"}'

    async def extract(_):
        return [{**LIMIT, "slot": "per_day_cap", "statement": "a daily cap of 2,00,000 INR"}]

    found = await operations.check_story(
        "BTS-25: a daily cap of 2,00,000 INR.", snapshot, judge=judge, extract=extract
    )
    assert found["verdict"] == "reworded" and text in prompts[0]
