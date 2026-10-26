"""Planning's memory in Memsy (2026-10-26): exact, encrypted, newest snapshot wins."""

import hashlib

import pytest

from src.memory import operations, record
from tests.memory.fake_memsy import FakeMemsy

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "2,00,000 INR a day",
    "values": ["2,00,000 INR"],
}
ON_TRANSFER = [{"service": "bank", "file_path": "TransferService.java", "symbol": "transfer"}]


@pytest.fixture
def memsy(monkeypatch):
    fake = FakeMemsy()
    monkeypatch.setattr(record, "_client", lambda: fake)
    return fake


def _record(**payload):
    return operations.remember({"issue_key": "BTS-25", **payload})


def test_the_memory_is_stored_encrypted_in_memsy(memsy):
    _record(
        tdd_version="v1.0",
        page_id="9",
        facts=[LIMIT],
        facts_hash="fh",
        decisions=[
            {"id": "paise", "title": "Paise", "choice": "long paise", "applies_to": ON_TRANSFER}
        ],
    )
    (event,) = memsy.stored("tdd:BTS-25")
    assert event.session_id == "memory:BTS-25:1:1" and event.kind == "app_event"
    assert "2,00,000" not in event.metadata and "long paise" not in event.metadata
    assert "enc:v1:" in event.metadata


def test_values_come_back_exactly_although_memsy_rewrites_its_text(memsy):
    _record(tdd_version="v1.0", page_id="9", facts=[LIMIT], facts_hash="fh")
    found = record.load_requirement("BTS-25")
    assert found["facts"] == [LIMIT] and found["facts_hash"] == "fh"
    assert record.current_version("BTS-25") == {"tdd_version": "v1.0", "page_id": "9"}


def test_the_newest_snapshot_wins_and_replaced_decisions_drop_out(memsy):
    _record(
        tdd_version="v1.0",
        page_id="9",
        facts=[LIMIT],
        facts_hash="fh",
        decisions=[{"id": "double", "choice": "double", "applies_to": ON_TRANSFER}],
    )
    _record(
        tdd_version="v1.1",
        page_id="9",
        decisions=[{"id": "paise", "choice": "long paise", "applies_to": ON_TRANSFER}],
        superseded=["double"],
    )
    assert record.current_version("BTS-25")["tdd_version"] == "v1.1"
    assert [d["id"] for d in record.decisions_in_force("BTS-25")] == ["paise"]
    assert record.load_requirement("BTS-25")["facts"] == [LIMIT]  # a page revision keeps facts


def test_a_large_memory_is_split_into_parts_and_joined_back(memsy):
    noise = [hashlib.sha512(str(i).encode()).hexdigest() for i in range(40)]
    many = [{**LIMIT, "slot": f"rule_{i}", "statement": noise[i]} for i in range(40)]
    _record(tdd_version="v1.0", page_id="9", facts=many, facts_hash="fh")
    parts = memsy.stored("tdd:BTS-25")
    assert len(parts) > 1 and all(len(p.metadata) <= 4096 for p in parts)
    assert len({p.session_id for p in parts}) == len(parts)  # Memsy merges one session
    assert len(record.load_requirement("BTS-25")["facts"]) == 40


def test_project_decisions_reach_a_later_story_on_the_same_service(memsy):
    _record(
        tdd_version="v1.0",
        page_id="9",
        decisions=[
            {"id": "paise", "title": "Paise", "choice": "long paise", "applies_to": ON_TRANSFER}
        ],
    )
    found = record.decisions_in_force("BTS-41", ["bank"])
    assert [(d["id"], d["story_key"]) for d in found] == [("paise", "BTS-25")]
    assert record.decisions_in_force("BTS-41", ["other-service"]) == []
    assert "[paise]" in record.decisions_block(found, "BTS-41")


def test_without_memsy_settings_the_memory_says_so(monkeypatch):
    monkeypatch.setattr(record, "secret", lambda name, default="": "")
    with pytest.raises(record.MemoryUnavailable):
        record.load_snapshot("BTS-25")


@pytest.mark.asyncio
async def test_an_unchanged_text_is_same_without_extracting(memsy):
    _record(
        tdd_version="v1.3", page_id="9", facts=[LIMIT], facts_hash="fh", story_text="Limit 2,00,000"
    )

    async def never(text):
        raise AssertionError("extracted")

    found = await operations.check_story("BTS-25", "limit 2,00,000", extract=never)
    assert found["verdict"] == "same" and found["current"]["tdd_version"] == "v1.3"


@pytest.mark.asyncio
async def test_a_story_with_no_memory_is_new(memsy):
    async def extract(text):
        return [LIMIT]

    found = await operations.check_story("BTS-25", "anything", extract=extract)
    assert found["verdict"] == "new" and found["facts"][0]["slot"] == "daily_limit"
