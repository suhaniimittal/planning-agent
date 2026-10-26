"""The memory is encrypted, and the facts check follows its rules (2026-10-26)."""

import pytest
import json

from src.ingestion import code_crypto
from src.memory import operations, record

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "Outgoing transfers are limited to 2,00,000 INR per IST day",
    "values": ["2,00,000 INR"],
}


class FakeResult(list):
    def single(self):
        return self[0] if self else None


class FakeSession:
    def __init__(self, answers=None):
        self.queries = []
        self.answers = list(answers or [])

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeResult(self.answers.pop(0) if self.answers else [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _driver(monkeypatch, answers=None):
    session = FakeSession(answers)
    monkeypatch.setattr(
        record, "get_driver", lambda: type("D", (), {"session": lambda s: session})()
    )
    return session


def test_facts_are_stored_encrypted(monkeypatch):
    session = _driver(monkeypatch)
    record.save_requirement("bts-25", [LIMIT], "fh", "th")
    rows = session.queries[1][1]["rows"]
    assert code_crypto.is_encrypted(rows[0]["statement"]) and code_crypto.is_encrypted(
        rows[0]["values"]
    )
    assert "2,00,000" not in json.dumps(rows)
    assert rows[0]["slot"] == "daily_limit"


def test_stored_facts_are_read_back_in_plain(monkeypatch):
    sealed = record._sealed_fact("BTS-25", LIMIT)
    _driver(monkeypatch, [[{"facts_hash": "fh", "story_text_hash": "th"}], [sealed]])
    found = record.load_requirement("BTS-25")
    assert found["facts"] == [{**LIMIT, "values": ["2,00,000 INR"]}]
    assert found["facts_hash"] == "fh"


def test_decisions_are_stored_encrypted_with_their_targets(monkeypatch):
    session = _driver(monkeypatch)
    decision = {
        "id": "money-integer-paise",
        "title": "Money as paise",
        "choice": "Sum in long paise",
        "reason": "No rounding",
        "applies_to": [{"service": "bank", "file_path": "T.java", "symbol": "transfer"}],
    }
    record.save_decisions("BTS-25", [decision], superseded=["money-double"])
    supersede, write = session.queries
    assert supersede[1] == {"uid": "BTS-25:money-double", "status": "superseded"}
    row = write[1]["rows"][0]
    assert row["uid"] == "BTS-25:money-integer-paise" and row["status"] == "agreed"
    assert code_crypto.is_encrypted(row["choice"]) and "paise" not in row["choice"]
    assert row["target_rows"] == [{"service": "bank", "file_path": "T.java", "symbol": "transfer"}]


def test_project_decisions_reach_a_later_story_on_the_same_service(monkeypatch):
    def stored(story, did, service):
        uid = f"{story}:{did}"
        sealed = code_crypto.encrypt_row(
            {"title": did, "choice": f"{did} choice", "reason": "r"},
            label="Decision",
            service="BTS",
            qualified_name=uid,
        )
        return {
            **sealed,
            "uid": uid,
            "id": did,
            "story_key": story,
            "project_key": "BTS",
            "status": "agreed",
            "override_by": "",
            "targets": json.dumps([{"service": service, "file_path": "T.java", "symbol": ""}]),
        }

    rows = [
        stored("BTS-25", "paise", "bank"),
        stored("BTS-40", "other", "ui"),
        stored("BTS-41", "mine", "bank"),
    ]
    _driver(monkeypatch, [rows])
    found = record.decisions_in_force("BTS-41", ["bank"])
    assert [d["id"] for d in found] == ["mine", "paise"]
    assert found[1]["choice"] == "paise choice"
    assert "[paise]" in record.decisions_block(found, "BTS-41")


@pytest.mark.asyncio
async def test_an_unchanged_text_is_same_without_extracting(monkeypatch):
    monkeypatch.setattr(
        record,
        "load_requirement",
        lambda k: {
            "facts": [LIMIT],
            "facts_hash": "fh",
            "story_text_hash": operations.facts.text_hash("Limit 2,00,000"),
        },
    )
    monkeypatch.setattr(
        record, "current_version", lambda k: {"tdd_version": "v1.3", "page_id": "9"}
    )

    async def never(text):
        raise AssertionError("extracted")

    found = await operations.check_story("BTS-25", "limit 2,00,000", extract=never)
    assert found["verdict"] == "same" and found["current"]["tdd_version"] == "v1.3"


@pytest.mark.asyncio
async def test_a_story_with_no_memory_is_new(monkeypatch):
    monkeypatch.setattr(record, "load_requirement", lambda k: None)
    monkeypatch.setattr(record, "current_version", lambda k: None)

    async def extract(text):
        return [LIMIT]

    found = await operations.check_story("BTS-25", "anything", extract=extract)
    assert found["verdict"] == "new" and found["facts"][0]["slot"] == "daily_limit"


def test_record_stores_facts_version_and_decisions(monkeypatch):
    calls = []
    monkeypatch.setattr(record, "create_memory_indexes", lambda: None)
    monkeypatch.setattr(record, "save_requirement", lambda *a: calls.append(("facts", a)) or 1)
    monkeypatch.setattr(record, "save_version", lambda *a: calls.append(("version", a)))
    monkeypatch.setattr(
        record, "save_decisions", lambda *a, **k: calls.append(("decisions", a, k)) or 2
    )
    out = operations.remember(
        {
            "issue_key": "bts-25",
            "tdd_version": "v1.4",
            "page_id": "9",
            "facts": [LIMIT],
            "facts_hash": "fh",
            "decisions": [{"id": "x", "choice": "c"}],
            "superseded": ["y"],
        }
    )
    assert out == {"facts": 1, "decisions": 2}
    assert [c[0] for c in calls] == ["facts", "version", "decisions"]
    assert calls[2][2]["superseded"] == ["y"]
