"""A TDD changes only for a valid modification; a bug is investigated in the graph (2026-10-26)."""

import pytest
from src.query import design_qa, investigate


def test_a_change_that_breaks_an_agreed_decision_is_refused():
    raw = (
        '{"answer": "Paise was agreed.", "verdict": "refuse", '
        '"refused": [{"change": "use double", "reason": "breaks money-integer-paise"}], '
        '"apply": ["ignored"], "new_decisions": [{"id": "x"}]}'
    )
    a = design_qa.parse_answer(raw)
    assert a.verdict == "refuse" and not a.change_recommended
    assert a.apply == [] and a.new_decisions == []
    assert a.refused[0]["reason"] == "breaks money-integer-paise"


def test_a_change_already_in_the_tdd_makes_no_version():
    a = design_qa.parse_answer('{"answer": "Section 6 has it.", "verdict": "already_present"}')
    assert a.verdict == "already_present" and not a.change_recommended


def test_a_partial_change_applies_only_the_valid_part():
    raw = (
        '{"answer": "a", "verdict": "partial", "apply": ["log rejections"], '
        '"refused": [{"change": "use double", "reason": "r"}], "supersedes": ["d1"], '
        '"new_decisions": [{"id": "log-rejections", "title": "t", "choice": "c", "reason": "r"}]}'
    )
    a = design_qa.parse_answer(raw)
    assert a.change_recommended and a.proposed_change == "log rejections"
    assert a.supersedes == ["d1"] and a.new_decisions[0]["id"] == "log-rejections"


@pytest.mark.asyncio
async def test_an_override_is_applied_and_marked_human():
    a = await design_qa.answer_question("Use double for money", "TDD", kind="update_override")
    assert a.verdict == "apply" and a.apply == ["Use double for money"]
    assert a.new_decisions[0]["override_by"] == "human"


def test_the_prompt_carries_the_facts_and_the_decisions():
    prompt = design_qa.build_user_prompt(
        "use double",
        "TDD",
        [],
        [],
        "update",
        [{"slot": "daily_limit", "statement": "2,00,000 INR a day"}],
        [{"id": "paise", "title": "Paise", "choice": "long paise", "reason": "rounding"}],
    )
    assert "KIND: update" in prompt and "[daily_limit]" in prompt and "[paise] Paise" in prompt


SYMBOLS = [
    {
        "service": "bank",
        "file_path": "src/T.java",
        "qualified_name": "T.java::T.transfer",
        "source_snippet": "void transfer() {}",
    }
]


def test_only_files_that_were_shown_are_kept():
    raw = {
        "root_cause": "c",
        "root_cause_confidence": "high",
        "code_relevant": True,
        "affected_files": [
            {"path": "bank/src/T.java", "why": "w"},
            {"path": "bank/src/Made.java"},
            {"path": "src/T.java", "why": "again"},
        ],
    }
    found = investigate.ground(raw, SYMBOLS, ["bank"], ["transfer"])
    assert found["affected_files"] == [{"path": "bank/src/T.java", "why": "w"}]
    assert found["root_cause_confidence"] == "high"
    assert found["code_note"].startswith("Code read from the code graph (bank)")


def test_no_relevant_code_is_said_plainly_and_caps_confidence():
    raw = {
        "root_cause": "untrimmed input",
        "root_cause_confidence": "high",
        "code_relevant": False,
        "affected_files": [{"path": "bank/src/T.java"}],
    }
    found = investigate.ground(raw, SYMBOLS, ["bank"], ["approveDeferredReport"])
    assert found["affected_files"] == [] and found["root_cause_confidence"] == "medium"
    assert "not found in the repositories Planning has (bank)" in found["code_note"]
    assert "approveDeferredReport" in found["code_note"]


def test_the_prompt_labels_each_function_with_its_file():
    prompt = investigate.build_user_prompt("Submit does nothing", SYMBOLS)
    assert "--- Code bank/src/T.java :: T.java::T.transfer" in prompt
    assert "none of the code" in investigate.build_user_prompt("x", [])
