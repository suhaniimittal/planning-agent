
from src.query import design_qa


def test_the_model_answer_is_read_from_json():
    raw = '{"answer": "A", "change_recommended": true, "proposed_change": "Use a table"}'
    # An older-style reply: a recommended change is read as verdict "apply".
    assert design_qa.parse_answer(raw) == design_qa.DesignAnswer(
        "A", True, "Use a table", verdict="apply", apply=["Use a table"]
    )


def test_a_fenced_answer_is_read_too():
    raw = '```json\n{"answer": "A", "change_recommended": false}\n```'
    assert design_qa.parse_answer(raw).answer == "A"


def test_plain_text_is_taken_as_the_answer():
    assert design_qa.parse_answer("Just text").answer == "Just text"


def test_the_prompt_carries_the_question_the_tdd_and_the_discussion():
    prompt = design_qa.build_user_prompt("Why X?", "TDD body", ["Q1"])
    assert prompt.startswith("KIND: question\nCOMMENT:\nWhy X?")
    assert "TDD body" in prompt and "Q1" in prompt
