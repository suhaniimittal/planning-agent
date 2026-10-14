import pytest

from src.query import design_qa


def test_the_model_answer_is_read_from_json():
    raw = '{"answer": "A", "change_recommended": true, "proposed_change": "Use a table"}'
    assert design_qa.parse_answer(raw) == design_qa.DesignAnswer("A", True, "Use a table")


def test_a_fenced_answer_is_read_too():
    raw = '```json\n{"answer": "A", "change_recommended": false}\n```'
    assert design_qa.parse_answer(raw).answer == "A"


def test_plain_text_is_taken_as_the_answer():
    assert design_qa.parse_answer("Just text").answer == "Just text"


def test_the_prompt_carries_the_question_the_tdd_the_discussion_and_the_code():
    prompt = design_qa.build_user_prompt("Why X?", "TDD body", ["Q1"], ["[bank] f\ncode"])
    assert prompt.startswith("QUESTION:\nWhy X?")
    assert "TDD body" in prompt and "Q1" in prompt and "[bank] f\ncode" in prompt


@pytest.mark.asyncio
async def test_no_code_graph_still_answers_from_the_tdd(monkeypatch):
    async def no_graph(text):
        raise ConnectionError("neo4j down")

    seen = {}

    async def call(system, user):
        seen["user"] = user
        return '{"answer": "From the TDD", "change_recommended": false}'

    monkeypatch.setattr(design_qa, "embed", no_graph)
    monkeypatch.setattr(design_qa.planner, "call_llm", call)
    answer = await design_qa.answer_question("Why X?", "TDD body")
    assert answer.answer == "From the TDD" and "RELATED CODE" not in seen["user"]
