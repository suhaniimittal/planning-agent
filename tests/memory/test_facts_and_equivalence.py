"""Same meaning gives the same requirement; a changed number never does (2026-10-26)."""

import pytest

from src.memory import equivalence, facts

LIMIT = {
    "slot": "daily_limit",
    "kind": "limit",
    "statement": "Outgoing transfers are limited to 2,00,000 INR per IST day",
    "values": ["2,00,000 INR", "IST"],
}
FEELING = {
    "slot": "reason",
    "kind": "rule",
    "statement": "The user says: I am not feeling well",
    "values": [],
}
AC = {
    "slot": "ac_reject",
    "kind": "acceptance",
    "statement": "Given 1,50,000 sent, when 60,000 is sent, then it is rejected",
    "values": ["1,50,000", "60,000"],
}


def judge_says(verdict, confidence=0.95, calls=None):
    async def judge(system, prompt):
        if calls is not None:
            calls.append(prompt)
        changes = '[{"slot": "reason", "why": "x"}]'
        return (
            f'{{"verdict": "{verdict}", "confidence": {confidence}, '
            f'"changes": {changes}, "reason": "r"}}'
        )

    return judge


def test_formatting_and_spacing_do_not_change_the_text_hash():
    assert facts.text_hash("*Daily*  limit 2,00,000 INR.") == facts.text_hash(
        "daily limit 2,00,000 inr"
    )


def test_amounts_written_differently_are_the_same_value():
    assert facts.normalise_value("₹2,00,000") == facts.normalise_value("2,00,000 INR")
    assert facts.normalise_value("200000 INR") == facts.normalise_value("2,00,000 INR")


def test_the_facts_hash_ignores_order_and_formatting():
    a = [LIMIT, AC]
    b = [
        dict(AC),
        {**LIMIT, "statement": "Outgoing transfers are limited to 2,00,000 INR per IST day."},
    ]
    assert facts.facts_hash(a) == facts.facts_hash(b)


@pytest.mark.asyncio
async def test_identical_facts_are_reworded_without_asking_a_model():
    calls = []
    found = await equivalence.compare(
        [LIMIT, AC], [AC, LIMIT], judge=judge_says("different", calls=calls)
    )
    assert found["verdict"] == "reworded" and calls == []


@pytest.mark.asyncio
async def test_a_paraphrase_is_judged_the_same():
    sick = {**FEELING, "statement": "The user says: I am feeling sick"}
    calls = []
    found = await equivalence.compare(
        [LIMIT, FEELING], [LIMIT, sick], judge=judge_says("same", calls=calls)
    )
    assert found["verdict"] == "reworded"
    assert len(calls) == 1 and "not feeling well" in calls[0] and "feeling sick" in calls[0]


@pytest.mark.asyncio
async def test_a_changed_number_is_a_change_without_a_model():
    calls = []
    raised = {
        **LIMIT,
        "statement": "Outgoing transfers are limited to 2,50,000 INR per IST day",
        "values": ["2,50,000 INR", "IST"],
    }
    found = await equivalence.compare([LIMIT], [raised], judge=judge_says("same", calls=calls))
    assert found["verdict"] == "changed" and calls == []
    assert found["changes"][0]["slot"] == "daily_limit"
    assert "2,50,000" in found["changes"][0]["new"]


@pytest.mark.asyncio
async def test_a_new_acceptance_criterion_is_a_change():
    extra = {
        **AC,
        "slot": "ac_allow",
        "statement": "Given 1,50,000 sent, when 50,000 is sent, then it succeeds",
        "values": ["1,50,000", "50,000"],
    }
    found = await equivalence.compare([LIMIT, AC], [LIMIT, AC, extra], judge=judge_says("same"))
    assert found["verdict"] == "changed"


@pytest.mark.asyncio
async def test_a_rule_changed_in_words_is_a_change():
    rule = {
        "slot": "counted",
        "kind": "rule",
        "statement": "Only successful transfers count",
        "values": [],
    }
    found = await equivalence.compare(
        [rule], [{**rule, "statement": "All transfers count"}], judge=judge_says("different")
    )
    assert found["verdict"] == "changed"


@pytest.mark.asyncio
async def test_a_judge_that_is_not_confident_is_unsure():
    sick = {**FEELING, "statement": "The user says: I am feeling sick"}
    found = await equivalence.compare([FEELING], [sick], judge=judge_says("same", confidence=0.5))
    assert found["verdict"] == "unsure"


@pytest.mark.asyncio
async def test_an_unreadable_judgement_is_unsure():
    async def broken(system, prompt):
        return "no idea"

    sick = {**FEELING, "statement": "The user says: I am feeling sick"}
    found = await equivalence.compare([FEELING], [sick], judge=broken)
    assert found["verdict"] == "unsure"


@pytest.mark.parametrize("raw", ['```json\n{"facts": []}\n```', 'Here: {"facts": []} done'])
def test_the_model_json_is_read_around_fences_and_text(raw):
    assert facts.parse_json(raw) == {"facts": []}


@pytest.mark.asyncio
async def test_the_judge_reads_the_new_story_text():
    calls = []
    sick = {**FEELING, "statement": "The user says: I am feeling sick"}
    await equivalence.compare(
        [FEELING],
        [sick],
        judge=judge_says("same", calls=calls),
        new_text="Story: I am feeling sick",
    )
    assert "NEW STORY TEXT:\nStory: I am feeling sick" in calls[0]


@pytest.mark.asyncio
async def test_a_changed_limit_shows_its_old_value_even_when_it_appears_elsewhere():
    ac = {
        "slot": "ac",
        "kind": "acceptance",
        "statement": "send 2,00,000",
        "values": ["2,00,000 INR"],
    }
    raised = {**LIMIT, "values": ["2,50,000 INR", "IST"]}
    found = await equivalence.compare([LIMIT, ac], [raised, ac], judge=judge_says("same"))
    change = next(c for c in found["changes"] if c["slot"] == "daily_limit")
    assert "2,00,000" in change["old"] and change["new"] == "2,50,000 INR"


def test_quotes_around_a_message_are_not_part_of_it():
    message = "Daily transfer limit of ₹2,00,000 reached. Try again tomorrow."
    assert facts.normalise_value(f'"{message}"') == facts.normalise_value(message)
