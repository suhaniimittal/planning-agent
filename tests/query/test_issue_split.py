import json

import pytest

from src.query import code_search, issue_split

ISSUE = (
    'Deferred order "Submit" silently fails with no error when a candidate-entered '
    "international phone number has leading/trailing whitespace.\n"
    'On the client-facing "Review Deferred Report" screen, clicking Submit does nothing. '
    "Instead of surfacing the error, it is only written to console.log.\n"
    "The underlying GraphQL mutation (approveDeferredReport) is never sent to the server, "
    "so the report stays stuck in DEFERRED status indefinitely.\n"
    "The employment-verification phone field is validated against the regex "
    "internationalPhoneNumber whenever non-empty. This regex is anchored at the start of "
    'the string, so any leading whitespace (e.g. a value like " +23480 330-20339") fails '
    "validation even though the number itself is valid."
)

CLEAN_CHUNKS = [
    "The internationalPhoneNumber regex is anchored at the start of the string, so a phone "
    "value with leading whitespace fails validation.",
    "The underlying GraphQL mutation approveDeferredReport is never sent to the server when "
    "Submit fails on the deferred report review screen.",
    "Because approveDeferredReport is never sent, the report stays stuck in DEFERRED status "
    "indefinitely with no visible cause.",
]


@pytest.fixture(autouse=True)
def clear_cache():
    issue_split._cache.clear()
    yield
    issue_split._cache.clear()


def _reply(chunks):
    return json.dumps({"chunks": chunks})


# ---------- parse_chunks ----------


def test_parse_chunks_reads_a_plain_json_object():
    assert issue_split.parse_chunks(_reply(["one", "two"])) == ["one", "two"]


def test_parse_chunks_strips_markdown_fences_the_gateway_path_can_add():
    raw = "```json\n" + _reply(["one", "two"]) + "\n```"
    assert issue_split.parse_chunks(raw) == ["one", "two"]


def test_parse_chunks_digs_the_object_out_of_surrounding_prose():
    raw = "Here are the chunks:\n" + _reply(["one"]) + "\nHope that helps."
    assert issue_split.parse_chunks(raw) == ["one"]


def test_parse_chunks_accepts_a_bare_array_too():
    assert issue_split.parse_chunks('["one", "two"]') == ["one", "two"]


def test_parse_chunks_drops_blanks_and_non_strings():
    assert issue_split.parse_chunks(_reply(["ok", "", "   ", 7, None])) == ["ok"]


@pytest.mark.parametrize("raw", ["", "not json at all", "{", '{"chunks": "a string"}'])
def test_parse_chunks_returns_empty_rather_than_raising(raw):
    assert issue_split.parse_chunks(raw) == []


# ---------- invented_tokens ----------


def test_invented_tokens_passes_a_rewrite_built_from_the_issues_own_words():
    assert issue_split.invented_tokens(CLEAN_CHUNKS, ISSUE) == []


def test_invented_tokens_catches_a_symbol_the_issue_never_contained():
    chunk = "The internationalPhoneNumber regex lives in PhoneValidator.ts and rejects whitespace."
    assert issue_split.invented_tokens([chunk], ISSUE) == ["PhoneValidator.ts"]


def test_invented_tokens_catches_an_invented_service_name():
    chunk = "The phone value is first accepted by candidate-service before the report defers."
    assert issue_split.invented_tokens([chunk], ISSUE) == ["candidate-service"]


def test_invented_tokens_ignores_ordinary_prose():
    chunk = "The user sees nothing at all after pressing the button on the review screen."
    assert issue_split.invented_tokens([chunk], ISSUE) == []


# ---------- usable ----------


def test_usable_accepts_a_good_split():
    kept, reason = issue_split.usable(CLEAN_CHUNKS, ISSUE)
    assert reason is None
    assert kept == CLEAN_CHUNKS


def test_usable_drops_a_short_fragment_without_failing_the_split():
    kept, reason = issue_split.usable([*CLEAN_CHUNKS, "Submit."], ISSUE)
    assert reason is None
    assert "Submit." not in kept
    assert len(kept) == len(CLEAN_CHUNKS)


def test_usable_rejects_a_split_that_never_really_split():
    unsplit = ["x" * (issue_split.MAX_CHUNK_CHARS + 1)]
    kept, reason = issue_split.usable(unsplit, ISSUE)
    assert kept == []
    assert "not really split" in reason


def test_usable_rejects_more_chunks_than_the_query_budget():
    many = [f"chunk number {i} padded out to clear the minimum length rule" for i in range(30)]
    kept, reason = issue_split.usable(many, ISSUE)
    assert kept == []
    assert "budget" in reason


def test_usable_rejects_too_few_chunks_for_a_long_issue():
    assert len(ISSUE) > issue_split.UNSPLIT_TEXT_CHARS
    kept, reason = issue_split.usable(CLEAN_CHUNKS[:1], ISSUE)
    assert kept == []
    assert "only 1 chunks" in reason


def test_usable_rejects_the_whole_split_when_one_chunk_invents_a_name():
    bad = [*CLEAN_CHUNKS, "The value reaches candidate-service before validation runs on it."]
    kept, reason = issue_split.usable(bad, ISSUE)
    assert kept == []
    assert "invented candidate-service" in reason


def test_usable_rejects_when_nothing_clears_the_minimum_length():
    kept, reason = issue_split.usable(["a", "b"], ISSUE)
    assert kept == []
    assert "minimum length" in reason


# ---------- split_issue ----------


@pytest.mark.asyncio
async def test_split_issue_returns_the_models_chunks(monkeypatch):
    async def call_llm(system, user):
        assert "self-contained" not in user  # the user prompt is just the issue
        assert ISSUE.strip() in user
        return _reply(CLEAN_CHUNKS)

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    assert await issue_split.split_issue(ISSUE) == ("bug", CLEAN_CHUNKS)


@pytest.mark.asyncio
async def test_split_issue_calls_the_model_once_per_ticket(monkeypatch):
    calls = []

    async def call_llm(system, user):
        calls.append(user)
        return _reply(CLEAN_CHUNKS)

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    first = await issue_split.split_issue(ISSUE)
    second = await issue_split.split_issue(ISSUE)

    assert first == second  # the same ticket must not split two ways in one run
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_split_issue_gives_up_quietly_when_the_provider_fails(monkeypatch):
    async def call_llm(system, user):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    assert await issue_split.split_issue(ISSUE) == ("bug", [])


@pytest.mark.asyncio
async def test_split_issue_gives_up_on_unparseable_output(monkeypatch):
    async def call_llm(system, user):
        return "I'm afraid I can't do that."

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    assert await issue_split.split_issue(ISSUE) == ("bug", [])


@pytest.mark.asyncio
async def test_split_issue_does_not_cache_a_rejected_split(monkeypatch):
    async def call_llm(system, user):
        return _reply(["The regex lives in PhoneValidator.ts and rejects leading whitespace."])

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    assert await issue_split.split_issue(ISSUE) == ("bug", [])
    assert issue_split._cache == {}


# ---------- issue_queries ----------


@pytest.mark.asyncio
async def test_issue_queries_puts_the_whole_ticket_first(monkeypatch):
    async def call_llm(system, user):
        return _reply(CLEAN_CHUNKS)

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    mode, queries = await issue_split.issue_queries(ISSUE)

    # query_flow scores services on vectors[0] alone — it must stay the whole ticket
    assert mode in issue_split.MODES
    assert queries[0] == ISSUE.strip()
    assert queries[1:] == CLEAN_CHUNKS


@pytest.mark.asyncio
async def test_issue_queries_falls_back_to_sentence_splitting(monkeypatch):
    async def call_llm(system, user):
        raise RuntimeError("down")

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    mode, queries = await issue_split.issue_queries(ISSUE)
    assert queries == code_search.issue_queries(ISSUE)
    assert mode in issue_split.MODES


@pytest.mark.asyncio
async def test_issue_queries_never_exceeds_the_embedding_budget(monkeypatch):
    async def call_llm(system, user):
        return _reply([f"chunk {i} padded out well past the minimum length" for i in range(40)])

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    _, queries = await issue_split.issue_queries(ISSUE)
    assert len(queries) <= code_search.MAX_QUERY_PARTS


# ---------- mode detection ----------

BUG_TICKET = ISSUE  # the deferred-submit report above

FEATURE_TICKET = (
    "Enhancement request for mass report printing\n"
    "Parity with Classic: the client user needs to print reports in mass from "
    "the report dashboard.\n"
    "As a client user, I want to select multiple completed reports and generate "
    "a single merged PDF, so that I can distribute report documentation "
    "without opening each report individually.\n"
    "In Scope: multi-select report printing, generation of a single merged PDF.\n"
    "Out of Scope: individual PDF generation for each selected report.\n"
    "Acceptance criteria\n"
    "Users shall be able to select multiple reports from the report dashboard.\n"
    "Given a user has selected multiple completed reports, when the user selects "
    "Bulk Print, then a single merged PDF shall be generated."
)


def test_keyword_mode_recognises_a_bug_report():
    """Guards the bug vocabulary. Measured on the real deferred-submit ticket:
    9 bug signals, 0 feature."""
    mode, bug, feature = issue_split.keyword_mode(BUG_TICKET)
    assert mode == "bug"
    assert bug > feature


def test_keyword_mode_recognises_a_feature_request():
    """Guards the feature vocabulary. Measured on the real bulk-print ticket:
    0 bug signals, 11 feature."""
    mode, bug, feature = issue_split.keyword_mode(FEATURE_TICKET)
    assert mode == "feature"
    assert feature > bug


def test_keyword_mode_falls_back_to_bug_on_a_tie():
    """A terse ticket scores nothing either way. Bug is the safer default: the
    bug prompt merely strains on a feature, while the feature prompt run on a
    real bug drops the value trace that locates where a bad value is written."""
    mode, bug, feature = issue_split.keyword_mode("Bulk print output looks odd.")
    assert (bug, feature) == (0, 0)
    assert mode == "bug"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"mode": "feature", "chunks": []}', "feature"),
        ('{"mode": "bug", "chunks": []}', "bug"),
        ('{"chunks": []}', None),
        ('{"mode": "banana", "chunks": []}', None),
        ('{"mode": 7, "chunks": []}', None),
        ("not json", None),
    ],
)
def test_parse_mode_accepts_only_a_known_mode(raw, expected):
    assert issue_split.parse_mode(raw) == expected


@pytest.mark.asyncio
async def test_split_issue_uses_the_mode_the_model_returned(monkeypatch):
    async def call_llm(system, user):
        assert "classify the report" in system
        return json.dumps({"mode": "feature", "chunks": CLEAN_CHUNKS})

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    mode, chunks = await issue_split.split_issue(ISSUE)
    assert mode == "feature"  # the model wins even though the text reads as a bug
    assert chunks == CLEAN_CHUNKS


@pytest.mark.asyncio
async def test_split_issue_falls_back_to_keywords_when_the_mode_is_missing(monkeypatch):
    async def call_llm(system, user):
        return _reply(CLEAN_CHUNKS)  # no "mode" key

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    mode, chunks = await issue_split.split_issue(ISSUE)
    assert mode == "bug"
    assert chunks == CLEAN_CHUNKS


@pytest.mark.asyncio
async def test_split_issue_rejects_an_unrecognised_mode(monkeypatch):
    async def call_llm(system, user):
        return json.dumps({"mode": "refactor", "chunks": CLEAN_CHUNKS})

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    mode, _ = await issue_split.split_issue(ISSUE)
    assert mode == "bug"  # not "refactor" — unknown values never reach the prompt


@pytest.mark.asyncio
async def test_mode_is_cached_with_the_chunks(monkeypatch):
    calls = []

    async def call_llm(system, user):
        calls.append(user)
        return json.dumps({"mode": "feature", "chunks": CLEAN_CHUNKS})

    monkeypatch.setattr(issue_split.planner, "call_llm", call_llm)
    first = await issue_split.split_issue(ISSUE)
    second = await issue_split.split_issue(ISSUE)
    assert first == second == ("feature", CLEAN_CHUNKS)
    assert len(calls) == 1  # one ticket cannot be classified two ways in a run
