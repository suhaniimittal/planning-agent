import asyncio
import json
from types import SimpleNamespace

from src.query import context as ctx
from src.query import tool_loop
from src.query import prompt as prompts

SCREEN = "src/apps/client/modules/Reports/pages/ReportsListScreen/view.tsx"
GRID = "src/apps/client/components/InfiniteScrollGrid/view.tsx"


class FakeReader:
    def __init__(self, sources):
        self.sources = sources

    def load_file_sources(self, pairs):
        return {p: {"source": self.sources[p], "line_count": self.sources[p].count("\n") + 1}
                for p in pairs if p in self.sources}

    def load_file_symbols(self, pairs):
        return {p: [] for p in pairs}

    def find_files(self, service, fragment):
        return sorted(p for s, p in self.sources if s == service and fragment in p)

    def importers_of(self, pairs, limit):
        return {pairs[0]: {"count": 3, "files": ["a.tsx", "b.tsx", "c.tsx"]}}


def _call(name, call_id="c1", **args):
    return SimpleNamespace(type="function_call", name=name, call_id=call_id,
                           arguments=json.dumps(args))


def _answer(text='{"title": "t"}'):
    return SimpleNamespace(id="final", output=[SimpleNamespace(type="message")], output_text=text)


class FakeClient:
    """Replies with the scripted outputs in order, and records each request."""

    def __init__(self, *outputs):
        self.outputs = list(outputs)
        self.requests = []
        self.responses = SimpleNamespace(create=self._create)

    async def _create(self, **request):
        self.requests.append(request)
        reply = self.outputs.pop(0)
        if isinstance(reply, list):
            return SimpleNamespace(id=f"r{len(self.requests)}", output=reply, output_text="")
        return reply


def _lookups(sources=None, files=()):
    sources = sources or {("ui", GRID): "<Grid item md={4} lg={4}>\n</Grid>"}
    return tool_loop.Lookups(services=["ui", "reports-api"],
                             context=ctx.Context(files=list(files)),
                             reader=FakeReader(sources))


def _run(client, lookups):
    return asyncio.run(tool_loop.run("system", "user", lookups, client=client))


def test_enough_in_the_starting_code_is_one_call():
    client = FakeClient(_answer())

    result = _run(client, _lookups())

    assert result.raw == '{"title": "t"}' and result.rounds == 1
    assert len(client.requests) == 1
    assert client.requests[0]["input"] == "user" and client.requests[0]["instructions"] == "system"


def test_a_file_asked_for_is_read_and_the_conversation_continues():
    client = FakeClient([_call("read_file", service="ui", path=GRID)], _answer())
    lookups = _lookups()

    result = _run(client, lookups)

    assert result.rounds == 2
    second = client.requests[1]
    # The second call continues the first: only the new result is sent.
    assert second["previous_response_id"] == "r1"
    [output] = second["input"]
    assert output["call_id"] == "c1"
    assert f"### ui · {GRID} · WHOLE · 2 lines" in output["output"]
    assert "md={4}" in output["output"]
    # What it read counts as shown, for verification.
    assert [f.path for f in lookups.files] == [GRID]


def test_the_last_round_must_answer():
    rounds = [[_call("find_importers", call_id=f"c{i}", service="ui", path=f"f{i}.tsx")]
              for i in range(tool_loop.MAX_ROUNDS - 1)]
    client = FakeClient(*rounds, _answer())

    result = _run(client, _lookups())

    assert result.rounds == tool_loop.MAX_ROUNDS
    last = client.requests[-1]
    assert last["tool_choice"] == "none"
    assert last["input"][-1] == {"role": "user", "content": tool_loop._FINAL_NOTE}
    assert all(r.get("tool_choice") != "none" for r in client.requests[:-1])


def test_lookups_past_the_limit_and_repeats_are_not_run():
    lookups = _lookups()
    lookups.count = tool_loop.MAX_LOOKUPS - 1

    first = asyncio.run(lookups.run("find_importers", {"service": "ui", "path": "a.tsx"}))
    again = asyncio.run(lookups.run("find_importers", {"service": "ui", "path": "a.tsx"}))
    over = asyncio.run(lookups.run("find_importers", {"service": "ui", "path": "b.tsx"}))

    assert first.startswith("3 files import a.tsx")
    assert again == "Same lookup as before; its result is above."
    assert over.startswith("The lookup limit is reached")
    assert lookups.count == tool_loop.MAX_LOOKUPS


def test_an_import_alias_is_answered_with_the_files_it_may_mean():
    lookups = _lookups()

    result = asyncio.run(lookups.run(
        "read_file", {"service": "ui", "path": "@client/components/InfiniteScrollGrid"}))

    assert "Files whose path contains 'components/InfiniteScrollGrid'" in result
    assert GRID in result
    assert lookups.files == []


def test_only_the_services_of_the_run_and_the_declared_lookups():
    lookups = _lookups()

    other = asyncio.run(lookups.run("read_file", {"service": "billing", "path": "x.kt"}))
    private = asyncio.run(lookups.run("outside", {"service": "ui"}))

    assert other.startswith("billing is not a service of this run")
    assert private == "No lookup named outside."


def test_a_file_over_the_limit_comes_back_as_its_symbols_and_a_range_is_allowed():
    big = "\n".join("x" * 100 for _ in range(tool_loop.MAX_RESULT_CHARS // 100 + 10))
    lookups = _lookups({("ui", "big.tsx"): big})

    whole = asyncio.run(lookups.run("read_file", {"service": "ui", "path": "big.tsx"}))
    ranged = asyncio.run(lookups.run(
        "read_file", {"service": "ui", "path": "big.tsx", "start_line": 10, "end_line": 12}))

    assert "OUTLINE" in whole and "ask for a line range" in whole
    assert "PARTIAL (lines 10–12)" in ranged
    assert [(f.path, f.status) for f in lookups.files] == [("big.tsx", "PARTIAL")]


def test_a_file_already_shown_whole_is_not_sent_again():
    shown = ctx.ShownFile(service="ui", path=GRID, tier="import", reason="r", text="x")
    lookups = _lookups(files=[shown])

    result = asyncio.run(lookups.run("read_file", {"service": "ui", "path": GRID}))

    assert result == f"{GRID} is already shown whole above."


def test_an_answer_the_check_rejects_is_sent_back_in_the_same_conversation():
    client = FakeClient(_answer('{"draft": 1}'), _answer('{"draft": 2}'))
    seen = []

    def check(raw):
        seen.append(raw)
        return "Point 1 has no answer." if raw == '{"draft": 1}' else None

    result = _run_checked(client, _lookups(), check)

    assert result.raw == '{"draft": 2}' and result.send_backs == 1 and result.rounds == 2
    second = client.requests[1]
    assert second["previous_response_id"] == "final"
    assert second["input"] == [{"role": "user", "content": "Point 1 has no answer."}]
    assert seen == ['{"draft": 1}', '{"draft": 2}']


def test_an_answer_is_sent_back_at_most_twice():
    client = FakeClient(*[_answer(f'{{"draft": {i}}}') for i in range(3)])

    result = _run_checked(client, _lookups(), lambda raw: "Still missing.")

    assert result.raw == '{"draft": 2}' and result.send_backs == tool_loop.MAX_SEND_BACKS
    assert len(client.requests) == 3


def test_an_answer_forced_out_on_the_last_round_can_still_be_sent_back():
    rounds = [[_call("find_importers", call_id=f"c{i}", service="ui", path=f"f{i}.tsx")]
              for i in range(tool_loop.MAX_ROUNDS - 1)]
    client = FakeClient(*rounds, _answer('{"draft": 1}'), _answer('{"draft": 2}'))

    result = _run_checked(client, _lookups(),
                          lambda raw: "Point 1 has no answer." if raw == '{"draft": 1}' else None)

    assert result.raw == '{"draft": 2}' and result.send_backs == 1
    assert result.rounds == tool_loop.MAX_ROUNDS + 1
    assert client.requests[-1]["input"][0] == {"role": "user", "content": "Point 1 has no answer."}


def test_exact_text_is_found_with_its_lines():
    class Reader(FakeReader):
        def find_text(self, services, text, limit):
            assert services == ["ui"]
            return [{"service": "ui", "file_path": GRID,
                     "hits": [{"line": 98, "text": "    <Grid item md={4} lg={4}>"}]}]

    lookups = tool_loop.Lookups(services=["ui"], context=ctx.Context(), reader=Reader({}))

    result = asyncio.run(lookups.run("find_text", {"text": "md={4}"}))
    short = asyncio.run(lookups.run("find_text", {"text": "md"}))

    assert result == f"'md={{4}}' found in 1 files:\nui · {GRID}\n    98: <Grid item md={{4}} lg={{4}}>"
    assert short.startswith("Give at least 3 characters")


def _run_checked(client, lookups, check, recheck=None):
    return asyncio.run(tool_loop.run("system", "user", lookups, client=client, check=check,
                                     recheck=recheck))


def test_an_accepted_answer_is_re_checked_once_and_unchanged_keeps_it():
    client = FakeClient(_answer('{"draft": 1}'), _answer('{"unchanged": true}'))

    result = _run_checked(client, _lookups(), lambda raw: None, recheck="Check your design.")

    assert result.raw == '{"draft": 1}' and result.rechecked and result.rounds == 2
    assert client.requests[1]["input"] == [{"role": "user", "content": "Check your design."}]


def test_a_re_checked_answer_that_changes_is_checked_again():
    client = FakeClient(_answer('{"draft": 1}'), _answer('{"draft": 2}'), _answer('{"draft": 3}'))
    seen = []

    def check(raw):
        seen.append(raw)
        return "Point 1 has no answer." if raw == '{"draft": 2}' else None

    result = _run_checked(client, _lookups(), check, recheck="Check your design.")

    # 1 accepted -> re-check -> 2 fails the check -> sent back -> 3 accepted,
    # and the re-check is not sent a second time.
    assert seen == ['{"draft": 1}', '{"draft": 2}', '{"draft": 3}']
    assert result.raw == '{"draft": 3}' and result.send_backs == 1 and result.rounds == 3


def test_no_re_check_once_the_time_is_spent(monkeypatch):
    monkeypatch.setattr(tool_loop, "MAX_SECONDS", 0)
    client = FakeClient(_answer('{"draft": 1}'))

    result = _run_checked(client, _lookups(), lambda raw: None, recheck="Check your design.")

    assert result.raw == '{"draft": 1}' and not result.rechecked and len(client.requests) == 1


def test_the_prompt_offers_lookups_only_with_the_loop():
    plain = prompts.build_system_prompt("bug")
    looping = prompts.build_system_prompt("bug", lookups=(6, 15))

    assert "LOOKUPS" not in plain
    assert "at most 6 rounds and 15 lookups" in looping
    assert 'instead of writing "not shown"' in looping


ANSWER = json.dumps({"title": "t", "services": [{"changes": [
    {"pseudocode_sketch": "title: actionButtonState.title", "edge_cases": [{"case": "a"}]}]}]})


def test_re_check_fixes_are_applied_to_the_design_it_reviewed():
    fixes = json.dumps({"fixes": [
        {"path": "/services/0/changes/0/pseudocode_sketch",
         "value": 'title: selectedReports.length === 0 ? "Archive/Unarchive" : actionButtonState.title'},
        {"path": "/services/0/changes/0/edge_cases/-", "value": {"case": "mixed selection"}},
        {"path": "/title", "value": "Reserve the widest label"},
    ]})

    fixed = json.loads(tool_loop._apply_fixes(ANSWER, fixes))

    change = fixed["services"][0]["changes"][0]
    assert change["pseudocode_sketch"].startswith("title: selectedReports.length === 0")
    assert change["edge_cases"] == [{"case": "a"}, {"case": "mixed selection"}]
    assert fixed["title"] == "Reserve the widest label"


def test_a_re_check_reply_that_is_not_fixes_is_handled():
    remove = json.dumps({"fixes": [{"path": "/services/0/changes/0/edge_cases/0", "op": "remove"}]})

    assert tool_loop._apply_fixes(ANSWER, '{"unchanged": true}') == ANSWER
    assert tool_loop._apply_fixes(ANSWER, '{"title": "whole"}') == '{"title": "whole"}'
    assert json.loads(tool_loop._apply_fixes(ANSWER, remove))["services"][0]["changes"][0][
        "edge_cases"] == []
    assert tool_loop._apply_fixes(ANSWER, '{"fixes": [{"path": "/services/9/x", "value": 1}]}') \
        is None


def test_the_loop_returns_the_fixed_design_and_keeps_it_when_fixes_fail():
    good = json.dumps({"fixes": [{"path": "/title", "value": "fixed"}]})
    client = FakeClient(_answer(ANSWER), _answer(good))

    result = _run_checked(client, _lookups(), lambda raw: None, recheck="Check your design.")

    assert json.loads(result.raw)["title"] == "fixed"

    client = FakeClient(_answer(ANSWER), _answer('{"fixes": [{"path": "/nope/0", "value": 1}]}'))

    result = _run_checked(client, _lookups(), lambda raw: None, recheck="Check your design.")

    assert result.raw == ANSWER


def test_screenshots_go_with_round_1_at_the_slot_before_the_code():
    from src.query.images import SLOT, TicketImage

    shot = TicketImage(name="after.png", media_type="image/png", data=b"\x89PNG\r\n\x1a\n")
    client = FakeClient(_answer())
    user = f"## Issue\nshift\n{SLOT}\n## Code\n..."

    asyncio.run(tool_loop.run("system", user, _lookups(), client=client, images=[shot]))

    [message] = client.requests[0]["input"]
    assert message["role"] == "user"
    assert message["content"][0] == {"type": "input_text", "text": f"## Issue\nshift\n{SLOT}"}
    assert message["content"][1]["type"] == "input_image"
    assert message["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert message["content"][2] == {"type": "input_text", "text": "\n## Code\n..."}
