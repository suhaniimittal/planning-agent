import json

import pytest

from src.query.planner import TddParseError, _strip_schema_titles, parse_tdd


def test_strip_schema_titles_keeps_the_real_title_field_but_drops_metadata():
    schema = {
        "title": "TechnicalDesignDoc",  # cosmetic class-name metadata — must be dropped
        "type": "object",
        "properties": {
            "title": {"title": "Title", "type": "string", "description": "keep me"},  # real field — must survive
        },
        "required": ["title"],
    }
    stripped = _strip_schema_titles(schema)

    assert "title" not in stripped  # cosmetic top-level metadata gone
    assert "title" in stripped["properties"]  # the real field itself survives
    assert stripped["properties"]["title"] == {"type": "string", "description": "keep me"}
    assert stripped["required"] == ["title"]  # untouched — it's a list, not a schema node


def test_parse_tdd_valid_json_returns_model():
    raw = (
        '{"title": "t", "issue_summary": "s", "services": [{"service": "orders", '
        '"changes": [{"file_path": "Orders.java", "function_or_symbol": "calculateTotal", '
        '"change_description": "fix summation", "implementation_notes": "loop and sum", '
        '"pseudocode_sketch": "total = sum(item.price for item in items)", '
        '"acceptance_criteria": ["total is correct for 3 items"], "reasoning": "r"}], '
        '"complexity": "medium", "reasoning": "r"}], '
        '"overall_reasoning": "o"}'
    )
    tdd = parse_tdd(raw)
    assert tdd.title == "t"
    assert tdd.issue_summary == "s"
    assert tdd.services[0].service == "orders"
    assert tdd.services[0].complexity == "medium"
    assert tdd.services[0].changes[0].file_path == "Orders.java"
    assert tdd.services[0].changes[0].change_description == "fix summation"
    assert tdd.services[0].changes[0].implementation_notes == "loop and sum"
    expected_pseudocode = "total = sum(item.price for item in items)"
    assert tdd.services[0].changes[0].pseudocode_sketch == expected_pseudocode
    assert tdd.services[0].changes[0].acceptance_criteria == ["total is correct for 3 items"]


def test_parse_tdd_file_path_defaults_to_none():
    raw = (
        '{"title": "t", "issue_summary": "s", "services": [{"service": "orders", '
        '"changes": [{"function_or_symbol": "calculateTotal", "change_description": "fix", '
        '"implementation_notes": "loop and sum", "pseudocode_sketch": "total = 0", '
        '"reasoning": "r"}], '
        '"complexity": "medium", "reasoning": "r"}], "overall_reasoning": "o"}'
    )
    tdd = parse_tdd(raw)
    assert tdd.services[0].changes[0].file_path is None
    assert tdd.services[0].changes[0].acceptance_criteria == []


def test_parse_tdd_strips_trailing_line_range_from_file_path():
    """Regression test: the model is shown each real code symbol's location
    as `path:START-END` in the input (see _format_code_symbol) and can echo
    that whole string back into `file_path` instead of the bare path — a
    real, observed failure mode that silently breaks every downstream
    file_path lookup (Neo4j, the cloned repo) since neither stores paths
    with a line-range suffix."""
    raw = (
        '{"title": "t", "issue_summary": "s", "services": [{"service": "orders", '
        '"changes": [{"file_path": "src/main/java/Item.java:45-49", '
        '"function_or_symbol": "Item.hashCode", "change_description": "add hashCode", '
        '"implementation_notes": "n", "pseudocode_sketch": "p", "reasoning": "r"}], '
        '"complexity": "low", "reasoning": "r"}], "overall_reasoning": "o"}'
    )
    tdd = parse_tdd(raw)
    assert tdd.services[0].changes[0].file_path == "src/main/java/Item.java"


def test_parse_tdd_leaves_file_path_without_line_range_untouched():
    raw = (
        '{"title": "t", "issue_summary": "s", "services": [{"service": "orders", '
        '"changes": [{"file_path": "src/main/java/Item.java", '
        '"function_or_symbol": "Item.hashCode", "change_description": "add hashCode", '
        '"implementation_notes": "n", "pseudocode_sketch": "p", "reasoning": "r"}], '
        '"complexity": "low", "reasoning": "r"}], "overall_reasoning": "o"}'
    )
    tdd = parse_tdd(raw)
    assert tdd.services[0].changes[0].file_path == "src/main/java/Item.java"


def test_parse_tdd_strips_markdown_code_fence():
    raw = (
        '```json\n{"title": "t", "issue_summary": "s", "services": [], '
        '"overall_reasoning": "o"}\n```'
    )
    tdd = parse_tdd(raw)
    assert tdd.issue_summary == "s"


def test_parse_tdd_malformed_json_raises_tdd_parse_error():
    with pytest.raises(TddParseError):
        parse_tdd("not json at all {{{")


def test_parse_tdd_repairs_missing_required_fields_instead_of_raising():
    """A model omitting one required field (schema is large now) shouldn't
    crash the entire run and produce nothing — it gets a clearly-labeled
    placeholder, and the gap is surfaced in open_questions, never hidden."""
    tdd = parse_tdd('{"issue_summary": "s"}')  # title, services, overall_reasoning all missing
    assert tdd.title
    assert tdd.services == []
    assert tdd.overall_reasoning
    assert any("title" in q for q in tdd.open_questions)
    assert any("services" in q for q in tdd.open_questions)
    assert any("overall_reasoning" in q for q in tdd.open_questions)
    assert all("verify this section manually" in q for q in tdd.open_questions)


def test_parse_tdd_repairs_missing_field_on_a_nested_service():
    raw = json.dumps(
        {
            "title": "t",
            "issue_summary": "s",
            "overall_reasoning": "o",
            "services": [{"service": "orders", "complexity": "low"}],  # reasoning missing
        }
    )
    tdd = parse_tdd(raw)
    assert tdd.services[0].service == "orders"
    assert tdd.services[0].reasoning
    assert any("services.0.reasoning" in q for q in tdd.open_questions)


def test_parse_tdd_wrong_type_still_raises_tdd_parse_error():
    """Repair only ever applies to a MISSING field — a field that's PRESENT
    but the wrong type is a real problem, not a gap worth papering over."""
    raw = json.dumps(
        {
            "title": "t",
            "issue_summary": "s",
            "overall_reasoning": "o",
            "services": "not-a-list",
        }
    )
    with pytest.raises(TddParseError):
        parse_tdd(raw)


def test_parse_tdd_defaults_risks_and_open_questions_to_empty_list():
    raw = '{"title": "t", "issue_summary": "s", "services": [], "overall_reasoning": "o"}'
    tdd = parse_tdd(raw)
    assert tdd.risks == []
    assert tdd.open_questions == []
    assert tdd.testing_notes is None


@pytest.mark.asyncio
async def test_both_llm_paths_use_a_low_temperature(monkeypatch):
    """The API default (1.0) made identical input produce very different
    TDDs from run to run."""
    import sys
    import types

    from src.query import planner

    sent = {}

    class Completions:
        async def create(self, **kwargs):
            sent["openai"] = kwargs
            message = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    # The laptop launcher installs the client class (src/ingestion/local_client.py).
    from src.ingestion import local_client

    monkeypatch.setattr(local_client, "_factory", FakeOpenAI)

    class FakeGateway:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def chat(self, **kwargs):
            sent["gateway"] = kwargs
            return {"content": "{}"}

    ai_module = types.SimpleNamespace(AiGatewayClient=FakeGateway)
    monkeypatch.setitem(sys.modules, "agent_lib.gateway.ai", ai_module)
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "k")
    monkeypatch.setenv("AI_GATEWAY_URL", "https://gateway.example")

    monkeypatch.setattr(planner, "_OPENAI_MODEL", "gpt-4o")
    await planner._plan_via_openai("system", "user")
    await planner._plan_via_gateway("system", "user")

    assert sent["openai"]["temperature"] == planner._TEMPERATURE
    assert sent["gateway"]["temperature"] == planner._TEMPERATURE
    assert planner._TEMPERATURE <= 0.3


def test_models_get_the_api_parameters_they_actually_accept():
    """Three families, verified against the live API: gpt-5 models think only
    when asked and then refuse `temperature`, gpt-5.5/5.6 also refuse `seed`,
    and pre-5 models take the plain parameters."""
    from src.query import planner

    refuses_seed = planner._openai_params("gpt-5.5")
    assert "temperature" not in refuses_seed and "seed" not in refuses_seed
    assert refuses_seed["reasoning_effort"] == "high"
    assert refuses_seed["max_completion_tokens"] >= 12000

    thinking = planner._openai_params("gpt-5.4")
    assert "temperature" not in thinking and thinking["seed"] == planner._SEED
    assert thinking["reasoning_effort"] == "high"
    assert thinking["max_completion_tokens"] >= 12000

    plain = planner._openai_params("gpt-4.1")
    assert plain == {"max_tokens": 12000, "temperature": 0, "seed": planner._SEED}


def test_tdd_model_thinks_before_writing():
    """With only temperature=0 the model spent zero reasoning tokens."""
    from src.query import planner

    assert planner._openai_params(planner._OPENAI_MODEL)["reasoning_effort"] == "high"


# ---------- bug / feature modes ----------


# ---------- test work as implementation work ----------


def _ui_symbol(file_path: str) -> dict:
    return {
        "file_path": file_path, "line_start": 1, "line_end": 9,
        "qualified_name": f"{file_path}::X", "name": "X", "signature": "X",
        "docstring": None, "language": "typescript", "matched_identifiers": [],
    }


