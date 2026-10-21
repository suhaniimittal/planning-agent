import json

import pytest

from src.query.planner import TddParseError, _strip_schema_titles, build_system_prompt, build_user_prompt, parse_tdd


def test_build_system_prompt_does_not_leak_schema_class_name_as_title_value():
    """Regression test: model_json_schema() puts the class's own cosmetic
    "title": "TechnicalDesignDoc" right next to the TDD's actual `title`
    FIELD's schema (which also carries a cosmetic "title": "Title") — a
    real, observed failure mode where the model echoed the schema's class
    name back as the TDD's title verbatim instead of writing a real one."""
    prompt = build_system_prompt()
    assert '"TechnicalDesignDoc"' not in prompt


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


def test_build_system_prompt_contains_schema_field_names():
    prompt = build_system_prompt()
    for field in (
        "title",
        "services",
        "changes",
        "file_path",
        "change_description",
        "reasoning",
        "issue_summary",
        "risks",
        "testing_notes",
        "open_questions",
        "implementation_notes",
        "acceptance_criteria",
        "complexity",
        "affected_data_model",
        "backward_compatibility_notes",
        "pseudocode_sketch",
    ):
        assert field in prompt


def test_build_system_prompt_balances_thoroughness_with_no_fabrication():
    prompt = build_system_prompt()
    assert "thorough" in prompt.lower()
    assert "open_questions" in prompt
    assert "invent" in prompt.lower()


def test_build_system_prompt_labels_pseudocode_conditionally():
    """pseudocode_sketch's disclaimer is now conditional: illustrative-only
    for doc-only services, but may track real source for services with a
    'Real code symbols' section — the model must not assume uniformity
    across a mixed candidate set."""
    prompt = build_system_prompt()
    assert "pseudocode_sketch" in prompt
    assert "real code symbols" in prompt.lower()
    assert "not seen its actual source" in prompt.lower()
    assert "not assume uniform" in prompt.lower()


def test_build_system_prompt_instructs_no_fabricated_file_paths():
    prompt = build_system_prompt()
    assert "never" in prompt.lower() and "file_path" in prompt


def test_build_system_prompt_instructs_llm_to_leave_service_dependencies_empty():
    prompt = build_system_prompt()
    assert "service_dependencies" in prompt
    assert "empty" in prompt.lower()


def test_build_user_prompt_includes_issue_text_and_all_candidate_docs():
    docs = [
        {"name": "orders", "summary": "Handles order creation and lookup."},
        {"name": "shipping", "summary": "Handles shipment scheduling."},
    ]
    prompt = build_user_prompt("shipping estimates are wrong", docs)
    assert "shipping estimates are wrong" in prompt
    assert "Handles order creation and lookup." in prompt
    assert "Handles shipment scheduling." in prompt
    assert "## Service: orders" in prompt
    assert "## Service: shipping" in prompt


def test_build_user_prompt_seed_with_no_code_gets_summary_only():
    docs = [{"name": "orders", "summary": "Handles order creation and lookup."}]
    prompt = build_user_prompt("issue", docs, seed_names={"orders"})
    assert "Handles order creation and lookup." in prompt
    assert "matched at the summary level only" in prompt.lower()


def test_build_user_prompt_notes_when_seed_has_real_code():
    """Once real code symbols exist for a seed service, the summary is kept
    only for business context — the note should say so, and never claim any
    'full documentation' concept (there isn't one anymore)."""
    docs = [{"name": "orders", "summary": "Handles order creation and lookup."}]
    code_symbols = {
        "orders": [
            {
                "qualified_name": "controller.py::create_order",
                "name": "create_order",
                "file_path": "controller.py",
                "line_start": 1,
                "line_end": 2,
                "signature": "def create_order():",
                "docstring": None,
                "language": "python",
            }
        ]
    }
    prompt = build_user_prompt("issue", docs, code_symbols, {}, seed_names={"orders"})
    assert "Handles order creation and lookup." in prompt
    assert "real code was found" in prompt.lower()
    assert "full documentation" not in prompt.lower()


def test_build_user_prompt_includes_known_dependencies_as_verified_input():
    """Regression test: the model had no confirmed cross-service ground
    truth at generation time, so it left architecture-diagram `depends_on`
    lists empty rather than guess — real edges must be shown as INPUT, not
    just used to overwrite the model's own output afterward."""
    docs = [{"name": "orders", "summary": "d"}]
    prompt = build_user_prompt(
        "issue", docs, known_dependencies=[{"source": "orders", "target": "shipping"}]
    )
    assert "Known service dependencies" in prompt
    assert "orders -> shipping" in prompt
    assert "already confirmed real" in prompt.lower()


def test_build_user_prompt_omits_known_dependencies_section_when_none_given():
    docs = [{"name": "orders", "summary": "d"}]
    prompt = build_user_prompt("issue", docs, known_dependencies=[])
    assert "Known service dependencies" not in prompt


def test_build_user_prompt_marks_non_seed_neighbor_as_background_context():
    docs = [{"name": "shipping", "summary": "Handles shipment scheduling."}]
    # "shipping" is a candidate (e.g. via a CALLS hop), but NOT in seed_names
    prompt = build_user_prompt("issue", docs, seed_names={"orders"})
    assert "Handles shipment scheduling." in prompt
    assert "graph-connected" in prompt.lower()


def test_build_user_prompt_treats_all_as_seeds_when_seed_names_omitted():
    """Default (no seed_names passed) preserves old behavior — every
    candidate is treated as a seed, for any caller that doesn't
    distinguish."""
    docs = [{"name": "shipping", "summary": "Handles shipment scheduling."}]
    prompt = build_user_prompt("issue", docs)
    assert "Handles shipment scheduling." in prompt
    assert "## Service: shipping" in prompt


def test_build_user_prompt_handles_missing_summary_gracefully():
    docs = [{"name": "orders", "summary": None}]
    prompt = build_user_prompt("issue", docs)
    assert "Not available." in prompt


def test_build_user_prompt_includes_real_code_symbols_section_when_present():
    docs = [{"name": "orders", "summary": "doc"}]
    code_symbols = {
        "orders": [
            {
                "qualified_name": "controller.py::OrderController.create_order",
                "name": "create_order",
                "file_path": "controller.py",
                "line_start": 4,
                "line_end": 7,
                "signature": "def create_order(self, user_id, items):",
                "docstring": "Creates a new order.",
                "language": "python",
            }
        ]
    }
    snippets = {
        ("orders", "controller.py::OrderController.create_order"):
            "def create_order(self, user_id, items):\n    return {}"
    }
    prompt = build_user_prompt("issue", docs, code_symbols, snippets)
    assert "Real code symbols for orders" in prompt
    assert "controller.py:4-7" in prompt
    assert "def create_order(self, user_id, items):" in prompt
    assert "return {}" in prompt


def test_build_user_prompt_omits_real_code_symbols_section_when_absent():
    docs = [{"name": "shipping", "summary": "doc"}]
    prompt = build_user_prompt("issue", docs)
    assert "Real code symbols" not in prompt


def test_build_user_prompt_handles_mixed_candidate_set():
    docs = [
        {"name": "orders", "summary": "orders doc"},
        {"name": "shipping", "summary": "shipping doc"},
    ]
    code_symbols = {
        "orders": [
            {
                "qualified_name": "controller.py::create_order",
                "name": "create_order",
                "file_path": "controller.py",
                "line_start": 1,
                "line_end": 2,
                "signature": "def create_order():",
                "docstring": None,
                "language": "python",
            }
        ]
    }
    prompt = build_user_prompt("issue", docs, code_symbols, {})
    assert "Real code symbols for orders" in prompt
    assert "Real code symbols for shipping" not in prompt


def test_build_system_prompt_mentions_real_code_symbols_and_has_real_source():
    prompt = build_system_prompt()
    assert "real code symbols" in prompt.lower()
    assert "has_real_source" in prompt


def test_build_system_prompt_insists_structured_facts_are_required_not_optional():
    """Regression test: these fields are all Optional/empty-list-default in
    the schema (so a genuinely single, isolated component can be left
    empty), which means a model can satisfy validation while skipping them
    entirely — the old ASCII-diagram version of this instruction hit
    exactly this failure in practice. The prompt must push back hard on
    treating empty as the default, not just mention the fields exist."""
    prompt = build_system_prompt()
    assert "not optional polish" in prompt.lower()
    assert "current_architecture_components" in prompt
    assert "proposed_architecture_components" in prompt
    assert "sequence_steps" in prompt


def test_build_system_prompt_requires_proposed_architecture_to_reflect_the_fix():
    """Regression test: a model that copies current_architecture_components
    verbatim into proposed_architecture_components — even when its own
    proposed_solution describes a brand new call/dependency — produces a
    document that contradicts itself. The prompt must call this out as an
    inconsistency to avoid, not just describe the two fields independently."""
    prompt = build_system_prompt()
    assert "must reflect what your own" in prompt.lower()
    assert "copying the current list" in prompt.lower()
    assert "inconsistency" in prompt.lower()


def test_build_system_prompt_instructs_facts_not_hand_drawn_diagrams():
    """The whole point of the restructure: the LLM states facts (components
    + dependencies, ordered steps, sequence interactions) and a separate
    rendering step builds the actual diagram — the model should never be
    asked to hand-draw ASCII art anymore."""
    prompt = build_system_prompt()
    assert "do not need to hand-draw" in prompt.lower() or "do not need to hand" in prompt.lower()
    assert "depends_on" in prompt
    assert "current_flow_steps" in prompt


def test_build_system_prompt_asks_for_short_specific_writing():
    """The TDDs had become 15-page walls of text that repeated themselves;
    the prompt must ask for brevity, one fact per field, one symbol per change."""
    prompt = build_system_prompt()
    assert "WRITE FOR A BUSY DEVELOPER" in prompt
    assert "multiple sentences" not in prompt
    assert "exactly ONE symbol name" in prompt
    assert "Leave `sequence_steps` empty" in prompt


def test_build_system_prompt_forbids_generic_domain_knowledge_as_fact():
    """Regression test: the model introduced ungrounded domain assumptions
    (a compliance standard, a common e-commerce pattern) that were never
    actually in the evidence — a real observed failure, not hypothetical."""
    prompt = build_system_prompt()
    assert "pci dss" in prompt.lower()
    assert "generic domain or compliance knowledge" in prompt.lower()
    assert "plausible-sounding guess" in prompt.lower()


def test_build_system_prompt_requires_tracing_real_code_before_root_cause():
    """Regression test: root cause analysis was describing functions
    generically instead of what the real code does — the model must still
    trace the real lines first, but write only the conclusions, each marked
    with its certainty and evidence."""
    prompt = build_system_prompt()
    assert "TRACE the real" in prompt
    assert "grounded in its actual lines" in prompt
    assert "never the walk-through itself" in prompt
    assert "Confirmed: <cause>" in prompt and "Not the cause:" in prompt


def test_build_system_prompt_requires_impact_analysis_to_name_specific_evidence():
    """Regression test: impact/consequence statements were generic filler
    ("this could impact payment processing and compliance") instead of
    naming specific other services/functions actually in evidence."""
    prompt = build_system_prompt()
    assert "generic concerns" in prompt.lower() or "generically-named concern" in prompt.lower()
    assert "not a checkable" in prompt.lower() or "sounds thorough without saying anything checkable" in prompt.lower()


def test_build_system_prompt_requires_using_known_dependencies_for_depends_on():
    """Regression test: the model left `depends_on` empty for services that
    actually do have real, known dependencies — the prompt must explicitly
    require using the 'Known service dependencies' input, not just mention
    that depends_on exists as a field."""
    prompt = build_system_prompt()
    assert "known service dependencies" in prompt.lower()
    assert "must reflect every one of those pairs" in prompt.lower()
    assert "wrong answer, not a cautious one" in prompt.lower()


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
    """Three families, verified against the live API: gpt-5.5/5.6 refuse
    `temperature` outright, other gpt-5 models accept it but still need
    `max_completion_tokens`, and pre-5 models take the plain parameters."""
    from src.query import planner

    refuses_temp = planner._openai_params("gpt-5.5")
    assert "temperature" not in refuses_temp and "seed" not in refuses_temp
    assert refuses_temp["max_completion_tokens"] >= 12000

    thinking = planner._openai_params("gpt-5.4")
    assert thinking["temperature"] == 0 and thinking["seed"] == planner._SEED
    assert thinking["max_completion_tokens"] >= 12000

    plain = planner._openai_params("gpt-4.1")
    assert plain == {"max_tokens": 12000, "temperature": 0, "seed": planner._SEED}


def test_tdd_model_is_deterministic_by_default():
    """The same ticket must not produce a visibly different TDD run to run."""
    from src.query import planner

    assert planner._TEMPERATURE == 0
    assert "temperature" in planner._openai_params(planner._OPENAI_MODEL)



def test_user_prompt_lists_what_else_is_defined_in_each_file():
    symbol = {
        "file_path": "a.ts", "line_start": 1, "line_end": 3, "qualified_name": "a.ts::validate",
        "name": "validate", "signature": "validate()", "docstring": None, "language": "typescript",
    }
    prompt = build_user_prompt(
        "issue",
        [{"name": "ui", "summary": "s"}],
        {"ui": [symbol]},
        {("ui", "a.ts::validate"): "code"},
        file_context={("ui", "a.ts"): ["validate", "submit", "Form.render"]},
    )
    assert "Also defined in this file: submit, Form.render" in prompt


def test_system_prompt_requires_an_answer_for_every_chosen_service():
    prompt = build_system_prompt()
    assert "Never silently leave one out" in prompt


def test_user_prompt_says_which_issue_names_a_symbol_contains():
    symbol = {
        "file_path": "a.ts", "line_start": 1, "line_end": 3, "qualified_name": "a.ts::schema",
        "name": "schema", "signature": "schema", "docstring": None, "language": "typescript",
        "matched_identifiers": ["internationalPhoneNumber"],
    }
    prompt = build_user_prompt("issue", [{"name": "ui", "summary": "s"}], {"ui": [symbol]}, {})
    assert "Contains `internationalPhoneNumber`, named in the issue itself." in prompt


def test_system_prompt_restricts_scope_to_what_the_ticket_reports():
    """Regression test: the prompt once told the model to fix every copy of a
    pattern and "where the bad input ENTERS and where the failure SURFACES".
    It obeyed — and produced a UI alert box nobody asked for, which a human
    reviewer rejected with "does anyone told you to add alert popup... this is
    not mentioned in the ticket description"."""
    prompt = build_system_prompt()
    assert "ONLY FIX WHAT THE TICKET REPORTS" in prompt
    assert "not an improvement you noticed along the way" in prompt
    assert "propose a change for EACH of them" not in prompt
    assert "where the bad input ENTERS" not in prompt


def test_system_prompt_allows_no_change_as_a_real_answer():
    """A service the report does not touch should get "no change needed",
    not invented work to look thorough."""
    prompt = build_system_prompt()
    assert '"no change needed" IS the correct answer' in prompt


def test_system_prompt_separates_analysis_scope_from_change_scope():
    """A service on the reproduction path must be EXAMINED and appear with a
    verdict, which is not the same as being changed. Conflating the two is
    how candidate-service first vanished from the document entirely, and
    then — once it was included — how it was waved through."""
    prompt = build_system_prompt()
    assert "ANALYSIS SCOPE" in prompt and "CHANGE SCOPE" in prompt
    assert "must be EXAMINED and must APPEAR" in prompt
    assert "Sitting on the data path is NOT by itself a reason to change" in prompt


def test_system_prompt_treats_where_a_bad_value_is_stored_as_a_root_cause():
    """Regression test: candidate-service persists the reported phone value
    untrimmed (EmploymentVerificationInput.toAvro -> .setPhone(phone)), and
    the client form later rejects it. The TDD cleared candidate-service
    anyway. Storing a value a consumer then rejects is a cause, not an
    adjacency.

    Phrased deliberately without "where the bad input ENTERS" — that older
    wording over-scoped into changes nobody asked for."""
    prompt = build_system_prompt()
    assert "ROOT CAUSE LOCATION, NOT A " in prompt
    assert "first accepted and written down" in prompt
    assert "where the bad input ENTERS" not in prompt
    # both fix sites must be surfaced rather than one silently chosen
    assert "entry-side fix and a consumer-side fix" in prompt
    assert "Never present one as settled" in prompt


def test_system_prompt_separates_verified_from_reported_and_inferred():
    prompt = build_system_prompt()
    for kind in ("VERIFIED", "REPORTED", "INFERRED", "UNKNOWN"):
        assert kind in prompt
    assert "merely EXISTING is not evidence" in prompt


def test_system_prompt_forbids_clearing_a_service_using_another_services_code():
    """Regression test: the TDD cleared candidate-service (Kotlin backend) by
    citing a trim in ui/src/apps/candidate/... — a different service's
    source for the same feature."""
    prompt = build_system_prompt()
    assert "only evidence about that service" in prompt
    assert "Never clear a service of needing a change" in prompt


def test_system_prompt_root_cause_format_has_hypothesis_and_unknown():
    prompt = build_system_prompt()
    assert "Hypothesis: <cause>" in prompt
    assert "Unknown: <what you could not determine>" in prompt
    assert "never establishes a cause on its own" in prompt


def test_system_prompt_requires_every_service_in_the_component_diagram():
    """Regression test: the rendered architecture showed three components and
    one edge, with two dead ends — the "half-finished" diagram. Every service
    in `services` must be a component, including no-change ones."""
    prompt = build_system_prompt()
    assert "EVERY service you listed in `services` MUST appear here" in prompt
    assert "half-finished diagram" in prompt
    assert "Appearing here is NOT a claim that a service changes" in prompt


def test_system_prompt_requires_the_two_flow_lists_to_be_comparable():
    prompt = build_system_prompt()
    assert "rendered side by\nside" in prompt or "rendered side by " in prompt
    assert "same journey at the same" in prompt
    assert "the rendered flow looks truncated" in prompt


def test_system_prompt_requires_a_failing_then_passing_regression_test():
    prompt = build_system_prompt()
    assert "FAILS" in prompt and "PASSES after your change" in prompt
    assert "TO ADD or an EXISTING test" in prompt


def test_system_prompt_forbids_claiming_tests_were_executed():
    """Nothing in this pipeline runs tests, so any claim that one passed is
    fabricated by construction."""
    prompt = build_system_prompt()
    assert "you have run nothing" in prompt
    assert "never write" in prompt.lower() and "passed, failed, ran or was verified" in prompt


def test_system_prompt_requires_each_implementation_step_to_name_a_location():
    prompt = build_system_prompt()
    assert "<service> <file_path> <symbol>" in prompt
    assert "is not an implementation step" in prompt
    assert "which `testing_plan` scenario" in prompt


def test_system_prompt_requires_reusing_what_already_exists():
    prompt = build_system_prompt()
    assert "EXISTING function, component" in prompt
    assert "adds a second mechanism beside one that already exists" in prompt


def test_system_prompt_ends_with_a_cross_section_self_check():
    """Sections that contradict each other cost the reader trust in all of
    them, so consistency is checked explicitly rather than hoped for."""
    prompt = build_system_prompt()
    assert "SELF-CHECK BEFORE YOU RESPOND" in prompt
    assert "no service you called \"no change needed\" also carries changes" in prompt
    assert "nothing claims a test was run" in prompt


def test_system_prompt_leaves_unrendered_diagram_fields_empty():
    """current_architecture_components is never read by the renderer, and the
    prompt used to demand it in detail while also saying sequence_steps was
    both required and to-be-left-empty. One instruction each now."""
    prompt = build_system_prompt()
    assert "Leave `sequence_steps` empty" in prompt
    assert "leave `current_architecture_" in prompt
    assert "whenever more than one component is involved" not in prompt


def test_system_prompt_requires_a_written_trace_of_the_reported_value():
    """Regression test: candidate-service persists the reported phone value
    untrimmed at EmploymentVerificationInput.toAvro -> .setPhone(phone). That
    whole class WAS in the prompt (7.6KB, under the class-widening
    threshold), but the model read isPhoneValid, saw non-US bypasses
    validation, wrote "Not the cause", and never looked at the builder four
    lines below.

    An in-your-head trace can be skipped silently. Four named output lines
    cannot: writing `Stored at:` forces the model to go find the write."""
    prompt = build_system_prompt()
    assert "TRACE THE REPORTED VALUE THROUGH ITS WHOLE LIFE" in prompt
    for stage in ("Accepted at:", "Validated at:", "Stored at:", "Consumed at:"):
        assert stage in prompt
    assert "never leave a stage out" in prompt


def test_system_prompt_says_a_passing_validation_is_not_exoneration():
    """The exact reasoning that cleared candidate-service — "validation does
    not reject it, so this service is not the cause" — must instead become a
    reason to keep tracing to the write."""
    prompt = build_system_prompt()
    assert "is not exoneration" in prompt
    assert "keep tracing to where it is written" in prompt
    assert "still gets its `Stored at:` line" in prompt


def test_system_prompt_defines_smallest_change_set_as_end_to_end():
    """Regression test: with "smallest set that fully addresses the reported
    problem", the model read "reported problem" as the SYMPTOM and proposed a
    ui-only fix, leaving candidate-service writing untrimmed phone values via
    EmploymentVerificationInput.toAvro -> .setPhone(phone). Redefining
    "smallest" as end-to-end is what puts the source fix back in scope."""
    prompt = build_system_prompt()
    assert "SMALLEST SET THAT CORRECTLY ADDRESSES THE ISSUE END TO END" in prompt
    assert "not the fewest files" in prompt
    assert "frontend-only or backend-only fix by default" in prompt
    assert "whether data is already persisted in an invalid or unnormalised state" in prompt


def test_system_prompt_puts_the_persisting_service_in_change_scope():
    prompt = build_system_prompt()
    assert "needs its own change, even when fixing the consumer alone" in prompt
    assert "Propose the normalisation " in prompt
    # the over-scoping guard must survive the widening
    assert "Sitting on the data path is NOT by itself a reason to change" in prompt
    assert "merely because its file or service is part of the same data flow" in prompt


# ---------- bug / feature modes ----------


def test_bug_mode_carries_the_value_trace_and_feature_mode_does_not():
    """The trace is what located EmploymentVerificationInput.toAvro on a bug
    ticket. It has nothing to trace on a feature request, where no reported
    value exists."""
    bug = build_system_prompt("bug")
    feature = build_system_prompt("feature")
    for probe in ("TRACE THE REPORTED VALUE", "Transformed at:", "Stored at:", "Not the cause:"):
        assert probe in bug
        assert probe not in feature


def test_feature_mode_replaces_root_cause_with_gap_analysis():
    """Regression test: the bulk-print feature ticket ran through the bug
    prompt and had to express "compliance-service already merges PDFs" as a
    `Not the cause:` line. Exists/Missing/Extend is the shape that fits."""
    feature = build_system_prompt("feature")
    bug = build_system_prompt("bug")
    assert "GAP ANALYSIS" in feature
    for probe in ("Exists: <capability>", "Missing: <capability", "Extend: <service>"):
        assert probe in feature
        assert probe not in bug


def test_bug_mode_asks_whether_the_invalid_state_can_recur():
    bug = build_system_prompt("bug")
    assert "can the same invalid state be produced again" in bug
    assert "UPSTREAM correction" in bug and "DOWNSTREAM correction" in bug


def test_bug_mode_addresses_already_stored_data():
    """Fixing the entry point does not clean rows already written. A bug TDD
    has to say which change handles those."""
    bug = build_system_prompt("bug")
    assert "PERSISTED DATA" in bug
    assert "whether existing records still fail after the change" in bug
    assert "Do not recommend a data repair or migration unless" in bug


def test_feature_mode_separates_stated_requirements_from_assumptions():
    feature = build_system_prompt("feature")
    assert "REQUIREMENTS BEFORE DESIGN" in feature
    assert "Never invent a business rule" in feature
    assert "acceptance criterion" in feature


def test_feature_mode_requires_building_on_existing_mechanisms():
    feature = build_system_prompt("feature")
    assert "BUILD ON WHAT IS ALREADY THERE" in feature
    assert "A second mechanism standing beside one that already works is wrong" in feature


def test_both_modes_share_the_create_and_no_change_rules():
    """These fix defects measured on the bulk-print run — a `(modify)` on a
    symbol the text said not to change, and two no-change files listed under
    FILES TO CHANGE. Neither is mode-specific."""
    for mode in ("bug", "feature"):
        prompt = build_system_prompt(mode)
        assert "WHEN A CHANGE ADDS SOMETHING THAT DOES NOT EXIST YET" in prompt
        assert "A CHANGE THAT CHANGES NOTHING IS NOT A CHANGE" in prompt


def test_both_modes_share_the_whole_base():
    """The base must not be duplicated per mode, or the two copies drift."""
    for mode in ("bug", "feature"):
        prompt = build_system_prompt(mode)
        for probe in (
            "FOUR KINDS OF KNOWING",
            "SELF-CHECK BEFORE YOU RESPOND",
            "WRITE FOR A BUSY DEVELOPER",
            "only evidence about that service",
            "EVERY service you listed in `services` MUST appear here",
            "you have run nothing",
            "<service> <file_path> <symbol>",
        ):
            assert probe in prompt, f"{probe!r} missing from {mode} mode"


def test_an_unknown_mode_falls_back_to_bug():
    """A malformed mode must never produce a prompt with neither block."""
    assert build_system_prompt("refactor") == build_system_prompt("bug")
    assert build_system_prompt("") == build_system_prompt("bug")


# ---------- test work as implementation work ----------


def test_prompt_requires_a_step_that_writes_the_tests():
    """Regression test: the implementation plans of four real TDDs listed only
    production edits. The coding agent worked through the plan, wrote no
    tests, and CI went red. A scenario in `testing_plan` instructs nobody."""
    for mode in ("bug", "feature"):
        prompt = build_system_prompt(mode)
        assert "TEST WORK IS PART OF THE CHANGE" in prompt
        assert "must ALSO appear as its own step in `implementation_plan`" in prompt


def test_prompt_lets_a_test_step_name_an_unverified_location():
    """The verified-location rule is what deleted the test step: test files are
    never in the evidence, so a step that writes one cannot name a location the
    model saw. Without this exception the new rule and the old one contradict."""
    prompt = build_system_prompt("bug")
    assert "The single exception is a step that writes or updates tests" in prompt
    assert "marked as proposed rather than existing" in prompt


def test_prompt_treats_a_sibling_test_file_as_a_contradiction_risk():
    """Regression test: a TDD told the coding agent to replace ReportForm's
    console logging. ReportForm/view.test.tsx asserted that logging. One test
    of 17,737 failed and blocked the merge."""
    prompt = build_system_prompt("bug")
    assert '"Tested by" LINE NAMES A TEST THAT ALREADY ASSERTS' in prompt
    assert "propose updating it in the same implementation step" in prompt
    assert "never quote a test, name a test function" in prompt


def test_prompt_does_not_read_absence_of_a_test_as_untested():
    """Two lookups can miss a test — the same-folder one and the by-name one
    — so silence from both still is not proof that nothing covers it."""
    prompt = build_system_prompt("bug")
    assert "no test file sits beside it AND names no test" in prompt
    assert "do not conclude it is untested" in prompt
    assert "a test may still exist that never names it directly" in prompt


def test_prompt_prefers_an_existing_test_over_a_new_one():
    """A new test file is written from nothing, imports included — and the
    existing one very likely asserts the behaviour being changed."""
    prompt = build_system_prompt("bug")
    assert "AN EXISTING TEST IS ALWAYS PREFERRED OVER A NEW ONE" in prompt
    assert "Existing test files that mention this symbol by name" in prompt
    assert "code that did not exist cannot already be tested" in prompt


def test_self_check_covers_test_work():
    prompt = build_system_prompt("bug")
    assert "every change has test work in `implementation_plan`" in prompt


def test_user_prompt_names_a_test_file_sitting_beside_a_symbol():
    symbol = {
        "file_path": "src/apps/client/modules/Reports/components/ReportForm/view.tsx",
        "line_start": 1, "line_end": 9, "qualified_name": "ReportForm/view.tsx::ReportForm",
        "name": "ReportForm", "signature": "ReportForm", "docstring": None,
        "language": "typescript", "matched_identifiers": [],
    }
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [symbol]}, {},
        sibling_tests={
            ("ui", symbol["file_path"]): [
                "src/apps/client/modules/Reports/components/ReportForm/view.test.tsx"
            ]
        },
    )
    assert "Tested by (existing test file beside it)" in prompt
    assert "ReportForm/view.test.tsx" in prompt


def test_user_prompt_says_so_when_no_test_sits_beside_a_symbol():
    symbol = {
        "file_path": "a/b/view.tsx", "line_start": 1, "line_end": 2,
        "qualified_name": "a/b/view.tsx::X", "name": "X", "signature": "X",
        "docstring": None, "language": "typescript", "matched_identifiers": [],
    }
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [symbol]}, {},
        sibling_tests={("ui", "a/b/view.tsx"): []},
    )
    assert "No test file sits beside this one in its folder." in prompt


def _ui_symbol(file_path: str) -> dict:
    return {
        "file_path": file_path, "line_start": 1, "line_end": 9,
        "qualified_name": f"{file_path}::X", "name": "X", "signature": "X",
        "docstring": None, "language": "typescript", "matched_identifiers": [],
    }


def test_a_symbol_with_no_sibling_test_is_told_where_a_new_one_belongs():
    """Without a path the model can only ask where tests live, and the test
    never gets written — a coding agent acts on file changes, not questions."""
    path = "src/forms/employmentVerificationForm.ts"
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [_ui_symbol(path)]}, {},
        sibling_tests={("ui", path): []},
        new_test_paths={("ui", path): "src/forms/employmentVerificationForm.test.ts"},
    )
    assert "No test file sits beside this one." in prompt
    assert "belongs at `src/forms/employmentVerificationForm.test.ts`" in prompt


def test_an_existing_sibling_test_wins_over_a_suggested_path():
    path = "src/forms/view.tsx"
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [_ui_symbol(path)]}, {},
        sibling_tests={("ui", path): ["src/forms/view.test.tsx"]},
        new_test_paths={("ui", path): "src/forms/view.test.tsx"},
    )
    assert "Tested by (existing test file beside it): `src/forms/view.test.tsx`" in prompt
    assert "belongs at" not in prompt


def test_a_service_with_no_measured_convention_still_just_says_there_is_no_test():
    """A mirrored test tree (src/test/kotlin) yields no convention, and the
    model is asked where tests live rather than handed a made-up path."""
    path = "src/main/kotlin/Foo.kt"
    prompt = build_user_prompt(
        "issue", [{"name": "reports-api", "summary": "s"}],
        {"reports-api": [_ui_symbol(path)]}, {},
        sibling_tests={("reports-api", path): []},
        new_test_paths={},
    )
    assert "No test file sits beside this one in its folder." in prompt
    assert "belongs at" not in prompt


def test_a_symbol_lists_the_tests_that_name_it_wherever_they_live():
    """Five test files covered a real component from another tree; the
    same-folder lookup found none, so the design had a new test written
    from scratch instead of the real one edited."""
    path = "src/common/.../AtsIntegrationSection/view.tsx"
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [_ui_symbol(path)]}, {},
        sibling_tests={("ui", path): []},
        tests_by_symbol={("ui", "X"): ["src/apps/internal/.../view.data.test.tsx"]},
    )
    assert "Existing test files that mention this symbol by name:" in prompt
    assert "src/apps/internal/.../view.data.test.tsx" in prompt


def test_nothing_is_claimed_when_no_test_names_the_symbol():
    path = "a/b/view.tsx"
    prompt = build_user_prompt(
        "issue", [{"name": "ui", "summary": "s"}], {"ui": [_ui_symbol(path)]}, {},
        sibling_tests={("ui", path): []},
        tests_by_symbol={},
    )
    assert "mention this symbol by name" not in prompt
