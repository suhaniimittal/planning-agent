import json

import pytest

from src.query.planner import TddParseError, build_system_prompt, build_user_prompt, parse_tdd


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
        {
            "name": "orders",
            "summary": "Handles order creation and lookup.",
            "full_doc": "orders full doc text",
            "key_symbols": ["newOrder"],
            "domain": "E-commerce",
        },
        {
            "name": "shipping",
            "summary": "Handles shipment scheduling.",
            "full_doc": "shipping full doc text",
            "key_symbols": [],
            "domain": "Shipping",
        },
    ]
    prompt = build_user_prompt("shipping estimates are wrong", docs)
    assert "shipping estimates are wrong" in prompt
    assert "Handles order creation and lookup." in prompt
    assert "Handles shipment scheduling." in prompt
    assert "## Service: orders" in prompt
    assert "## Service: shipping" in prompt


def test_build_user_prompt_gives_seed_services_summary_not_full_doc_by_default():
    """A seed with no code_symbols passed at all (the common case for any
    caller that hasn't run selection yet) gets the same summary-only
    treatment as a seed with no MATCHED code — full_doc is never the
    default for a seed anymore, matched code or not."""
    docs = [
        {
            "name": "orders",
            "summary": "Handles order creation and lookup.",
            "full_doc": "orders full doc text",
            "key_symbols": [],
            "domain": "E-commerce",
        }
    ]
    prompt = build_user_prompt("issue", docs, seed_names={"orders"})
    assert "Handles order creation and lookup." in prompt
    assert "orders full doc text" not in prompt
    assert "Full documentation:" not in prompt


def test_build_user_prompt_uses_summary_not_full_doc_when_seed_has_real_code():
    """Once real code symbols exist for a seed service, the detailed
    file-by-file full_doc is superseded by it — only the short business-
    context summary is still worth including alongside the real code."""
    docs = [
        {
            "name": "orders",
            "summary": "Handles order creation and lookup.",
            "full_doc": "orders full doc text " * 200,  # simulate a large doc
            "key_symbols": [],
            "domain": "E-commerce",
        }
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
    prompt = build_user_prompt("issue", docs, code_symbols, {}, seed_names={"orders"})
    assert "Handles order creation and lookup." in prompt
    assert "orders full doc text" not in prompt
    assert "Full documentation:" not in prompt
    assert "superseded" in prompt.lower()


def test_build_user_prompt_seed_without_matched_code_gets_summary_not_full_doc():
    """Regression test: a seed only means "scored in the top-k service
    search" — that can be a weak, coincidental match, especially when
    several services' summary embeddings score close together on a small
    graph. Rewarding an unconfirmed seed with its entire full_doc was
    backwards; it now gets the same summary-only treatment as any other
    seed, and is explicitly told to treat itself with the same caution as
    a graph-connected neighbor unless its own summary/key symbols justify
    otherwise."""
    docs = [
        {
            "name": "orders",
            "summary": "Handles order creation and lookup.",
            "full_doc": "orders full doc text",
            "key_symbols": [],
            "domain": "E-commerce",
        }
    ]
    prompt = build_user_prompt("issue", docs, code_symbols_by_service={}, seed_names={"orders"})
    assert "Handles order creation and lookup." in prompt
    assert "orders full doc text" not in prompt
    assert "Full documentation:" not in prompt
    assert "graph-connected neighbor" in prompt.lower()


def test_build_user_prompt_includes_known_dependencies_as_verified_input():
    """Regression test: the model had no confirmed cross-service ground
    truth at generation time, so it left architecture-diagram `depends_on`
    lists empty rather than guess — real edges must be shown as INPUT, not
    just used to overwrite the model's own output afterward."""
    docs = [{"name": "orders", "full_doc": "d", "key_symbols": [], "domain": None}]
    prompt = build_user_prompt(
        "issue", docs, known_dependencies=[{"source": "orders", "target": "shipping"}]
    )
    assert "Known service dependencies" in prompt
    assert "orders -> shipping" in prompt
    assert "already confirmed real" in prompt.lower()


def test_build_user_prompt_omits_known_dependencies_section_when_none_given():
    docs = [{"name": "orders", "full_doc": "d", "key_symbols": [], "domain": None}]
    prompt = build_user_prompt("issue", docs, known_dependencies=[])
    assert "Known service dependencies" not in prompt


def test_build_user_prompt_omits_full_doc_for_non_seed_neighbor():
    docs = [{"name": "shipping", "full_doc": "shipping full doc text", "key_symbols": [], "domain": "Shipping"}]
    # "shipping" is a candidate (e.g. via a CALLS hop), but NOT in seed_names
    prompt = build_user_prompt("issue", docs, seed_names={"orders"})
    assert "shipping full doc text" not in prompt
    assert "graph-connected" in prompt.lower()


def test_build_user_prompt_treats_all_as_seeds_when_seed_names_omitted():
    """Default (no seed_names passed) preserves old behavior — every
    candidate is treated as a seed, for any caller that doesn't
    distinguish. Content is still summary-only without matched code,
    same as an explicit seed."""
    docs = [
        {
            "name": "shipping",
            "summary": "Handles shipment scheduling.",
            "full_doc": "shipping full doc text",
            "key_symbols": [],
            "domain": "Shipping",
        }
    ]
    prompt = build_user_prompt("issue", docs)
    assert "Handles shipment scheduling." in prompt
    assert "## Service: shipping" in prompt


def test_build_user_prompt_includes_key_symbols_and_domain():
    docs = [
        {
            "name": "orders",
            "full_doc": "doc",
            "key_symbols": ["newOrder", "parseId"],
            "domain": "E-commerce",
        }
    ]
    prompt = build_user_prompt("issue", docs)
    assert "newOrder, parseId" in prompt


def test_build_user_prompt_includes_real_code_symbols_section_when_present():
    docs = [{"name": "orders", "full_doc": "doc", "key_symbols": [], "domain": "E-commerce"}]
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
    docs = [{"name": "shipping", "full_doc": "doc", "key_symbols": [], "domain": "Shipping"}]
    prompt = build_user_prompt("issue", docs)
    assert "Real code symbols" not in prompt


def test_build_user_prompt_handles_mixed_candidate_set():
    docs = [
        {"name": "orders", "full_doc": "orders doc", "key_symbols": [], "domain": "E-commerce"},
        {"name": "shipping", "full_doc": "shipping doc", "key_symbols": [], "domain": "Shipping"},
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


def test_build_system_prompt_asks_for_reasoning_depth_now_that_diagrams_are_offloaded():
    prompt = build_system_prompt()
    assert "freed-up effort" in prompt.lower() or "freed up effort" in prompt.lower()
    assert "reasoning chain" in prompt.lower()


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
    generically ("processes the request") instead of what the actual
    retrieved snippet does — the prompt must force a literal trace first."""
    prompt = build_system_prompt()
    assert "trace" in prompt.lower()
    assert "quoting or closely paraphrasing" in prompt.lower()
    assert "processes the request" in prompt.lower()  # the explicit bad example


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
