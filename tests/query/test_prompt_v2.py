import json

from src.query import context_v2, planner, prompt_v2
from src.query.models import TechnicalDesignDoc


def test_the_system_prompt_is_a_third_of_v1():
    assert len(prompt_v2.build_system_prompt("bug")) < len(planner.build_system_prompt("bug")) / 2.5


def test_optional_sections_appear_only_when_asked():
    plain = prompt_v2.build_system_prompt("bug")
    full = prompt_v2.build_system_prompt("bug", value_trace=True, stored_data=True, test_writing=True)

    for marker in ("Accepted at:", "STORED DATA.", "seed it in the setup"):
        assert marker not in plain
        assert marker in full


def test_feature_mode_swaps_scope_and_analysis_and_never_traces_a_value():
    feature = prompt_v2.build_system_prompt("feature", value_trace=True)

    assert "BUILD WHAT THE TICKET SPECIFIES" in feature and "Exists:" in feature
    assert "ONLY WHAT THE TICKET REPORTS" not in feature
    assert "Accepted at:" not in feature


def test_tests_are_decided_per_change_and_none_is_allowed():
    prompt = prompt_v2.build_system_prompt("bug")

    assert '"NO TEST" IS A NORMAL ANSWER' in prompt
    assert "`Tests: <decision> — <reason>`" in prompt
    assert "may be empty" in prompt
    # v1's rules that forced test work on every change are gone.
    assert "Every change you propose maps to at least one scenario" not in prompt
    assert "At least one entry is a `regression` scenario" not in prompt


def test_the_schema_the_model_sees_leaves_out_fields_code_fills_in():
    schema = json.loads(prompt_v2.llm_schema())
    change = schema["$defs"]["FileChange"]["properties"]

    for name in ("service_dependencies", "sequence_steps", "generated_at",
                 "current_architecture_components"):
        assert name not in schema["properties"]
    for name in ("has_real_source", "verification", "verification_note"):
        assert name not in change
    # What the model returns still parses, the omitted fields defaulted.
    tdd = TechnicalDesignDoc.model_validate(
        {"title": "t", "issue_summary": "s", "services": [], "overall_reasoning": "r"}
    )
    assert tdd.service_dependencies == [] and tdd.sequence_steps == []


def test_user_prompt_marks_files_and_gives_every_service_a_section():
    context = context_v2.Context(
        files=[
            context_v2.ShownFile(
                service="ui", path="a/view.tsx", tier="focus", reason="best match", text="code",
                line_count=1, symbols=[{"name": "View", "qualified_name": "a/view.tsx::View",
                                        "label": "Function", "symbol_kind": "function",
                                        "line_start": 1, "line_end": 1}],
                tested_by=["a/view.test.tsx"],
            ),
            context_v2.ShownFile(service="ui", path="a/big.ts", tier="import",
                                 reason="imported by a/view.tsx", status="OUTLINE", line_count=900),
        ],
        tables=[{"service": "reports-api", "name": "reports.reports", "columns": ["id uuid not null"],
                 "foreign_keys": [], "defined_in": ["001.sql"], "used_by": ["R.kt reads"]}],
        used=4,
    )
    prompt = prompt_v2.build_user_prompt(
        "Controls shift.",
        [{"name": "ui", "summary": "The UI."}, {"name": "kafka", "summary": "x" * 5000}],
        context,
        [{"source": "ui", "target": "reports-api", "via": ["graphql"], "evidence": ["Query.reports"]}],
    )

    assert "- ui -> reports-api via graphql (Query.reports)" in prompt
    assert "### ui (code shown below)" in prompt
    assert "### kafka (summary only, no code shown)" in prompt and "[… summary cut here]" in prompt
    assert "### ui · a/view.tsx · WHOLE · 1 lines" in prompt
    assert "Symbols: View (function, 1–1)" in prompt
    assert "Tested by: a/view.test.tsx" in prompt
    assert "### ui · a/big.ts · OUTLINE · 900 lines" in prompt and "(Not shown" in prompt
    assert "Columns: id uuid not null" in prompt


def test_the_ticket_points_are_numbered_for_the_model_to_answer():
    prompt = prompt_v2.build_user_prompt(
        "i", [{"name": "ui", "summary": "s"}], context_v2.Context(),
        points=["White space is added.", "Controls shift down."],
    )
    schema = json.loads(prompt_v2.llm_schema())

    assert "## Ticket points (answer each in `ticket_coverage`)\n1. White space is added.\n" \
        "2. Controls shift down." in prompt
    # The model answers the points; it does not get to restate them.
    assert "ticket_coverage" in schema["properties"]
    assert "ticket_points" not in schema["properties"]
    assert "`ticket_coverage`: one entry per numbered point" in prompt_v2.build_system_prompt("bug")


def test_a_file_block_lists_who_imports_it():
    context = context_v2.Context(files=[context_v2.ShownFile(
        service="ui", path="grid/view.tsx", tier="import", reason="imported", text="x",
        line_count=1, used_by={"count": 3, "files": ["a.tsx", "b.tsx"]},
    )])

    prompt = prompt_v2.build_user_prompt("i", [{"name": "ui", "summary": "s"}], context)

    assert "Used by (3): a.tsx, b.tsx, … 1 more" in prompt


def test_each_change_traces_its_edge_cases_and_none_is_not_an_easy_test_answer():
    prompt = prompt_v2.build_system_prompt("bug")
    change = json.loads(prompt_v2.llm_schema())["$defs"]["FileChange"]["properties"]

    assert "edge_cases" in change
    assert "`edge_cases`: trace the change through every state" in prompt
    assert 'Before you choose "none", name what the change makes observable' in prompt
    assert "CSS layout under jsdom, timing). Put" not in prompt  # no longer a reason for none


def test_the_re_check_asks_for_the_slips_structural_checks_miss():
    for check in ("does exactly what its `change_description`", "traced through the real lines",
                  'is not "fixed"', "checked with a lookup", "`Tests: none`",
                  '{"unchanged": true}'):
        assert check in prompt_v2.RECHECK


def test_without_new_tests_existing_ones_are_kept_passing_and_new_ones_recommended():
    prompt = prompt_v2.build_system_prompt("bug", new_tests=False)

    assert "NEW TESTS ARE RECOMMENDED, NOT WRITTEN" in prompt
    assert "1. update existing:" in prompt
    assert '"Recommended tests (not part of this change):"' in prompt
    assert "3. new file:" not in prompt and "2. add case:" not in prompt
