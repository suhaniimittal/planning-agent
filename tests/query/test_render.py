from src.query.models import (
    ArchComponent,
    FileChange,
    FlowStep,
    Scenario,
    SequenceStep,
    ServiceDependency,
    ServicePlan,
    TechnicalDesignDoc,
)
from src.query.render import render_markdown, render_text


def _tdd(**overrides) -> TechnicalDesignDoc:
    defaults = dict(
        title="Fix order totals",
        issue_summary="Orders show wrong total price",
        services=[
            ServicePlan(
                service="orders",
                changes=[
                    FileChange(
                        file_path="src/.../OrdersController.java",
                        function_or_symbol="calculateTotal",
                        change_description="Fix summation of item prices",
                        implementation_notes="Loop over items and sum unit price * quantity",
                        pseudocode_sketch="total = 0\nfor item in items:\n    total += item.price",
                        reasoning="This is where totals are computed",
                    )
                ],
                complexity="medium",
                reasoning="Orders owns total calculation",
            )
        ],
        overall_reasoning="The bug is in total calculation logic",
    )
    defaults.update(overrides)
    return TechnicalDesignDoc(**defaults)


def test_render_markdown_title_becomes_h1_heading():
    md = render_markdown(_tdd())
    assert md.startswith("# Fix order totals")


def test_render_markdown_service_becomes_h3_heading():
    md = render_markdown(_tdd())
    assert "### orders" in md


def test_render_markdown_change_with_file_path_renders_inline():
    md = render_markdown(_tdd())
    assert "`src/.../OrdersController.java`" in md
    assert "**calculateTotal**" in md
    assert "Fix summation of item prices" in md


def test_render_markdown_change_without_file_path_omits_backticks():
    tdd = _tdd()
    tdd.services[0].changes[0].file_path = None
    md = render_markdown(tdd)
    assert "None" not in md
    assert "**calculateTotal**" in md


def test_render_markdown_omits_empty_risks_testing_open_questions_sections():
    md = render_markdown(_tdd())
    assert "## Risks" not in md
    assert "## Testing Notes" not in md
    assert "## Open Questions" not in md


def test_render_markdown_includes_populated_optional_sections():
    tdd = _tdd(
        risks=["Discount logic isn't described in the docs"],
        testing_notes="Add a multi-item cart test case",
        open_questions=["What happens with zero-quantity items?"],
    )
    md = render_markdown(tdd)
    assert "## Risks" in md
    assert "Discount logic isn't described in the docs" in md
    assert "## Testing Notes" in md
    assert "Add a multi-item cart test case" in md
    assert "## Open Questions" in md
    assert "What happens with zero-quantity items?" in md


def test_render_markdown_includes_overall_reasoning():
    md = render_markdown(_tdd())
    assert "## Overall Reasoning" in md
    assert "The bug is in total calculation logic" in md


def test_render_markdown_omits_dependency_diagram_when_no_edges():
    md = render_markdown(_tdd())
    assert "## Service Dependencies" not in md
    assert "```mermaid" not in md


def test_render_markdown_includes_mermaid_diagram_for_edges():
    tdd = _tdd(
        service_dependencies=[ServiceDependency(source="orders", target="shipping")]
    )
    md = render_markdown(tdd)
    assert "## Service Dependencies" in md
    assert "```mermaid" in md
    assert "flowchart LR" in md
    assert "orders --> shipping" in md


def test_render_markdown_renders_multiple_edges():
    tdd = _tdd(
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping"),
            ServiceDependency(source="orders", target="payment"),
        ]
    )
    md = render_markdown(tdd)
    assert "orders --> shipping" in md
    assert "orders --> payment" in md


def test_render_markdown_includes_implementation_notes():
    md = render_markdown(_tdd())
    assert "Loop over items and sum unit price * quantity" in md


def test_render_markdown_includes_acceptance_criteria_when_present():
    tdd = _tdd()
    tdd.services[0].changes[0].acceptance_criteria = [
        "Total matches sum of item prices for a 3-item cart"
    ]
    md = render_markdown(tdd)
    assert "Acceptance criteria" in md
    assert "Total matches sum of item prices for a 3-item cart" in md


def test_render_markdown_omits_acceptance_criteria_when_empty():
    md = render_markdown(_tdd())
    assert "Acceptance criteria" not in md


def test_render_markdown_includes_complexity_and_data_model():
    tdd = _tdd()
    tdd.services[0].affected_data_model = "MongoDB: CustomerOrder"
    md = render_markdown(tdd)
    assert "Complexity: medium" in md
    assert "Data model: MongoDB: CustomerOrder" in md


def test_render_markdown_omits_data_model_when_none():
    md = render_markdown(_tdd())
    assert "Data model:" not in md


def test_render_markdown_omits_backward_compatibility_section_when_none():
    md = render_markdown(_tdd())
    assert "## Backward Compatibility" not in md


def test_render_markdown_includes_backward_compatibility_when_present():
    tdd = _tdd(backward_compatibility_notes="Order total field type is unchanged")
    md = render_markdown(tdd)
    assert "## Backward Compatibility" in md
    assert "Order total field type is unchanged" in md


def test_render_markdown_includes_pseudocode_in_fenced_code_block():
    md = render_markdown(_tdd())
    assert "```" in md
    assert "total = 0" in md
    assert "total += item.price" in md


def test_render_markdown_labels_pseudocode_as_illustrative_not_real_code():
    md = render_markdown(_tdd())
    assert "illustrative pseudocode, not real code" in md


def test_render_markdown_renders_all_pseudocode_lines():
    tdd = _tdd()
    tdd.services[0].changes[0].pseudocode_sketch = "line1\nline2\nline3"
    md = render_markdown(tdd)
    assert "line1" in md
    assert "line2" in md
    assert "line3" in md


def test_render_markdown_tags_changes_with_real_source():
    tdd = _tdd()
    tdd.services[0].changes[0].has_real_source = True
    md = render_markdown(tdd)
    assert "(grounded in real source)" in md
    assert "may reflect real source shown to the model" in md


def test_render_markdown_omits_real_source_tag_when_false():
    md = render_markdown(_tdd())
    assert "grounded in real source" not in md
    assert "illustrative pseudocode, not real code" in md


# --- render_text (the rich, section-numbered .txt renderer) ------------------


def test_render_text_includes_title_and_issue_summary():
    text = render_text(_tdd())
    assert "Fix order totals" in text
    assert "Orders show wrong total price" in text


def test_render_text_omits_empty_sections():
    text = _tdd_render()
    # Sections with no supporting data (root cause, sequence diagram, etc.)
    # must be omitted entirely, not printed with an empty body.
    assert "SEQUENCE / INTERACTION FLOW" not in text
    assert "ROOT CAUSE" not in text
    assert "SECURITY CONSIDERATIONS" not in text


def _tdd_render(**overrides) -> str:
    return render_text(_tdd(**overrides))


def test_render_text_builds_architecture_diagram_from_components():
    """The diagram is BUILT by our code from stated facts, not hand-drawn
    by the LLM — this is the whole point of the structured-facts restructure."""
    text = _tdd_render(
        current_architecture="Orders calls payment directly.",
        current_architecture_components=[
            ArchComponent(name="Orders", depends_on=["Payment"]),
            ArchComponent(name="Payment"),
        ],
    )
    assert "CURRENT SYSTEM / ARCHITECTURE" in text
    assert "[Orders]" in text
    assert "--> [Payment]" in text
    assert "(no further dependencies)" in text


def test_render_text_builds_flow_diagram_from_steps():
    text = _tdd_render(
        current_flow_steps=[
            FlowStep(step="Validate input", detail="Checks required fields are present"),
            FlowStep(step="Process request"),
        ]
    )
    assert "CURRENT FLOW" in text
    assert "1. Validate input" in text
    assert "Checks required fields are present" in text
    assert "2. Process request" in text
    assert "Validate input" in text.split("Flow:")[1]


def test_render_text_builds_sequence_diagram_from_steps():
    text = _tdd_render(
        sequence_steps=[
            SequenceStep(from_participant="Client", to_participant="Orders", action="Create order"),
            SequenceStep(
                from_participant="Orders",
                to_participant="Client",
                action="Order confirmation",
                direction="response",
            ),
        ]
    )
    assert "SEQUENCE / INTERACTION FLOW" in text
    assert "Client --- Create order --> Orders" in text
    assert "Orders <-- Order confirmation --- Client" in text


def test_render_text_omits_diagrams_when_no_components_or_steps_given():
    text = _tdd_render()
    assert "CURRENT FLOW" not in text
    assert "SEQUENCE / INTERACTION FLOW" not in text


def test_render_text_includes_requirements_with_not_specified_fallback():
    text = _tdd_render(functional_requirements=["Must not double-charge"])
    assert "Must not double-charge" in text
    assert "Not specified." in text  # non-functional requirements, left empty


def test_render_text_renders_testing_plan_as_table():
    text = _tdd_render(
        testing_plan=[
            Scenario(category="unit", scenario="Empty cart", expected_result="Rejected with 400")
        ]
    )
    assert "TESTING STRATEGY" in text
    assert "Category" in text and "Scenario" in text and "Expected Result" in text
    assert "Empty cart" in text
    assert "Rejected with 400" in text


def test_render_text_change_summary_table_derived_from_changes():
    text = _tdd_render()
    assert "CHANGE SUMMARY" in text
    assert "calculateTotal" in text
    assert "src/.../OrdersController.java" in text


def test_render_text_groups_files_by_change_type():
    tdd = _tdd()
    tdd.services[0].changes[0].change_type = "create"
    text = render_text(tdd)
    assert "Files to Create:" in text
    assert "src/.../OrdersController.java" in text.split("Files to Create:")[1].split("Files to Modify:")[0]


def test_render_text_includes_real_service_dependencies_not_llm_claims():
    tdd = _tdd(service_dependencies=[ServiceDependency(source="orders", target="shipping")])
    text = render_text(tdd)
    assert "INTER-SERVICE COMMUNICATION" in text
    assert "orders -> shipping" in text


def test_render_text_includes_implementation_plan_as_ordered_list():
    text = _tdd_render(implementation_plan=["Fix the loop", "Add a test"])
    assert "1. Fix the loop" in text
    assert "2. Add a test" in text


def test_render_text_pseudocode_label_reflects_has_real_source():
    tdd = _tdd()
    tdd.services[0].changes[0].has_real_source = True
    text = render_text(tdd)
    assert "grounded in real source" in text
