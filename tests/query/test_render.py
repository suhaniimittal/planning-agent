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
from src.query.render import render_confluence, render_markdown, render_text


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


def _md(**overrides) -> str:
    return render_markdown(_tdd(**overrides))


def test_render_markdown_title_becomes_h1_heading():
    assert _md().startswith("# Fix order totals")


def test_render_markdown_service_becomes_h3_heading():
    assert "### orders" in _md()


def test_render_markdown_change_includes_file_path_symbol_and_description():
    md = _md()
    assert "`src/.../OrdersController.java`" in md
    assert "`calculateTotal`" in md
    assert "Fix summation of item prices" in md


def test_render_markdown_change_without_file_path_omits_backtick_path():
    tdd = _tdd()
    tdd.services[0].changes[0].file_path = None
    md = render_markdown(tdd)
    assert "`calculateTotal`" in md


def test_render_markdown_omits_empty_optional_sections():
    md = _md()
    assert "## 21. Risks" not in md
    assert "## 19. Testing Strategy" not in md
    assert "## 22. Open Questions" not in md
    assert "## 6. Root Cause" not in md
    assert "## 18. Backward Compatibility" not in md


def test_render_markdown_includes_populated_optional_sections():
    md = _md(
        risks=["Discount logic isn't described in the docs"],
        testing_notes="Add a multi-item cart test case",
        open_questions=["What happens with zero-quantity items?"],
    )
    assert "## 21. Risks" in md
    assert "Discount logic isn't described in the docs" in md
    assert "## 19. Testing Strategy" in md
    assert "Add a multi-item cart test case" in md
    assert "## 22. Open Questions" in md
    assert "What happens with zero-quantity items?" in md


def test_render_markdown_includes_overall_reasoning():
    md = _md()
    assert "## 25. Overall Reasoning" in md
    assert "The bug is in total calculation logic" in md


def test_render_markdown_omits_dependency_diagram_when_no_edges():
    md = _md()
    assert "## 15. Inter-Service Communication" not in md
    assert "```mermaid" not in md


def test_render_markdown_includes_mermaid_diagram_for_service_dependencies():
    md = _md(service_dependencies=[ServiceDependency(source="orders", target="shipping")])
    assert "## 15. Inter-Service Communication" in md
    assert "```mermaid" in md
    assert "flowchart LR" in md
    assert 'orders["orders"]' in md
    assert 'shipping["shipping"]' in md
    assert "orders --> shipping" in md


def test_render_markdown_renders_multiple_dependency_edges():
    md = _md(
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping"),
            ServiceDependency(source="orders", target="payment"),
        ]
    )
    assert "orders --> shipping" in md
    assert "orders --> payment" in md


def test_render_markdown_labels_kafka_based_dependency_edge():
    md = _md(
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping", via="kafka:orders.created")
        ]
    )
    assert "orders -- kafka:orders.created --> shipping" in md


def test_render_markdown_architecture_diagram_uses_mermaid_flowchart():
    md = _md(
        current_architecture="Orders calls payment directly.",
        current_architecture_components=[
            ArchComponent(name="Orders", depends_on=["Payment"]),
            ArchComponent(name="Payment"),
        ],
    )
    assert "## 4. Current System / Architecture" in md
    assert "```mermaid" in md
    assert "flowchart LR" in md
    assert 'Orders["Orders"]' in md
    assert 'Payment["Payment"]' in md
    assert "Orders --> Payment" in md


def test_render_markdown_architecture_diagram_sanitizes_ids_with_spaces_and_punctuation():
    md = _md(
        current_architecture_components=[
            ArchComponent(name="Order Service", depends_on=["Payment-Gateway (v2)"]),
        ],
    )
    assert 'Order_Service["Order Service"]' in md
    assert "Order_Service -->" in md


def test_render_markdown_flow_diagram_uses_mermaid_flowchart_td():
    md = _md(
        current_flow_steps=[
            FlowStep(step="Validate input", detail="Checks required fields are present"),
            FlowStep(step="Process request"),
        ]
    )
    assert "## 5. Current Flow" in md
    assert "flowchart TD" in md
    assert "Validate input" in md
    assert "Checks required fields are present" in md
    assert "s0 --> s1" in md


def test_render_markdown_sequence_diagram_uses_mermaid_sequence_diagram():
    md = _md(
        sequence_steps=[
            SequenceStep(from_participant="Client", to_participant="Orders", action="Create order"),
            SequenceStep(
                from_participant="Orders", to_participant="Client", action="Order confirmation", direction="response"
            ),
        ]
    )
    assert "## 10. Sequence / Interaction Flow" in md
    assert "sequenceDiagram" in md
    assert "Client->>Orders: Create order" in md
    assert "Orders-->>Client: Order confirmation" in md


def test_render_markdown_omits_diagram_sections_when_no_data():
    md = _md()
    assert "## 5. Current Flow" not in md
    assert "## 9. Proposed Flow" not in md
    assert "## 10. Sequence" not in md


def test_render_markdown_includes_implementation_notes():
    assert "Loop over items and sum unit price * quantity" in _md()


def test_render_markdown_includes_acceptance_criteria_when_present():
    tdd = _tdd()
    tdd.services[0].changes[0].acceptance_criteria = ["Total matches sum of item prices for a 3-item cart"]
    md = render_markdown(tdd)
    assert "Acceptance criteria" in md
    assert "Total matches sum of item prices for a 3-item cart" in md


def test_render_markdown_omits_acceptance_criteria_when_empty():
    assert "Acceptance criteria" not in _md()


def test_render_markdown_includes_complexity_and_data_model():
    tdd = _tdd()
    tdd.services[0].affected_data_model = "MongoDB: CustomerOrder"
    md = render_markdown(tdd)
    assert "Complexity: medium" in md
    assert "Data model: MongoDB: CustomerOrder" in md


def test_render_markdown_omits_data_model_line_when_none():
    assert "| Data model:" not in _md()


def test_render_markdown_omits_backward_compatibility_section_when_none():
    assert "## 18. Backward Compatibility" not in _md()


def test_render_markdown_includes_backward_compatibility_when_present():
    md = _md(backward_compatibility_notes="Order total field type is unchanged")
    assert "## 18. Backward Compatibility" in md
    assert "Order total field type is unchanged" in md


def test_render_markdown_includes_pseudocode_in_fenced_code_block():
    md = _md()
    assert "```" in md
    assert "total = 0" in md
    assert "total += item.price" in md


def test_render_markdown_labels_pseudocode_as_illustrative_not_real_code():
    assert "illustrative pseudocode, not real code" in _md()


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
    md = _md()
    assert "(grounded in real source)" not in md
    assert "illustrative pseudocode, not real code" in md


def test_render_markdown_omits_affected_services_section_when_no_services():
    tdd = _tdd(services=[])
    md = render_markdown(tdd)
    assert "## 11. Affected Services" not in md
    assert "## 12. Affected Files" not in md


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


def test_render_text_labels_kafka_based_dependency_edge():
    tdd = _tdd(
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping", via="kafka:orders.created")
        ]
    )
    text = render_text(tdd)
    assert "orders -> shipping (via kafka:orders.created)" in text


def test_render_text_includes_implementation_plan_as_ordered_list():
    text = _tdd_render(implementation_plan=["Fix the loop", "Add a test"])
    assert "1. Fix the loop" in text
    assert "2. Add a test" in text


def test_render_text_pseudocode_label_reflects_has_real_source():
    tdd = _tdd()
    tdd.services[0].changes[0].has_real_source = True
    text = render_text(tdd)
    assert "grounded in real source" in text


# --- render_confluence -------------------------------------------------


def _cf(**overrides) -> str:
    return render_confluence(_tdd(**overrides))


def test_render_confluence_issue_summary_is_a_real_heading_and_paragraph():
    cf = _cf()
    assert "<h2>2. Issue Summary</h2>" in cf
    assert "<p>Orders show wrong total price</p>" in cf


def test_render_confluence_change_includes_file_path_symbol_and_code_block():
    cf = _cf()
    assert "<code>src/.../OrdersController.java</code>" in cf
    assert "<code>calculateTotal</code>" in cf
    assert "Fix summation of item prices" in cf
    assert '<ac:structured-macro ac:name="code">' in cf
    assert "<![CDATA[total = 0\nfor item in items:\n    total += item.price]]>" in cf


def test_render_confluence_change_summary_is_a_real_table():
    cf = _cf()
    assert "<table><tbody>" in cf
    assert "<th>Service</th>" in cf
    assert "<td>orders</td>" in cf


def test_render_confluence_omits_empty_optional_sections():
    cf = _cf()
    assert "<h2>21. Risks</h2>" not in cf
    assert "<h2>19. Testing Strategy</h2>" not in cf
    assert "<h2>6. Root Cause" not in cf


def test_render_confluence_drops_architecture_component_diagram():
    cf = _cf(
        current_architecture="Orders talks to shipping directly.",
        current_architecture_components=[ArchComponent(name="orders", depends_on=["shipping"])],
    )
    # the prose survives, but the component-diagram's own node/edge markup
    # (the bracketed mermaid node syntax render_markdown would produce) never
    # appears — render_confluence only ever uses `current_architecture` text
    assert "Orders talks to shipping directly." in cf
    assert '["orders"]' not in cf
    assert "-->" not in cf


def test_render_confluence_flow_steps_render_as_ordered_list_no_arrow_chain():
    cf = _cf(current_flow_steps=[FlowStep(step="Validate input"), FlowStep(step="Save order")])
    assert "<ol>" in cf
    assert "<li>Validate input</li>" in cf
    assert " --> " not in cf
    assert " | \n   v" not in cf


def test_render_confluence_sequence_steps_render_as_list_no_mermaid():
    cf = _cf(
        sequence_steps=[
            SequenceStep(from_participant="Client", to_participant="Orders", action="POST /orders", direction="call")
        ]
    )
    assert "```mermaid" not in cf
    assert "Client" in cf and "Orders" in cf and "POST /orders" in cf


def test_render_confluence_labels_kafka_based_dependency_edge():
    cf = _cf(
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping", via="kafka:orders.created")
        ]
    )
    assert "orders → shipping (via kafka:orders.created)" in cf


def test_render_confluence_escapes_html_special_characters():
    cf = _cf(issue_summary="Totals break when amount < 0 & currency == \"USD\"")
    assert "&lt;" in cf
    assert "&amp;" in cf
    assert "amount < 0" not in cf


def test_render_confluence_includes_overall_reasoning():
    cf = _cf()
    assert "<h2>25. Overall Reasoning</h2>" in cf
    assert "The bug is in total calculation logic" in cf


def test_change_blocks_show_the_code_check_only_when_verified():
    from src.query.models import FileChange, ServicePlan, TechnicalDesignDoc
    from src.query.render import render_confluence, render_text

    def tdd(verification, note):
        change = FileChange(
            file_path="src/a.tsx",
            function_or_symbol="fn",
            change_description="d",
            implementation_notes="i",
            pseudocode_sketch="p",
            reasoning="r",
            verification=verification,
            verification_note=note,
        )
        plan = ServicePlan(service="ui", complexity="low", reasoning="r", changes=[change])
        return TechnicalDesignDoc(title="t", issue_summary="s", overall_reasoning="r", services=[plan])

    flagged = tdd("unverified", "`fn` was not found in `ui`.")
    assert "Code check: UNVERIFIED - check manually — `fn` was not found in `ui`." in render_text(flagged)
    assert "Code check: UNVERIFIED" in render_confluence(flagged)
    assert "Code check" not in render_text(tdd(None, None))
