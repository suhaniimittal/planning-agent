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


# --- render_text (the plain-text TDD behind the PDF) ------------------------


def _tdd_render(**overrides) -> str:
    return render_text(_tdd(**overrides))


def test_render_text_includes_title_and_summary():
    text = render_text(_tdd())
    assert "Fix order totals" in text
    assert "1. SUMMARY" in text
    assert "Orders show wrong total price" in text


def test_render_text_numbers_sections_consecutively_and_skips_empty_ones():
    text = _tdd_render(functional_requirements=["Must not double-charge"])
    headings = [line for line in text.splitlines() if line[:1].isdigit() and ". " in line and line.isupper()]
    numbers = [int(h.split(".")[0]) for h in headings]
    assert numbers == list(range(1, len(numbers) + 1))
    for absent in ("ROOT CAUSE", "SECURITY", "TESTING", "RISKS"):
        assert absent not in text


def test_render_text_shows_each_change_as_one_block_with_file_and_check():
    tdd = _tdd()
    change = tdd.services[0].changes[0]
    change.verification = "verified"
    change.verification_note = "`calculateTotal` found in `src/.../OrdersController.java`."
    text = render_text(tdd)
    assert "SERVICES AND CHANGES" in text
    assert "orders  (complexity: medium)" in text
    assert "Change 1: calculateTotal  (modify)" in text
    assert "    File: src/.../OrdersController.java" in text
    assert "    Code check: VERIFIED\n" in text  # no note repeating the file
    assert "    Change: Fix summation of item prices" in text


def test_render_text_says_why_a_service_needs_no_change():
    tdd = _tdd()
    tdd.services.append(ServicePlan(service="payments", complexity="low", reasoning="Never reached."))
    text = render_text(tdd)
    assert "payments — no change\n    Never reached." in text


def test_render_text_lists_files_to_change_per_service_without_tables():
    tdd = _tdd()
    tdd.services[0].changes[0].change_type = "create"
    text = render_text(tdd)
    files = text.split("FILES TO CHANGE")[1]
    assert "orders\n  - create: src/.../OrdersController.java" in files
    assert "------" not in text  # no table rulers anywhere


def test_render_text_flow_shows_today_and_after():
    text = _tdd_render(
        current_flow_steps=[FlowStep(step="Validate input", detail="Checks required fields")],
        proposed_flow_steps=[FlowStep(step="Trim input"), FlowStep(step="Validate input")],
    )
    flow = text.split("FLOW")[1]
    assert "Today:\n1. Validate input\n   Checks required fields" in flow
    assert "After the change:\n1. Trim input\n2. Validate input" in flow


def test_render_text_architecture_uses_proposed_components_and_real_dependencies():
    text = _tdd_render(
        proposed_architecture="Orders calls payment directly.",
        proposed_architecture_components=[
            ArchComponent(name="Orders", depends_on=["Payment"]),
            ArchComponent(name="Payment"),
        ],
        service_dependencies=[
            ServiceDependency(source="orders", target="shipping", via="kafka:orders.created")
        ],
    )
    # One merged list, each edge carrying its provenance. The previous version
    # printed the model's components and the graph's verified edges as two
    # separate blocks of arrows, with nothing telling the reader which to
    # believe, and rendered a component that calls nothing as
    # "(no further dependencies)" — which reads like a failure, not a fact.
    assert "Components:" in text
    assert "Orders, Payment, orders, shipping" in text
    assert "Orders --> Payment" in text
    assert "(proposed by this design)" in text
    assert "orders --> shipping" in text
    assert "(verified in the code graph, via kafka:orders.created)" in text
    assert "no further dependencies" not in text


def test_render_text_testing_is_a_list_not_a_table():
    text = _tdd_render(
        testing_plan=[Scenario(category="unit", scenario="Empty cart", expected_result="Rejected with 400")]
    )
    assert "- [unit] Empty cart\n    Expected: Rejected with 400" in text


def test_render_text_includes_implementation_plan_as_ordered_list():
    text = _tdd_render(implementation_plan=["Fix the loop", "Add a test"])
    assert "1. Fix the loop" in text
    assert "2. Add a test" in text


def test_render_text_sketch_label_reflects_has_real_source():
    tdd = _tdd()
    assert "Sketch (illustrative, not real code):" in render_text(tdd)
    tdd.services[0].changes[0].has_real_source = True
    assert "    Sketch:" in render_text(tdd)


def test_render_text_leaves_out_restated_sections():
    text = _tdd_render(
        sequence_steps=[SequenceStep(from_participant="A", to_participant="B", action="call")],
        current_architecture="restated architecture prose",
    )
    for gone in ("SEQUENCE", "CHANGE SUMMARY", "OVERALL REASONING", "restated architecture prose"):
        assert gone not in text


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


def test_render_text_puts_each_root_cause_finding_on_its_own_line():
    text = _tdd_render(
        root_cause_analysis="Confirmed: the regex is anchored — a.ts. Likely: input not trimmed. "
        "Not the cause: the server — never reached."
    )
    cause = text.split("ROOT CAUSE")[1]
    assert "\n\nLikely: input not trimmed." in cause
    assert "\n\nNot the cause: the server" in cause


def test_render_text_keeps_the_note_for_a_corrected_change():
    tdd = _tdd()
    change = tdd.services[0].changes[0]
    change.verification = "corrected"
    change.verification_note = "Path corrected: `calculateTotal` is in `B.java`, not `A.java`."
    assert "Code check: CORRECTED — Path corrected" in render_text(tdd)


def _reference_change() -> FileChange:
    return FileChange(
        file_path="src/.../ReportStatus.kt",
        function_or_symbol="ReportStatus",
        change_description="New code reads this enum to decide printability",
        implementation_notes="This file is NOT edited.",
        pseudocode_sketch="",
        reasoning="The caller needs the existing statuses",
        change_type="reference",
    )


def test_render_text_keeps_reference_files_out_of_files_to_change():
    """A "reference" change is a symbol the design reads without editing it.
    Listing it under FILES TO CHANGE sent coding_agent to open files the TDD
    itself said not to touch."""
    tdd = _tdd()
    tdd.services[0].changes.append(_reference_change())
    text = render_text(tdd)
    files = text.split("FILES TO CHANGE")[1].split("FILES READ BUT NOT CHANGED")[0]
    assert "ReportStatus.kt" not in files
    assert "OrdersController.java" in files


def test_render_text_lists_reference_files_in_their_own_section():
    tdd = _tdd()
    tdd.services[0].changes.append(_reference_change())
    text = render_text(tdd)
    read_only = text.split("FILES READ BUT NOT CHANGED")[1]
    assert "orders\n  - src/.../ReportStatus.kt" in read_only


def test_render_text_omits_the_read_only_section_when_nothing_is_referenced():
    assert "FILES READ BUT NOT CHANGED" not in render_text(_tdd())


def test_render_text_spells_out_that_a_reference_is_not_a_change():
    tdd = _tdd()
    tdd.services[0].changes.append(_reference_change())
    assert "ReportStatus  (reference (read, not changed))" in render_text(tdd)


def test_render_markdown_and_confluence_separate_reference_files():
    tdd = _tdd()
    tdd.services[0].changes.append(_reference_change())
    for body in (render_markdown(tdd), render_confluence(tdd)):
        files = body.split("Files to Create / Modify / Delete")[1]
        create_to_delete = files.split("Read but NOT changed")[0]
        assert "ReportStatus.kt" not in create_to_delete
        assert "Read but NOT changed" in files
        assert "ReportStatus.kt" in files.split("Read but NOT changed")[1]


def test_root_cause_splits_every_label_the_prompts_emit():
    """These labels arrive inside one string field, often as a run-on
    paragraph; a label the regex doesn't know renders as a wall of text."""
    trace = (
        "Accepted at: ui/view.tsx:onSubmit - read raw. "
        "Validated at: ui/schema.ts:phone - anchored regex. "
        "Transformed at: not shown in the evidence. "
        "Stored at: cs/X.kt:setPhone - written as-is. "
        "Consumed at: cs/Y.kt:read - expects trimmed. "
        "Confirmed: leading space - ui/schema.ts. "
        "Hypothesis: paste adds it - unproven. "
        "Exists: a validator - ui/schema.ts. "
        "Missing: trimming - no shown code does it. "
        "Extend: ui/schema.ts - the existing validator. "
        "Out of scope: server-side rules - the ticket excludes them. "
        "Unknown: where tests live. "
        "Not the cause: the server - it never receives the value."
    )
    text = render_text(_tdd(root_cause_analysis=trace))
    for label in (
        "Accepted at:",
        "Validated at:",
        "Transformed at:",
        "Stored at:",
        "Consumed at:",
        "Confirmed:",
        "Hypothesis:",
        "Exists:",
        "Missing:",
        "Extend:",
        "Out of scope:",
        "Unknown:",
        "Not the cause:",
    ):
        assert f"\n{label}" in text, label
