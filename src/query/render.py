"""Renders a TechnicalDesignDoc as a readable Markdown document. Pure
function, no I/O — the structured TechnicalDesignDoc object itself remains
available unchanged for a coding agent that wants to consume it
programmatically; this is purely an additive, human-readable form.
"""

from __future__ import annotations

from .models import TechnicalDesignDoc


def _service_dependency_diagram(tdd: TechnicalDesignDoc) -> str | None:
    """A Mermaid flowchart of the real CALLS edges among the candidate
    services — grounded in graph data (query_flow.build_tdd fills this in
    after parsing), never guessed by the LLM. None if there are no edges."""
    if not tdd.service_dependencies:
        return None
    lines = ["```mermaid", "flowchart LR"]
    for e in tdd.service_dependencies:
        lines.append(f"    {e.source} --> {e.target}")
    lines.append("```")
    return "\n".join(lines)


def render_markdown(tdd: TechnicalDesignDoc) -> str:
    lines = [f"# {tdd.title}", "", f"**Issue:** {tdd.issue_summary}", ""]

    diagram = _service_dependency_diagram(tdd)
    if diagram:
        lines += ["## Service Dependencies", "", diagram, ""]

    lines += ["## Affected Services", ""]
    for s in tdd.services:
        data_model = f" | Data model: {s.affected_data_model}" if s.affected_data_model else ""
        lines += [
            f"### {s.service}",
            "",
            f"Complexity: {s.complexity}{data_model}",
            "",
            s.reasoning,
            "",
        ]
        for c in s.changes:
            loc = f"`{c.file_path}` — " if c.file_path else ""
            source_tag = " _(grounded in real source)_" if c.has_real_source else ""
            pseudocode_label = (
                "  - Proposed approach (may reflect real source shown to the model):"
                if c.has_real_source
                else "  - Proposed approach (illustrative pseudocode, not real code):"
            )
            lines += [
                f"- {loc}**{c.function_or_symbol}**{source_tag}",
                f"  - Change: {c.change_description}",
                f"  - Implementation: {c.implementation_notes}",
                pseudocode_label,
                "    ```",
                *[f"    {line}" for line in c.pseudocode_sketch.splitlines()],
                "    ```",
                f"  - Why: {c.reasoning}",
            ]
            if c.acceptance_criteria:
                lines.append("  - Acceptance criteria:")
                lines += [f"    - {ac}" for ac in c.acceptance_criteria]
            lines.append("")

    if tdd.backward_compatibility_notes:
        lines += ["## Backward Compatibility", "", tdd.backward_compatibility_notes, ""]

    if tdd.risks:
        lines += ["## Risks", ""] + [f"- {r}" for r in tdd.risks] + [""]

    if tdd.testing_notes:
        lines += ["## Testing Notes", "", tdd.testing_notes, ""]

    if tdd.open_questions:
        lines += ["## Open Questions", ""] + [f"- {q}" for q in tdd.open_questions] + [""]

    lines += ["## Overall Reasoning", "", tdd.overall_reasoning]

    return "\n".join(lines)


# --- Plain-text renderer: the full, section-numbered TDD -------------------


def _txt_section(number: int, title: str, *body_parts: str | None) -> str | None:
    """One numbered section, or None if every part is empty — omitted
    entirely rather than printed with nothing under it."""
    parts = [p.strip() for p in body_parts if p and p.strip()]
    if not parts:
        return None
    heading = f"{number}. {title.upper()}"
    return heading + "\n" + ("=" * len(heading)) + "\n\n" + "\n\n".join(parts)


def _txt_bullets(items: list[str]) -> str | None:
    if not items:
        return None
    return "\n".join(f"- {i}" for i in items)


def _txt_table(headers: list[str], rows: list[list[str]]) -> str | None:
    """A simple aligned plain-text table — no markdown pipes, readable in
    any plain .txt viewer."""
    if not rows:
        return None
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))
    ]
    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))
    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    lines += [_fmt_row(row) for row in rows]
    return "\n".join(lines)


def _ascii_architecture_diagram(components) -> str | None:
    """Built from stated facts (name + what it depends on), never hand-
    drawn by the LLM. Deliberately a simple per-component dependency
    listing rather than a true 2D box layout — an arbitrary dependency
    graph (multiple components each depending on several others) doesn't
    have one canonical box arrangement, and a listing renders correctly
    for ANY graph shape without needing layout logic that could get it
    wrong, which was exactly the reliability problem this replaced."""
    if not components:
        return None
    blocks = []
    for c in components:
        if c.depends_on:
            lines = [f"[{c.name}]"] + [f"   --> [{dep}]" for dep in c.depends_on]
        else:
            lines = [f"[{c.name}]", "   (no further dependencies)"]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _ascii_flow_diagram(steps) -> str | None:
    """A numbered breakdown (with each step's optional detail) plus a
    simple vertical arrow chain — both built from the same ordered list of
    facts, so nothing is stated twice in two different shapes."""
    if not steps:
        return None
    numbered = []
    for i, s in enumerate(steps):
        line = f"{i + 1}. {s.step}"
        if s.detail:
            line += f"\n   {s.detail}"
        numbered.append(line)
    chain = "\n   |\n   v\n".join(s.step for s in steps)
    return "\n".join(numbered) + "\n\nFlow:\n" + chain


def _ascii_sequence_diagram(steps) -> str | None:
    """One line per interaction (`A --- action --> B` for a call, `A <--
    action --- B` for a response) — deliberately not a column-aligned
    multi-participant table. True column alignment gets fragile fast with
    more than 2-3 participants (exactly the kind of hand-drawn-diagram
    unreliability this whole restructure was meant to remove); a plain
    per-line arrow is unambiguous and always renders correctly regardless
    of how many participants are involved."""
    if not steps:
        return None
    lines = []
    for s in steps:
        if s.direction == "response":
            lines.append(f"{s.from_participant} <-- {s.action} --- {s.to_participant}")
        else:
            lines.append(f"{s.from_participant} --- {s.action} --> {s.to_participant}")
    return "\n".join(lines)


def _txt_change_block(c) -> str:
    lines = [
        f"Symbol: {c.function_or_symbol}",
        f"File: {c.file_path or 'Unknown'}",
        f"Change type: {c.change_type}",
    ]
    if c.current_behavior:
        lines.append(f"Current behavior: {c.current_behavior}")
    lines += [
        f"Change: {c.change_description}",
        f"Reason: {c.reasoning}",
        f"Implementation: {c.implementation_notes}",
    ]
    source_tag = "grounded in real source" if c.has_real_source else "illustrative only, not verified code"
    lines.append(f"Pseudocode ({source_tag}):")
    lines += ["  " + line for line in c.pseudocode_sketch.splitlines()]
    if c.acceptance_criteria:
        lines.append("Acceptance criteria:")
        lines += [f"  - {ac}" for ac in c.acceptance_criteria]
    return "\n".join(lines)


def render_text(tdd: TechnicalDesignDoc) -> str:
    """The full, section-numbered plain-text TDD — meant to be saved
    directly as a `.txt` file and read start to finish by a developer or a
    coding agent, without re-analyzing the repository. Every section is
    built from the structured TechnicalDesignDoc; nothing here is invented
    by the renderer — a section with no supporting data is simply omitted,
    same "say Unknown rather than guess" principle the LLM itself follows.
    """
    header = [
        "TECHNICAL DESIGN DOCUMENT",
        tdd.title,
    ]
    if tdd.generated_at:
        header.append(f"Generated: {tdd.generated_at}")
    top = "\n".join(header) + "\n\n" + ("#" * 60)

    sections: list[str | None] = [top]

    sections.append(_txt_section(2, "Issue Summary", tdd.issue_summary))

    sections.append(
        _txt_section(
            3,
            "Requirements",
            "Functional Requirements:\n"
            + (_txt_bullets(tdd.functional_requirements) or "Not specified."),
            "Non-Functional Requirements:\n"
            + (_txt_bullets(tdd.non_functional_requirements) or "Not specified."),
        )
    )

    current_arch_diagram = _ascii_architecture_diagram(tdd.current_architecture_components)
    sections.append(
        _txt_section(
            4,
            "Current System / Architecture",
            tdd.current_architecture,
            f"Diagram:\n{current_arch_diagram}" if current_arch_diagram else None,
        )
    )

    current_flow_diagram = _ascii_flow_diagram(tdd.current_flow_steps)
    sections.append(_txt_section(5, "Current Flow", current_flow_diagram))

    sections.append(_txt_section(6, "Root Cause / Problem Analysis", tdd.root_cause_analysis))

    sections.append(_txt_section(7, "Proposed Solution", tdd.proposed_solution))

    proposed_arch_diagram = _ascii_architecture_diagram(tdd.proposed_architecture_components)
    sections.append(
        _txt_section(
            8,
            "Proposed Architecture",
            tdd.proposed_architecture,
            f"Diagram:\n{proposed_arch_diagram}" if proposed_arch_diagram else None,
        )
    )

    proposed_flow_diagram = _ascii_flow_diagram(tdd.proposed_flow_steps)
    sections.append(_txt_section(9, "Proposed Flow", proposed_flow_diagram))

    sequence_diagram = _ascii_sequence_diagram(tdd.sequence_steps)
    sections.append(_txt_section(10, "Sequence / Interaction Flow", sequence_diagram))

    service_blocks = []
    for s in tdd.services:
        block = [f"Service: {s.service}", f"Complexity: {s.complexity}"]
        if s.responsibility_change:
            block.append(f"Responsibility change: {s.responsibility_change}")
        block.append(f"Reason: {s.reasoning}")
        if s.affected_data_model:
            block.append(f"Affected data model: {s.affected_data_model}")
        if s.api_changes:
            block.append("API changes:\n" + _txt_bullets(s.api_changes))
        service_blocks.append("\n".join(block))
    sections.append(_txt_section(11, "Affected Services", *service_blocks))

    change_blocks = []
    for s in tdd.services:
        for c in s.changes:
            change_blocks.append(f"[{s.service}]\n" + _txt_change_block(c))
    sections.append(_txt_section(12, "Affected Files, Classes & Functions", *change_blocks))

    api_blocks = [
        f"{s.service}:\n" + _txt_bullets(s.api_changes) for s in tdd.services if s.api_changes
    ]
    sections.append(
        _txt_section(
            13,
            "API Changes",
            *api_blocks if api_blocks else ["No API changes identified from the available documentation."],
        )
    )

    data_model_blocks = [
        f"{s.service}: {s.affected_data_model}" for s in tdd.services if s.affected_data_model
    ]
    sections.append(
        _txt_section(
            14,
            "Data Model / Database Changes",
            "\n".join(data_model_blocks) if data_model_blocks else "Affected data model: Unknown",
        )
    )

    comm_parts = []
    if tdd.service_dependencies:
        comm_parts.append(
            "Real service-to-service dependencies (from the graph):\n"
            + "\n".join(f"- {d.source} -> {d.target}" for d in tdd.service_dependencies)
        )
    if tdd.communication_notes:
        comm_parts.append("Notes:\n" + _txt_bullets(tdd.communication_notes))
    sections.append(_txt_section(15, "Inter-Service Communication", *comm_parts))

    sections.append(_txt_section(16, "Error Handling & Edge Cases", tdd.error_handling_notes))
    sections.append(_txt_section(17, "Security Considerations", tdd.security_considerations))
    sections.append(_txt_section(18, "Backward Compatibility", tdd.backward_compatibility_notes))

    testing_parts = []
    if tdd.testing_notes:
        testing_parts.append(tdd.testing_notes)
    if tdd.testing_plan:
        table = _txt_table(
            ["Category", "Scenario", "Expected Result"],
            [[t.category, t.scenario, t.expected_result] for t in tdd.testing_plan],
        )
        if table:
            testing_parts.append(table)
    sections.append(_txt_section(19, "Testing Strategy", *testing_parts))

    if tdd.implementation_plan:
        sections.append(
            _txt_section(
                20,
                "Implementation Plan",
                "\n".join(f"{i + 1}. {step}" for i, step in enumerate(tdd.implementation_plan)),
            )
        )

    sections.append(_txt_section(21, "Risks", _txt_bullets(tdd.risks)))
    sections.append(_txt_section(22, "Open Questions", _txt_bullets(tdd.open_questions)))

    summary_rows = [
        [s.service, c.file_path or "Unknown", c.function_or_symbol, c.change_type, s.complexity]
        for s in tdd.services
        for c in s.changes
    ]
    summary_table = _txt_table(["Service", "File", "Symbol", "Change Type", "Complexity"], summary_rows)
    sections.append(_txt_section(23, "Change Summary", summary_table))

    created = [c.file_path for s in tdd.services for c in s.changes if c.change_type == "create" and c.file_path]
    modified = [c.file_path for s in tdd.services for c in s.changes if c.change_type == "modify" and c.file_path]
    deleted = [c.file_path for s in tdd.services for c in s.changes if c.change_type == "delete" and c.file_path]
    sections.append(
        _txt_section(
            24,
            "Files to Create / Modify / Delete",
            "Files to Create:\n" + (_txt_bullets(sorted(set(created))) or "None identified."),
            "Files to Modify:\n" + (_txt_bullets(sorted(set(modified))) or "None identified."),
            "Files to Delete:\n" + (_txt_bullets(sorted(set(deleted))) or "None identified."),
        )
    )

    sections.append(_txt_section(25, "Overall Reasoning", tdd.overall_reasoning))

    return "\n\n\n".join(s for s in sections if s)
