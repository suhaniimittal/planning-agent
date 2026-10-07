"""Renders a TechnicalDesignDoc as a readable Markdown document. Pure
function, no I/O — the structured TechnicalDesignDoc object itself remains
available unchanged for a coding agent that wants to consume it
programmatically; this is purely an additive, human-readable form.

Diagrams here are real Mermaid (```mermaid fences) — rendered as actual
boxes/arrows by GitHub, VSCode's Markdown preview, Notion, etc. — built
entirely from the same structured facts (ArchComponent/FlowStep/
SequenceStep/ServiceDependency) the LLM/graph already produced; nothing
here is hand-drawn or invented by the renderer.
"""

from __future__ import annotations

import re

from .models import ArchComponent, FlowStep, SequenceStep, ServiceDependency, TechnicalDesignDoc

_VERIFICATION_LABELS = {
    "verified": "VERIFIED",
    "corrected": "CORRECTED",
    "new": "NEW",
    "unverified": "UNVERIFIED - check manually",
}


def _code_check(c) -> str | None:
    """One line saying whether this change's file/symbol was confirmed in
    the real code graph (see query.verify) — None for a TDD that was never
    verified, so older/unverified TDDs render exactly as before."""
    if not c.verification:
        return None
    label = _VERIFICATION_LABELS.get(c.verification, c.verification.upper())
    return f"{label} — {c.verification_note}" if c.verification_note else label


def _mermaid_id(name: str) -> str:
    """A safe Mermaid node id for an arbitrary display name — ids can't
    contain spaces or most punctuation, so this derives one deterministically
    while the real name still shows as the node's quoted label."""
    slug = re.sub(r"[^A-Za-z0-9_]", "_", name).strip("_") or "n"
    return f"n_{slug}" if slug[0].isdigit() else slug


def _mermaid_label(text: str) -> str:
    """Mermaid node/edge labels break on a literal quote — swap for a
    plain apostrophe rather than trying to escape it."""
    return text.replace('"', "'")


def _mermaid_architecture_diagram(components: list[ArchComponent]) -> str | None:
    """One node per named component, one edge per stated dependency — built
    from facts the LLM states (name + depends_on), never a diagram it hand-
    draws itself."""
    if not components:
        return None
    lines = ["```mermaid", "flowchart LR"]
    declared: set[str] = set()

    def _declare(name: str) -> str:
        node_id = _mermaid_id(name)
        if node_id not in declared:
            lines.append(f'    {node_id}["{_mermaid_label(name)}"]')
            declared.add(node_id)
        return node_id

    for c in components:
        source_id = _declare(c.name)
        for dep in c.depends_on:
            lines.append(f"    {source_id} --> {_declare(dep)}")
    lines.append("```")
    return "\n".join(lines)


def _mermaid_service_dependency_diagram(dependencies: list[ServiceDependency]) -> str | None:
    """The real CALLS edges among the candidate services — grounded in
    graph data (query_flow.build_tdd fills this in after parsing), never
    guessed by the LLM. None if there are no edges."""
    if not dependencies:
        return None
    lines = ["```mermaid", "flowchart LR"]
    declared: set[str] = set()

    def _declare(name: str) -> str:
        node_id = _mermaid_id(name)
        if node_id not in declared:
            lines.append(f'    {node_id}["{_mermaid_label(name)}"]')
            declared.add(node_id)
        return node_id

    for e in dependencies:
        if e.via:
            lines.append(f"    {_declare(e.source)} -- {e.via} --> {_declare(e.target)}")
        else:
            lines.append(f"    {_declare(e.source)} --> {_declare(e.target)}")
    lines.append("```")
    return "\n".join(lines)


def _mermaid_flow_diagram(steps: list[FlowStep]) -> str | None:
    """A straight-line flowchart, one node per ordered step — the detail
    (when given) renders inside the same node, under the step label."""
    if not steps:
        return None
    lines = ["```mermaid", "flowchart TD"]
    ids = [f"s{i}" for i in range(len(steps))]
    for i, s in enumerate(steps):
        label = _mermaid_label(s.step)
        if s.detail:
            label += f"<br/><i>{_mermaid_label(s.detail)}</i>"
        lines.append(f'    {ids[i]}["{label}"]')
    for i in range(len(steps) - 1):
        lines.append(f"    {ids[i]} --> {ids[i + 1]}")
    lines.append("```")
    return "\n".join(lines)


def _mermaid_sequence_diagram(steps: list[SequenceStep]) -> str | None:
    """A real Mermaid sequenceDiagram — a solid arrow for a call, a dashed
    arrow for a response, participants declared implicitly in the order
    they first appear (Mermaid's own default)."""
    if not steps:
        return None
    lines = ["```mermaid", "sequenceDiagram"]
    for s in steps:
        arrow = "-->>" if s.direction == "response" else "->>"
        action = s.action.replace(":", " -")  # a literal colon breaks the message syntax
        lines.append(f"    {s.from_participant}{arrow}{s.to_participant}: {action}")
    lines.append("```")
    return "\n".join(lines)


def _md_table(headers: list[str], rows: list[list[str]]) -> str | None:
    if not rows:
        return None
    esc = lambda cell: cell.replace("|", "\\|").replace("\n", "<br/>")  # noqa: E731
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines += ["| " + " | ".join(esc(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _md_change_block(c) -> list[str]:
    loc = f"`{c.file_path}` — " if c.file_path else ""
    source_tag = " _(grounded in real source)_" if c.has_real_source else " _(illustrative only, not verified code)_"
    pseudocode_label = (
        "Proposed approach (may reflect real source shown to the model):"
        if c.has_real_source
        else "Proposed approach (illustrative pseudocode, not real code):"
    )
    lines = [f"##### {loc}`{c.function_or_symbol}`{source_tag}", ""]
    if c.current_behavior:
        lines += [f"- **Current behavior:** {c.current_behavior}"]
    lines += [f"- **Change type:** {c.change_type}"]
    if _code_check(c):
        lines += [f"- **Code check:** {_code_check(c)}"]
    lines += [
        f"- **Change:** {c.change_description}",
        f"- **Implementation:** {c.implementation_notes}",
        f"- **Why:** {c.reasoning}",
        "",
        pseudocode_label,
        "```",
        c.pseudocode_sketch,
        "```",
    ]
    if c.acceptance_criteria:
        lines.append("**Acceptance criteria:**")
        lines += [f"- {ac}" for ac in c.acceptance_criteria]
    lines.append("")
    return lines


def _md_section(number: int, title: str, *body_parts: str | None) -> str | None:
    """One numbered Markdown section, or None if every part is empty —
    omitted entirely rather than printed as a bare header, mirroring
    render_text()'s _txt_section discipline exactly."""
    parts = [p.strip() for p in body_parts if p and p.strip()]
    if not parts:
        return None
    return f"## {number}. {title}\n\n" + "\n\n".join(parts)


def render_markdown(tdd: TechnicalDesignDoc) -> str:
    """The full TDD as Markdown, with real Mermaid diagrams — mirrors
    render_text()'s section coverage and numbering (for easy cross-
    reference between the two output formats) but renders every
    architecture/flow/sequence diagram as an actual Mermaid figure instead
    of plain ASCII, and omits any section with nothing to show exactly the
    way render_text() does."""
    header = [f"# {tdd.title}"]
    if tdd.generated_at:
        header.append(f"*Generated: {tdd.generated_at}*")

    sections: list[str | None] = ["\n".join(header)]

    sections.append(_md_section(2, "Issue Summary", tdd.issue_summary))

    sections.append(
        _md_section(
            3,
            "Requirements",
            "**Functional:**\n\n" + ("\n".join(f"- {r}" for r in tdd.functional_requirements) or "Not specified."),
            "**Non-functional:**\n\n"
            + ("\n".join(f"- {r}" for r in tdd.non_functional_requirements) or "Not specified."),
        )
    )

    current_arch_diagram = _mermaid_architecture_diagram(tdd.current_architecture_components)
    sections.append(_md_section(4, "Current System / Architecture", tdd.current_architecture, current_arch_diagram))

    current_flow_diagram = _mermaid_flow_diagram(tdd.current_flow_steps)
    sections.append(_md_section(5, "Current Flow", current_flow_diagram))

    sections.append(_md_section(6, "Root Cause / Problem Analysis", tdd.root_cause_analysis))

    sections.append(_md_section(7, "Proposed Solution", tdd.proposed_solution))

    proposed_arch_diagram = _mermaid_architecture_diagram(tdd.proposed_architecture_components)
    sections.append(_md_section(8, "Proposed Architecture", tdd.proposed_architecture, proposed_arch_diagram))

    proposed_flow_diagram = _mermaid_flow_diagram(tdd.proposed_flow_steps)
    sections.append(_md_section(9, "Proposed Flow", proposed_flow_diagram))

    sequence_diagram = _mermaid_sequence_diagram(tdd.sequence_steps)
    sections.append(_md_section(10, "Sequence / Interaction Flow", sequence_diagram))

    service_blocks = []
    for s in tdd.services:
        data_model = f" | Data model: {s.affected_data_model}" if s.affected_data_model else ""
        block = [f"### {s.service}", "", f"Complexity: {s.complexity}{data_model}", "", s.reasoning]
        if s.responsibility_change:
            block += ["", f"**Responsibility change:** {s.responsibility_change}"]
        service_blocks.append("\n".join(block))
    sections.append(_md_section(11, "Affected Services", *service_blocks))

    change_blocks = []
    for s in tdd.services:
        if not s.changes:
            continue
        change_texts = ["\n".join(_md_change_block(c)) for c in s.changes]
        change_blocks.append(f"#### {s.service}\n\n" + "\n\n".join(change_texts))
    sections.append(_md_section(12, "Affected Files, Classes & Functions", *change_blocks))

    api_blocks = [
        f"**{s.service}:**\n\n" + "\n".join(f"- {a}" for a in s.api_changes) for s in tdd.services if s.api_changes
    ]
    sections.append(
        _md_section(
            13,
            "API Changes",
            *api_blocks if api_blocks else ["No API changes identified from the available documentation."],
        )
    )

    dm_lines = [f"- **{s.service}:** {s.affected_data_model}" for s in tdd.services if s.affected_data_model]
    sections.append(
        _md_section(14, "Data Model / Database Changes", "\n".join(dm_lines) if dm_lines else "Affected data model: Unknown")
    )

    comm_parts = []
    dependency_diagram = _mermaid_service_dependency_diagram(tdd.service_dependencies)
    if dependency_diagram:
        comm_parts.append("Real service-to-service dependencies (from the graph):\n\n" + dependency_diagram)
    if tdd.communication_notes:
        comm_parts.append("**Notes:**\n\n" + "\n".join(f"- {n}" for n in tdd.communication_notes))
    sections.append(_md_section(15, "Inter-Service Communication", *comm_parts))

    sections.append(_md_section(16, "Error Handling & Edge Cases", tdd.error_handling_notes))
    sections.append(_md_section(17, "Security Considerations", tdd.security_considerations))
    sections.append(_md_section(18, "Backward Compatibility", tdd.backward_compatibility_notes))

    testing_parts = []
    if tdd.testing_notes:
        testing_parts.append(tdd.testing_notes)
    testing_table = _md_table(
        ["Category", "Scenario", "Expected Result"],
        [[t.category, t.scenario, t.expected_result] for t in tdd.testing_plan],
    )
    if testing_table:
        testing_parts.append(testing_table)
    sections.append(_md_section(19, "Testing Strategy", *testing_parts))

    if tdd.implementation_plan:
        sections.append(
            _md_section(
                20,
                "Implementation Plan",
                "\n".join(f"{i + 1}. {step}" for i, step in enumerate(tdd.implementation_plan)),
            )
        )

    sections.append(_md_section(21, "Risks", "\n".join(f"- {r}" for r in tdd.risks) if tdd.risks else None))
    sections.append(
        _md_section(22, "Open Questions", "\n".join(f"- {q}" for q in tdd.open_questions) if tdd.open_questions else None)
    )

    summary_rows = [
        [s.service, c.file_path or "Unknown", c.function_or_symbol, c.change_type, s.complexity]
        for s in tdd.services
        for c in s.changes
    ]
    summary_table = _md_table(["Service", "File", "Symbol", "Change Type", "Complexity"], summary_rows)
    sections.append(_md_section(23, "Change Summary", summary_table))

    created = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "create" and c.file_path})
    modified = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "modify" and c.file_path})
    deleted = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "delete" and c.file_path})
    sections.append(
        _md_section(
            24,
            "Files to Create / Modify / Delete",
            "**Create:**\n\n" + ("\n".join(f"- `{f}`" for f in created) or "None identified."),
            "**Modify:**\n\n" + ("\n".join(f"- `{f}`" for f in modified) or "None identified."),
            "**Delete:**\n\n" + ("\n".join(f"- `{f}`" for f in deleted) or "None identified."),
        )
    )

    sections.append(_md_section(25, "Overall Reasoning", tdd.overall_reasoning))

    return "\n\n".join(s for s in sections if s)


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
    if _code_check(c):
        lines.append(f"Code check: {_code_check(c)}")
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


# --- Confluence renderer: same section coverage/numbering as render_text(),
# rendered as Confluence's own "storage format" (XHTML-like markup Confluence's
# REST API expects), with real headings/lists/tables/code blocks instead of
# plain text — and every architecture/flow/sequence DIAGRAM section dropped
# entirely rather than converted, since Confluence has no Mermaid support and
# an ASCII-art fallback doesn't read as a real diagram on a real page anyway.
# The underlying facts those diagrams were built from (e.g. real service
# dependencies) are kept as plain lists where they're informative on their own.


def _cf_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cf_section(number: int, title: str, *body_parts: str | None) -> str | None:
    """One numbered Confluence section, or None if every part is empty —
    omitted entirely, same discipline as _txt_section/_md_section."""
    parts = [p for p in body_parts if p]
    if not parts:
        return None
    return f"<h2>{number}. {_cf_escape(title)}</h2>\n" + "\n".join(parts)


def _cf_p(text: str | None) -> str | None:
    if not text:
        return None
    return f"<p>{_cf_escape(text)}</p>"


def _cf_bullets(items: list[str]) -> str | None:
    if not items:
        return None
    return "<ul>\n" + "\n".join(f"<li>{_cf_escape(i)}</li>" for i in items) + "\n</ul>"


def _cf_numbered(items: list[str]) -> str | None:
    if not items:
        return None
    return "<ol>\n" + "\n".join(f"<li>{_cf_escape(i)}</li>" for i in items) + "\n</ol>"


def _cf_table(headers: list[str], rows: list[list[str]]) -> str | None:
    if not rows:
        return None
    head = "<tr>" + "".join(f"<th>{_cf_escape(h)}</th>" for h in headers) + "</tr>"
    body = "\n".join(
        "<tr>" + "".join(f"<td>{_cf_escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><tbody>\n{head}\n{body}\n</tbody></table>"


def _cf_code(text: str, language: str = "none") -> str:
    # CDATA needs no escaping of &/</> — the one thing it can't contain
    # literally is the sequence "]]>" itself, split here so a pseudocode
    # snippet containing that exact text (vanishingly unlikely, but real
    # code can contain almost anything) can't break the macro.
    safe = text.replace("]]>", "]]]]><![CDATA[>")
    return (
        f'<ac:structured-macro ac:name="code"><ac:parameter ac:name="language">{language}'
        f"</ac:parameter><ac:plain-text-body><![CDATA[{safe}]]></ac:plain-text-body>"
        "</ac:structured-macro>"
    )


def _cf_flow_list(steps) -> str | None:
    """The ordered steps only — no arrow-chain diagram, unlike
    _ascii_flow_diagram/_mermaid_flow_diagram."""
    if not steps:
        return None
    items = [f"{s.step} — {s.detail}" if s.detail else s.step for s in steps]
    return _cf_numbered(items)


def _cf_sequence_list(steps) -> str | None:
    """Same interaction facts as _ascii_sequence_diagram, as a plain list
    instead of arrow art."""
    if not steps:
        return None
    items = []
    for s in steps:
        if s.direction == "response":
            items.append(f"{s.from_participant} ← {s.action} — {s.to_participant}")
        else:
            items.append(f"{s.from_participant} → {s.action} — {s.to_participant}")
    return _cf_bullets(items)


def _cf_change_block(c) -> str:
    loc = f"<code>{_cf_escape(c.file_path)}</code> — " if c.file_path else ""
    source_tag = " (grounded in real source)" if c.has_real_source else " (illustrative only, not verified code)"
    lines = [f"<h4>{loc}<code>{_cf_escape(c.function_or_symbol)}</code>{source_tag}</h4>"]
    facts = []
    if c.current_behavior:
        facts.append(f"Current behavior: {c.current_behavior}")
    facts += [f"Change type: {c.change_type}"]
    if _code_check(c):
        facts += [f"Code check: {_code_check(c)}"]
    facts += [
        f"Change: {c.change_description}",
        f"Implementation: {c.implementation_notes}",
        f"Why: {c.reasoning}",
    ]
    lines.append(_cf_bullets(facts))
    lines.append(_cf_code(c.pseudocode_sketch))
    if c.acceptance_criteria:
        lines.append("<p><strong>Acceptance criteria:</strong></p>")
        lines.append(_cf_bullets(c.acceptance_criteria))
    return "\n".join(lines)


def render_confluence(tdd: TechnicalDesignDoc) -> str:
    """The full TDD as Confluence storage-format XHTML — same section
    coverage/numbering as render_text(), for easy cross-reference, but with
    real Confluence headings/lists/tables/code blocks and every diagram
    section dropped (see module docstring above)."""
    sections: list[str | None] = []

    header = [f"<p><em>Generated: {_cf_escape(tdd.generated_at)}</em></p>"] if tdd.generated_at else []
    if header:
        sections.append("\n".join(header))

    sections.append(_cf_section(2, "Issue Summary", _cf_p(tdd.issue_summary)))

    sections.append(
        _cf_section(
            3,
            "Requirements",
            "<p><strong>Functional:</strong></p>\n" + (_cf_bullets(tdd.functional_requirements) or "<p>Not specified.</p>"),
            "<p><strong>Non-functional:</strong></p>\n"
            + (_cf_bullets(tdd.non_functional_requirements) or "<p>Not specified.</p>"),
        )
    )

    sections.append(_cf_section(4, "Current System / Architecture", _cf_p(tdd.current_architecture)))
    sections.append(_cf_section(5, "Current Flow", _cf_flow_list(tdd.current_flow_steps)))
    sections.append(_cf_section(6, "Root Cause / Problem Analysis", _cf_p(tdd.root_cause_analysis)))
    sections.append(_cf_section(7, "Proposed Solution", _cf_p(tdd.proposed_solution)))
    sections.append(_cf_section(8, "Proposed Architecture", _cf_p(tdd.proposed_architecture)))
    sections.append(_cf_section(9, "Proposed Flow", _cf_flow_list(tdd.proposed_flow_steps)))
    sections.append(_cf_section(10, "Sequence / Interaction Flow", _cf_sequence_list(tdd.sequence_steps)))

    service_blocks = []
    for s in tdd.services:
        data_model = f" | Data model: {s.affected_data_model}" if s.affected_data_model else ""
        block = [f"<h3>{_cf_escape(s.service)}</h3>", f"<p>Complexity: {_cf_escape(s.complexity)}{_cf_escape(data_model)}</p>"]
        if s.responsibility_change:
            block.append(f"<p><strong>Responsibility change:</strong> {_cf_escape(s.responsibility_change)}</p>")
        block.append(_cf_p(s.reasoning) or "")
        service_blocks.append("\n".join(block))
    sections.append(_cf_section(11, "Affected Services", *service_blocks))

    change_blocks = []
    for s in tdd.services:
        if not s.changes:
            continue
        blocks = "\n".join(_cf_change_block(c) for c in s.changes)
        change_blocks.append(f"<h3>{_cf_escape(s.service)}</h3>\n{blocks}")
    sections.append(_cf_section(12, "Affected Files, Classes & Functions", *change_blocks))

    api_blocks = [
        f"<p><strong>{_cf_escape(s.service)}:</strong></p>\n" + _cf_bullets(s.api_changes)
        for s in tdd.services
        if s.api_changes
    ]
    sections.append(
        _cf_section(
            13,
            "API Changes",
            *(api_blocks or ["<p>No API changes identified from the available documentation.</p>"]),
        )
    )

    dm_lines = [f"{s.service}: {s.affected_data_model}" for s in tdd.services if s.affected_data_model]
    sections.append(
        _cf_section(14, "Data Model / Database Changes", _cf_bullets(dm_lines) or "<p>Affected data model: Unknown</p>")
    )

    comm_parts = []
    if tdd.service_dependencies:
        comm_parts.append("<p>Real service-to-service dependencies (from the graph):</p>")
        comm_parts.append(
            _cf_bullets(
                [
                    f"{d.source} → {d.target}" + (f" (via {d.via})" if d.via else "")
                    for d in tdd.service_dependencies
                ]
            )
        )
    if tdd.communication_notes:
        comm_parts.append("<p><strong>Notes:</strong></p>")
        comm_parts.append(_cf_bullets(tdd.communication_notes))
    sections.append(_cf_section(15, "Inter-Service Communication", *comm_parts))

    sections.append(_cf_section(16, "Error Handling & Edge Cases", _cf_p(tdd.error_handling_notes)))
    sections.append(_cf_section(17, "Security Considerations", _cf_p(tdd.security_considerations)))
    sections.append(_cf_section(18, "Backward Compatibility", _cf_p(tdd.backward_compatibility_notes)))

    testing_parts = []
    if tdd.testing_notes:
        testing_parts.append(_cf_p(tdd.testing_notes))
    testing_table = _cf_table(
        ["Category", "Scenario", "Expected Result"],
        [[t.category, t.scenario, t.expected_result] for t in tdd.testing_plan],
    )
    if testing_table:
        testing_parts.append(testing_table)
    sections.append(_cf_section(19, "Testing Strategy", *testing_parts))

    if tdd.implementation_plan:
        sections.append(_cf_section(20, "Implementation Plan", _cf_numbered(tdd.implementation_plan)))

    sections.append(_cf_section(21, "Risks", _cf_bullets(tdd.risks)))
    sections.append(_cf_section(22, "Open Questions", _cf_bullets(tdd.open_questions)))

    summary_rows = [
        [s.service, c.file_path or "Unknown", c.function_or_symbol, c.change_type, s.complexity]
        for s in tdd.services
        for c in s.changes
    ]
    summary_table = _cf_table(["Service", "File", "Symbol", "Change Type", "Complexity"], summary_rows)
    sections.append(_cf_section(23, "Change Summary", summary_table))

    created = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "create" and c.file_path})
    modified = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "modify" and c.file_path})
    deleted = sorted({c.file_path for s in tdd.services for c in s.changes if c.change_type == "delete" and c.file_path})
    sections.append(
        _cf_section(
            24,
            "Files to Create / Modify / Delete",
            "<p><strong>Create:</strong></p>\n" + (_cf_bullets(created) or "<p>None identified.</p>"),
            "<p><strong>Modify:</strong></p>\n" + (_cf_bullets(modified) or "<p>None identified.</p>"),
            "<p><strong>Delete:</strong></p>\n" + (_cf_bullets(deleted) or "<p>None identified.</p>"),
        )
    )

    sections.append(_cf_section(25, "Overall Reasoning", _cf_p(tdd.overall_reasoning)))

    return "\n".join(s for s in sections if s)


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
            + "\n".join(
                f"- {d.source} -> {d.target}" + (f" (via {d.via})" if d.via else "")
                for d in tdd.service_dependencies
            )
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
