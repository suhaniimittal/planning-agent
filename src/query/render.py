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


def _change_type_label(c) -> str:
    """A bare "reference" reads like a change; spell out that it is not."""
    return "reference (read, not changed)" if c.change_type == "reference" else c.change_type


def _is_edit(c) -> bool:
    """True for a change that actually edits code. A "reference" change is a
    symbol the design reads, calls, or depends on without touching it, so it
    must stay out of every "files to change" list — listing it there sends a
    reader (or a coding agent) to open a file the design explicitly said to
    leave alone."""
    return c.change_type != "reference"


def _reference_paths(changes) -> set[str]:
    return {c.file_path for c in changes if c.change_type == "reference" and c.file_path}


def _code_check(c) -> str | None:
    """One line saying whether this change's file/symbol was confirmed in
    the real code graph (see query.verify) — None for a TDD that was never
    verified, so older/unverified TDDs render exactly as before."""
    if not c.verification:
        return None
    label = _VERIFICATION_LABELS.get(c.verification, c.verification.upper())
    # A verified change's note only repeats the file and symbol shown right
    # beside it; the other outcomes' notes say what was corrected or missing.
    if c.verification == "verified" or not c.verification_note:
        return label
    return f"{label} — {c.verification_note}"


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
    source_tag = (
        " _(grounded in real source)_"
        if c.has_real_source
        else " _(illustrative only, not verified code)_"
    )
    pseudocode_label = (
        "Proposed approach (may reflect real source shown to the model):"
        if c.has_real_source
        else "Proposed approach (illustrative pseudocode, not real code):"
    )
    lines = [f"##### {loc}`{c.function_or_symbol}`{source_tag}", ""]
    if c.current_behavior:
        lines += [f"- **Current behavior:** {c.current_behavior}"]
    lines += [f"- **Change type:** {_change_type_label(c)}"]
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
            "**Functional:**\n\n"
            + ("\n".join(f"- {r}" for r in tdd.functional_requirements) or "Not specified."),
            "**Non-functional:**\n\n"
            + ("\n".join(f"- {r}" for r in tdd.non_functional_requirements) or "Not specified."),
        )
    )

    current_arch_diagram = _mermaid_architecture_diagram(tdd.current_architecture_components)
    sections.append(
        _md_section(
            4, "Current System / Architecture", tdd.current_architecture, current_arch_diagram
        )
    )

    current_flow_diagram = _mermaid_flow_diagram(tdd.current_flow_steps)
    sections.append(_md_section(5, "Current Flow", current_flow_diagram))

    sections.append(_md_section(6, "Root Cause / Problem Analysis", tdd.root_cause_analysis))

    sections.append(_md_section(7, "Proposed Solution", tdd.proposed_solution))

    proposed_arch_diagram = _mermaid_architecture_diagram(tdd.proposed_architecture_components)
    sections.append(
        _md_section(8, "Proposed Architecture", tdd.proposed_architecture, proposed_arch_diagram)
    )

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
        f"**{s.service}:**\n\n" + "\n".join(f"- {a}" for a in s.api_changes)
        for s in tdd.services
        if s.api_changes
    ]
    sections.append(
        _md_section(
            13,
            "API Changes",
            *(
                api_blocks
                if api_blocks
                else ["No API changes identified from the available documentation."]
            ),
        )
    )

    dm_lines = [
        f"- **{s.service}:** {s.affected_data_model}" for s in tdd.services if s.affected_data_model
    ]
    sections.append(
        _md_section(
            14,
            "Data Model / Database Changes",
            "\n".join(dm_lines) if dm_lines else "Affected data model: Unknown",
        )
    )

    comm_parts = []
    dependency_diagram = _mermaid_service_dependency_diagram(tdd.service_dependencies)
    if dependency_diagram:
        comm_parts.append(
            "Real service-to-service dependencies (from the graph):\n\n" + dependency_diagram
        )
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

    sections.append(
        _md_section(21, "Risks", "\n".join(f"- {r}" for r in tdd.risks) if tdd.risks else None)
    )
    sections.append(
        _md_section(
            22,
            "Open Questions",
            "\n".join(f"- {q}" for q in tdd.open_questions) if tdd.open_questions else None,
        )
    )

    summary_rows = [
        [s.service, c.file_path or "Unknown", c.function_or_symbol, c.change_type, s.complexity]
        for s in tdd.services
        for c in s.changes
    ]
    summary_table = _md_table(
        ["Service", "File", "Symbol", "Change Type", "Complexity"], summary_rows
    )
    sections.append(_md_section(23, "Change Summary", summary_table))

    created = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "create" and c.file_path
        }
    )
    modified = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "modify" and c.file_path
        }
    )
    deleted = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "delete" and c.file_path
        }
    )
    referenced = sorted({f for s in tdd.services for f in _reference_paths(s.changes)})
    parts_24 = [
        "**Create:**\n\n" + ("\n".join(f"- `{f}`" for f in created) or "None identified."),
        "**Modify:**\n\n" + ("\n".join(f"- `{f}`" for f in modified) or "None identified."),
        "**Delete:**\n\n" + ("\n".join(f"- `{f}`" for f in deleted) or "None identified."),
    ]
    if referenced:
        parts_24.append("**Read but NOT changed:**\n\n" + "\n".join(f"- `{f}`" for f in referenced))
    sections.append(_md_section(24, "Files to Create / Modify / Delete", *parts_24))

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
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    lines += [_fmt_row(row) for row in rows]
    return "\n".join(lines)


def _architecture_parts(components, service_dependencies) -> list[str]:
    """The Architecture section's structured half: what the parts are, and
    which of them call which.

    Built from stated facts, never hand-drawn by the LLM — a listing
    renders correctly for ANY graph shape, where a 2D box layout would
    need arrangement logic that gets arbitrary graphs wrong.

    This replaced a version that printed the model's component list and the
    graph's verified edges as two separate blocks of arrows, one after the
    other. A reader saw the same relationship twice in two formats with
    nothing saying which to believe, and a component that called nothing
    rendered as "(no further dependencies)", which reads like something
    failed rather than like a fact. Here the two sources are merged into one
    aligned list and each edge carries its own provenance, so the reader can
    tell a verified call from one this design is proposing."""
    if not components and not service_dependencies:
        return []

    verified = {(d.source, d.target): d for d in service_dependencies or []}
    names = [c.name for c in components or []]
    for source, target in verified:
        for name in (source, target):
            if name not in names:
                names.append(name)

    parts: list[str] = []
    if names:
        parts.append("Components:\n  " + ", ".join(names))

    edges: list[tuple[str, str]] = []
    for c in components or []:
        edges.extend((c.name, dep) for dep in c.depends_on or [])
    for pair in verified:
        if pair not in edges:
            edges.append(pair)
    if not edges:
        parts.append("Calls:\n  No calls between these components are shown in the evidence.")
        return parts

    source_width = max(len(source) for source, _ in edges)
    target_width = max(len(target) for _, target in edges)
    lines = []
    for source, target in edges:
        edge = verified.get((source, target))
        if edge is None:
            note = "proposed by this design"
        elif edge.via:
            note = f"verified in the code graph, via {edge.via}"
        else:
            note = "verified in the code graph"
        lines.append(f"  {source:<{source_width}} --> {target:<{target_width}}   ({note})")
    parts.append("Calls:\n" + "\n".join(lines))
    return parts


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


def _txt_change_block(number: str, c) -> str:
    """One change as a short labelled block: what, where, whether the code
    graph confirmed it, then what/how/why and when it's done."""
    status = _code_check(c)
    lines = [
        f"{number} {c.function_or_symbol}  ({_change_type_label(c)})",
        f"    File: {c.file_path or 'Unknown'}",
    ]
    if status:
        lines.append(f"    Code check: {status}")
    if c.current_behavior:
        lines.append(f"    Today: {c.current_behavior}")
    lines.append(f"    Change: {c.change_description}")
    lines.append(f"    How: {c.implementation_notes}")
    lines.append(f"    Why: {c.reasoning}")
    if c.pseudocode_sketch and c.pseudocode_sketch.strip():
        label = "Sketch" if c.has_real_source else "Sketch (illustrative, not real code)"
        lines.append(f"    {label}:")
        lines += ["        " + line for line in c.pseudocode_sketch.strip().splitlines()]
    if c.acceptance_criteria:
        lines.append("    Done when:")
        lines += [f"      - {ac}" for ac in c.acceptance_criteria]
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
    source_tag = (
        " (grounded in real source)"
        if c.has_real_source
        else " (illustrative only, not verified code)"
    )
    lines = [f"<h4>{loc}<code>{_cf_escape(c.function_or_symbol)}</code>{source_tag}</h4>"]
    facts = []
    if c.current_behavior:
        facts.append(f"Current behavior: {c.current_behavior}")
    facts += [f"Change type: {_change_type_label(c)}"]
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

    header = (
        [f"<p><em>Generated: {_cf_escape(tdd.generated_at)}</em></p>"] if tdd.generated_at else []
    )
    if header:
        sections.append("\n".join(header))

    sections.append(_cf_section(2, "Issue Summary", _cf_p(tdd.issue_summary)))

    sections.append(
        _cf_section(
            3,
            "Requirements",
            "<p><strong>Functional:</strong></p>\n"
            + (_cf_bullets(tdd.functional_requirements) or "<p>Not specified.</p>"),
            "<p><strong>Non-functional:</strong></p>\n"
            + (_cf_bullets(tdd.non_functional_requirements) or "<p>Not specified.</p>"),
        )
    )

    sections.append(
        _cf_section(4, "Current System / Architecture", _cf_p(tdd.current_architecture))
    )
    sections.append(_cf_section(5, "Current Flow", _cf_flow_list(tdd.current_flow_steps)))
    sections.append(_cf_section(6, "Root Cause / Problem Analysis", _cf_p(tdd.root_cause_analysis)))
    sections.append(_cf_section(7, "Proposed Solution", _cf_p(tdd.proposed_solution)))
    sections.append(_cf_section(8, "Proposed Architecture", _cf_p(tdd.proposed_architecture)))
    sections.append(_cf_section(9, "Proposed Flow", _cf_flow_list(tdd.proposed_flow_steps)))
    sections.append(
        _cf_section(10, "Sequence / Interaction Flow", _cf_sequence_list(tdd.sequence_steps))
    )

    service_blocks = []
    for s in tdd.services:
        data_model = f" | Data model: {s.affected_data_model}" if s.affected_data_model else ""
        block = [
            f"<h3>{_cf_escape(s.service)}</h3>",
            f"<p>Complexity: {_cf_escape(s.complexity)}{_cf_escape(data_model)}</p>",
        ]
        if s.responsibility_change:
            block.append(
                f"<p><strong>Responsibility change:</strong> {_cf_escape(s.responsibility_change)}</p>"
            )
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

    dm_lines = [
        f"{s.service}: {s.affected_data_model}" for s in tdd.services if s.affected_data_model
    ]
    sections.append(
        _cf_section(
            14,
            "Data Model / Database Changes",
            _cf_bullets(dm_lines) or "<p>Affected data model: Unknown</p>",
        )
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
    sections.append(
        _cf_section(18, "Backward Compatibility", _cf_p(tdd.backward_compatibility_notes))
    )

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
        sections.append(
            _cf_section(20, "Implementation Plan", _cf_numbered(tdd.implementation_plan))
        )

    sections.append(_cf_section(21, "Risks", _cf_bullets(tdd.risks)))
    sections.append(_cf_section(22, "Open Questions", _cf_bullets(tdd.open_questions)))

    summary_rows = [
        [s.service, c.file_path or "Unknown", c.function_or_symbol, c.change_type, s.complexity]
        for s in tdd.services
        for c in s.changes
    ]
    summary_table = _cf_table(
        ["Service", "File", "Symbol", "Change Type", "Complexity"], summary_rows
    )
    sections.append(_cf_section(23, "Change Summary", summary_table))

    created = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "create" and c.file_path
        }
    )
    modified = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "modify" and c.file_path
        }
    )
    deleted = sorted(
        {
            c.file_path
            for s in tdd.services
            for c in s.changes
            if c.change_type == "delete" and c.file_path
        }
    )
    referenced = sorted({f for s in tdd.services for f in _reference_paths(s.changes)})
    parts_24 = [
        "<p><strong>Create:</strong></p>\n" + (_cf_bullets(created) or "<p>None identified.</p>"),
        "<p><strong>Modify:</strong></p>\n" + (_cf_bullets(modified) or "<p>None identified.</p>"),
        "<p><strong>Delete:</strong></p>\n" + (_cf_bullets(deleted) or "<p>None identified.</p>"),
    ]
    if referenced:
        parts_24.append("<p><strong>Read but NOT changed:</strong></p>\n" + _cf_bullets(referenced))
    sections.append(_cf_section(24, "Files to Create / Modify / Delete", *parts_24))

    sections.append(_cf_section(25, "Overall Reasoning", _cf_p(tdd.overall_reasoning)))

    return "\n".join(s for s in sections if s)


def render_text(tdd: TechnicalDesignDoc) -> str:
    """The plain-text TDD behind the PDF — read top to bottom by a developer,
    and read back by coding_agent. Ordered by what a reader needs: the
    problem, its cause, the fix, then every change per service with its file.
    Sections are numbered consecutively and omitted when empty; no tables
    (they don't survive line wrapping) and no fact stated twice."""
    header = ["TECHNICAL DESIGN DOCUMENT", tdd.title]
    if tdd.generated_at:
        header.append(f"Generated: {tdd.generated_at}")

    body: list[tuple[str, list[str | None]]] = []
    body.append(("Summary", [tdd.issue_summary]))

    requirements = [_txt_bullets(tdd.functional_requirements)]
    if tdd.non_functional_requirements:
        requirements.append("Non-functional:\n" + _txt_bullets(tdd.non_functional_requirements))
    body.append(("Requirements", requirements))

    body.append(("Root Cause", [_one_finding_per_line(tdd.root_cause_analysis)]))
    body.append(("Proposed Solution", [tdd.proposed_solution]))

    service_parts = []
    change_number = 0
    for s in tdd.services:
        if not s.changes:
            service_parts.append(f"{s.service} — no change\n    {s.reasoning}")
            continue
        head = [f"{s.service}  (complexity: {s.complexity})", f"    Why: {s.reasoning}"]
        if s.api_changes:
            head.append("    API changes:\n" + "\n".join(f"      - {a}" for a in s.api_changes))
        if s.affected_data_model:
            head.append(f"    Data model: {s.affected_data_model}")
        blocks = []
        for c in s.changes:
            change_number += 1
            blocks.append(_txt_change_block(f"Change {change_number}:", c))
        service_parts.append("\n".join(head) + "\n\n" + "\n\n".join(blocks))
    body.append(("Services and Changes", service_parts))

    file_parts = []
    reference_parts = []
    for s in tdd.services:
        rows = sorted(
            {(c.change_type, c.file_path) for c in s.changes if _is_edit(c) and c.file_path}
        )
        if rows:
            listed = "\n".join(f"  - {kind}: {path}" for kind, path in rows)
            file_parts.append(f"{s.service}\n{listed}")
        refs = sorted(_reference_paths(s.changes))
        if refs:
            listed = "\n".join(f"  - {path}" for path in refs)
            reference_parts.append(f"{s.service}\n{listed}")
    body.append(("Files to Change", file_parts))
    # Its own section, never merged into the one above: the whole point of a
    # "reference" change is that the reader must NOT open these to edit them.
    body.append(("Files Read But Not Changed", reference_parts))

    flow = []
    if tdd.current_flow_steps:
        flow.append("Today:\n" + _txt_steps(tdd.current_flow_steps))
    if tdd.proposed_flow_steps:
        flow.append("After the change:\n" + _txt_steps(tdd.proposed_flow_steps))
    body.append(("Flow", flow))

    arch = [tdd.proposed_architecture]
    arch.extend(_architecture_parts(tdd.proposed_architecture_components, tdd.service_dependencies))
    if tdd.communication_notes:
        arch.append("Communication:\n" + _txt_bullets(tdd.communication_notes))
    body.append(("Architecture", arch))

    testing = [tdd.testing_notes]
    if tdd.testing_plan:
        testing.append(
            "\n".join(
                f"- [{t.category}] {t.scenario}\n    Expected: {t.expected_result}"
                for t in tdd.testing_plan
            )
        )
    body.append(("Testing", testing))

    plan = "\n".join(f"{i}. {step}" for i, step in enumerate(tdd.implementation_plan, start=1))
    body.append(("Implementation Plan", [plan]))
    body.append(("Risks", [_txt_bullets(tdd.risks)]))
    body.append(("Open Questions", [_txt_bullets(tdd.open_questions)]))

    notes = []
    if tdd.error_handling_notes:
        notes.append("Error handling: " + tdd.error_handling_notes)
    if tdd.security_considerations:
        notes.append("Security: " + tdd.security_considerations)
    if tdd.backward_compatibility_notes:
        notes.append("Backward compatibility: " + tdd.backward_compatibility_notes)
    body.append(("Other Notes", notes))

    sections = ["\n".join(header) + "\n\n" + ("#" * 60)]
    number = 0
    for title, parts in body:
        if not any(p and p.strip() for p in parts):
            continue
        number += 1
        sections.append(_txt_section(number, title, *parts))
    return "\n\n\n".join(sections)


# Every label the planner prompts actually emit — the bug mode's value trace
# and graded findings, and the feature mode's gap analysis. They arrive inside
# ONE string field, often as a single run-on paragraph, so a label missing
# from this list renders as a wall of text. "Likely" is retained only so
# older TDDs, written before the prompt said "Hypothesis", still split.
_FINDING_LABELS = (
    "Accepted at",
    "Validated at",
    "Transformed at",
    "Stored at",
    "Consumed at",
    "Confirmed",
    "Hypothesis",
    "Likely",
    "Not the cause",
    "Exists",
    "Missing",
    "Extend",
    "Out of scope",
    "Unknown",
)
_FINDING_LABEL = re.compile(r"\s+(?=(?:" + "|".join(_FINDING_LABELS) + r"):)")


def _one_finding_per_line(text: str | None) -> str | None:
    """Root cause findings ("Confirmed: ...", "Likely: ...", "Not the cause:
    ...") on separate lines — the model often runs them into one paragraph."""
    if not text:
        return text
    return _FINDING_LABEL.sub("\n\n", text.strip())


def _txt_steps(steps) -> str:
    lines = []
    for i, step in enumerate(steps, start=1):
        lines.append(f"{i}. {step.step}")
        if step.detail:
            lines.append(f"   {step.detail}")
    return "\n".join(lines)
