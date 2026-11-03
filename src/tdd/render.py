"""Step 9 of the TDD flow (2026-11-02): the TDD as a document people and the
coding agent read.

One outline, drawn twice: HTML (Confluence, and the PDF through
src/query/pdf_render.py) and plain text (what Orchestration hands to Coding).
It reads as a design and an instruction guide — what is wrong and why, which
repositories to clone at which commit, what was agreed, how it works now and
after, then the steps, the tests and the work order. What the planner read
and skipped is the log's and Orchestration's (`files_read`, `not_read`,
`warnings`), not the document's.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from .models import Step, TddDocument

if TYPE_CHECKING:
    from .attachments import Attachment
    from .inputs import PlanInput

# The coding agent's copy of the ticket: one attachment's text, and the
# whole copy (its prompt also holds the code), the longest attachments cut first.
_ATTACHMENT_CHARS = 8_000
_ISSUE_CHARS = 30_000


@dataclass
class Block:
    """One piece of a section. kind: "p", "note", "ul", "ol", "kv" (label/value
    rows), "table" (head + rows), "steps", "h3"."""

    kind: str
    content: object
    head: tuple[str, ...] = ()
    widths: tuple[int, ...] = ()  # a table's column widths, in percent


def _outline(doc: TddDocument) -> list[tuple[str, list[Block]]]:
    """[(heading, blocks)], in reading order; an empty section is left out."""
    sections: list[tuple[str, list[Block]]] = []

    def add(title: str, *blocks: Block) -> None:
        kept = [b for b in blocks if b.content]
        if kept:
            sections.append((title, kept))

    # Written from file paths only, the cause and the fix are the planner's
    # best reading, for the developer or the coding agent to confirm.
    cause, fix = ("Likely cause", "Suggested approach") if doc.high_level else ("Cause", "Fix")
    add("Summary", Block("kv", [("Problem", doc.problem), (cause, doc.cause), (fix, doc.fix)]
                         if (doc.problem or doc.cause or doc.fix) else []))

    pinned = {p.service: p for p in doc.pinned}
    planned = [p.service for p in doc.per_service_plan] or list(pinned)
    what = {p.service: p.what_changes for p in doc.per_service_plan}
    rows = []
    for name in dict.fromkeys(planned):
        pin = pinned.get(name)
        rows.append([
            name,
            pin.repo if pin else "not readable",
            pin.branch if pin else "-",
            pin.sha[:12] if pin else "-",
            (pin.path or "whole repository") if pin else "-",
            what.get(name, ""),
        ])
    add(
        "Services to change",
        Block("p", "Clone each repository at the commit shown; the paths below are inside it."
              if rows else ""),
        Block("table", rows, ("Service", "Repository", "Branch", "Commit", "Folder",
                              "What changes"), (11, 17, 11, 13, 14, 34)),
    )

    agreed = doc.requirements or [f"{d.title} {d.choice}" for d in doc.card_decisions()]
    design = [
        f"{d.title}: {d.choice} Why: {d.reason}"
        + (f" Applies to {', '.join(t.file_path for t in d.applies_to)}." if d.applies_to else "")
        for d in doc.design_decisions()
    ]
    add(
        "Agreed on the ticket",
        Block("ul", agreed),
        Block("h3", "Design decisions" if design else ""),
        Block("ul", design),
    )

    several = len({p.service for p in doc.per_service_plan}) > 1

    def flow(steps) -> list[str]:
        return [f"{s.step} ({s.service})" if several and s.service else s.step for s in steps]

    add("How it works today", Block("ol", flow(doc.current_flow)))
    add("How it will work", Block("ol", flow(doc.flow_after_change)))

    for plan in doc.per_service_plan:
        add(
            f"Implementation guide: {plan.service}",
            Block("p", plan.what_changes),
            Block("steps", plan.steps),
            Block("h3", "Files to look at" if plan.files_to_look_at else ""),
            Block("ul", [f"{f.path}: {f.why}" if f.why else f.path
                         for f in plan.files_to_look_at]),
            Block("p", f"Reference change: {plan.reference_pr}" if plan.reference_pr else ""),
        )

    add("Order of work", Block("ol", doc.order_of_work))
    add("Test plan", Block("ol", doc.risks_and_tests.tests))
    add("Risks", Block("ul", doc.risks_and_tests.risks))
    add(
        "Open questions",
        Block("ul", [
            q.question
            + (" (from the ticket)" if q.source == "card" else "")
            + (f" Why it matters: {q.why_it_matters}" if q.why_it_matters else "")
            for q in doc.open_questions
        ]),
    )

    wo = doc.work_order
    add(
        "Work order for the coding agent",
        Block("table", [
            [str(s.order), s.service,
             f"{s.repo} @ {s.base_sha[:12]}" if s.repo else "not readable",
             s.path or ("whole repository" if s.repo else "-"), s.summary,
             s.reference_pr or "-"]
            for s in wo.steps
        ], ("#", "Service", "Repository @ commit", "Folder", "Task", "Reference"),
           (5, 10, 22, 11, 41, 11)),
        Block("p", "Ready for the coding agent." if wo.ready
              else f"Not ready for the coding agent: {wo.not_ready_reason}." if wo.steps else ""),
    )
    return sections


def coding_task(doc: TddDocument, issue: dict | None = None) -> dict:
    """The TDD as data for coding_agent (2026-11-02), kept inside the PDF as
    hidden metadata (pdf_render.pdf_from_html): what the pages say, without
    their layout, so the coding agent reads it exactly and with no model
    call. Commits are full shas here; the pages show 12 characters.
    ``issue`` (from issue()) is the ticket itself, which the pages do not show."""
    pinned = {p.service: p for p in doc.pinned}
    planned = [p.service for p in doc.per_service_plan] or list(pinned)
    task = {
        "version": 1,
        "ticket": doc.ticket,
        "ticket_type": doc.ticket_type,
        "status": doc.status,
        "high_level": doc.high_level,
        "title": doc.title,
        "problem": doc.problem,
        "cause": doc.cause,
        "fix": doc.fix,
        "services": [
            {"service": name, "repo": pin.repo, "branch": pin.branch, "commit": pin.sha,
             "folder": pin.path}
            for name in dict.fromkeys(planned)
            if (pin := pinned.get(name))
        ],
        "requirements": doc.requirements
        or [f"{d.title} {d.choice}" for d in doc.card_decisions()],
        "design_decisions": [
            f"{d.title}: {d.choice} Why: {d.reason}" for d in doc.design_decisions()
        ],
        "flow_today": [s.step for s in doc.current_flow],
        "flow_after": [s.step for s in doc.flow_after_change],
        "steps": [
            {"service": plan.service, "title": s.title,
             "where": f"{s.where} (new file)" if s.where and s.new_file else s.where,
             "what": s.what, "why": s.why, "done_when": s.done_when}
            for plan in doc.per_service_plan
            for s in plan.steps
        ],
        "files_to_look_at": [
            {"service": plan.service, "path": f.path, "why": f.why}
            for plan in doc.per_service_plan
            for f in plan.files_to_look_at
        ],
        "order_of_work": doc.order_of_work,
        "tests": doc.risks_and_tests.tests,
        "risks": doc.risks_and_tests.risks,
        "open_questions": [
            q.question + (f" Why it matters: {q.why_it_matters}" if q.why_it_matters else "")
            for q in doc.open_questions
        ],
        # The same questions with who raised them: "card", asked on the ticket and
        # unanswered, is a person's to decide; "planner", raised while planning,
        # is usually one the code can answer.
        "open_questions_detail": [
            {"question": q.question, "why_it_matters": q.why_it_matters, "source": q.source}
            for q in doc.open_questions
        ],
    }
    if issue:
        task["issue"] = issue
    return task


def issue(plan: PlanInput, attachments: list[Attachment]) -> dict:
    """The ticket as it was given, for the coding agent (2026-11-02): the TDD
    is a reference, the ticket is what must be true. Its description, each
    attachment's description and text (for a screenshot on the platform, the
    vision call's account of it), and the card's answered questions word for
    word; the open ones are the TDD's open_questions. At most _ISSUE_CHARS
    in all: what is left after the description and the answers is shared by
    the attachments' texts, the longest cut first."""
    card = plan.card
    description = card.description[:_ISSUE_CHARS]
    answers = [
        {"question": q.question, "answers": list(q.answers)}
        for q in plan.questions
        if q.answered
    ]
    used = len(description) + sum(
        len(a["question"]) + sum(len(x) for x in a["answers"]) for a in answers
    )
    texts = [a.text[:_ATTACHMENT_CHARS] for a in attachments]
    fitted = _fit(texts, _ISSUE_CHARS - used - sum(len(a.description) for a in attachments))
    return {
        "key": card.key,
        "type": card.type,
        "title": card.title,
        "description": description,
        "attachments": [
            {"name": a.name, "description": a.description, "text": text,
             "cut": len(text) < len(a.text)}
            for a, text in zip(attachments, fitted, strict=True)
        ],
        "answers": answers,
    }


def _fit(texts: list[str], budget: int) -> list[str]:
    """The texts within ``budget`` characters in all: the longest are cut to
    one even length, the shorter ones kept whole."""
    budget = max(budget, 0)
    if sum(len(t) for t in texts) <= budget:
        return texts
    sizes = sorted(len(t) for t in texts)
    left, cap = budget, 0
    for i, size in enumerate(sizes):
        share = left // (len(sizes) - i)
        if size > share:
            cap = share
            break
        left -= size
    return [t[:cap] for t in texts]


def _header(doc: TddDocument) -> tuple[str, list[str]]:
    """(the line under the title, notes above the first section)."""
    status = "Draft" if doc.status == "draft" else "Final"
    meta = " · ".join(p for p in (doc.ticket, doc.ticket_type, status, _day(doc.generated_at)) if p)
    notes = []
    if doc.status == "draft":
        notes.append(f"Draft: {doc.draft_reason}")
    if doc.high_level:
        notes.append("High-level: written without reading the source code. Confirm each step "
                     "in the file it names before changing it.")
    return meta, notes


def _footer(doc: TddDocument) -> str:
    if not doc.pinned:
        return "Based on the ticket only: no repository could be read."
    if doc.high_level:
        where = ", ".join(f"{p.repo} at {p.sha[:12]}" for p in doc.pinned)
        return f"Based on the file lists of {where}."
    count = {p.service: 0 for p in doc.pinned}
    for line in doc.files_read:
        service = line.split("/", 1)[0]
        if service in count:
            count[service] += 1
    return "Based on " + "; ".join(
        f"{count[p.service]} files of {p.repo} at commit {p.sha[:12]}" for p in doc.pinned
    ) + "."


def _day(iso: str) -> str:
    try:
        when = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    return f"{when.day} {when:%b %Y}"


def _step_rows(step: Step) -> list[tuple[str, str]]:
    where = f"{step.where} (new file)" if step.where and step.new_file else step.where
    return [(k, v) for k, v in (("Where", where), ("What", step.what), ("Why", step.why),
                                ("Done when", step.done_when)) if v]


# --- HTML -------------------------------------------------------------------------------


def _e(text: object) -> str:
    return html.escape(str(text))


def _html_block(block: Block) -> str:
    if block.kind == "p":
        return f"<p>{_e(block.content)}</p>"
    if block.kind == "h3":
        return f"<h3>{_e(block.content)}</h3>"
    if block.kind in ("ul", "ol"):
        items = "".join(f"<li>{_e(i)}</li>" for i in block.content)
        return f"<{block.kind}>{items}</{block.kind}>"
    if block.kind == "kv":
        rows = "".join(f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in block.content if v)
        return f'<table class="kv"><tbody>{rows}</tbody></table>'
    if block.kind == "table":
        widths = block.widths or (0,) * len(block.head)
        head = "".join(
            f'<th style="width: {w}%">{_e(h)}</th>' if w else f"<th>{_e(h)}</th>"
            for h, w in zip(block.head, widths, strict=True)
        )
        body = "".join(
            "<tr>" + "".join(f"<td>{_e(c)}</td>" for c in row) + "</tr>" for row in block.content
        )
        return f'<table class="grid"><tbody><tr>{head}</tr>{body}</tbody></table>'
    if block.kind == "steps":
        out = []
        for n, step in enumerate(block.content, start=1):
            title = f"Step {n}" + (f". {step.title}" if step.title else "")
            rows = "".join(
                f"<tr><th>{_e(k)}</th><td>{_e(v)}</td></tr>" for k, v in _step_rows(step)
            )
            out.append(f'<h4>{_e(title)}</h4><table class="kv"><tbody>{rows}</tbody></table>')
        return "".join(out)
    raise ValueError(f"unknown block {block.kind}")


def render_html(doc: TddDocument) -> str:
    """The document's body as HTML: Confluence takes it as it is; the PDF adds a style."""
    meta, notes = _header(doc)
    parts = [f"<h1>{_e(doc.title)}</h1>", f'<p class="meta">{_e(meta)}</p>']
    parts += [f'<p class="note">{_e(n)}</p>' for n in notes]
    for number, (title, blocks) in enumerate(_outline(doc), start=1):
        parts.append(f"<h2>{number}. {_e(title)}</h2>")
        parts += [_html_block(b) for b in blocks]
    parts.append(f'<p class="foot">{_e(_footer(doc))}</p>')
    return "\n".join(parts)


# --- text -------------------------------------------------------------------------------


def _text_block(block: Block) -> list[str]:
    if block.kind == "p":
        return [f"   {block.content}"]
    if block.kind == "h3":
        return ["", f"   {block.content}"]
    if block.kind == "ul":
        return [f"   - {i}" for i in block.content]
    if block.kind == "ol":
        return [f"   {n}. {i}" for n, i in enumerate(block.content, start=1)]
    if block.kind == "kv":
        return [f"   {k + ':':<11}{v}" for k, v in block.content if v]
    if block.kind == "table":
        return [
            "   - " + "; ".join(f"{h}: {c}" for h, c in zip(block.head, row, strict=True) if c)
            for row in block.content
        ]
    if block.kind == "steps":
        lines = []
        for n, step in enumerate(block.content, start=1):
            lines.append(f"   Step {n}" + (f". {step.title}" if step.title else ""))
            lines += [f"     {k + ':':<11}{v}" for k, v in _step_rows(step)]
        return lines
    raise ValueError(f"unknown block {block.kind}")


def render_text(doc: TddDocument) -> str:
    meta, notes = _header(doc)
    lines = [doc.title, meta, *notes, ""]
    for number, (title, blocks) in enumerate(_outline(doc), start=1):
        lines.append(f"{number}. {title.upper()}")
        for block in blocks:
            lines += _text_block(block)
        lines.append("")
    lines.append(_footer(doc))
    return "\n".join(lines) + "\n"
