"""Step 8 of the TDD flow (2026-11-02): what every TDD must pass to be final.

1. Its services are a subset of the input's services.
2. Every file it cites was read, or is in that service's pinned file list.
3. Every card question is accounted for: an answered one as a decision, an
   open one as an open question.
4. The work order is usable: steps in order, each for an input service, every
   planned service with at least one step.
5. No code anywhere in it, and no talk of the planner's own reading ("the
   read portion", "unread"): the reader has the repository, not the input.

Each problem says where it is and what is wrong, never quoting the offending
text: the list goes back to the model in the repair call, and into the
draft's reason.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .models import TddDocument

# What reads as code rather than prose: a fence, a line starting like a
# statement, a tag with attributes (<Grid item md={4}>), a prop with a value
# (md={4}), an arrow function, a line ending in ; or {.
_CODE = re.compile(
    r"```"
    r"|^\s*(?:import|export|def|class|function|const|let|var|return|public|private)\s"
    r"|^\s*from \S+ import\s"
    r"|^\s*(?:if|for|while|switch) ?\("
    r"|<[A-Za-z][\w.]*\s+[\w-]+=[\"'{]"
    r"|\b[\w-]+=\{"
    r"|=>"
    r"|[;{]\s*$",
    re.M,
)
# The planner describing its sources instead of the code.
_READING = re.compile(
    r"\b(?:read|unread) (?:portion|part|section)\b|\bunread (?:render|file|code|lines?)\b"
    r"|\bfiles? shown\b|\bshown (?:files?|code|excerpts?)\b|\bthe excerpts?\b",
    re.I,
)
# Ids and paths are names, not prose; they are not scanned.
_NOT_SCANNED = {"id", "path", "file_path", "symbol", "service", "reference_pr", "repo",
                "base_branch", "base_sha", "sha", "branch", "where", "new_file"}


@dataclass
class CheckContext:
    input_services: set[str]
    readable_services: set[str]  # services whose repository was read
    known: Callable[[str, str], bool]  # (service, path): read, or in the pinned file list
    decision_ids: list[str]  # the card's answered questions
    open_questions: list[str]  # the card's unanswered questions


def check(doc: TddDocument, ctx: CheckContext) -> list[str]:
    problems: list[str] = []
    problems += _services(doc, ctx)
    problems += _citations(doc, ctx)
    problems += _questions(doc, ctx)
    problems += _work_order(doc, ctx)
    problems += _code(doc, ctx)
    return problems


def _services(doc: TddDocument, ctx: CheckContext) -> list[str]:
    out = []
    for i, plan in enumerate(doc.per_service_plan):
        if plan.service not in ctx.input_services:
            out.append(
                f"per_service_plan[{i}] is for '{plan.service}', which is not one of the input "
                f"services ({', '.join(sorted(ctx.input_services))})"
            )
    return out


def _citations(doc: TddDocument, ctx: CheckContext) -> list[str]:
    out = []
    for i, plan in enumerate(doc.per_service_plan):
        for j, step in enumerate(plan.steps):
            where = f"per_service_plan[{i}].steps[{j}].where"
            if not step.where or step.new_file:
                continue
            if plan.service not in ctx.readable_services:
                out.append(f"{where} names a file of '{plan.service}', whose repo was not read")
            elif not ctx.known(plan.service, step.where):
                out.append(
                    f"{where}: '{step.where}' is not a file of {plan.service}; use a path given, "
                    "or set new_file for a file to create"
                )
        for j, f in enumerate(plan.files_to_look_at):
            where = f"per_service_plan[{i}].files_to_look_at[{j}]"
            if plan.service not in ctx.readable_services:
                out.append(f"{where} cites a file of '{plan.service}', whose repo was not read")
            elif not ctx.known(plan.service, f.path):
                out.append(
                    f"{where}: '{f.path}' is not a file of {plan.service}; cite only paths given"
                )
    for i, decision in enumerate(doc.decisions):
        for j, target in enumerate(decision.applies_to):
            where = f"decisions[{i}].applies_to[{j}]"
            if target.service not in ctx.readable_services:
                out.append(f"{where} cites a file of '{target.service}', whose repo was not read")
            elif not ctx.known(target.service, target.file_path):
                out.append(
                    f"{where}: '{target.file_path}' is not a file of {target.service}; "
                    "cite only paths given"
                )
    return out


def _questions(doc: TddDocument, ctx: CheckContext) -> list[str]:
    ids = {d.id for d in doc.decisions}
    asked = {_flat(q.question) for q in doc.open_questions}
    out = [f"the card's answered question '{i}' is missing from decisions"
           for i in ctx.decision_ids if i not in ids]
    out += [f"the card's open question '{q[:80]}' is missing from open_questions"
            for q in ctx.open_questions if _flat(q) not in asked]
    return out


def _work_order(doc: TddDocument, ctx: CheckContext) -> list[str]:
    steps = doc.work_order.steps
    out = []
    if doc.per_service_plan and not steps:
        out.append("work_order has no steps")
    orders = [s.order for s in steps]
    if orders != list(range(1, len(steps) + 1)):
        out.append("work_order steps must be numbered 1, 2, 3, ... in the order to do them")
    for i, step in enumerate(steps):
        if step.service not in ctx.input_services:
            out.append(f"work_order.steps[{i}] is for '{step.service}', not an input service")
        if not step.summary.strip():
            out.append(f"work_order.steps[{i}] has no summary")
    stepped = {s.service for s in steps}
    for plan in doc.per_service_plan:
        if plan.service not in stepped:
            out.append(f"'{plan.service}' is in per_service_plan but has no work_order step")
    return out


def _code(doc: TddDocument, ctx: CheckContext) -> list[str]:
    """Code in anything the model wrote. The card's own answers and open
    questions are a person's words, copied as they are: not scanned."""
    out = []

    def walk(value: object, where: str, key: str) -> None:
        if isinstance(value, str):
            if key in _NOT_SCANNED:
                return
            if _CODE.search(value):
                out.append(f"{where} looks like code; describe the change in words instead")
            if _READING.search(value):
                out.append(
                    f"{where} talks about what was read; state the fact, or say 'likely' "
                    "and name the file where it is confirmed"
                )
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{where}.{k}" if where else k, k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{where}[{i}]", key)

    # Only what the model wrote: the flow's own fields hold paths and notes.
    written = doc.model_dump(
        include={"title", "problem", "cause", "fix", "requirements", "decisions",
                 "current_flow", "flow_after_change", "per_service_plan", "order_of_work",
                 "risks_and_tests", "open_questions", "work_order"}
    )
    card_ids = set(ctx.decision_ids)
    written["decisions"] = [d if d["id"] not in card_ids else {} for d in written["decisions"]]
    written["open_questions"] = [
        q if q["source"] != "card" else {} for q in written["open_questions"]
    ]
    walk(written, "", "")
    return out


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())
