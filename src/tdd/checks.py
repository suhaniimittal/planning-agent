"""Step 8 of the TDD flow (2026-11-02): what every TDD must pass to be final.

1. Its services are a subset of the input's services.
2. Every step's area is a folder of that service at its pinned commit, and
   no file is named anywhere (2026-11-06): with no code read, a file name is
   a guess. A file a reference change changed is a fact, and may be named.
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
from dataclasses import dataclass, field

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
# A file name: a name with a code or config extension.
_FILE_NAME = re.compile(
    r"\b[\w@-]+(?:\.[\w-]+)*\.(?:tsx?|jsx?|mjs|kts?|java|py|go|rb|cs|sql|graphqls?|gql|ya?ml"
    r"|json|xml|s?css|html|properties|gradle|avsc|proto)\b"
)
# Ids and paths are names, not prose; they are not scanned.
_NOT_SCANNED = {"id", "path", "file_path", "symbol", "service", "reference_pr", "repo",
                "base_branch", "base_sha", "sha", "branch", "areas", "new_area", "confirm_in"}


@dataclass
class CheckContext:
    input_services: set[str]
    readable_services: set[str]  # services whose repository was read
    known: Callable[[str, str], bool]  # (service, path): in the pinned file list
    known_folder: Callable[[str, str], bool]  # (service, folder): in the pinned file list
    decision_ids: list[str]  # the card's answered questions
    open_questions: list[str]  # the card's unanswered questions
    # File names a reference change changed: facts, so they may be named.
    reference_files: set[str] = field(default_factory=set)


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
            for k, area in enumerate(step.areas):
                where = f"per_service_plan[{i}].steps[{j}].areas[{k}]"
                out += _folder(where, plan.service, area, step.new_area and k == 0, ctx)
    for i, need in enumerate(doc.data_needed):
        if need.confirm_in:
            out += _folder(f"data_needed[{i}].confirm_in", need.service, need.confirm_in,
                           False, ctx)
    for i, decision in enumerate(doc.decisions):
        if decision.applies_to and decision.id not in ctx.decision_ids:
            out.append(f"decisions[{i}].applies_to must be empty: name no files")
    return out


def _folder(where: str, service: str, folder: str, new: bool, ctx: CheckContext) -> list[str]:
    """Why ``folder`` cannot be cited for ``service``; empty when it can."""
    if service not in ctx.readable_services:
        return [f"{where} names a folder of '{service}', whose repo was not read"]
    parent = folder.rsplit("/", 1)[0] if new else folder
    if ctx.known_folder(service, parent):
        return []
    return [
        f"{where}: '{folder}' is not a folder of {service}; give a folder that appears in the "
        "paths given (set new_area for a folder to create inside one)"
    ]


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
            named = {m.group(0) for m in _FILE_NAME.finditer(value)} - ctx.reference_files
            if named:
                out.append(
                    f"{where} names a file; describe the component in words, and give its "
                    "folder as the step's area"
                )
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
                 "current_flow", "per_service_plan", "data_needed", "risks_and_tests",
                 "open_questions", "work_order"}
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
