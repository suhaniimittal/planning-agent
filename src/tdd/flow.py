"""The TDD flow, start to end (2026-11-02).

    1-2  prepare     which services can be read; one pinned commit each
    3    service map the run's services only; paths only with code reads off
    4    evidence    reference changes, candidate files, reads within the caps
    5    questions   answered -> decisions, unanswered -> open (never answered here)
    6    scout       optional model call: extra files from the path list
    7    writer      model call: the TDD as JSON, no code
    8    checks      services, citations, questions, work order, no code;
                     one repair call, then a draft with the reason
    9-10 the flow's own fields: pins, what was and was not read, the work order
    11   every file text and patch dropped, whatever happened

GitHub I/O is synchronous (github.py) and runs in a thread. No usable
service is not an error: the TDD is written from the ticket alone, as a
draft whose work order is not ready. Nothing here writes a file or logs a
line of code: the log holds names, paths, counts and sizes.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime

from pydantic import ValidationError

from src.query import planner

from . import prompts, questions
from .attachments import Attachment
from .checks import CheckContext, check
from .config import Settings, describe
from .evidence import Evidence, Gatherer
from .github import GitHubReader
from .inputs import PlanInput
from .models import OpenQuestion, PinnedService, TddDocument, WorkStep
from .prepare import Prepared, prepare
from .service_map.slice import MapSlice, build_slice

# Fields the flow sets; whatever the model put there is discarded.
_FLOW_FIELDS = ("ticket", "ticket_type", "status", "draft_reason", "high_level", "pinned",
                "files_read", "not_read", "warnings", "generated_at")
_LINE_RANGE = re.compile(r"(?::\d+(?:-\d+)?|#L\d+(?:-L\d+)?)$")


async def run(
    plan: PlanInput,
    settings: Settings,
    attachments: list[Attachment],
    memory: str = "",
    reader: GitHubReader | None = None,
) -> TddDocument:
    print(f"  [TDD] {plan.card.key}: {describe(settings)}")
    reader = reader or GitHubReader(settings)
    gatherer: Gatherer | None = None
    try:
        prepared = await asyncio.to_thread(prepare, plan, reader)
        map_slice = MapSlice(paths_only=not settings.allow_code_reads)
        if prepared.services:
            map_slice = await asyncio.to_thread(build_slice, reader, prepared.services, settings)
        gatherer = Gatherer(reader, settings, prepared.services, map_slice)
        ev = gatherer.ev
        if prepared.services:
            ev = await asyncio.to_thread(gatherer.gather, plan)
        decided, still_open = questions.split(plan.questions)
        map_view = map_slice.view()
        if settings.scout and prepared.services and not gatherer.good_enough():
            await _scout(plan, attachments, map_view, gatherer, prepared)
        elif prepared.services:
            print("  [TDD] scout: skipped, the evidence already has a strong lead per service")
        if prepared.services:
            # The room kept for the scout, when it was skipped, failed or picked few.
            extra = await asyncio.to_thread(gatherer.fill)
            if extra:
                print(f"  [TDD] filled the scout's room: {extra} more file(s) read")
        return await _write(
            plan, settings, attachments, memory, prepared, map_slice, map_view, ev,
            decided, still_open,
        )
    finally:
        # Step 11: the run's code leaves memory with this run. Python frees a
        # string once nothing refers to it; this removes every reference
        # the run kept.
        if gatherer is not None:
            gatherer.ev.drop()
        reader.close()


# --- step 6: scout -------------------------------------------------------------------------


async def _scout(
    plan: PlanInput, attachments: list[Attachment], map_view: dict, gatherer: Gatherer,
    prepared: Prepared,
) -> None:
    user = prompts.scout_user(plan, attachments, map_view, gatherer.ev, prepared)
    print(f"  [TDD] scout: asking for files ({len(user):,} chars of prompt)")
    try:
        raw = await planner.call_llm(prompts.scout_system(), user)
        picks = _json_object(raw).get("files") or []
    except Exception as error:  # noqa: BLE001 - the scout only adds; the TDD goes on without it
        gatherer.ev.warnings.append(f"the scout call failed ({type(error).__name__}); skipped")
        return
    picks = [p for p in picks if isinstance(p, dict)][: prompts.SCOUT_PICKS]
    before = len(gatherer.ev.files)
    refused = await asyncio.to_thread(gatherer.read_more, picks)
    gatherer.ev.warnings += refused
    print(f"  [TDD] scout: {len(picks)} picked, {len(gatherer.ev.files) - before} read, "
          f"{len(refused)} refused")


# --- steps 7 to 10: write, check, complete -----------------------------------------------


async def _write(
    plan: PlanInput,
    settings: Settings,
    attachments: list[Attachment],
    memory: str,
    prepared: Prepared,
    map_slice: MapSlice,
    map_view: dict,
    ev: Evidence,
    decided: list,
    still_open: list[OpenQuestion],
) -> TddDocument:
    high_level = not settings.allow_code_reads or not ev.files
    images = [a.image for a in attachments if a.image is not None]
    system = prompts.writer_system(plan.card.is_bug, high_level)
    user = prompts.writer_user(
        plan, attachments, decided, still_open, memory, prepared, map_view, ev, bool(images)
    )
    print(f"  [TDD] writer: {len(system):,} + {len(user):,} chars of prompt, "
          f"{len(images)} image(s)")
    ctx = CheckContext(
        input_services={s.service for s in plan.services},
        readable_services=set(ev.trees),
        known=lambda service, path: ev.known(service, path),
        decision_ids=[d.id for d in decided],
        open_questions=[q.question for q in still_open],
    )

    def finish(doc: TddDocument) -> list[str]:
        _complete(doc, plan, prepared, map_slice, ev, decided, still_open, high_level)
        return check(doc, ctx)

    raw = await _ask(system, user, images)
    doc, problems = _parse(raw)
    if doc is not None:
        problems = finish(doc)
    if problems:
        print(f"  [TDD] checks: {len(problems)} problem(s); one repair call")
        raw = await _ask(system, prompts.repair_user(user, raw, problems), images)
        repaired, unreadable = _parse(raw)
        if repaired is not None:
            doc, problems = repaired, finish(repaired)
        elif doc is None:
            problems = unreadable
        # else: the repair could not be read; the first answer and its problems stand.
    if doc is None:
        doc = _ticket_only(plan, decided, still_open)
        finish(doc)
        _draft(doc, "the model's answer could not be read as a TDD: " + "; ".join(problems[:5]))
    elif problems:
        _drop_bad_citations(doc, ctx)
        _draft(doc, "it did not pass every check after one repair: " + "; ".join(problems[:8]))
    if not prepared.services:
        _draft(doc, "no repository could be read: " + "; ".join(prepared.skipped or ["none given"]))
    _ready(doc, plan)
    print(f"  [TDD] {doc.status}: {len(doc.per_service_plan)} service plan(s), "
          f"{len(doc.decisions)} decisions, {len(doc.open_questions)} open questions, "
          f"work order {'ready' if doc.work_order.ready else 'not ready'}")
    # What the TDD was based on: for the log, not the document.
    for line in doc.files_read:
        print(f"  [TDD] read {line}")
    for line in doc.not_read[:30]:
        print(f"  [TDD] not read: {line}")
    for line in doc.warnings:
        print(f"  [TDD] warning: {line}")
    return doc


async def _ask(system: str, user: str, images: list) -> str:
    # Images only when there are some: call_llm's two-argument form otherwise.
    args = (system, user, images) if images else (system, user)
    return await planner.call_llm(*args)


def _json_object(raw: str) -> dict:
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the answer")
    return json.loads(text[start : end + 1])


def _parse(raw: str) -> tuple[TddDocument | None, list[str]]:
    """(the TDD, []) or (None, why it could not be read). Never quotes the answer."""
    try:
        data = _json_object(raw)
    except ValueError as error:
        return None, [f"the answer is not one JSON object ({type(error).__name__})"]
    for name in _FLOW_FIELDS:
        data.pop(name, None)
    try:
        return TddDocument.model_validate(data), []
    except ValidationError as error:
        return None, [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()[:15]
        ]


def _complete(
    doc: TddDocument,
    plan: PlanInput,
    prepared: Prepared,
    map_slice: MapSlice,
    ev: Evidence,
    decided: list,
    still_open: list[OpenQuestion],
    high_level: bool,
) -> None:
    """The flow's own fields, and the card's questions put back word for word."""
    card = plan.card
    doc.ticket, doc.ticket_type = card.key, card.type
    doc.title = doc.title.strip() or card.title
    doc.generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    doc.high_level = high_level
    doc.pinned = [
        PinnedService(service=s.name, repo=s.repo, branch=s.branch, sha=s.sha,
                      path=map_slice.folders.get(s.name, s.path))
        for s in prepared.services
    ]
    doc.files_read = [f"{f.service}/{f.path}" + (" (cut)" if f.cut else "") for f in ev.files]
    doc.not_read = list(ev.not_read) + list(ev.unread_leads)
    doc.warnings = list(dict.fromkeys(
        prepared.warnings + [f"skipped {s}" for s in prepared.skipped]
        + map_slice.notes + ev.warnings
    ))
    _fix_paths(doc, ev)
    # Decisions: the card's, word for word, then the model's own.
    card_ids = {d.id for d in decided}
    own = [d for d in doc.decisions if d.id and d.choice and d.id not in card_ids]
    doc.decisions = list(decided) + list({d.id: d for d in own}.values())
    # Open questions: the card's, word for word, then the model's own.
    asked = {_flat(q.question) for q in still_open}
    mine = [OpenQuestion(question=q.question, why_it_matters=q.why_it_matters, source="planner")
            for q in doc.open_questions if q.question and _flat(q.question) not in asked]
    doc.open_questions = list(still_open) + mine
    # The work order: renumbered, and where to work filled from the input.
    by_name = {s.name: s for s in prepared.services}
    doc.work_order.ticket = card.key
    doc.work_order.steps = sorted(doc.work_order.steps, key=lambda s: s.order or 10_000)
    for order, step in enumerate(doc.work_order.steps, start=1):
        step.order = order
        svc = by_name.get(step.service)
        step.repo = svc.repo if svc else ""
        step.path = svc.path if svc else ""
        step.base_branch = svc.branch if svc else ""
        step.base_sha = svc.sha if svc else ""
    doc.work_order.open_questions = [q.question for q in doc.open_questions]


def _fix_paths(doc: TddDocument, ev: Evidence) -> None:
    """A path the model wrote as "ui/src/x.tsx" (as the prompt shows it), with
    a leading slash or a line range, back to the tree's own "src/x.tsx"."""

    def fixed(service: str, path: str) -> str:
        path = _LINE_RANGE.sub("", path.strip()).lstrip("/")
        if not ev.known(service, path) and path.startswith(service + "/"):
            trimmed = path[len(service) + 1 :]
            if ev.known(service, trimmed):
                return trimmed
        return path

    for plan in doc.per_service_plan:
        for step in plan.steps:
            if step.where:
                step.where = fixed(plan.service, step.where)
        for f in plan.files_to_look_at:
            f.path = fixed(plan.service, f.path)
    for decision in doc.decisions:
        for target in decision.applies_to:
            target.file_path = fixed(target.service, target.file_path)


def _drop_bad_citations(doc: TddDocument, ctx: CheckContext) -> None:
    """In a draft, a cited file that does not exist is removed (and said so),
    so nobody is sent to look for it."""
    for plan in doc.per_service_plan:
        readable = plan.service in ctx.readable_services
        for step in plan.steps:
            if step.where and not step.new_file and not (
                readable and ctx.known(plan.service, step.where)
            ):
                doc.warnings.append(f"removed a step's file that is not in {plan.service}: "
                                    f"{step.where}")
                step.where = ""
        kept = []
        for f in plan.files_to_look_at:
            if readable and ctx.known(plan.service, f.path):
                kept.append(f)
            else:
                doc.warnings.append(f"removed a cited file that is not in {plan.service}: {f.path}")
        plan.files_to_look_at = kept
    for decision in doc.decisions:
        decision.applies_to = [
            t for t in decision.applies_to
            if t.service in ctx.readable_services and ctx.known(t.service, t.file_path)
        ]


def _ticket_only(
    plan: PlanInput, decided: list, still_open: list[OpenQuestion]
) -> TddDocument:
    """The document when the model's answer could not be used at all: the
    ticket, its decisions and its open questions, and no plan."""
    card = plan.card
    return TddDocument(
        title=card.title,
        problem=card.description or card.title,
        decisions=list(decided),
        open_questions=list(still_open),
        work_order={"steps": [WorkStep(order=0, service=s.service, summary=s.reason or "-")
                              for s in plan.ranked_services()]},
    )


def _draft(doc: TddDocument, reason: str) -> None:
    doc.status = "draft"
    doc.draft_reason = f"{doc.draft_reason}; {reason}" if doc.draft_reason else reason
    print(f"  [TDD] draft: {reason}")


def _ready(doc: TddDocument, plan: PlanInput) -> None:
    """Whether the coding agent can start: a final TDD, steps that each have a
    repository, and no question from the card still open."""
    wo = doc.work_order
    reasons = []
    if doc.status == "draft":
        reasons.append("the TDD is a draft")
    if not wo.steps:
        reasons.append("there are no steps")
    if any(not s.repo for s in wo.steps):
        reasons.append("a step's service has no repository that could be read")
    open_card = [q for q in doc.open_questions if q.source == "card"]
    if open_card:
        reasons.append(f"{len(open_card)} question(s) on the card are still open")
    wo.ready = not reasons
    wo.not_ready_reason = "; ".join(reasons)


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())
