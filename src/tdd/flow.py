"""The TDD flow, start to end (2026-11-02).

    1-2  prepare     which services can be read; one pinned commit each
    3-4  evidence    reference changes, candidate files and contract files, by path
    5    questions   answered -> decisions, unanswered -> open (never answered here)
    6    writer      model call: the TDD as JSON, no code
    7    checks      services, citations, questions, work order, no code;
                     one repair call, then a draft with the reason
    8-9  the flow's own fields: pins, the leads, the work order

No file's contents are read (2026-11-03): the TDD is high-level, and the
coding agent reads the code.

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
from .evidence import Evidence, Gatherer, is_test
from .github import GitHubReader
from .inputs import PlanInput
from .models import CARD_DECISION, OpenQuestion, PinnedService, TddDocument, WorkStep
from .prepare import Prepared, prepare

# Fields the flow sets; whatever the model put there is discarded.
_FLOW_FIELDS = ("ticket", "ticket_type", "status", "draft_reason", "high_level", "pinned",
                "files_read", "not_read", "hints", "warnings", "generated_at")
_LINE_RANGE = re.compile(r"(?::\d+(?:-\d+)?|#L\d+(?:-L\d+)?)$")
# A requirement about where to check, not what the app does: no edit can make
# "verified in Chrome 130" true (2026-11-03). Moved to the tests when it names
# a browser, device, size or zoom and either says to check or has no "must".
_ENVIRONMENT = re.compile(
    r"\b(?:chrome|edge|firefox|safari|browsers?|windows|macos|ios|android|zoom"
    r"|\d{3,4}\s*[x×]\s*\d{3,4})\b",
    re.I,
)
_VERIFY = re.compile(r"\b(?:verif\w*|check\w*|tested|in-scope)\b", re.I)
_MUST = re.compile(r"\bmust\b", re.I)
_PATHISH = re.compile(r"[\w@.-]+(?:/[\w@.-]+)+")
# Words that say nothing about what a question asks.
_QUESTION_WORDS = {
    "what", "which", "where", "when", "should", "would", "could", "does", "there", "this",
    "that", "with", "from", "have", "will", "they", "them", "their", "about", "also", "only",
    "into", "than", "then", "been", "being", "after", "before", "each", "other", "need",
    "needs", "happen", "happens", "still", "confirm", "specified", "story", "ticket",
}
# A question of the planner's that shares this much of its telling words with
# a card question repeats it.
_SAME_QUESTION = 0.6
# An id worth replacing in prose: "d3", "q2", or a slug with a hyphen; a
# one-word slug could be an ordinary word.
_CODE_ID = re.compile(r"^(?:[a-z]{1,2}\d+|[a-z0-9]+(?:-[a-z0-9]+)+)$")


async def run(
    plan: PlanInput,
    settings: Settings,
    attachments: list[Attachment],
    memory: str = "",
    reader: GitHubReader | None = None,
) -> TddDocument:
    print(f"  [TDD] {plan.card.key}: {describe(settings)}")
    reader = reader or GitHubReader(settings)
    try:
        prepared = await asyncio.to_thread(prepare, plan, reader)
        gatherer = Gatherer(reader, settings, prepared.services)
        ev = gatherer.ev
        if prepared.services:
            ev = await asyncio.to_thread(gatherer.gather, plan)
        decided, still_open = questions.split(plan.questions)
        return await _write(
            plan, attachments, memory, prepared, ev, decided, still_open,
        )
    finally:
        reader.close()


# --- steps 6 to 9: write, check, complete -----------------------------------------------


async def _write(
    plan: PlanInput,
    attachments: list[Attachment],
    memory: str,
    prepared: Prepared,
    ev: Evidence,
    decided: list,
    still_open: list[OpenQuestion],
) -> TddDocument:
    images = [a.image for a in attachments if a.image is not None]
    system = prompts.writer_system(plan.card.is_bug)
    user = prompts.writer_user(
        plan, attachments, decided, still_open, memory, prepared, ev, bool(images)
    )
    print(f"  [TDD] writer: {len(system):,} + {len(user):,} chars of prompt, "
          f"{len(images)} image(s)")
    ctx = CheckContext(
        input_services={s.service for s in plan.services},
        readable_services=set(ev.trees),
        known=lambda service, path: ev.known(service, path),
        known_folder=lambda service, folder: ev.known_folder(service, folder),
        decision_ids=[d.id for d in decided],
        open_questions=[q.question for q in still_open],
        reference_files={
            f.path.rsplit("/", 1)[-1] for r in ev.references for f in r.files
        },
    )

    def finish(doc: TddDocument) -> list[str]:
        _complete(doc, plan, prepared, ev, decided, still_open)
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
    for line in ev.not_read[:30]:
        print(f"  [TDD] note: {line}")
    for line in ev.unread_leads[:30]:
        print(f"  [TDD] lead: {line}")
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
    ev: Evidence,
    decided: list,
    still_open: list[OpenQuestion],
) -> None:
    """The flow's own fields, and the card's questions put back word for word."""
    card = plan.card
    doc.ticket, doc.ticket_type = card.key, card.type
    doc.title = doc.title.strip() or card.title
    doc.generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    doc.high_level = True  # no file's contents are read
    doc.pinned = [
        PinnedService(service=s.name, repo=s.repo, branch=s.branch, sha=s.sha,
                      path=ev.folders.get(s.name, s.path))
        for s in prepared.services
    ]
    doc.not_read = list(ev.not_read)
    doc.hints = list(ev.unread_leads)
    doc.warnings = list(dict.fromkeys(
        prepared.warnings + [f"skipped {s}" for s in prepared.skipped]
        + ev.warnings
    ))
    _fix_areas(doc, ev)
    # Decisions: the card's, word for word, then the model's own, naming no files.
    card_ids = {d.id for d in decided}
    unsettled = [q for q in still_open if q.answer]
    own = [
        d for d in doc.decisions
        if d.id and d.choice and d.id not in card_ids and not CARD_DECISION.match(d.id)
        # Not one about a question the card leaves to confirm (KS-16908: a
        # column "to be decided during design" came back as a decision).
        and not _repeats(d.title, [f"{q.question} {q.answer}" for q in unsettled])
    ]
    for d in own:
        d.applies_to = []
    doc.decisions = list(decided) + list({d.id: d for d in own}.values())
    # Open questions: the card's, word for word, each answered one with the
    # writer's one-line assumption, then the writer's own that are new.
    card_open = [q.model_copy() for q in still_open]
    _attach_assumptions(card_open, doc.assumptions)
    mine: list[OpenQuestion] = []
    for q in doc.open_questions:
        if not q.question or q.same_as or _repeats(
            q.question, [o.question for o in card_open + mine]
        ):
            continue
        mine.append(OpenQuestion(question=q.question, why_it_matters=q.why_it_matters,
                                 source="planner"))
    doc.open_questions = card_open + mine
    doc.assumptions = []
    _drop_test_files(doc)
    _environment_to_tests(doc)
    _no_ids(doc)
    # Said once (2026-11-05): a bug's flow is its problem and cause, the flow
    # after the change is the requirements, and the work order is the order.
    if card.is_bug:
        doc.current_flow = []
    doc.flow_after_change = []
    if doc.work_order.steps:
        doc.order_of_work = []
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
    doc.work_order.open_questions = [q.question for q in doc.open_questions if not q.answer]


def _words(text: str) -> set[str]:
    """A question's telling words, in the singular."""
    words = set()
    for w in re.findall(r"[a-z]{4,}", text.lower()):
        if w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 4:
            w = w[:-1]
        if w not in _QUESTION_WORDS:
            words.add(w)
    return words


def _repeats(text: str, others: list[str]) -> bool:
    """Does ``text`` ask what one of ``others`` asks? Words every one of them
    shares (the ticket's subject) do not count."""
    if not others:
        return False
    sets = [_words(o) for o in others]
    common = set.intersection(*sets) if len(sets) > 2 else set()
    mine = _words(text) - common
    if not mine:
        return False
    return any(
        len(mine & (other - common)) / min(len(mine), len(other - common) or 1) >= _SAME_QUESTION
        for other in sets
    )


def _attach_assumptions(card_open: list[OpenQuestion], assumptions: list) -> None:
    """Each card question with an answer so far gets the writer's one-line
    assumption: by its words, else the closest; none found, the page shows
    the question."""
    by_text = {_flat(a.question): a.assumption.strip() for a in assumptions if a.assumption}
    for q in card_open:
        if not q.answer:
            continue
        q.assumption = by_text.get(_flat(q.question), "")
        if not q.assumption:
            for a in assumptions:
                if a.assumption and _repeats(a.question, [q.question]):
                    q.assumption = a.assumption.strip()
                    break


def _fix_areas(doc: TddDocument, ev: Evidence) -> None:
    """A folder written as "ui/src/x/" (as the prompt shows paths), with a
    leading or trailing slash, back to the tree's own "src/x"; a file given
    as a folder becomes its folder. Steps' areas and data_needed alike."""

    def fixed(service: str, folder: str) -> str:
        folder = _LINE_RANGE.sub("", folder.strip()).strip("/")
        if folder.startswith(service + "/") and not ev.known_folder(service, folder):
            folder = folder[len(service) + 1 :]
        if ev.known(service, folder):
            folder = folder.rsplit("/", 1)[0] if "/" in folder else ""
        return folder

    for plan in doc.per_service_plan:
        for step in plan.steps:
            areas = [fixed(plan.service, a) for a in step.areas if a.strip()]
            step.areas = list(dict.fromkeys(a for a in areas if a))
    services = {p.service for p in doc.per_service_plan} | set(ev.trees)
    for need in doc.data_needed:
        if not need.service and len(services) == 1:
            need.service = next(iter(services))
        if need.confirm_in:
            need.confirm_in = fixed(need.service, need.confirm_in)


def _names_test(text: str) -> bool:
    return any(is_test(p) for p in _PATHISH.findall(text))


def _drop_test_files(doc: TddDocument) -> None:
    """No step in a test folder: the coding agent writes no tests, and the
    checks are in the requirements and the test plan."""
    for plan in doc.per_service_plan:
        steps = [s for s in plan.steps if not any(is_test(a + "/") for a in s.areas)]
        if len(steps) < len(plan.steps):
            doc.warnings.append(f"{plan.service}: {len(plan.steps) - len(steps)} step(s) in a "
                                "test folder removed")
        plan.steps = steps
    doc.work_order.steps = [s for s in doc.work_order.steps if not _names_test(s.summary)]
    doc.order_of_work = [line for line in doc.order_of_work if not _names_test(line)]


def _environment_to_tests(doc: TddDocument) -> None:
    """A requirement about where to check it goes to the test plan."""
    kept, moved = [], []
    tests = doc.risks_and_tests.tests
    for req in doc.requirements:
        line = req.requirement
        if _ENVIRONMENT.search(line) and (_VERIFY.search(line) or not _MUST.search(line)):
            if line not in tests:
                moved.append(line)
        else:
            kept.append(req)
    doc.requirements = kept
    tests[:0] = moved  # first in the test plan, in the ticket's order


def _no_ids(doc: TddDocument) -> None:
    """A decision's id in prose ("d3 says") becomes its title."""
    titles = {d.id: d.title or d.choice for d in doc.decisions if _CODE_ID.match(d.id or "")}
    if not titles:
        return
    pattern = re.compile(
        r"(?<![\w-])(" + "|".join(map(re.escape, sorted(titles, key=len, reverse=True)))
        + r")(?![\w-])"
    )

    def plain(text: str) -> str:
        return pattern.sub(lambda m: f"the decision '{titles[m.group(1)]}'", text)

    doc.problem, doc.cause, doc.fix = plain(doc.problem), plain(doc.cause), plain(doc.fix)
    for req in doc.requirements:
        req.requirement, req.check = plain(req.requirement), plain(req.check)
    doc.order_of_work = [plain(t) for t in doc.order_of_work]
    doc.risks_and_tests.risks = [plain(t) for t in doc.risks_and_tests.risks]
    doc.risks_and_tests.tests = [plain(t) for t in doc.risks_and_tests.tests]
    for d in doc.decisions:
        if not CARD_DECISION.match(d.id):
            d.choice, d.reason = plain(d.choice), plain(d.reason)
    for step in [*doc.current_flow, *doc.flow_after_change]:
        step.step = plain(step.step)
    for plan in doc.per_service_plan:
        plan.what_changes = plain(plan.what_changes)
        for s in plan.steps:
            s.title, s.what, s.component, s.done_when = (
                plain(s.title), plain(s.what), plain(s.component), plain(s.done_when)
            )
    for q in doc.open_questions:
        if q.source == "planner":
            q.why_it_matters = plain(q.why_it_matters)
    for need in doc.data_needed:
        need.what, need.sources = plain(need.what), [plain(x) for x in need.sources]
    for w in doc.work_order.steps:
        w.summary = plain(w.summary)


def _drop_bad_citations(doc: TddDocument, ctx: CheckContext) -> None:
    """In a draft, an area that is not a folder is removed (and said so), so
    nobody is sent to look for it."""
    for plan in doc.per_service_plan:
        readable = plan.service in ctx.readable_services
        for step in plan.steps:
            kept = []
            for k, area in enumerate(step.areas):
                folder = area.rsplit("/", 1)[0] if step.new_area and k == 0 else area
                if readable and ctx.known_folder(plan.service, folder):
                    kept.append(area)
                else:
                    doc.warnings.append(f"removed a step's folder that is not in "
                                        f"{plan.service}: {area}")
            step.areas = kept
    for need in doc.data_needed:
        if need.confirm_in and not ctx.known_folder(need.service, need.confirm_in):
            doc.warnings.append(f"removed a folder that is not in {need.service}: "
                                f"{need.confirm_in}")
            need.confirm_in = ""


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
    # A card answer still "to confirm" is planned on; only an unanswered one blocks.
    open_card = [q for q in doc.open_questions if q.source == "card" and not q.answer]
    if open_card:
        reasons.append(f"{len(open_card)} question(s) on the card are still open")
    wo.ready = not reasons
    wo.not_ready_reason = "; ".join(reasons)


def _flat(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())
