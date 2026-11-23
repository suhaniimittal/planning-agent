"""The prompts of the TDD flow's writer call, and of its one repair call (2026-11-02).

The writer is told the TDD is an instruction guide with NO code in it, that it
may cite only the files it was given, that decisions from the card are
settled and that open questions are not its to answer. checks.py then checks
each of those in code: a prompt is a request, the checks are the rule.
"""

from __future__ import annotations

from src.query import images as ticket_images

from .attachments import Attachment
from .evidence import Evidence
from .inputs import PlanInput
from .models import OpenQuestion
from .prepare import Prepared

# Leads shown to the writer per service.
_LEADS = 50

_SHAPE = """{
  "title": "a short title naming the problem (a bug) or the need (a feature), not the fix",
  "problem": "what is wrong (or what is needed), as the user sees it: 1-3 sentences; when an \
attachment shows the screen, what appears where on it",
  "cause": "why it happens, in the code: 1-2 sentences, 'likely' when inferred",
  "fix": "the change to make, in one or two sentences",
  "requirements": [{"requirement": "one thing that must be true after the change",
                    "check": "how a tester confirms it, with the expected result"}],
  "decisions": [{"id": "short-slug", "title": "the choice in a few words",
                 "choice": "what is decided", "reason": "why, in one sentence",
                 "applies_to": []}],
  "current_flow": [{"step": "what happens today, one step", "service": "..."}],
  "per_service_plan": [{"service": "...",
                        "what_changes": "what changes in this service, one sentence",
                        "steps": [{"title": "a few words",
                                   "what": "the change, as an instruction: 'Add the type to the \
package template, with a migration that sets existing rows to Pre-Employment'",
                                   "component": "what changes, in words: 'the package template \
entity (the template itself, not its link to clients)'",
                                   "areas": ["the folder it lives in, from the paths given; \
a few narrow folders rather than one broad one"],
                                   "new_area": false,
                                   "done_when": "how to tell it is done"}],
                        "reference_pr": "#142, or empty"}],
  "data_needed": [{"service": "...", "what": "a value the change needs, in words",
                   "sources": ["each place it may come from that the ticket allows, best \
first: 'the package's settings', 'the order's own settings'"],
                   "confirm_in": "the contract folder (schema, API) to confirm it in, from the \
paths given"}],
  "risks_and_tests": {"risks": ["a real risk of this change, one sentence"],
                      "tests": ["only a check no requirement covers (API-level, regression)"]},
  "assumptions": [{"question": "an open question from the card that has an answer so far, \
word for word", "assumption": "what the plan assumes, one short line: 'Existing templates and \
packages default to Pre-Employment'"}],
  "open_questions": [{"question": "only a NEW question that blocks a step",
                      "why_it_matters": "...",
                      "same_as": "the card question it repeats, word for word, or empty"}],
  "work_order": {"steps": [{"order": 1, "service": "...", "summary": "what to do in this \
service, one sentence", "reference_pr": ""}]}
}"""

_BUG = """This card is a BUG. Leave current_flow empty: problem and cause already say how \
the fault happens; say "likely" for anything inferred. The plan is the \
smallest change that meets the card's acceptance check and its decisions, in every place the \
card says it happens. For a layout bug (a gap appears, things \
shift or wrap), the fix must say to find where the space comes from (a row that wraps, a fixed \
width or column share, a container that grows) in the component that lays out that row: \
always showing or hiding what triggers it is not enough on its own."""

_FEATURE = """This card is a FEATURE or change. In current_flow, describe how the area works \
today, in a few steps. The plan covers every service the change touches, and what each must \
keep working.

A feature usually needs data it does not show today, and with no code read you do not know \
whether that data exists where it is needed. List each such value in data_needed: what it is, \
every source the ticket allows (best first), and the contract folder to confirm it in (a \
schema or API folder from "Contract files"). Then the FIRST step of each service that needs one \
is "Confirm the data": confirm the value exists in that contract, and say what to do if it \
does not (the next source; or, if none exists, the service that must add it). Every later \
step that uses the value says which source it uses: "the package's setting if the order does \
not expose its own"."""

_HIGH_LEVEL = """No source code is available for this TDD, only file paths, and a path is a \
match by name, not a confirmed place: two files can have the same words in their names and do \
different jobs (a details page and the form, a link table and the entity, a service and the \
resolver that lists). So NEVER name a file. Each step says the change, as an instruction; the \
component, in words, by the job it does; and its areas, the folders it lives in, as deep as \
the paths given make sure (the module's pages folder rather than one page's folder you are not \
sure of). The coding agent opens that folder and finds the file. State the cause as likely. \
Never start a step with "Look for". Do not describe code you have not seen."""


def writer_system(is_bug: bool) -> str:
    parts = [
        """You write a Technical Design Document (TDD) for one Jira card. It is an instruction \
guide for a developer or a coding agent who will open the code themselves.

Rules:
1. Plain English only. NO CODE: no snippets, diffs, pseudo-code, JSX or markup, config \
blocks, commands or code fences, not even one line. Describe components, routes, tables and \
settings by what they do, and say in words what to change.
2. NO FILE NAMES anywhere: not in a step, a decision, a risk, a test or a question. The only \
exception is a file a reference change changed, which is a fact. A step's areas, and a \
data_needed confirm_in, are folders that appear in the paths below (under "Leads", "Contract \
files" or "Reference changes"), written as they appear there; a folder to create is new_area \
true (the first area), inside one that appears. A decision's applies_to is always empty.
3. Services: only those under "Services". A service that was skipped may be in the plan, but \
give no area for it.
4. "Decisions already made" are settled: follow every one. Add a decision of your own (a new \
short id) only for a real design choice.
5. "Open questions" from the card are NOT yours to answer. One with an answer so far: plan on \
that answer, and give it one line in assumptions, saying what the plan assumes. One with none: \
where the plan depends on it, say so. Never ask them again: add a question of your own only \
if no card question covers it and a step depends on it; one that repeats a card question \
names it in same_as.
6. A reference change shows how a similar change was made: follow its pattern and name it in \
reference_pr.
7. Say what you know as known; say what you infer as "likely".
8. Write about the code and the change, never about your sources: no "the read portion", \
"unread", "the files shown", "the excerpt" or "the screenshots suggest". The reader has the \
repository, not your input. Say what a screenshot shows as a fact of the screen: "a blank band \
appears beside the title and the search row drops below it".
9. requirements: what the app must do, or must not do, after the change: every requirement \
of the ticket and its answers, true to their words, each with the check that confirms it. \
Every symptom the title or description names is a "must not" requirement ("No blank band \
appears above the search row, in any selection state"). Where to check it (browsers, devices, \
screen size, zoom) is not a requirement: put it in tests. Mark a requirement you derived, that \
the ticket does not state, with "(derived)". Do not repeat requirements as decisions.
10. One step per real change, as many steps as the change needs and no more: a rule, its \
schema and the place it is shown are ONE change, not three steps. `what` is the instruction; \
`component` names what changes by its job; `areas` its folder or folders; `done_when` is \
checkable. No step only passes a value from one place to another unless the ticket requires \
it: how data reaches a component is the code's to show.
11. work_order: one step per service, in the order to do them, each one sentence; the steps \
above hold the detail.
12. No step is in a test folder (tests, specs, fixtures, Cypress, e2e), and no step is about \
writing tests: the checks go in requirements and tests.
13. Prefer areas of the app the ticket names. When you do not know which component owns what \
changes, describe it ("the shared grid or toolbar component the client screen renders").
14. Never refer to a decision or a question by its id: name it.
15. An answer on the ticket that proposes a fix is the first fix to plan: build on it, and add \
another guard only where the ticket asks for one. Add no condition or scope the ticket does not \
state; when the scope is unclear, ask it once as an open question.
16. Say each thing once. A service's what_changes is one sentence the steps do not repeat. A \
long list (settings, fields) is written in full once; elsewhere refer to it ("the same \
settings as the template"). tests hold only checks no requirement covers. A risk never repeats \
an open question. A message in two languages is one check.
17. A decision is a design choice the plan needs. Something the ticket leaves "to be decided" \
or "to confirm" is an open question, never a decision.
18. A name is not a fact: never take a step's order, a screen's position or a component's \
behaviour from a file or folder name ("OpeningTextForm" is not the first step). Describe a \
component by what the ticket says it does: "the step where the letter templates are chosen".
19. Never state an assumption as a fact in a step: a step that rests on one says what to do \
if it does not hold.""",
        _BUG if is_bug else _FEATURE,
        _HIGH_LEVEL,
    ]
    parts.append("Return ONE JSON object and nothing else, in this shape:\n" + _SHAPE)
    return "\n\n".join(parts)


def _ticket(plan: PlanInput, attachments: list[Attachment], with_images: bool) -> list[str]:
    card = plan.card
    lines = ["## Ticket", f"{card.key} ({card.type}): {card.title}", "", card.description or "-"]
    if attachments:
        lines += ["", "## Attachments"]
        for n, a in enumerate(attachments, start=1):
            lines.append(f"{n}. {a.name}" + (f": {a.description}" if a.description else ""))
            if a.text:
                lines.append(f"   Text of {a.name}:\n{_indent(a.text)}")
            if a.image is not None:
                lines.append(f"   (the image {a.name} is attached below)")
            if a.note:
                lines.append(f"   Note: {a.note}")
        if with_images:
            lines.append(ticket_images.SLOT)
    return lines


def writer_user(
    plan: PlanInput,
    attachments: list[Attachment],
    decided: list,
    still_open: list[OpenQuestion],
    memory: str,
    prepared: Prepared,
    ev: Evidence,
    with_images: bool,
) -> str:
    lines = _ticket(plan, attachments, with_images)

    lines += ["", "## Decisions already made (keep each, same id)"]
    lines += [f"- [{d.id}] {d.title} -> {d.choice}" for d in decided] or ["- none"]
    if memory:
        lines += ["", "## From this ticket's earlier versions and its project", memory]
    lines += ["", "## Open questions (do not answer these; plan on an answer so far)"]
    lines += [
        f"- {q.question}" + (f" (answer so far: {q.answer})" if q.answer else "")
        for q in still_open
    ] or ["- none"]

    lines += ["", "## Services (rank 1 is the most relevant)"]
    for svc in prepared.services:
        folder = f", folder {svc.path}" if svc.path else ""
        lines.append(
            f"- {svc.name} (rank {svc.rank}): {svc.repo}, branch {svc.branch} at "
            f"{svc.sha[:12]}{folder}. Why: {svc.reason or '-'}"
        )
    lines += [f"- skipped {line}" for line in prepared.skipped]

    lines += ["", "## Contract files (schemas and API definitions, by path; where a value is "
              "confirmed)"]
    for svc in prepared.services:
        matching = ev.contracts.get(svc.name, [])
        folders = ev.contract_folders.get(svc.name, [])
        if matching:
            lines.append(f"- {svc.name}, matching the ticket: " + ", ".join(matching))
        if folders:
            lines.append(f"- {svc.name}, folders holding contracts: " + ", ".join(folders))
        if not matching and not folders:
            lines.append(f"- {svc.name}: none found")

    if ev.references:
        lines += ["", "## Reference changes"]
        for r in ev.references:
            lines.append(f"### {r.service} {r.ref} ({r.state}){': ' + r.title if r.title else ''}")
            if r.card:
                lines.append(f"Card: {r.card}")
            if r.body:
                lines.append(f"Description:\n{_indent(r.body)}")
            if r.commit_lines:
                lines.append("Commits: " + " | ".join(r.commit_lines))
            lines.append("Files changed:")
            lines += [
                f"- {f.path} ({f.status}, +{f.additions} -{f.deletions})" for f in r.files[:60]
            ]

    lines += ["", "## Leads (paths matching the ticket's words, by name only: use them to "
              "choose areas, never name the files)"]
    any_lead = False
    for svc in prepared.services:
        for c in ev.leads(svc.name, _LEADS):
            why = [*c.why, c.note] if c.note else c.why
            lines.append(f"- {svc.name}/{c.path}: {'; '.join(why)}")
            any_lead = True
    if not any_lead:
        lines.append("- none")

    notes = ev.not_read + ev.warnings
    if notes:
        lines += ["", "## Limits of this evidence"]
        lines += [f"- {n}" for n in notes[:60]]
    return "\n".join(lines)


def repair_user(original_user: str, previous: str, problems: list[str]) -> str:
    return (
        original_user
        + "\n\n## Your previous answer\n"
        + previous
        + "\n\n## Problems found in it\n"
        + "\n".join(f"- {p}" for p in problems)
        + "\n\nReturn the corrected JSON object only, fixing every problem above and keeping "
        "everything else."
    )


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())
