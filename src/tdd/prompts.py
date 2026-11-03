"""The prompts of the TDD flow's two model calls, and of the one repair call (2026-11-02).

The writer is told the TDD is an instruction guide with NO code in it, that it
may cite only the files it was given, that decisions from the card are
settled and that open questions are not its to answer. checks.py then checks
each of those in code: a prompt is a request, the checks are the rule.
"""

from __future__ import annotations

import json

from src.query import images as ticket_images

from .attachments import Attachment
from .evidence import Evidence, is_test, readable
from .inputs import PlanInput
from .models import OpenQuestion
from .prepare import Prepared

# How many paths per service the scout sees, and how many files it may pick.
SCOUT_PATHS = 2_500
SCOUT_PICKS = 10
# Not-yet-read leads shown to the writer per service.
_LEADS = 25

_SHAPE = """{
  "title": "a short title naming the problem (a bug) or the need (a feature), not the fix",
  "problem": "what is wrong (or what is needed), as the user sees it: 1-3 sentences; when an \
attachment shows the screen, what appears where on it",
  "cause": "why it happens, in the code: 1-3 sentences; when unsure, 'likely' + where to confirm",
  "fix": "the suggested change, in one or two sentences",
  "requirements": ["one thing the ticket or its answers require, as a plain sentence"],
  "decisions": [{"id": "short-slug", "title": "the choice in a few words",
                 "choice": "what is decided", "reason": "why, in one sentence",
                 "applies_to": [{"service": "...", "file_path": "...", "symbol": ""}]}],
  "current_flow": [{"step": "what happens today, one step", "service": "..."}],
  "flow_after_change": [{"step": "what will happen, one step", "service": "..."}],
  "per_service_plan": [{"service": "...",
                        "what_changes": "what changes in this service, one sentence",
                        "steps": [{"title": "a few words",
                                   "where": "the file to change, a path given below",
                                   "new_file": false,
                                   "what": "what to change, in words",
                                   "why": "why",
                                   "done_when": "how to tell it is done"}],
                        "files_to_look_at": [{"path": "...", "why": "..."}],
                        "reference_pr": "#142, or empty"}],
  "order_of_work": ["which service or step first, and why"],
  "risks_and_tests": {"risks": ["..."],
                      "tests": ["a check a tester can run, with its expected result"]},
  "open_questions": [{"question": "...", "why_it_matters": "..."}],
  "work_order": {"steps": [{"order": 1, "service": "...", "summary": "what to do",
                            "reference_pr": ""}]}
}"""

_BUG = """This card is a BUG. In current_flow, describe how the fault happens today, step by \
step, and where it comes from as far as the evidence below shows it; say "likely" for \
anything inferred. The plan is the smallest change that meets the card's acceptance check and its \
decisions, in every place the card says it happens. For a layout bug (a gap appears, things \
shift or wrap), the fix must say to find where the space comes from (a row that wraps, a fixed \
width or column share, a container that grows) in the component that lays out that row: \
always showing or hiding what triggers it is not enough on its own."""

_FEATURE = """This card is a FEATURE or change. In current_flow, describe how the area works \
today; in flow_after_change, how it will work. The plan covers every service the change \
touches, and what each must keep working."""

_HIGH_LEVEL = """No source code is available for this TDD, only file paths. State the cause as \
likely and name where to confirm it. For each step, give the file to change from the paths \
given and what to look for in it. Do not describe code you have not seen."""


def writer_system(is_bug: bool, high_level: bool) -> str:
    parts = [
        """You write a Technical Design Document (TDD) for one Jira card. It is an instruction \
guide for a developer or a coding agent who will open the code themselves.

Rules:
1. Plain English only. NO CODE: no snippets, diffs, pseudo-code, JSX or markup, config \
blocks, commands or code fences, not even one line. Name files, components, functions, props, \
routes, tables and settings by name, and say in words what to change.
2. Use only what you were given. Cite a file (files_to_look_at, applies_to) only if its path \
appears below: under "Files read", "Leads not read", "Service map" or "Reference changes". \
Never invent or shorten a path.
3. Services: only those under "Services". A service that was skipped may be in the plan, but \
cite no files for it.
4. "Decisions already made" are settled: follow every one. Add a decision of your own (a new \
short id) only for a real design choice.
5. "Open questions" from the card are NOT yours to answer: assume no answer anywhere; where the \
plan depends on one, say so. You may add open questions of your own.
6. A reference change shows how a similar change was made: follow its pattern and name it in \
reference_pr.
7. Say what you know as known; say what you infer as "likely", and name the file where the \
developer confirms it.
8. Write about the code and the change, never about your sources: no "the read portion", \
"unread", "the files shown", "the excerpt" or "the screenshots suggest". The reader has the \
repository, not your input. Say what a screenshot shows as a fact of the screen: "a blank band \
appears beside the title and the search row drops below it".
9. requirements: what the app must do, or must not do, after the change: every requirement \
of the ticket and its answers, one plain sentence each, true to their words (behaviour, scope, \
acceptance). Every symptom the title or description names is a "must not" requirement \
("No blank band appears above the search row, in any selection state"). Where to check it \
(browsers, devices, screen size, zoom) is not a requirement: put it in tests. Do not repeat \
requirements as decisions; decisions are only design choices you make.
10. Each step: `where` is one file path given below (or a new file, with new_file true); \
`what` says what to change in words; `done_when` is checkable.
11. work_order: one step per piece of work, in the order to do it; every service in \
per_service_plan has at least one step.
12. No step, decision or file to look at is a test file (tests, specs, Cypress, e2e): the \
checks a tester runs go in tests.
13. Prefer files of the app the ticket names. When you do not know which file lays out or owns \
what changes, describe it ("the shared grid or toolbar component the client screen renders") \
instead of naming a file of another app.
14. Never refer to a decision or a question by its id: name it.""",
        _BUG if is_bug else _FEATURE,
    ]
    if high_level:
        parts.append(_HIGH_LEVEL)
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
    map_view: dict,
    ev: Evidence,
    with_images: bool,
) -> str:
    lines = _ticket(plan, attachments, with_images)

    lines += ["", "## Decisions already made (keep each, same id)"]
    lines += [f"- [{d.id}] {d.title} -> {d.choice}" for d in decided] or ["- none"]
    if memory:
        lines += ["", "## From this ticket's earlier versions and its project", memory]
    lines += ["", "## Open questions (do not answer these)"]
    lines += [f"- {q.question}" for q in still_open] or ["- none"]

    lines += ["", "## Services (rank 1 is the most relevant)"]
    for svc in prepared.services:
        folder = f", folder {svc.path}" if svc.path else ""
        lines.append(
            f"- {svc.name} (rank {svc.rank}): {svc.repo}, branch {svc.branch} at "
            f"{svc.sha[:12]}{folder}. Why: {svc.reason or '-'}"
        )
    lines += [f"- skipped {line}" for line in prepared.skipped]

    lines += ["", "## Service map (only these services; names and paths, no code)"]
    lines.append(json.dumps(map_view, indent=1, ensure_ascii=False))

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
            for path, patch in r.patches.items():
                lines.append(f"Patch of {path} (to read, never to copy into the TDD):")
                lines.append(_indent(patch))

    lines += ["", "## Files read (at the pinned commit; to read, never to copy into the TDD)"]
    if not ev.files:
        lines.append("- none")
    for f in ev.files:
        cut = " (long: its middle is left out)" if f.cut else ""
        lines.append(f"### {f.service}/{f.path}{cut}\nWhy it was read: {f.why}")
        lines.append(_indent(f.text))

    lines += ["", "## Leads not read (paths only: name them as files to inspect)"]
    any_lead = False
    for svc in prepared.services:
        for c in ev.leads(svc.name, _LEADS * 2):
            why = [*c.why, c.note] if c.note else c.why
            lines.append(f"- {svc.name}/{c.path}: {'; '.join(why)}")
            any_lead = True
    if not any_lead:
        lines.append("- none")

    notes = ev.not_read + ev.unread_leads + ev.warnings
    if notes:
        lines += ["", "## Limits of this evidence"]
        lines += [f"- {n}" for n in notes[:60]]
    return "\n".join(lines)


def scout_system() -> str:
    return f"""You pick the source files worth reading for one Jira card, before a design is \
written. You see the ticket, a service map, the leads found so far, and each service's file \
paths (no contents).

Pick at most {SCOUT_PICKS} files, best first: the screen, component, route, handler or rule the \
ticket is about, and the files that lay out, call or configure it. No tests, generated files, \
assets or files already read. Copy each path EXACTLY from a list below.

Return ONE JSON object and nothing else:
{{"files": [{{"service": "...", "path": "...", "why": "one short reason"}}]}}"""


def scout_user(
    plan: PlanInput, attachments: list[Attachment], map_view: dict, ev: Evidence,
    prepared: Prepared,
) -> str:
    lines = _ticket(plan, attachments, with_images=False)
    answered = [q for q in plan.questions if q.answered]
    if answered:
        lines += ["", "## Answers on the card"]
        lines += [f"- {q.question} -> {' / '.join(q.answers)}" for q in answered]
    lines += ["", "## Service map", json.dumps(map_view, indent=1, ensure_ascii=False)]
    lines += ["", "## Already read"]
    lines += [f"- {f.service}/{f.path}" for f in ev.files] or ["- none"]
    lines += ["", "## Leads so far"]
    for svc in prepared.services:
        lines += [
            f"- {svc.name}/{c.path}: {'; '.join(c.why)}"
            for c in ev.ranked(svc.name)[:_LEADS]
            if c.score >= 1
        ]
    for svc in prepared.services:
        paths, cut = scout_paths(ev, svc.name)
        lines += ["", f"## File paths of {svc.name}" + (f" ({cut} more not shown)" if cut else "")]
        lines += paths
    return "\n".join(lines)


def scout_paths(ev: Evidence, service: str) -> tuple[list[str], int]:
    """The paths the scout may pick from: readable, not tests, in the folder;
    over SCOUT_PATHS, those whose path holds a ticket term first, then the
    shallowest. (paths, how many were left out)."""
    tree = ev.trees.get(service)
    if tree is None:
        return [], 0
    folder = ev.folders.get(service, "")
    paths = [
        p for p in tree.paths
        if (not folder or p.startswith(folder + "/")) and readable(p) and not is_test(p)
    ]
    if len(paths) <= SCOUT_PATHS:
        return paths, 0
    keys = [k for k in (t.lower().replace(" ", "") for t in ev.terms) if len(k) >= 4]

    def rank(path: str) -> tuple:
        flat = path.lower().replace("-", "").replace("_", "")
        return (not any(k in flat for k in keys), path.count("/"), path)

    paths.sort(key=rank)
    return paths[:SCOUT_PATHS], len(paths) - SCOUT_PATHS


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
