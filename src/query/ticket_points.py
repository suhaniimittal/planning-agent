"""The ticket's points, and whether a TDD answers each of them.

Before planning, one small model call lists what the ticket says is wrong
(or asks for), one point each. The planner is shown that list and must say,
for every point, how far its changes answer it: fixed, partly, needs a
decision the ticket does not make, or needs no change. Code then checks the
answer: a point without one is sent back to the model in the same
conversation (tool_loop), and one still unanswered after that is written
into the TDD's open questions — never dropped silently. A point answered
"partly" or "needs_decision" is a valid answer, and is written into the open
questions too, for the ticket owner.

Why a list made by code and not by the planner: on the Archive-button
ticket ("White space added and controls shift when report selected") TDD
after TDD fixed the shift and never mentioned the white space, while the
file holding it was in the prompt. A model asked to list what it covered
lists what it covered.

Why four answers and not "fixed or no change": with only those two, the
next TDD listed the shift fix against the white space too — which, at the
reported screen width, it does not remove. Removing it meant moving the
button beside the title, a design choice the ticket did not make; a model
that can say so has no reason to claim "fixed".

What this checks is that every point has an answer, not that the answer is
right.
"""

from __future__ import annotations

from . import issue_split, planner
from .models import PointCoverage, TechnicalDesignDoc

# More than this and the list is being split too finely to check against.
MAX_POINTS = 12
MAX_POINT_CHARS = 300

POINTS_SYSTEM_PROMPT = (
    "You list the points of a software ticket that a fix or an implementation must answer.\n\n"
    "A point is one distinct thing the ticket says is wrong, or one distinct thing it asks "
    "for. Two different symptoms are two points, even in one sentence. The same symptom "
    "described twice is one point.\n\n"
    "Write each point in the ticket's own words, as one short sentence. Do not add causes, "
    "fixes, files or components the ticket does not state. Leave out headers, steps to "
    "reproduce on their own, and background with nothing to answer.\n\n"
    'Respond with ONLY a JSON object of the form {"points": ["...", "..."]}, no markdown '
    "fences and no prose outside the JSON."
)

_STATUS_LABELS = {
    "fixed": "fixed",
    "partly": "partly fixed",
    "needs_decision": "needs a decision",
    "no_change": "no change needed",
}


def parse_points(raw: str) -> list[str]:
    data = issue_split._parse_object(raw)
    points = data.get("points") if isinstance(data, dict) else None
    if not isinstance(points, list):
        return []
    kept = [p.strip() for p in points if isinstance(p, str) and p.strip()]
    if not kept or len(kept) > MAX_POINTS or any(len(p) > MAX_POINT_CHARS for p in kept):
        return []
    return kept


async def extract(issue_text: str) -> list[str]:
    """The ticket's points, or [] when they cannot be had — the check is
    then skipped, never the run."""
    try:
        raw = await planner.call_llm(
            POINTS_SYSTEM_PROMPT, f"<ticket>\n{issue_text.strip()}\n</ticket>"
        )
    except Exception as error:  # noqa: BLE001 - any provider failure skips the check
        print(f"  [TDD] ticket points unavailable ({type(error).__name__}); no coverage check")
        return []
    points = parse_points(raw)
    if not points:
        print("  [TDD] ticket points unusable; no coverage check")
    return points


def label(status: str | None) -> str:
    return _STATUS_LABELS.get(status or "", "not answered")


def _answers(entry: PointCoverage, changed: set[str]) -> bool:
    """Whether one coverage entry is a complete answer: a status, and what
    that status needs — a change in the design for fixed and partly, an
    explanation for the others."""
    explained = bool((entry.how or "").strip())
    names_a_change = any(ref.file_path in changed for ref in entry.addressed_by)
    if entry.status == "fixed":
        return names_a_change
    if entry.status == "partly":
        return names_a_change and explained
    if entry.status in ("needs_decision", "no_change"):
        return explained
    return False


def uncovered(tdd: TechnicalDesignDoc, points: list[str]) -> list[int]:
    """Numbers (1-based) of the points the TDD leaves without a complete
    answer (see _answers)."""
    changed = {c.file_path for s in tdd.services for c in s.changes if c.file_path}
    answered = {e.point for e in tdd.ticket_coverage if _answers(e, changed)}
    return [n for n in range(1, len(points) + 1) if n not in answered]


def feedback(points: list[str], missing: list[int]) -> str:
    """What the model is told when its answer leaves points unanswered."""
    listed = "\n".join(f"{n}. {points[n - 1]}" for n in missing)
    return (
        "Your design leaves these ticket points without a complete `ticket_coverage` answer:\n"
        f"{listed}\n\n"
        "Each needs a `status`: fixed or partly, naming the entries in `changes` that do it in "
        "`addressed_by` (partly also says in `how` what remains); needs_decision, with the "
        "choice and its options in `how`; or no_change, with the reason in `how`. Look the code "
        "up if you need to."
    )


def note_uncovered(tdd: TechnicalDesignDoc, points: list[str]) -> list[int]:
    """Record the points, and put what the ticket owner must know in
    open_questions: each point only partly fixed, each needing a decision,
    and each still unanswered. Returns the unanswered ones."""
    tdd.ticket_points = points
    missing = uncovered(tdd, points)
    by_point = {e.point: e for e in tdd.ticket_coverage}
    for n, point in enumerate(points, start=1):
        entry = by_point.get(n)
        how = (entry.how or "").strip() if entry else ""
        if n in missing:
            tdd.open_questions.append(f'Ticket point {n} has no complete answer: "{point}"')
        elif entry.status == "partly":
            tdd.open_questions.append(f'Ticket point {n} is only partly fixed ("{point}"): {how}')
        elif entry.status == "needs_decision":
            tdd.open_questions.append(f'Ticket point {n} needs a decision ("{point}"): {how}')
    return missing

