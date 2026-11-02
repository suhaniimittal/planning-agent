"""Step 5 of the TDD flow (2026-11-02): the card's questions.

An answered question becomes a decision, the answer copied word for word: a
person settled it, so the TDD follows it. An unanswered one stays an open
question, also word for word, and the planner is never asked to answer it —
the flow puts both back into the document itself, whatever the model wrote.

Decision ids are "q<number>-<slug>", stable for the same question, so the
memory recognises the same decision on the ticket's next version.
"""

from __future__ import annotations

import re

from src.query.models import DesignDecision

from .inputs import Question
from .models import OpenQuestion

_SLUG = re.compile(r"[^a-z0-9]+")


def question_id(number: int, question: str) -> str:
    slug = _SLUG.sub("-", question.lower()).strip("-")[:40].rstrip("-")
    return f"q{number}-{slug}" if slug else f"q{number}"


def split(questions: list[Question]) -> tuple[list[DesignDecision], list[OpenQuestion]]:
    """(decisions from the answered questions, the unanswered ones), in card order."""
    decisions: list[DesignDecision] = []
    still_open: list[OpenQuestion] = []
    for number, q in enumerate(questions, start=1):
        if not q.answered:
            still_open.append(OpenQuestion(question=q.question, source="card"))
            continue
        if len(q.answers) == 1:
            choice = q.answers[0]
            reason = "Answered on the card."
        else:
            choice = " ".join(f"({i}) {a}" for i, a in enumerate(q.answers, start=1))
            reason = "Answered on the card more than once; every answer is kept, check they agree."
        decisions.append(
            DesignDecision(
                id=question_id(number, q.question),
                title=q.question[:120],
                choice=choice,
                reason=reason,
            )
        )
    return decisions, still_open
