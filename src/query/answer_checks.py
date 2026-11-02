"""What code checks in the planner's answer before accepting it.

Each check finds something missing, not something wrong: a ticket point
without an answer (ticket_points.py), and a change that does not trace its
edge cases. Whether the answer is RIGHT is left to the re-check round
(prompt.RECHECK), which the model does itself — code cannot read a
sketch and see that `[].every()` makes its label "Unarchive".

A failed check goes back to the model inside the lookup loop (tool_loop),
so the missing part is filled in before the design is accepted.
"""

from __future__ import annotations

from src.ingestion.test_paths import is_test_path

from . import planner, ticket_points
from .models import TechnicalDesignDoc


def edge_case_gaps(tdd: TechnicalDesignDoc) -> list[str]:
    """The changes to code that list no edge cases, as "service file symbol".
    A reference changes nothing, and a test's cases are the test itself."""
    gaps = []
    for s in tdd.services:
        for c in s.changes:
            if c.change_type not in ("create", "modify") or not c.file_path:
                continue
            if is_test_path(c.file_path) or c.edge_cases:
                continue
            gaps.append(" ".join(p for p in (s.service, c.file_path, c.function_or_symbol) if p))
    return gaps


def _edge_case_feedback(gaps: list[str]) -> str:
    return (
        "These changes list no `edge_cases`:\n" + "\n".join(f"- {g}" for g in gaps) + "\n\n"
        "For each, trace the change through the states where it could behave differently "
        "(empty or missing input, first load, after a reset, both sides of each condition it "
        "touches), from the lines you were shown, and fix the sketch if a state comes out wrong."
    )


def new_test_files(tdd: TechnicalDesignDoc) -> list[str]:
    """Test files the design creates, as "service file"."""
    return [
        f"{s.service} {c.file_path}"
        for s in tdd.services for c in s.changes
        if c.change_type == "create" and c.file_path and is_test_path(c.file_path)
    ]


def check_for(points: list[str], new_tests: bool = True):
    """A tool_loop check: what the answer is missing, or None to accept it.
    With `new_tests` off, a new test file is sent back too."""

    def check(raw: str) -> str | None:
        try:
            tdd = planner.parse_tdd(raw)
        except Exception:  # noqa: BLE001 - an unparseable answer is handled by the caller
            return None
        notes = []
        if points and (missing := ticket_points.uncovered(tdd, points)):
            notes.append(ticket_points.feedback(points, missing))
        if gaps := edge_case_gaps(tdd):
            notes.append(_edge_case_feedback(gaps))
        if not new_tests and (created := new_test_files(tdd)):
            notes.append(
                "New tests are not written in this change, but these changes create test "
                "files:\n" + "\n".join(f"- {c}" for c in created) + "\n\nRemove them from "
                "`changes`, and describe each test under \"Recommended tests (not part of this "
                "change):\" in `testing_notes` instead."
            )
        if not notes:
            return None
        return "\n\n".join(notes) + "\n\nThen respond with the whole JSON object again."

    return check
