import json

from src.query import ticket_points
from src.query.models import TechnicalDesignDoc
from src.query.render import render_text

GRID = "src/apps/client/components/InfiniteScrollGrid/view.tsx"
POINTS = [
    "White space is added when a report is selected.",
    "Controls shift down when the Archive button is displayed.",
    "The shift can cause mis-clicks.",
]
FIXED = {"status": "fixed", "how": "The slot is always there.", "addressed_by": [{"file_path": GRID}]}


def _tdd(coverage, changed=(GRID,)):
    return TechnicalDesignDoc.model_validate({
        "title": "t", "issue_summary": "s", "overall_reasoning": "r",
        "services": [{"service": "ui", "complexity": "low", "reasoning": "r", "changes": [
            {"file_path": path, "function_or_symbol": "InfiniteScrollGrid",
             "change_description": "c", "implementation_notes": "n", "pseudocode_sketch": "p",
             "reasoning": "r"}
            for path in changed
        ]}],
        "ticket_coverage": coverage,
    })


def test_points_parse_and_an_unusable_list_is_dropped():
    assert ticket_points.parse_points('{"points": ["a b", " c "]}') == ["a b", "c"]
    assert ticket_points.parse_points("not json") == []
    assert ticket_points.parse_points(json.dumps({"points": ["x"] * 13})) == []
    assert ticket_points.parse_points(json.dumps({"points": ["x" * 301]})) == []


def test_each_status_needs_what_it_claims():
    tdd = _tdd([
        {**FIXED, "point": 1},
        {"point": 2, "status": "needs_decision", "how": "Move the button beside the title?"},
        {"point": 3, "status": "no_change", "how": "It follows from point 2."},
    ])

    assert ticket_points.uncovered(tdd, POINTS) == []


def test_an_incomplete_answer_is_no_answer():
    tdd = _tdd([
        # fixed, but by a file the design does not change
        {"point": 1, "status": "fixed", "addressed_by": [{"file_path": "styles.ts"}]},
        # partly, without saying what remains
        {"point": 2, "status": "partly", "addressed_by": [{"file_path": GRID}]},
        # no status at all is never read as "fixed"
        {"point": 3, "how": "x", "addressed_by": [{"file_path": GRID}]},
    ])

    assert ticket_points.uncovered(tdd, POINTS) == [1, 2, 3]


def test_what_the_owner_must_know_goes_to_open_questions_and_the_document_shows_each():
    tdd = _tdd([
        {"point": 1, "status": "needs_decision",
         "how": "Only moving the button beside the title removes it (InfiniteScrollGrid/view.tsx)."},
        {**FIXED, "point": 2,
         "addressed_by": [{"file_path": GRID, "function_or_symbol": "InfiniteScrollGrid"}]},
    ])

    assert ticket_points.note_uncovered(tdd, POINTS) == [3]

    assert tdd.open_questions == [
        'Ticket point 1 needs a decision ("White space is added when a report is selected."): '
        "Only moving the button beside the title removes it (InfiniteScrollGrid/view.tsx).",
        'Ticket point 3 has no complete answer: "The shift can cause mis-clicks."',
    ]
    text = render_text(tdd)
    assert "TICKET COVERAGE" in text.upper()
    assert "1. White space is added when a report is selected.\n   [needs a decision] Only " in text
    assert f"[fixed] The slot is always there.\n   -> {GRID} (InfiniteScrollGrid)" in text
    assert "3. The shift can cause mis-clicks.\n   [not answered]" in text
