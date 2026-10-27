from src.query import answer_checks
from src.query.models import TechnicalDesignDoc

GRID = "src/apps/client/components/InfiniteScrollGrid/view.tsx"
POINTS = ["White space is added.", "Controls shift down.", "Mis-clicks."]
TRACED = [{"case": "nothing selected", "today": "no button", "after": "hidden, widest label"}]


def _change(path, symbol="InfiniteScrollGrid", change_type="modify", edge_cases=TRACED):
    return {"file_path": path, "function_or_symbol": symbol, "change_type": change_type,
            "change_description": "c", "implementation_notes": "n", "pseudocode_sketch": "p",
            "reasoning": "r", "edge_cases": edge_cases}


def _tdd(changes, coverage):
    return TechnicalDesignDoc.model_validate({
        "title": "t", "issue_summary": "s", "overall_reasoning": "r",
        "services": [{"service": "ui", "complexity": "low", "reasoning": "r", "changes": changes}],
        "ticket_coverage": coverage,
    }).model_dump_json()


def _fixed(n):
    return {"point": n, "status": "fixed", "how": "h", "addressed_by": [{"file_path": GRID}]}


def test_a_complete_answer_is_accepted():
    check = answer_checks.check_for(POINTS)

    assert check(_tdd([_change(GRID)], [_fixed(n) for n in (1, 2, 3)])) is None
    assert check("not json") is None


def test_unanswered_points_and_untraced_changes_are_sent_back_together():
    check = answer_checks.check_for(POINTS)
    changes = [_change(GRID, edge_cases=[]),
               _change("src/x/view.test.tsx", "X", edge_cases=[]),   # a test: its cases are itself
               _change("src/y.ts", "Y", change_type="reference", edge_cases=[])]  # changes nothing

    note = check(_tdd(changes, [_fixed(2)]))

    assert "1. White space is added." in note and "3. Mis-clicks." in note
    assert f"These changes list no `edge_cases`:\n- ui {GRID} InfiniteScrollGrid\n\n" in note
    assert "view.test.tsx" not in note and "src/y.ts" not in note
    assert note.endswith("Then respond with the whole JSON object again.")


def test_without_ticket_points_the_edge_cases_are_still_checked():
    check = answer_checks.check_for([])

    assert check(_tdd([_change(GRID)], [])) is None
    assert "list no `edge_cases`" in check(_tdd([_change(GRID, edge_cases=[])], []))


def test_the_document_shows_each_traced_state():
    from src.query.render import render_text

    text = render_text(TechnicalDesignDoc.model_validate_json(
        _tdd([_change(GRID)], [_fixed(1)])))

    assert "    Edge cases:\n      - nothing selected: today no button; after hidden, widest label" \
        in text


def test_without_new_tests_a_created_test_file_is_sent_back():
    changes = [_change(GRID), _change("src/x/view.test.tsx", "X", change_type="create",
                                      edge_cases=[])]
    answer = _tdd(changes, [_fixed(n) for n in (1, 2, 3)])

    note = answer_checks.check_for(POINTS, new_tests=False)(answer)

    assert "these changes create test files:\n- ui src/x/view.test.tsx" in note
    assert answer_checks.check_for(POINTS, new_tests=True)(answer) is None
