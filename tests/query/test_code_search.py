from src.query import code_search

ISSUE = (
    'Deferred order "Submit" silently fails when a phone number has a leading space.\n'
    'On the "Review Deferred Report" screen, errors are only logged as "Form errors: ...".\n'
    "The approveDeferredReport mutation is never sent; internationalPhoneNumber rejects it.\n"
    "Data comes from EmploymentVerificationInput, not GraphQL."
)


def test_issue_identifiers_keep_code_names_and_quoted_phrases_only():
    found = code_search.issue_identifiers(ISSUE)
    assert "approveDeferredReport" in found
    assert "internationalPhoneNumber" in found
    assert "EmploymentVerificationInput" in found
    assert "Review Deferred Report" in found
    assert "Form errors" in found
    # single words are too common to be a signal
    assert "Submit" not in found
    assert "GraphQL" not in found


def _hit(fn, score, identifiers=(), text=None):
    return {
        "function_qualified_name": fn,
        "text": text or fn,
        "score": score,
        "identifiers": list(identifiers),
    }


def test_rank_functions_rolls_chunks_up_to_their_best_chunk():
    ranked = code_search.rank_functions(
        [_hit("a::f", 0.70, text="weak"), _hit("a::f", 0.80, text="strong"), _hit("a::g", 0.75)]
    )
    assert [r["function_qualified_name"] for r in ranked] == ["a::f", "a::g"]
    assert ranked[0]["text"] == "strong"
    assert ranked[0]["score"] == 0.80


def test_rank_functions_boosts_exact_identifiers():
    ranked = code_search.rank_functions(
        [
            _hit("a::meaning_only", 0.80),
            _hit("a::has_name", 0.75, identifiers=["approveDeferredReport"]),
        ]
    )
    assert ranked[0]["function_qualified_name"] == "a::has_name"
    assert ranked[0]["identifiers"] == ["approveDeferredReport"]


def test_rank_functions_ignores_an_identifier_that_is_everywhere():
    common = [_hit(f"a::f{i}", 0.5, identifiers=["everywhere"]) for i in range(30)]
    ranked = code_search.rank_functions(common + [_hit("a::best", 0.6)])
    assert ranked[0]["function_qualified_name"] == "a::best"
    assert all(r["identifiers"] == [] for r in ranked)


def test_rank_functions_keeps_the_best_scores_and_caps_the_count():
    """No relevance floor. One was removed: being relative to each service's
    own best score, it cut a service whose matches dropped off sharply after
    a strong hit, and spared one whose matches were uniformly mediocre —
    measured on a real ticket, 20 functions down to 6 for the service the
    bug was actually in, while two irrelevant services kept all 20."""
    hits = [_hit("a::top", 0.90), _hit("a::close", 0.85), _hit("a::weak", 0.70)]
    assert [r["function_qualified_name"] for r in code_search.rank_functions(hits)] == [
        "a::top",
        "a::close",
        "a::weak",
    ]

    many = [_hit(f"a::f{i}", 0.9 - i * 0.001) for i in range(20)]
    assert len(code_search.rank_functions(many, max_functions=8)) == 8


def test_an_identifier_match_still_beats_a_higher_score():
    """The cap is now the only thing that cuts, so a named function has to
    win its place on the identifier bonus rather than on a floor."""
    weak_but_named = _hit("a::named", 0.50, identifiers=["approveDeferredReport"])
    strong = [_hit(f"a::f{i}", 0.90) for i in range(8)]
    kept = code_search.rank_functions(strong + [weak_but_named], max_functions=3)
    assert "a::named" in [r["function_qualified_name"] for r in kept]


def test_rank_functions_handles_no_hits():
    assert code_search.rank_functions([]) == []


def test_services_named_in_matches_whole_names_only():
    names = ["ui", "candidate-service", "reports-api"]
    assert code_search.services_named_in("fix it in candidate-service and the UI", names) == [
        "ui",
        "candidate-service",
    ]
    assert code_search.services_named_in("the guide is wrong", names) == []
    assert code_search.services_named_in("see candidate-service-v2", names) == []


def test_a_function_containing_an_issue_identifier_always_gets_a_place():
    hits = [_hit(f"a::meaning{i}", 0.90 - i * 0.001) for i in range(10)]
    hits.append(_hit("a::the_screen", 0.60, identifiers=["Form errors"]))

    ranked = code_search.rank_functions(hits, max_functions=8)

    names = [r["function_qualified_name"] for r in ranked]
    assert "a::the_screen" in names  # far below the score floor, but named in the issue
    assert len(names) == 8


def test_issue_identifiers_drop_trailing_punctuation_from_quoted_phrases():
    found = code_search.issue_identifiers('Click "Deselect All," then "Export to PDF."')
    assert "Deselect All" in found
    assert "Export to PDF" in found


def test_issue_queries_keeps_a_sentence_whole_across_an_abbreviation():
    text = (
        "The regex is anchored at the start, so any leading whitespace (e.g. a value "
        'like " +23480 330-20339" pasted by a candidate) fails validation.\n'
        "Open the deferred report review screen and click Submit to see it."
    )
    parts = code_search.issue_queries(text)[1:]
    assert any(p.startswith("The regex is anchored") and p.endswith("validation.") for p in parts)
    assert not any(p.endswith("(e.g.") for p in parts)


def test_issue_queries_still_splits_after_a_sentence_ending_in_an_acronym():
    text = (
        "The report stays stuck in DEFERRED status indefinitely with no cause shown.\n"
        "The approveDeferredReport mutation is never sent to the server at all."
    )
    parts = code_search.issue_queries(text)[1:]
    assert len(parts) == 2


def test_issue_queries_does_not_split_hard_wrapped_text_at_every_newline():
    text = (
        "The deferred report review screen swallows validation errors instead of\n"
        "surfacing them, so the approveDeferredReport mutation is never sent and the\n"
        "report remains in DEFERRED status with no indication to the user."
    )
    assert code_search.issue_queries(text)[1:] == []  # one sentence, identical to the whole


def test_issue_queries_always_leads_with_the_whole_ticket():
    text = (
        "First sentence that is comfortably long enough to survive.\n\n"
        "Second one, also long enough."
    )
    assert code_search.issue_queries(text)[0] == text.strip()
