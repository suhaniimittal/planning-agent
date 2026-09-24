from src.query.candidates import (
    _tokenize,
    merge_candidates,
    select_relevant_functions_by_embedding,
    select_relevant_symbols,
)


def test_merge_candidates_vector_hits_ordered_by_score_desc():
    hits = [{"name": "a", "score": 0.5}, {"name": "b", "score": 0.9}]
    assert merge_candidates(hits, []) == ["b", "a"]


def test_merge_candidates_neighbors_ordered_by_hop_distance():
    neighbors = [{"name": "far", "hop_distance": 2}, {"name": "near", "hop_distance": 1}]
    assert merge_candidates([], neighbors) == ["near", "far"]


def test_merge_candidates_dedupes_name_present_in_both_lists():
    hits = [{"name": "orders", "score": 0.9}]
    neighbors = [{"name": "orders", "hop_distance": 1}, {"name": "shipping", "hop_distance": 1}]
    result = merge_candidates(hits, neighbors)
    assert result.count("orders") == 1
    assert result == ["orders", "shipping"]


def test_merge_candidates_empty_neighbors_returns_vector_hits_only():
    hits = [{"name": "user", "score": 0.7}]
    assert merge_candidates(hits, []) == ["user"]


def test_merge_candidates_empty_vector_hits_returns_neighbors_only():
    neighbors = [{"name": "shipping", "hop_distance": 1}]
    assert merge_candidates([], neighbors) == ["shipping"]


# --- _tokenize: camelCase/PascalCase splitting -------------------------------


def test_tokenize_splits_pascal_case_symbol_name_into_subwords():
    """Regression test: a PascalCase name like "EmailLogRow" used to become
    ONE glued token ("emaillogrow") that could never match issue text
    spelling the same idea out as separate words — so a symbol whose name
    obviously relates to the issue could score zero overlap and be
    excluded before ranking even started."""
    tokens = _tokenize("EmailLogRow")
    assert "email" in tokens
    assert "log" in tokens
    assert "row" in tokens


def test_tokenize_splits_camel_case_method_name_into_subwords():
    tokens = _tokenize("getShippingUri")
    assert "get" in tokens
    assert "shipping" in tokens
    assert "uri" in tokens


def test_tokenize_keeps_acronym_run_intact_not_letter_by_letter():
    tokens = _tokenize("HTTPMonitoringInterceptor")
    assert "http" in tokens
    assert "monitoring" in tokens
    assert "interceptor" in tokens
    # Never split down to single letters — "h", "t", "p" would be noise.
    assert "h" not in tokens
    assert "t" not in tokens


def test_tokenize_still_keeps_the_whole_glued_name_as_a_token():
    """An exact full-name mention must still match exactly as before —
    splitting only adds recall, it never removes the old behavior."""
    tokens = _tokenize("EmailLogRow")
    assert "emaillogrow" in tokens


def test_tokenize_does_not_over_split_plain_snake_case_or_lowercase_words():
    assert _tokenize("create_order") == {"create", "order"}


def test_tokenize_single_word_with_no_case_transition_is_not_fragmented():
    """No camelCase to split here, but normalization still applies to any
    plain word — "orders" -> "order" — same as it would to a symbol's
    subwords, so a plural issue mention and a singular symbol name (or
    vice versa) still overlap."""
    tokens = _tokenize("orders")
    assert tokens == {"orders", "order"}


def test_tokenize_normalizes_plural_and_past_tense_to_match_base_form():
    """Regression test: the exact real-world case this was found from — an
    issue written as "emails ... logged" must overlap with a symbol whose
    camelCase subwords are the base forms "email"/"log"."""
    issue_tokens = _tokenize("emails were not logged")
    assert "email" in issue_tokens
    assert "log" in issue_tokens


# --- select_relevant_symbols -------------------------------------------------


def test_select_relevant_symbols_matches_pascal_case_name_via_subwords():
    """The exact real-world regression this was found from: a data-model
    class named "EmailLogRow", no docstring, genuinely relevant to an issue
    about email logging — previously excluded with zero keyword overlap
    purely because its name is one glued PascalCase token."""
    symbols = [_symbol("galleryflow", "EmailLogRow", docstring=None, signature="interface EmailLogRow")]
    result = select_relevant_symbols(symbols, "Sent emails' subject line is not getting logged in the database.")
    assert [s["name"] for s in result] == ["EmailLogRow"]


def _symbol(service, name, docstring="", signature=""):
    return {
        "service": service,
        "qualified_name": f"{service}.py::{name}",
        "name": name,
        "docstring": docstring,
        "signature": signature,
    }


def test_select_relevant_symbols_ranks_by_keyword_overlap():
    symbols = [
        _symbol("orders", "create_order", docstring="Creates a new order for checkout"),
        _symbol("orders", "unrelated_helper", docstring="Formats currency strings"),
    ]
    result = select_relevant_symbols(symbols, "checkout order is failing")
    names = [s["name"] for s in result]
    assert names == ["create_order"]


def test_select_relevant_symbols_drops_symbols_with_zero_overlap():
    symbols = [_symbol("orders", "unrelated_helper", docstring="Formats currency strings")]
    result = select_relevant_symbols(symbols, "checkout order is failing")
    assert result == []


def test_select_relevant_symbols_caps_per_service():
    symbols = [
        _symbol("orders", f"order_fn_{i}", docstring="order order order") for i in range(5)
    ]
    result = select_relevant_symbols(symbols, "order", max_per_service=2, max_total=100)
    assert len(result) == 2


def test_select_relevant_symbols_default_cap_is_5_per_service():
    """Locks in the lowered default (was 30) — a prompt-bloat fix: full
    docs + many symbols per service, multiplied across several candidate
    services, was overwhelming the LLM call with mostly-irrelevant content."""
    symbols = [
        _symbol("orders", f"order_fn_{i}", docstring="order order order order") for i in range(10)
    ]
    result = select_relevant_symbols(symbols, "order")
    assert len(result) == 5


def test_select_relevant_symbols_caps_globally_across_services():
    symbols = [_symbol("orders", "a", docstring="order")] + [
        _symbol("shipping", "b", docstring="order")
    ]
    result = select_relevant_symbols(symbols, "order", max_per_service=10, max_total=1)
    assert len(result) == 1


def test_select_relevant_symbols_empty_issue_text_returns_empty():
    symbols = [_symbol("orders", "create_order", docstring="Creates a new order")]
    assert select_relevant_symbols(symbols, "") == []


def test_select_relevant_symbols_empty_symbols_returns_empty():
    assert select_relevant_symbols([], "order issue") == []


# --- select_relevant_functions_by_embedding ----------------------------------


def _chunk_hit(service, function_qname, score, text="code"):
    return {
        "service": service,
        "function_qualified_name": function_qname,
        "text": text,
        "score": score,
    }


def test_select_relevant_functions_by_embedding_keeps_best_chunk_per_function():
    hits = [
        _chunk_hit("orders", "a.py::fn", 0.4, text="chunk0"),
        _chunk_hit("orders", "a.py::fn", 0.9, text="chunk1"),  # same function, higher score
    ]
    result = select_relevant_functions_by_embedding(hits)
    assert len(result) == 1
    assert result[0]["text"] == "chunk1"
    assert result[0]["score"] == 0.9


def test_select_relevant_functions_by_embedding_ranks_by_score_descending():
    hits = [
        _chunk_hit("orders", "a.py::low", 0.2),
        _chunk_hit("orders", "a.py::high", 0.8),
    ]
    result = select_relevant_functions_by_embedding(hits)
    assert [r["function_qualified_name"] for r in result] == ["a.py::high", "a.py::low"]


def test_select_relevant_functions_by_embedding_caps_per_service():
    hits = [_chunk_hit("orders", f"a.py::fn{i}", 0.5) for i in range(5)]
    result = select_relevant_functions_by_embedding(hits, max_per_service=2, max_total=100)
    assert len(result) == 2


def test_select_relevant_functions_by_embedding_default_cap_is_5_per_service():
    hits = [_chunk_hit("orders", f"a.py::fn{i}", 0.5) for i in range(10)]
    result = select_relevant_functions_by_embedding(hits)
    assert len(result) == 5


def test_select_relevant_functions_by_embedding_caps_globally_across_services():
    hits = [_chunk_hit("orders", "a.py::fn", 0.5), _chunk_hit("shipping", "b.py::fn", 0.5)]
    result = select_relevant_functions_by_embedding(hits, max_per_service=10, max_total=1)
    assert len(result) == 1


def test_select_relevant_functions_by_embedding_empty_hits_returns_empty():
    assert select_relevant_functions_by_embedding([]) == []
