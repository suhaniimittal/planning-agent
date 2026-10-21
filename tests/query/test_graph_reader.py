from src.query import graph_reader


class FakeSession:
    def __init__(self, records):
        self.records = records
        self.last_query = None
        self.last_params = None

    def run(self, query, **params):
        self.last_query = query
        self.last_params = params
        return self.records

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, records):
        self._session = FakeSession(records)

    def session(self):
        return self._session


def _patch_driver(monkeypatch, records):
    fake_driver = FakeDriver(records)
    monkeypatch.setattr(graph_reader, "get_driver", lambda: fake_driver)
    return fake_driver


def test_vector_search_services_passes_k_and_embedding_params(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"name": "orders", "score": 0.9}])
    result = graph_reader.vector_search_services([0.1, 0.2], k=3)
    assert driver._session.last_params == {"k": 3, "embedding": [0.1, 0.2]}
    assert result == [{"name": "orders", "score": 0.9}]


def test_vector_search_services_query_uses_correct_index_name(monkeypatch):
    driver = _patch_driver(monkeypatch, [])
    graph_reader.vector_search_services([0.1], k=5)
    assert "service_doc_index" in driver._session.last_query


def test_expand_neighbors_query_uses_undirected_calls_pattern(monkeypatch):
    driver = _patch_driver(monkeypatch, [])
    graph_reader.expand_neighbors(["orders"], hops=2)
    assert "-[:CALLS*1..2]-" in driver._session.last_query
    assert "-[:CALLS*1..2]->" not in driver._session.last_query


def test_expand_neighbors_excludes_seed_names_from_results(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"name": "shipping", "hop_distance": 1}])
    result = graph_reader.expand_neighbors(["orders"], hops=1)
    assert result == [{"name": "shipping", "hop_distance": 1}]
    assert "NOT neighbor.name IN $seed_names" in driver._session.last_query


def test_expand_neighbors_empty_seeds_returns_empty_without_query(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"name": "shouldnt-be-seen", "hop_distance": 1}])
    result = graph_reader.expand_neighbors([], hops=2)
    assert result == []
    assert driver._session.last_query is None


def test_get_edges_among_returns_source_target_pairs(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"source": "orders", "target": "shipping"}])
    result = graph_reader.get_edges_among(["orders", "shipping"])
    assert result == [{"source": "orders", "target": "shipping"}]
    assert driver._session.last_params == {"names": ["orders", "shipping"]}


def test_get_edges_among_empty_names_returns_empty_without_query(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"source": "shouldnt-be-seen", "target": "x"}])
    result = graph_reader.get_edges_among([])
    assert result == []
    assert driver._session.last_query is None


def test_load_candidate_docs_returns_name_and_summary_for_given_names(monkeypatch):
    driver = _patch_driver(
        monkeypatch,
        [
            {
                "name": "orders",
                "summary": "Handles order creation and lookup.",
            }
        ],
    )
    result = graph_reader.load_candidate_docs(["orders"])
    assert result == [
        {
            "name": "orders",
            "summary": "Handles order creation and lookup.",
        }
    ]
    assert "s.doc AS summary" in driver._session.last_query
    assert "full_doc" not in driver._session.last_query
    assert driver._session.last_params == {"names": ["orders"]}


def test_load_code_symbols_for_services_queries_class_or_function_by_service(monkeypatch):
    driver = _patch_driver(
        monkeypatch,
        [
            {
                "service": "orders",
                "qualified_name": "controller.py::OrderController.create_order",
                "name": "create_order",
                "file_path": "controller.py",
                "line_start": 4,
                "line_end": 7,
                "signature": "def create_order(self, user_id, items):",
                "docstring": "Creates a new order.",
                "language": "python",
                "is_method": True,
                "parent_class": "controller.py::OrderController",
                "kind": "Function",
            }
        ],
    )
    result = graph_reader.load_code_symbols_for_services(["orders"])
    assert result[0]["qualified_name"] == "controller.py::OrderController.create_order"
    assert "source_snippet" not in driver._session.last_query
    assert driver._session.last_params == {"names": ["orders"]}


def test_load_code_symbols_for_services_empty_names_returns_empty_without_query(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"service": "shouldnt-be-seen"}])
    result = graph_reader.load_code_symbols_for_services([])
    assert result == []
    assert driver._session.last_query is None


def test_load_source_snippets_keys_result_by_service_and_qualified_name(monkeypatch):
    driver = _patch_driver(
        monkeypatch,
        [
            {
                "service": "orders",
                "qualified_name": "controller.py::OrderController.create_order",
                "source_snippet": "def create_order(self, user_id, items):\n    ...",
            }
        ],
    )
    result = graph_reader.load_source_snippets(
        [("orders", "controller.py::OrderController.create_order")]
    )
    assert result == {
        ("orders", "controller.py::OrderController.create_order"):
            "def create_order(self, user_id, items):\n    ..."
    }
    assert driver._session.last_params == {
        "rows": [{"service": "orders", "qualified_name": "controller.py::OrderController.create_order"}]
    }


def test_load_source_snippets_empty_pairs_returns_empty_without_query(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"service": "shouldnt-be-seen"}])
    result = graph_reader.load_source_snippets([])
    assert result == {}
    assert driver._session.last_query is None


def test_vector_search_chunks_queries_chunk_index_and_rolls_up_to_function(monkeypatch):
    driver = _patch_driver(
        monkeypatch,
        [
            {
                "service": "orders",
                "function_qualified_name": "controller.py::OrderController.create_order",
                "text": "def create_order(self, user_id, items):\n    ...",
                "score": 0.87,
            }
        ],
    )
    result = graph_reader.vector_search_chunks([0.1, 0.2], ["orders"], k=10)
    assert result == [
        {
            "service": "orders",
            "function_qualified_name": "controller.py::OrderController.create_order",
            "text": "def create_order(self, user_id, items):\n    ...",
            "score": 0.87,
        }
    ]
    assert "chunk_embedding_index" in driver._session.last_query
    assert "HAS_CHUNK" in driver._session.last_query
    assert driver._session.last_params == {"k": 10, "embedding": [0.1, 0.2], "service_names": ["orders"]}


def test_vector_search_chunks_empty_service_names_returns_empty_without_query(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"service": "shouldnt-be-seen"}])
    result = graph_reader.vector_search_chunks([0.1], [])
    assert result == []
    assert driver._session.last_query is None


def test_file_exists_matches_exact_service_and_path(monkeypatch):
    class One:
        def single(self):
            return {"n": 1}

    driver = _patch_driver(monkeypatch, One())
    assert graph_reader.file_exists("ui", "src/a.tsx") is True
    assert driver._session.last_params == {"service": "ui", "file_path": "src/a.tsx"}


def test_folder_exists_searches_by_prefix_with_trailing_slash(monkeypatch):
    class Some:
        def single(self):
            return {"file_path": "src/report/a.tsx"}

    driver = _patch_driver(monkeypatch, Some())
    assert graph_reader.folder_exists("ui", "src/report/") is True
    assert driver._session.last_params == {"service": "ui", "prefix": "src/report/"}


def test_find_symbol_files_matches_name_or_qualified_tail(monkeypatch):
    driver = _patch_driver(
        monkeypatch, [{"file_path": "b.tsx"}, {"file_path": "a.tsx"}, {"file_path": "a.tsx"}]
    )
    assert graph_reader.find_symbol_files("ui", "Modal.view") == ["a.tsx", "b.tsx"]
    assert driver._session.last_params == {
        "service": "ui",
        "symbol": "Modal.view",
        "by_file": "::Modal.view",
        "by_class": ".Modal.view",
    }


def test_summary_scores_scopes_to_the_given_services(monkeypatch):
    driver = _patch_driver(monkeypatch, [{"name": "ui", "score": 0.6}])
    assert graph_reader.summary_scores([0.1], ["ui"]) == {"ui": 0.6}
    assert driver._session.last_params == {"names": ["ui"], "embedding": [0.1]}


def test_search_service_chunks_merges_semantic_and_identifier_hits(monkeypatch):
    class Session:
        def __init__(self):
            self.queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            if "identifiers" in params:
                return [{"function_qualified_name": "a::g", "text": "t", "score": 0.5,
                         "identifiers": ["approveDeferredReport"]}]
            return [{"function_qualified_name": "a::f", "text": "t", "score": 0.9, "identifiers": []}]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    session = Session()
    monkeypatch.setattr(graph_reader, "get_driver", lambda: type("D", (), {"session": lambda self: session})())

    rows = graph_reader.search_service_chunks("ui", [0.1], ["approveDeferredReport"])

    assert [r["function_qualified_name"] for r in rows] == ["a::f", "a::g"]
    assert all(p["service"] == "ui" for _, p in session.queries)
    assert all("is_test" in q for q, _ in session.queries)  # test code always excluded


def test_search_service_chunks_skips_the_identifier_read_when_there_are_none(monkeypatch):
    driver = _patch_driver(monkeypatch, [])
    graph_reader.search_service_chunks("ui", [0.1], [])
    assert "identifiers" not in driver._session.last_params


def test_load_symbols_and_file_context_skip_the_query_for_nothing(monkeypatch):
    driver = _patch_driver(monkeypatch, [])
    assert graph_reader.load_symbols([]) == []
    assert graph_reader.load_file_context([]) == {}
    assert driver._session.last_query is None


def test_load_file_context_returns_sorted_unique_names(monkeypatch):
    _patch_driver(monkeypatch, [{"service": "ui", "file_path": "a.ts", "names": ["g", "f", "Form.f", "f"]}])
    assert graph_reader.load_file_context([("ui", "a.ts")]) == {("ui", "a.ts"): ["Form.f", "f", "g"]}


def test_load_sibling_tests_skips_pairs_with_no_folder_or_already_a_test(monkeypatch):
    """Guards the pre-query filter: a bare filename has no folder to look in,
    and a test file's own folder is not a lookup worth making."""
    called = []
    monkeypatch.setattr(
        graph_reader, "get_driver", lambda: (_ for _ in ()).throw(AssertionError("queried"))
    )
    assert graph_reader.load_sibling_tests([("ui", "view.tsx")]) == {}
    assert graph_reader.load_sibling_tests([("ui", "a/b/view.test.tsx")]) == {}
    assert called == []


def test_new_test_path_puts_the_convention_infix_before_the_extension():
    assert (
        graph_reader.new_test_path("a/b/employmentVerificationForm.ts", ".test.")
        == "a/b/employmentVerificationForm.test.ts"
    )
    assert graph_reader.new_test_path("view.tsx", ".spec.") == "view.spec.tsx"


def test_new_test_path_is_none_for_a_file_with_no_extension():
    """Nothing to put the infix in front of — better no suggestion than a
    path the planner would then present as measured."""
    assert graph_reader.new_test_path("a/Makefile", ".test.") is None
    assert graph_reader.new_test_path("a/.gitignore", ".test.") is None


def test_a_service_with_no_doc_scores_zero_rather_than_vanishing(monkeypatch):
    """Omitting it would delete the service from the run: the search
    shortlist is built from these names, so a service missing here is never
    searched and never appears in the design, with nothing said about it."""
    driver = _patch_driver(monkeypatch, [{"name": "ui", "score": 0.4}, {"name": "kafka", "score": 0.0}])

    scores = graph_reader.summary_scores([0.1, 0.2], ["ui", "kafka"])

    assert scores == {"ui": 0.4, "kafka": 0.0}
    query = driver._session.last_query
    assert "s.doc_embedding IS NULL THEN 0.0" in query
    assert "WHERE s.doc_embedding IS NOT NULL" not in query


def test_tests_naming_finds_tests_by_content_not_by_folder(monkeypatch):
    """The sibling lookup only sees the same folder. On a project that keeps
    src/common/ apart from src/apps/ that finds nothing, and a design that
    believes no test exists has a new one written from scratch."""
    driver = _patch_driver(
        monkeypatch,
        [
            {"service": "ui", "symbol": "handleCcEmailAddressesBlur",
             "file_path": "src/apps/internal/.../ApplicationSettingsTab/view.data.test.tsx"},
            {"service": "ui", "symbol": "handleCcEmailAddressesBlur",
             "file_path": "src/apps/client/.../AtsIntegrationSettings/view.test.tsx"},
        ],
    )

    found = graph_reader.tests_naming([("ui", "handleCcEmailAddressesBlur", [])])

    assert found == {
        ("ui", "handleCcEmailAddressesBlur"): [
            "src/apps/client/.../AtsIntegrationSettings/view.test.tsx",
            "src/apps/internal/.../ApplicationSettingsTab/view.data.test.tsx",
        ]
    }
    query = driver._session.last_query
    # the file's own stored text, not its chunks: a test file's chunks are a
    # fraction of it and do not contain the fixture
    assert "coalesce(f.source, '') CONTAINS term" in query
    # a plain CONTAINS, not a vector search — this runs after the files to
    # change are chosen, so it can never take a slot from real code
    assert "queryNodes" not in query


def test_tests_naming_ignores_test_path_files_that_are_not_specs(monkeypatch):
    """A generated GraphQL type declaration under cypress/support/@types/ is
    test-path code but is not a test. Eight of them match the ATS ticket's
    field, and they are the SHORTEST matches, so smallest-first would show
    the planner nothing but type stubs."""
    driver = _patch_driver(monkeypatch, [])

    graph_reader.tests_naming([("ui", "X", ["field"])])

    assert "f.file_path =~ $spec_re" in driver._session.last_query
    spec_re = driver._session.last_params["spec_re"]
    import re

    assert re.fullmatch(spec_re, "a/view.test.tsx")
    assert not re.fullmatch(spec_re, "a/cypress/support/graphql/@types/Settings.ts")


def test_tests_naming_also_matches_the_issue_identifiers_a_symbol_touches(monkeypatch):
    """No test in the ui repo names AtsIntegrationSectionView — a component
    is rendered through a wrapper — while four name the field the ticket is
    about, and those four hold the working fixture."""
    driver = _patch_driver(monkeypatch, [])

    graph_reader.tests_naming(
        [("ui", "AtsIntegrationSectionView", ["sendReportRequestedLinkCcEmailAddresses"])]
    )

    assert driver._session.last_params["lookups"] == [
        {
            "service": "ui",
            "symbol": "AtsIntegrationSectionView",
            # the symbol's own name is always one of the terms
            "terms": ["AtsIntegrationSectionView", "sendReportRequestedLinkCcEmailAddresses"],
        }
    ]


def test_tests_naming_asks_nothing_when_there_are_no_symbols(monkeypatch):
    assert graph_reader.tests_naming([]) == {}
    assert graph_reader.tests_naming([("ui", "", ["x"])]) == {}


def test_load_test_sources_returns_the_stored_text_per_file(monkeypatch):
    """A path alone taught the planner nothing about the fixture, which is
    the one thing it cannot guess and the one thing it got wrong."""
    driver = _patch_driver(
        monkeypatch,
        [{"service": "ui", "file_path": "a/view.test.tsx", "source": "const settings = {...}"}],
    )

    found = graph_reader.load_test_sources([("ui", "a/view.test.tsx")])

    assert found == {("ui", "a/view.test.tsx"): "const settings = {...}"}
    query = driver._session.last_query
    # only files that actually carry text — production files never do
    assert "f.source IS NOT NULL AND f.source <> ''" in query


def test_load_test_sources_asks_nothing_for_an_empty_list(monkeypatch):
    assert graph_reader.load_test_sources([]) == {}
