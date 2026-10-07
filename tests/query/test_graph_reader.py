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
    assert "service_summary_index" in driver._session.last_query


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
    assert "s.summary AS summary" in driver._session.last_query
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
