from src.ingestion import code_graph_writer


class FakeSession:
    def __init__(self):
        self.queries = []

    def run(self, query, **params):
        self.queries.append((query, params))
        return []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self):
        self._session = FakeSession()

    def session(self):
        return self._session


def _patch_driver(monkeypatch):
    fake_driver = FakeDriver()
    monkeypatch.setattr(code_graph_writer, "get_driver", lambda: fake_driver)
    return fake_driver


def test_upsert_classes_batches_via_unwind(monkeypatch):
    driver = _patch_driver(monkeypatch)
    rows = [{"qualified_name": "a.py::A", "name": "A"}, {"qualified_name": "b.py::B", "name": "B"}]
    code_graph_writer.upsert_classes("orders", rows)

    query, params = driver._session.queries[0]
    assert "UNWIND $rows AS row" in query
    assert "MERGE (c:Class {service: $service, qualified_name: row.qualified_name})" in query
    assert params == {"service": "orders", "rows": rows}


def test_upsert_classes_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_classes("orders", [])
    assert driver._session.queries == []


def test_upsert_functions_batches_via_unwind(monkeypatch):
    driver = _patch_driver(monkeypatch)
    rows = [{"qualified_name": "a.py::fn"}]
    code_graph_writer.upsert_functions("orders", rows)

    query, params = driver._session.queries[0]
    assert "MERGE (f:Function {service: $service, qualified_name: row.qualified_name})" in query
    assert params == {"service": "orders", "rows": rows}


def test_create_contains_edges_uses_given_label(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_contains_edges("orders", ["a.py::A"], "Class")

    query, params = driver._session.queries[0]
    assert "MATCH (n:Class {service: $service, qualified_name: qname})" in query
    assert "MERGE (s)-[:CONTAINS]->(n)" in query
    assert params == {"service": "orders", "qnames": ["a.py::A"]}


def test_create_contains_edges_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_contains_edges("orders", [], "Function")
    assert driver._session.queries == []


def test_create_declares_edges_builds_pair_rows(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_declares_edges(
        "orders", [("a.py::A", "a.py::A.method"), ("a.py::B", "a.py::B.method")]
    )

    query, params = driver._session.queries[0]
    assert "MERGE (c)-[:DECLARES]->(f)" in query
    assert params["rows"] == [
        {"class_qname": "a.py::A", "method_qname": "a.py::A.method"},
        {"class_qname": "a.py::B", "method_qname": "a.py::B.method"},
    ]


def test_prune_stale_code_nodes_detaches_and_deletes(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.prune_stale_code_nodes("orders", ["a.py::A"])

    query, params = driver._session.queries[0]
    assert "DETACH DELETE n" in query
    assert "NOT n.qualified_name IN $current_qualified_names" in query
    assert params == {"service": "orders", "current_qualified_names": ["a.py::A"]}


def test_create_code_indexes_creates_both_indexes(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_code_indexes()

    queries = [q for q, _ in driver._session.queries]
    assert any("FOR (c:Class) ON (c.service)" in q for q in queries)
    assert any("FOR (f:Function) ON (f.service)" in q for q in queries)


def test_upsert_chunks_batches_via_unwind(monkeypatch):
    driver = _patch_driver(monkeypatch)
    rows = [{"qualified_name": "a.py::fn::chunk0", "text": "..."}]
    code_graph_writer.upsert_chunks("orders", rows)

    query, params = driver._session.queries[0]
    assert "MERGE (c:Chunk {service: $service, qualified_name: row.qualified_name})" in query
    assert params == {"service": "orders", "rows": rows}


def test_upsert_chunks_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_chunks("orders", [])
    assert driver._session.queries == []


def test_create_has_chunk_edges_builds_pair_rows(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_has_chunk_edges(
        "orders", [("a.py::fn", "a.py::fn::chunk0"), ("a.py::fn", "a.py::fn::chunk1")]
    )

    query, params = driver._session.queries[0]
    assert "MATCH (f:Function {service: $service, qualified_name: row.function_qname})" in query
    assert "MERGE (f)-[:HAS_CHUNK]->(c)" in query
    assert params["rows"] == [
        {"function_qname": "a.py::fn", "chunk_qname": "a.py::fn::chunk0"},
        {"function_qname": "a.py::fn", "chunk_qname": "a.py::fn::chunk1"},
    ]


def test_create_has_chunk_edges_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_has_chunk_edges("orders", [])
    assert driver._session.queries == []


def test_create_chunk_vector_index_uses_given_dimensions(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_chunk_vector_index(dimensions=1536)

    query, params = driver._session.queries[0]
    assert "FOR (c:Chunk) ON (c.embedding)" in query
    assert params == {"dimensions": 1536}


def test_prune_stale_code_nodes_covers_chunk_label_too(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.prune_stale_code_nodes("orders", ["a.py::A"])

    query, _ = driver._session.queries[0]
    assert "n:Chunk" in query
