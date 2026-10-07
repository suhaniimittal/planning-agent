from src.ingestion import code_graph_writer


class FakeSession:
    def __init__(self, rows=None):
        self.queries = []
        self._rows = rows if rows is not None else []

    def run(self, query, **params):
        self.queries.append((query, params))
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, rows=None):
        self._session = FakeSession(rows)

    def session(self):
        return self._session


def _patch_driver(monkeypatch, rows=None):
    fake_driver = FakeDriver(rows)
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


def test_upsert_files_batches_via_unwind(monkeypatch):
    driver = _patch_driver(monkeypatch)
    rows = [{"file_path": "a.py", "language": "python"}]
    code_graph_writer.upsert_files("orders", rows)

    query, params = driver._session.queries[0]
    assert "MERGE (f:File {service: $service, file_path: row.file_path})" in query
    assert params == {"service": "orders", "rows": rows}


def test_upsert_files_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_files("orders", [])
    assert driver._session.queries == []


def test_create_has_file_edges_builds_query(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_has_file_edges("orders", ["a.py"])

    query, params = driver._session.queries[0]
    assert "MATCH (f:File {service: $service, file_path: file_path})" in query
    assert "MERGE (s)-[:HAS_FILE]->(f)" in query
    assert params == {"service": "orders", "file_paths": ["a.py"]}


def test_create_has_file_edges_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_has_file_edges("orders", [])
    assert driver._session.queries == []


def test_create_defines_edges_uses_given_label(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_defines_edges("orders", [("a.py", "a.py::A")], "Class")

    query, params = driver._session.queries[0]
    assert "MATCH (file:File {service: $service, file_path: row.file_path})" in query
    assert "MATCH (n:Class {service: $service, qualified_name: row.qname})" in query
    assert "MERGE (file)-[:DEFINES]->(n)" in query
    assert params["rows"] == [{"file_path": "a.py", "qname": "a.py::A"}]


def test_create_defines_edges_skips_query_for_empty_list(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_defines_edges("orders", [], "Function")
    assert driver._session.queries == []


def test_prune_stale_files_detaches_and_deletes(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.prune_stale_files("orders", ["a.py"])

    query, params = driver._session.queries[0]
    assert "DETACH DELETE f" in query
    assert "NOT f.file_path IN $current_file_paths" in query
    assert params == {"service": "orders", "current_file_paths": ["a.py"]}


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


def test_create_code_indexes_creates_all_indexes(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_code_indexes()

    queries = [q for q, _ in driver._session.queries]
    assert any("FOR (f:File) ON (f.service)" in q for q in queries)
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


# --- Phase 4: git-diff incremental update support ---------------------------


def test_prune_stale_code_nodes_for_file_scopes_by_file_path(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.prune_stale_code_nodes_for_file("orders", "a.py", ["a.py::A"])

    query, params = driver._session.queries[0]
    assert "n.file_path = $file_path" in query
    assert "NOT n.qualified_name IN $current_qualified_names" in query
    assert params == {"service": "orders", "file_path": "a.py", "current_qualified_names": ["a.py::A"]}


def test_load_file_blob_shas_returns_mapping(monkeypatch):
    rows = [
        {"file_path": "a.py", "blob_sha": "sha_a"},
        {"file_path": "b.py", "blob_sha": "sha_b"},
    ]
    _patch_driver(monkeypatch, rows)

    shas = code_graph_writer.load_file_blob_shas("orders")

    assert shas == {"a.py": "sha_a", "b.py": "sha_b"}


def test_load_file_blob_shas_skips_null_shas(monkeypatch):
    rows = [{"file_path": "a.py", "blob_sha": None}]
    _patch_driver(monkeypatch, rows)

    assert code_graph_writer.load_file_blob_shas("orders") == {}


def test_delete_file_and_descendants_scopes_by_service_and_file_path(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.delete_file_and_descendants("orders", "a.py")

    query, params = driver._session.queries[0]
    assert "DETACH DELETE n" in query
    assert "n:File OR n:Class OR n:Function OR n:Chunk" in query
    assert params == {"service": "orders", "file_path": "a.py"}


def test_rename_file_updates_file_path_property(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.rename_file("orders", "old.py", "new.py")

    query, params = driver._session.queries[0]
    assert "SET n.file_path = $new_path" in query
    assert params == {"service": "orders", "old_path": "old.py", "new_path": "new.py"}


def test_get_aggregated_file_references_returns_distinct_targets(monkeypatch):
    rows = [{"target": "payment"}, {"target": "shipping"}]
    _patch_driver(monkeypatch, rows)

    targets = code_graph_writer.get_aggregated_file_references("orders")

    assert targets == ["payment", "shipping"]


def test_load_current_symbols_splits_by_kind(monkeypatch):
    rows = [
        {
            "qualified_name": "a.py::A",
            "name": "A",
            "file_path": "a.py",
            "language": "python",
            "docstring": None,
            "is_method": None,
            "parent_class": None,
            "kind": "Class",
        },
        {
            "qualified_name": "a.py::fn",
            "name": "fn",
            "file_path": "a.py",
            "language": "python",
            "docstring": None,
            "is_method": False,
            "parent_class": None,
            "kind": "Function",
        },
    ]
    _patch_driver(monkeypatch, rows)

    classes, functions = code_graph_writer.load_current_symbols("orders")

    assert [c["name"] for c in classes] == ["A"]
    assert [f["name"] for f in functions] == ["fn"]
    assert "kind" not in classes[0]
    assert "kind" not in functions[0]


def test_upsert_chunks_writes_in_slices(monkeypatch):
    driver = _patch_driver(monkeypatch)
    monkeypatch.setattr(code_graph_writer, "_CHUNK_WRITE_BATCH", 2)
    rows = [{"qualified_name": f"a.py::fn::chunk{i}"} for i in range(5)]

    code_graph_writer.upsert_chunks("orders", rows)

    sent = [params["rows"] for _, params in driver._session.queries]
    assert sent == [rows[0:2], rows[2:4], rows[4:5]]
    assert all(params["service"] == "orders" for _, params in driver._session.queries)


def test_create_has_file_edges_writes_in_slices(monkeypatch):
    driver = _patch_driver(monkeypatch)
    monkeypatch.setattr(code_graph_writer, "_WRITE_BATCH", 2)

    code_graph_writer.create_has_file_edges("orders", ["a.py", "b.py", "c.py"])

    assert [p["file_paths"] for _, p in driver._session.queries] == [["a.py", "b.py"], ["c.py"]]
