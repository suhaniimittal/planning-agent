from src.ingestion import code_crypto, code_graph_writer


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
    rows = [{"qualified_name": "a.py::fn::chunk0", "text": "total = applyDiscount(x)"}]
    code_graph_writer.upsert_chunks("orders", rows)

    query, params = driver._session.queries[0]
    assert "MERGE (c:Chunk {service: $service, qualified_name: row.qualified_name})" in query
    assert params["service"] == "orders"
    (stored,) = params["rows"]
    # stored encrypted, with the hashes of its words for the exact-name search
    assert code_crypto.is_encrypted(stored["text"]) and "applyDiscount" not in stored["text"]
    assert stored["text_terms"] == code_crypto.text_terms("orders", "total = applyDiscount(x)")
    assert rows[0]["text"] == "total = applyDiscount(x)"  # the caller's row is untouched


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
    # Any symbol that owns lines owns chunks: a function, a class's fields,
    # a large constant.
    assert "MATCH (f:Symbol {service: $service, qualified_name: row.function_qname})" in query
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
    assert "n:File OR n:Class OR n:Function OR n:Variable OR n:Chunk" in query
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


# --- pipeline version 4 ------------------------------------------------------


def test_every_symbol_upsert_adds_the_symbol_label(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_classes("orders", [{"qualified_name": "a.py::A"}])
    code_graph_writer.upsert_functions("orders", [{"qualified_name": "a.py::f"}])
    code_graph_writer.upsert_variables("orders", [{"qualified_name": "a.py::X"}])

    queries = [q for q, _ in driver._session.queries]
    assert "SET c += row, c:Symbol" in queries[0]
    assert "SET f += row, f:Symbol" in queries[1]
    assert "MERGE (v:Variable {service: $service, qualified_name: row.qualified_name})" in queries[2]
    assert "v:Symbol" in queries[2]


def test_create_contains_edges_links_any_two_symbols(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.create_contains_edges("ui", [("v.tsx::Screen", "v.tsx::Screen.save")])

    query, params = driver._session.queries[0]
    assert "MERGE (p)-[:CONTAINS]->(c)" in query
    assert params["rows"] == [{"parent": "v.tsx::Screen", "child": "v.tsx::Screen.save"}]


def test_load_chunk_embeddings_looks_up_by_text_hash(monkeypatch):
    driver = _patch_driver(monkeypatch, rows=[{"text_hash": "h1", "embedding": [0.5]}])

    found = code_graph_writer.load_chunk_embeddings("ui", ["h1", "h2", "h1"])

    query, params = driver._session.queries[0]
    assert "MATCH (c:Chunk {service: $service, text_hash: h})" in query
    assert params == {"service": "ui", "hashes": ["h1", "h2"]}
    assert found == {"h1": [0.5]}


def test_load_chunk_embeddings_asks_nothing_for_no_hashes(monkeypatch):
    driver = _patch_driver(monkeypatch)
    assert code_graph_writer.load_chunk_embeddings("ui", []) == {}
    assert driver._session.queries == []


def test_write_service_links_clears_then_writes_each_kind(monkeypatch):
    from src.ingestion.db_schema import Column, Table
    from src.ingestion.import_resolver import ResolvedImport
    from src.ingestion.links import Operation, ServiceLinks

    driver = _patch_driver(monkeypatch)
    links = ServiceLinks(
        imports=[ResolvedImport("a.tsx", "b.tsx", ["B"], ["index.ts"])],
        unresolved_imports={"a.tsx": ["./gone"]},
        references=[("a.tsx::A", "b.tsx::B")],
        tests=[("a.test.tsx", "a.tsx")],
        tables=[Table("rhino.reports", {"id": Column("id", "uuid", False)}, ["001.sql"], ["rhino.clients"])],
        maps_to=[("E.kt::E", "rhino.reports")],
        operations=[Operation("Query", "reports", "s.graphqls")],
        resolves=[("R.kt::R.reports", "Query", "reports")],
        endpoints=[("C.kt::C.get", "GET", "/x")],
        kafka={"H.kt::H": {"handles": ["com.a.M"], "input_topics": ["Topics.domain.orders"]}},
    )

    code_graph_writer.write_service_links("rhino-service", links)

    queries = [q for q, _ in driver._session.queries]
    joined = "\n".join(queries)
    # Old links go first, so a removed import or route does not linger.
    assert queries[0].strip().startswith("MATCH (:File {service: $service})-[r:IMPORTS|TESTS]->()")
    assert "DETACH DELETE n" in queries[2]
    for fragment in (
        "MERGE (a)-[r:IMPORTS]->(b)", "SET f.unresolved_imports", "MERGE (a)-[:TESTS]->(b)",
        "MERGE (a)-[:REFERENCES]->(b)", "CREATE (t:Table", "MERGE (a)-[:FOREIGN_KEY]->(b)",
        "MERGE (a)-[:MAPS_TO]->(t)", "MERGE (o:Operation", "MERGE (a)-[:RESOLVES]->(o)",
        "MERGE (a)-[:SERVES]->(e)", "SET c.kafka_handles",
    ):
        assert fragment in joined, fragment
    table_params = next(p for q, p in driver._session.queries if "CREATE (t:Table" in q)
    assert table_params["rows"][0]["columns"] == ["id uuid not null"]


def test_write_cross_service_links_clears_only_the_given_services_topic_links(monkeypatch):
    from src.ingestion.service_links import CrossServiceLinks

    driver = _patch_driver(monkeypatch)
    code_graph_writer.write_cross_service_links(["rhino-service"], CrossServiceLinks())

    first, params = driver._session.queries[0]
    assert "-[r:ON_TOPIC]->" in first and "WHERE c.service IN $services" in first
    assert params == {"services": ["rhino-service"]}
    assert not any(
        q.strip() == "MATCH (:Message)-[r:ON_TOPIC]->(:Topic) DELETE r"
        for q, _ in driver._session.queries
    )


def test_upsert_files_encrypts_imports_and_test_text_by_path(monkeypatch):
    driver = _patch_driver(monkeypatch)
    rows = [
        {"file_path": "a/v.test.ts", "language": "ts", "imports": "import x;", "source": "it()"}
    ]
    code_graph_writer.upsert_files("ui", rows)

    (stored,) = driver._session.queries[0][1]["rows"]
    for field, plain in (("imports", "import x;"), ("source", "it()")):
        assert code_crypto.is_encrypted(stored[field])
        assert (
            code_crypto.decrypt(
                stored[field], label="File", service="ui", qualified_name="a/v.test.ts", field=field
            )
            == plain
        )
    assert rows[0]["source"] == "it()"  # the caller's row is untouched


def test_upsert_files_stores_the_hashes_of_the_sources_words(monkeypatch):
    """What find_text matches, since Neo4j cannot search the encrypted text."""
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_files(
        "ui", [{"file_path": "v.tsx", "source": "<Grid item md={4}>"}, {"file_path": "x.png"}]
    )

    with_source, without = driver._session.queries[0][1]["rows"]
    assert with_source["text_terms"] == code_crypto.text_terms("ui", "Grid item md")
    assert "text_terms" not in without


def test_upsert_variables_encrypts_their_code(monkeypatch):
    driver = _patch_driver(monkeypatch)
    code_graph_writer.upsert_variables(
        "ui", [{"qualified_name": "s.ts::Wrap", "source_snippet": "styled.div``"}]
    )

    (stored,) = driver._session.queries[0][1]["rows"]
    assert code_crypto.is_encrypted(stored["source_snippet"])
    assert code_crypto.decrypt(
        stored["source_snippet"], label="Variable", service="ui", qualified_name="s.ts::Wrap",
        field="source_snippet",
    ) == "styled.div``"


def test_rename_file_reseals_the_files_text_under_its_new_path(monkeypatch):
    sealed = code_crypto.encrypt_row(
        {"source": "it()", "imports": None}, label="File", service="ui", qualified_name="old.ts"
    )
    driver = _patch_driver(monkeypatch, [sealed])

    code_graph_writer.rename_file("ui", "old.ts", "new.ts")

    query, params = driver._session.queries[-1]
    assert "SET f.source = $source, f.imports = $imports" in query
    assert params["imports"] is None
    assert (
        code_crypto.decrypt(
            params["source"], label="File", service="ui", qualified_name="new.ts", field="source"
        )
        == "it()"
    )
