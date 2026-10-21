from src.ingestion import graph_writer


class FakeResult(list):
    def single(self):
        return self[0] if self else None


class FakeSession:
    def __init__(self, rows=None):
        self.queries = []
        self._rows = rows if rows is not None else []

    def run(self, query, **params):
        self.queries.append((query, params))
        return FakeResult(self._rows)

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
    monkeypatch.setattr(graph_writer, "get_driver", lambda: fake_driver)
    return fake_driver


def test_upsert_service_node_sets_name_and_repo(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.upsert_service_node("orders", "org/orders-service")

    query, params = driver._session.queries[0]
    assert "MERGE (s:Service {name: $name})" in query
    assert "s.repo = $repo" in query
    assert params == {"name": "orders", "repo": "org/orders-service"}


def test_upsert_service_node_removes_stale_md_derived_properties(monkeypatch):
    """A Service node ingested by the old RepoDocAI/md pipeline still has
    full_doc/domain/entry_points/external_systems/key_symbols/
    summary_is_fallback — re-ingesting it via the new manifest-driven path
    must strip these, not just leave them sitting there forever."""
    driver = _patch_driver(monkeypatch)

    graph_writer.upsert_service_node("orders", "org/orders-service")

    query, _ = driver._session.queries[0]
    assert "REMOVE s.full_doc" in query
    assert "s.domain" in query
    assert "s.entry_points" in query
    assert "s.external_systems" in query
    assert "s.key_symbols" in query
    assert "s.summary_is_fallback" in query


def test_upsert_service_node_removes_the_generated_summary_it_replaced(monkeypatch):
    """A service matched against its committed SERVICE.md must not keep the
    LLM-written summary that preceded it: the two describe the same service
    differently, and only `doc` is read. Eleven nodes still carried one."""
    driver = _patch_driver(monkeypatch)

    graph_writer.upsert_service_node("orders", "org/orders-service")

    query, _ = driver._session.queries[0]
    assert "s.summary" in query
    assert "s.summary_embedding" in query


def test_update_service_doc_sets_doc_embedding_and_hash(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.update_service_doc("orders", "# orders\n\nPlacing orders.", [0.1, 0.2], "abc123")

    query, params = driver._session.queries[0]
    assert "MATCH (s:Service {name: $name})" in query
    assert "s.doc = $doc" in query
    assert "s.doc_embedding = $embedding" in query
    # stored so the next run can tell an edited SERVICE.md from an unchanged
    # one — the file, not the commit, decides whether this moves.
    assert "s.doc_hash = $doc_hash" in query
    assert params == {
        "name": "orders",
        "doc": "# orders\n\nPlacing orders.",
        "embedding": [0.1, 0.2],
        "doc_hash": "abc123",
    }


def test_set_service_dependencies_passes_source_and_targets(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.set_service_dependencies("orders", ["payment", "shipping"])

    query, params = driver._session.queries[0]
    assert "MATCH (a:Service {name: $source})" in query
    assert "DELETE r" in query
    assert "MERGE (a)-[:CALLS]->(b)" in query
    assert params == {"source": "orders", "targets": ["payment", "shipping"]}


def test_set_service_dependencies_with_empty_targets_still_runs(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.set_service_dependencies("orders", [])

    _, params = driver._session.queries[0]
    assert params == {"source": "orders", "targets": []}


def test_create_vector_index_passes_dimensions(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.create_vector_index(dimensions=1536)

    query, params = driver._session.queries[0]
    assert "CREATE VECTOR INDEX service_doc_index" in query
    assert params == {"dimensions": 1536}


def test_load_service_state_returns_commit_sha_when_node_exists(monkeypatch):
    _patch_driver(
        monkeypatch,
        [
            {
                "commit_sha": "abc123",
                "local_fingerprint": "fp1",
                "services_fingerprint": "sf1",
                "pipeline_version": 1,
            }
        ],
    )

    state = graph_writer.load_service_state("orders")

    assert state == {
        "commit_sha": "abc123",
        "local_fingerprint": "fp1",
        "services_fingerprint": "sf1",
        # which version of the parsing/embedding code built this graph —
        # a mismatch forces a rebuild from the same unchanged source
        "pipeline_version": 1,
    }


def test_load_service_state_returns_none_when_service_never_ingested(monkeypatch):
    _patch_driver(monkeypatch, [])

    assert graph_writer.load_service_state("orders") is None


def test_load_service_state_handles_node_with_no_commit_sha_yet(monkeypatch):
    _patch_driver(
        monkeypatch,
        [
            {
                "commit_sha": None,
                "local_fingerprint": None,
                "services_fingerprint": None,
                "pipeline_version": None,
            }
        ],
    )

    assert graph_writer.load_service_state("orders") == {
        "commit_sha": None,
        "local_fingerprint": None,
        "services_fingerprint": None,
        # None for a service ingested before this was recorded — unequal to
        # any version, so it rebuilds rather than being trusted
        "pipeline_version": None,
    }


def test_set_service_commit_writes_commit_sha_and_timestamp(monkeypatch):
    driver = _patch_driver(monkeypatch)

    graph_writer.set_service_commit("orders", "abc123")

    query, params = driver._session.queries[0]
    assert "s.commit_sha = $commit_sha" in query
    assert "s.last_ingested_at = $last_ingested_at" in query
    assert params["commit_sha"] == "abc123"
    assert params["name"] == "orders"
    assert isinstance(params["last_ingested_at"], str) and params["last_ingested_at"]
