import pytest

from src.ingestion import github_api, ingest
from src.ingestion.code_parser import CodeChunk, ParsedSymbol, RepoGraph
from src.ingestion.manifest import ServiceSpec


@pytest.fixture(autouse=True)
def _no_real_embedding_calls(monkeypatch):
    """No test may reach the real embedding API. Tests that don't care about
    embeddings get a call that fails at once (a chunk embed failing is
    always survivable) with no retry waits; tests that do care override
    these via _patch_embed or their own monkeypatch."""

    async def unavailable(*args, **kwargs):
        raise RuntimeError("embedding API not available in tests")

    monkeypatch.setattr(ingest, "embed", unavailable)
    monkeypatch.setattr(ingest, "embed_many", unavailable)
    monkeypatch.setattr(ingest, "_EMBEDDING_RETRY_DELAYS", ())


def _patch_embed(monkeypatch, fake_embed):
    """Stubs both the single and the batched embedding call with the same
    per-text fake — a batch fails as a whole if any one text in it fails,
    just like the real API."""

    async def fake_embed_many(texts):
        return [await fake_embed(t) for t in texts]

    monkeypatch.setattr(ingest, "embed", fake_embed)
    monkeypatch.setattr(ingest, "embed_many", fake_embed_many)
    monkeypatch.setattr(ingest, "_EMBEDDING_RETRY_DELAYS", ())


def _fn_symbol(name="create_order", chunks=None, file_path="controller.py") -> ParsedSymbol:
    return ParsedSymbol(
        kind="function",
        name=name,
        qualified_name=f"{file_path}::{name}",
        file_path=file_path,
        line_start=1,
        line_end=3,
        signature=f"def {name}():",
        docstring=None,
        language="python",
        source_snippet=f"def {name}():\n    pass",
        chunks=chunks if chunks is not None else [],
    )


def _class_symbol(name, file_path, qualified_name=None) -> ParsedSymbol:
    return ParsedSymbol(
        kind="class",
        name=name,
        qualified_name=qualified_name or f"{file_path}::{name}",
        file_path=file_path,
        line_start=1,
        line_end=10,
        signature=f"class {name}:",
        docstring=None,
        language="python",
        source_snippet=f"class {name}: ...",
    )


def test_chunk_row_builds_expected_shape():
    fn = _fn_symbol()
    chunk = CodeChunk(index=0, line_start=1, line_end=3, text="def create_order():\n    pass")
    row = ingest._chunk_row(fn, chunk, [0.1, 0.2])

    assert row == {
        "qualified_name": "controller.py::create_order::chunk0",
        "function_qualified_name": "controller.py::create_order",
        "chunk_index": 0,
        "file_path": "controller.py",
        "line_start": 1,
        "line_end": 3,
        "text": "def create_order():\n    pass",
        "embedding": [0.1, 0.2],
        "language": "python",
    }


@pytest.mark.asyncio
async def test_embed_chunks_returns_rows_and_edge_pairs(monkeypatch):
    chunk = CodeChunk(index=0, line_start=1, line_end=3, text="def create_order():\n    pass")
    fn = _fn_symbol(chunks=[chunk])

    async def fake_embed(text):
        return [0.5, 0.6]

    _patch_embed(monkeypatch, fake_embed)

    rows, edges = await ingest._embed_chunks("orders", [fn])

    assert len(rows) == 1
    assert rows[0]["qualified_name"] == "controller.py::create_order::chunk0"
    assert rows[0]["embedding"] == [0.5, 0.6]
    assert edges == [("controller.py::create_order", "controller.py::create_order::chunk0")]


@pytest.mark.asyncio
async def test_embed_chunks_skips_failed_embedding_without_aborting_others(monkeypatch):
    good_chunk = CodeChunk(index=0, line_start=1, line_end=2, text="good")
    bad_chunk = CodeChunk(index=0, line_start=1, line_end=2, text="bad")
    good_fn = _fn_symbol(name="good_fn", chunks=[good_chunk])
    bad_fn = _fn_symbol(name="bad_fn", chunks=[bad_chunk])

    async def flaky_embed(text):
        if text == "bad":
            raise RuntimeError("rate limited")
        return [0.1]

    _patch_embed(monkeypatch, flaky_embed)

    rows, edges = await ingest._embed_chunks("orders", [good_fn, bad_fn])

    assert len(rows) == 1
    assert rows[0]["function_qualified_name"] == "controller.py::good_fn"
    assert len(edges) == 1


@pytest.mark.asyncio
async def test_embed_chunks_handles_multiple_chunks_per_function(monkeypatch):
    chunks = [
        CodeChunk(index=0, line_start=1, line_end=5, text="def big():\npart one"),
        CodeChunk(index=1, line_start=6, line_end=10, text="def big():\npart two"),
    ]
    fn = _fn_symbol(name="big", chunks=chunks)

    async def fake_embed(text):
        return [len(text) * 1.0]

    _patch_embed(monkeypatch, fake_embed)

    rows, edges = await ingest._embed_chunks("orders", [fn])

    assert {r["qualified_name"] for r in rows} == {
        "controller.py::big::chunk0",
        "controller.py::big::chunk1",
    }
    assert set(edges) == {
        ("controller.py::big", "controller.py::big::chunk0"),
        ("controller.py::big", "controller.py::big::chunk1"),
    }


@pytest.mark.asyncio
async def test_embed_chunks_sends_chunks_in_batches(monkeypatch):
    monkeypatch.setattr(ingest, "_EMBEDDING_BATCH_SIZE", 2)
    fns = [
        _fn_symbol(
            name=f"f{i}", chunks=[CodeChunk(index=0, line_start=1, line_end=2, text=f"t{i}")]
        )
        for i in range(5)
    ]
    batch_sizes = []

    async def fake_embed_many(texts):
        batch_sizes.append(len(texts))
        return [[float(t[1:])] for t in texts]

    async def must_not_be_called(text):
        raise AssertionError("per-chunk embed used although the batch succeeded")

    monkeypatch.setattr(ingest, "embed_many", fake_embed_many)
    monkeypatch.setattr(ingest, "embed", must_not_be_called)

    rows, edges = await ingest._embed_chunks("orders", fns)

    assert sorted(batch_sizes) == [1, 2, 2]
    assert {r["function_qualified_name"]: r["embedding"] for r in rows} == {
        f"controller.py::f{i}": [float(i)] for i in range(5)
    }
    assert len(edges) == 5


@pytest.mark.asyncio
async def test_embed_chunks_closes_a_batch_at_the_char_limit(monkeypatch):
    monkeypatch.setattr(ingest, "_EMBEDDING_BATCH_MAX_CHARS", 10)
    fns = [
        _fn_symbol(
            name=f"f{i}", chunks=[CodeChunk(index=0, line_start=1, line_end=2, text="x" * 6)]
        )
        for i in range(3)
    ]
    batch_sizes = []

    async def fake_embed_many(texts):
        batch_sizes.append(len(texts))
        return [[0.0] for _ in texts]

    monkeypatch.setattr(ingest, "embed_many", fake_embed_many)

    rows, _ = await ingest._embed_chunks("orders", fns)

    assert batch_sizes == [1, 1, 1]
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_embed_chunks_wrong_length_batch_reply_falls_back_to_per_chunk(monkeypatch):
    fns = [
        _fn_symbol(
            name=f"f{i}", chunks=[CodeChunk(index=0, line_start=1, line_end=2, text=f"t{i}")]
        )
        for i in range(2)
    ]

    async def short_embed_many(texts):
        return [[0.0]]

    async def fake_embed(text):
        return [1.0]

    monkeypatch.setattr(ingest, "embed_many", short_embed_many)
    monkeypatch.setattr(ingest, "embed", fake_embed)

    rows, _ = await ingest._embed_chunks("orders", fns)

    assert [r["embedding"] for r in rows] == [[1.0], [1.0]]


@pytest.mark.asyncio
async def test_embed_chunks_retries_a_failed_batch_before_falling_back(monkeypatch):
    fn = _fn_symbol(chunks=[CodeChunk(index=0, line_start=1, line_end=2, text="t")])
    attempts = []
    sleeps = []

    async def rate_limited_once(texts):
        attempts.append(len(texts))
        if len(attempts) == 1:
            raise RuntimeError("429 rate limited")
        return [[7.0]]

    async def must_not_be_called(text):
        raise AssertionError("per-chunk fallback used although the retry succeeded")

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(ingest, "embed_many", rate_limited_once)
    monkeypatch.setattr(ingest, "embed", must_not_be_called)
    monkeypatch.setattr(ingest, "_EMBEDDING_RETRY_DELAYS", (3,))
    monkeypatch.setattr(ingest.asyncio, "sleep", fake_sleep)

    rows, _ = await ingest._embed_chunks("orders", [fn])

    assert attempts == [1, 1]
    assert sleeps == [3]
    assert rows[0]["embedding"] == [7.0]


@pytest.mark.asyncio
async def test_embed_chunks_empty_functions_returns_empty(monkeypatch):
    rows, edges = await ingest._embed_chunks("orders", [])
    assert rows == []
    assert edges == []


def test_language_for_known_and_unknown_extensions():
    assert ingest._language_for("app.py") == "python"
    assert ingest._language_for("config.yaml") is None


# --- _ingest_full: first-time clone + parse ----------------------------------


def _patch_code_graph_writer_noop(monkeypatch, calls=None):
    calls = calls if calls is not None else {}
    for name in (
        "prune_stale_code_nodes",
        "prune_stale_files",
        "upsert_classes",
        "upsert_functions",
        "upsert_chunks",
        "create_declares_edges",
        "create_has_chunk_edges",
    ):
        monkeypatch.setattr(ingest.code_graph_writer, name, lambda *a, **k: None)

    monkeypatch.setattr(
        ingest.code_graph_writer, "upsert_files", lambda service, rows: calls.setdefault("upsert_files", rows)
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "create_has_file_edges",
        lambda service, paths: calls.setdefault("create_has_file_edges", paths),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "create_defines_edges",
        lambda service, pairs, label: calls.setdefault(f"create_defines_edges_{label}", pairs),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "get_aggregated_file_references",
        lambda service: calls.get("aggregated_references", []),
    )
    monkeypatch.setattr(
        ingest.graph_writer,
        "set_service_dependencies",
        lambda source, targets: calls.setdefault("set_service_dependencies", (source, targets)),
    )
    return calls


@pytest.mark.asyncio
async def test_ingest_full_writes_files_and_returns_commit_sha(monkeypatch):
    cls = _class_symbol("OrderController", "controller.py")
    top_level_fn = _fn_symbol(name="format_total")
    method = _fn_symbol(name="create_order")
    method.qualified_name = "controller.py::OrderController.create_order"
    method.is_method = True
    method.parent_class = "controller.py::OrderController"
    graph = RepoGraph(classes=[cls], functions=[top_level_fn, method])

    def fake_clone_and_parse_full(spec, github_token, known_service_names):
        return graph, {"controller.py": {"payment"}}, "commit123", {"controller.py": "blobsha1"}

    monkeypatch.setattr(ingest, "_clone_and_parse_full", fake_clone_and_parse_full)

    async def fake_embed(text):
        return [0.1]

    _patch_embed(monkeypatch, fake_embed)
    calls = _patch_code_graph_writer_noop(monkeypatch)
    calls["aggregated_references"] = ["payment"]

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result_graph, commit_sha = await ingest._ingest_full(spec, None, {"orders", "payment"})

    assert result_graph is graph
    assert commit_sha == "commit123"
    assert calls["upsert_files"] == [
        {
            "file_path": "controller.py",
            "language": "python",
            "blob_sha": "blobsha1",
            "references": ["payment"],
        }
    ]
    assert calls["create_has_file_edges"] == ["controller.py"]
    assert calls["create_defines_edges_Class"] == [("controller.py", "controller.py::OrderController")]
    # Only the TOP-LEVEL function gets a DEFINES edge — the method is
    # reached via its class's DECLARES edge instead.
    assert calls["create_defines_edges_Function"] == [("controller.py", "controller.py::format_total")]
    assert calls["set_service_dependencies"] == ("orders", ["payment"])


@pytest.mark.asyncio
async def test_ingest_full_streams_chunks_after_functions_and_prunes_last(monkeypatch):
    chunks = [CodeChunk(index=i, line_start=1, line_end=2, text=f"t{i}") for i in range(3)]
    fn = _fn_symbol(chunks=chunks)
    graph = RepoGraph(classes=[], functions=[fn])
    monkeypatch.setattr(
        ingest, "_clone_and_parse_full", lambda *a: (graph, {}, "commit123", {})
    )
    monkeypatch.setattr(ingest, "_EMBEDDING_BATCH_SIZE", 2)

    async def fake_embed(text):
        return [0.1]

    _patch_embed(monkeypatch, fake_embed)
    _patch_code_graph_writer_noop(monkeypatch)
    order = []
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "upsert_functions",
        lambda service, rows: order.append("functions"),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "upsert_chunks",
        lambda service, rows: order.append(("chunks", len(rows))),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "prune_stale_code_nodes",
        lambda service, names: order.append(("prune", sorted(names))),
    )

    await ingest._ingest_full(ServiceSpec(name="orders", repo="org/orders"), None, {"orders"})

    # Functions first (HAS_CHUNK edges need them), then one chunk write per
    # embedded batch, then the prune — which must keep every new chunk.
    assert order[0] == "functions"
    assert sorted(o for o in order[1:-1]) == [("chunks", 1), ("chunks", 2)]
    assert order[-1] == (
        "prune",
        sorted(
            ["controller.py::create_order"]
            + [f"controller.py::create_order::chunk{i}" for i in range(3)]
        ),
    )


@pytest.mark.asyncio
async def test_embed_chunks_with_write_batch_hands_off_rows_instead_of_keeping_them(monkeypatch):
    chunks = [CodeChunk(index=i, line_start=1, line_end=2, text=f"t{i}") for i in range(2)]
    fn = _fn_symbol(chunks=chunks)

    async def fake_embed(text):
        return [0.1]

    _patch_embed(monkeypatch, fake_embed)
    written = []

    rows, edges = await ingest._embed_chunks(
        "orders", [fn], write_batch=lambda rows, pairs: written.append((rows, pairs))
    )

    assert rows == []
    assert len(edges) == 2
    assert [r["qualified_name"] for r in written[0][0]] == [
        "controller.py::create_order::chunk0",
        "controller.py::create_order::chunk1",
    ]
    assert written[0][1] == edges


@pytest.mark.asyncio
async def test_embed_chunks_never_runs_two_batch_writes_at_once(monkeypatch):
    import threading
    import time

    monkeypatch.setattr(ingest, "_EMBEDDING_BATCH_SIZE", 1)
    chunks = [CodeChunk(index=i, line_start=1, line_end=2, text=f"t{i}") for i in range(6)]
    fn = _fn_symbol(chunks=chunks)

    async def fake_embed(text):
        return [0.1]

    _patch_embed(monkeypatch, fake_embed)
    lock = threading.Lock()
    active = [0]
    max_active = [0]

    def slow_write(rows, pairs):
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        time.sleep(0.01)
        with lock:
            active[0] -= 1

    await ingest._embed_chunks("orders", [fn], write_batch=slow_write)

    assert max_active[0] == 1


@pytest.mark.asyncio
async def test_embed_chunks_does_not_retry_an_input_the_api_rejects(monkeypatch):
    fn = _fn_symbol(chunks=[CodeChunk(index=0, line_start=1, line_end=2, text="t")])

    class BadRequest(Exception):
        status_code = 400

    attempts = []

    async def rejected(texts):
        attempts.append(texts)
        raise BadRequest("maximum input length is 8192 tokens")

    async def fake_embed(text):
        return [2.0]

    async def must_not_sleep(seconds):
        raise AssertionError("waited to retry an error that can never succeed")

    monkeypatch.setattr(ingest, "embed_many", rejected)
    monkeypatch.setattr(ingest, "embed", fake_embed)
    monkeypatch.setattr(ingest, "_EMBEDDING_RETRY_DELAYS", (5, 15))
    monkeypatch.setattr(ingest.asyncio, "sleep", must_not_sleep)

    rows, _ = await ingest._embed_chunks("orders", [fn])

    assert len(attempts) == 1
    assert rows[0]["embedding"] == [2.0]


def test_is_retryable_only_for_rate_limits_and_non_client_errors():
    def err(status):
        e = Exception()
        e.status_code = status
        return e

    assert ingest._is_retryable(err(429))
    assert ingest._is_retryable(err(500))
    assert ingest._is_retryable(Exception("connection reset"))
    assert not ingest._is_retryable(err(400))
    assert not ingest._is_retryable(err(413))


@pytest.mark.asyncio
async def test_ingest_full_returns_none_on_repo_fetch_error(monkeypatch):
    def fake_clone_and_parse_full(spec, github_token, known_service_names):
        raise ingest.RepoFetchError("clone failed")

    monkeypatch.setattr(ingest, "_clone_and_parse_full", fake_clone_and_parse_full)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    graph, commit_sha = await ingest._ingest_full(spec, None, {"orders"})

    assert graph is None
    assert commit_sha is None


@pytest.mark.asyncio
async def test_ingest_full_returns_none_when_no_symbols_parsed(monkeypatch):
    def fake_clone_and_parse_full(spec, github_token, known_service_names):
        return RepoGraph(classes=[], functions=[]), {}, "commit123", {}

    monkeypatch.setattr(ingest, "_clone_and_parse_full", fake_clone_and_parse_full)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    graph, commit_sha = await ingest._ingest_full(spec, None, {"orders"})

    assert graph is None
    assert commit_sha is None


@pytest.mark.asyncio
async def test_ingest_full_does_not_skip_when_only_tracked_files_exist(monkeypatch):
    """A repo with zero classes/functions but real tracked-only files
    (.html/.css/.csv) must still be ingested — those files are still worth
    a File node even with no code-shaped symbols."""
    graph = RepoGraph(classes=[], functions=[], tracked_files={"index.html": "html"})

    def fake_clone_and_parse_full(spec, github_token, known_service_names):
        return graph, {}, "commit123", {}

    monkeypatch.setattr(ingest, "_clone_and_parse_full", fake_clone_and_parse_full)
    calls = _patch_code_graph_writer_noop(monkeypatch)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result_graph, commit_sha = await ingest._ingest_full(spec, None, {"orders"})

    assert result_graph is graph
    assert commit_sha == "commit123"
    assert calls["upsert_files"] == [
        {
            "file_path": "index.html",
            "language": "html",
            "blob_sha": None,
            "references": [],
        }
    ]


# --- _ingest_incremental: per-file git-diff sync -----------------------------


def _patch_incremental_writers(monkeypatch, calls):
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "delete_file_and_descendants",
        lambda service, path: calls.setdefault("deleted", []).append(path),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "rename_file",
        lambda service, old, new: calls.setdefault("renamed", []).append((old, new)),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "upsert_files",
        lambda service, rows: calls.setdefault("upserted_files", []).extend(rows),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "create_has_file_edges",
        lambda service, paths: calls.setdefault("has_file_edges", []).extend(paths),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "upsert_classes",
        lambda service, rows: calls.setdefault("upserted_classes", []).extend(rows),
    )
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "upsert_functions",
        lambda service, rows: calls.setdefault("upserted_functions", []).extend(rows),
    )
    monkeypatch.setattr(ingest.code_graph_writer, "upsert_chunks", lambda *a: None)
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "prune_stale_code_nodes_for_file",
        lambda service, path, qnames: calls.setdefault("pruned_files", []).append(path),
    )
    monkeypatch.setattr(ingest.code_graph_writer, "create_defines_edges", lambda *a: None)
    monkeypatch.setattr(ingest.code_graph_writer, "create_declares_edges", lambda *a: None)
    monkeypatch.setattr(ingest.code_graph_writer, "create_has_chunk_edges", lambda *a: None)
    monkeypatch.setattr(
        ingest.code_graph_writer,
        "get_aggregated_file_references",
        lambda service: calls.get("aggregated_references", []),
    )
    monkeypatch.setattr(
        ingest.graph_writer,
        "set_service_dependencies",
        lambda source, targets: calls.setdefault("set_service_dependencies", (source, targets)),
    )


@pytest.mark.asyncio
async def test_ingest_incremental_deletes_removed_file(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "old.py", "status": "removed", "sha": None, "previous_path": None}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {})
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert calls["deleted"] == ["old.py"]
    assert "upserted_files" not in calls


@pytest.mark.asyncio
async def test_ingest_incremental_renames_file_with_unchanged_content(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "new.py", "status": "renamed", "sha": "sha1", "previous_path": "old.py"}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {"old.py": "sha1"})

    def fail_fetch(*a, **k):
        raise AssertionError("must not fetch content for a pure rename")

    monkeypatch.setattr(github_api, "fetch_file_content", fail_fetch)
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert calls["renamed"] == [("old.py", "new.py")]


@pytest.mark.asyncio
async def test_ingest_incremental_reparses_renamed_file_when_content_changed(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "new.py", "status": "renamed", "sha": "sha2", "previous_path": "old.py"}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {"old.py": "sha1"})
    monkeypatch.setattr(github_api, "fetch_file_content", lambda repo, path, ref, token: "def handler(): pass")
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert "renamed" not in calls
    assert calls["pruned_files"] == ["new.py"]
    assert any(f["name"] == "handler" for f in calls["upserted_functions"])


@pytest.mark.asyncio
async def test_ingest_incremental_skips_when_blob_sha_matches_despite_modified_status(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "a.py", "status": "modified", "sha": "sha1", "previous_path": None}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {"a.py": "sha1"})

    def fail_fetch(*a, **k):
        raise AssertionError("must not fetch content when blob sha proves nothing changed")

    monkeypatch.setattr(github_api, "fetch_file_content", fail_fetch)
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert "upserted_files" not in calls
    assert "pruned_files" not in calls


@pytest.mark.asyncio
async def test_ingest_incremental_reparses_modified_file(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "app.py", "status": "modified", "sha": "sha2", "previous_path": None}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {"app.py": "sha1"})
    monkeypatch.setattr(
        github_api,
        "fetch_file_content",
        lambda repo, path, ref, token: "class OrderService:\n    def create(self):\n        pass\n",
    )
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert calls["pruned_files"] == ["app.py"]
    assert any(c["name"] == "OrderService" for c in calls["upserted_classes"])
    assert any(f["name"] == "create" for f in calls["upserted_functions"])
    assert calls["upserted_files"][0]["blob_sha"] == "sha2"


@pytest.mark.asyncio
async def test_ingest_incremental_handles_non_source_file_without_reparse(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "config.yaml", "status": "modified", "sha": "sha2", "previous_path": None}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {})
    monkeypatch.setattr(
        github_api, "fetch_file_content", lambda repo, path, ref, token: "url: http://payment-service/graphql"
    )
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders", "payment"}, "base", "head")

    assert "pruned_files" not in calls
    assert "upserted_classes" not in calls
    assert calls["upserted_files"] == [
        {
            "file_path": "config.yaml",
            "language": None,
            "blob_sha": "sha2",
            "references": ["payment"],
        }
    ]


@pytest.mark.asyncio
async def test_ingest_incremental_labels_tracked_only_extension_with_real_language(monkeypatch):
    """Unlike a truly unparseable file (language=None), a changed .html/
    .css/.csv file should get its real language label on the File row, not
    a bare None — it's tracked deliberately, not just tolerated."""
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "index.html", "status": "modified", "sha": "sha2", "previous_path": None}
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {})
    monkeypatch.setattr(github_api, "fetch_file_content", lambda repo, path, ref, token: "<html></html>")
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert calls["upserted_files"] == [
        {
            "file_path": "index.html",
            "language": "html",
            "blob_sha": "sha2",
            "references": [],
        }
    ]


@pytest.mark.asyncio
async def test_ingest_incremental_skips_file_on_fetch_error_without_aborting(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        github_api,
        "get_changed_files",
        lambda repo, base, head, token: [
            {"path": "bad.py", "status": "modified", "sha": "sha2", "previous_path": None},
            {"path": "good.py", "status": "modified", "sha": "sha3", "previous_path": None},
        ],
    )
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {})

    def fake_fetch(repo, path, ref, token):
        if path == "bad.py":
            raise github_api.GitHubApiError("boom")
        return "def handler(): pass"

    monkeypatch.setattr(github_api, "fetch_file_content", fake_fetch)
    _patch_incremental_writers(monkeypatch, calls)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert calls["pruned_files"] == ["good.py"]


@pytest.mark.asyncio
async def test_ingest_incremental_returns_early_on_compare_api_error(monkeypatch):
    def fail_get_changed_files(*a, **k):
        raise github_api.GitHubApiError("boom")

    monkeypatch.setattr(github_api, "get_changed_files", fail_get_changed_files)
    load_blob_shas_called = []
    monkeypatch.setattr(
        ingest.code_graph_writer, "load_file_blob_shas", lambda service: load_blob_shas_called.append(service)
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert load_blob_shas_called == []


@pytest.mark.asyncio
async def test_ingest_incremental_noop_when_no_files_changed(monkeypatch):
    monkeypatch.setattr(github_api, "get_changed_files", lambda repo, base, head, token: [])
    monkeypatch.setattr(ingest.code_graph_writer, "load_file_blob_shas", lambda service: {})

    set_deps_called = []
    monkeypatch.setattr(
        ingest.graph_writer, "set_service_dependencies", lambda *a: set_deps_called.append(a)
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest._ingest_incremental(spec, None, {"orders"}, "base", "head")

    assert set_deps_called == []


# --- sync_service: routes to skip / full ingest / incremental ---------------


@pytest.mark.asyncio
async def test_sync_service_skips_when_up_to_date(monkeypatch):
    monkeypatch.setattr(github_api, "get_remote_head_sha", lambda repo, branch, token: "abc")
    monkeypatch.setattr(ingest.graph_writer, "load_service_state", lambda name: {"commit_sha": "abc"})

    async def fail_full(*a):
        raise AssertionError("must not do a full ingest when up to date")

    async def fail_incremental(*a):
        raise AssertionError("must not do an incremental sync when up to date")

    monkeypatch.setattr(ingest, "_ingest_full", fail_full)
    monkeypatch.setattr(ingest, "_ingest_incremental", fail_incremental)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is None


@pytest.mark.asyncio
async def test_sync_service_does_full_ingest_when_never_ingested(monkeypatch):
    monkeypatch.setattr(github_api, "get_remote_head_sha", lambda repo, branch, token: "head_sha")
    monkeypatch.setattr(ingest.graph_writer, "load_service_state", lambda name: None)

    graph = RepoGraph(classes=[], functions=[])

    async def fake_full(spec, github_token, known_service_names):
        return graph, "head_sha"

    monkeypatch.setattr(ingest, "_ingest_full", fake_full)

    set_commit_calls = []
    monkeypatch.setattr(
        ingest.graph_writer, "set_service_commit", lambda name, sha: set_commit_calls.append((name, sha))
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is graph
    assert set_commit_calls == [("orders", "head_sha")]


@pytest.mark.asyncio
async def test_sync_service_does_full_ingest_when_state_has_no_commit_sha(monkeypatch):
    monkeypatch.setattr(github_api, "get_remote_head_sha", lambda repo, branch, token: "head_sha")
    monkeypatch.setattr(ingest.graph_writer, "load_service_state", lambda name: {"commit_sha": None})

    async def fake_full(spec, github_token, known_service_names):
        return None, "head_sha"

    monkeypatch.setattr(ingest, "_ingest_full", fake_full)
    set_commit_calls = []
    monkeypatch.setattr(
        ingest.graph_writer, "set_service_commit", lambda name, sha: set_commit_calls.append((name, sha))
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    await ingest.sync_service(spec, None, {"orders"})

    assert set_commit_calls == [("orders", "head_sha")]


@pytest.mark.asyncio
async def test_sync_service_does_not_advance_commit_when_full_ingest_fails(monkeypatch):
    monkeypatch.setattr(github_api, "get_remote_head_sha", lambda repo, branch, token: "head_sha")
    monkeypatch.setattr(ingest.graph_writer, "load_service_state", lambda name: None)

    async def fake_full(spec, github_token, known_service_names):
        return None, None

    monkeypatch.setattr(ingest, "_ingest_full", fake_full)
    set_commit_calls = []
    monkeypatch.setattr(
        ingest.graph_writer, "set_service_commit", lambda name, sha: set_commit_calls.append((name, sha))
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is None
    assert set_commit_calls == []


@pytest.mark.asyncio
async def test_sync_service_does_incremental_when_commit_differs(monkeypatch):
    monkeypatch.setattr(github_api, "get_remote_head_sha", lambda repo, branch, token: "head_sha")
    monkeypatch.setattr(ingest.graph_writer, "load_service_state", lambda name: {"commit_sha": "base_sha"})

    incremental_calls = []

    async def fake_incremental(spec, github_token, known_service_names, base_sha, head_sha):
        incremental_calls.append((base_sha, head_sha))

    monkeypatch.setattr(ingest, "_ingest_incremental", fake_incremental)
    set_commit_calls = []
    monkeypatch.setattr(
        ingest.graph_writer, "set_service_commit", lambda name, sha: set_commit_calls.append((name, sha))
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is None
    assert incremental_calls == [("base_sha", "head_sha")]
    assert set_commit_calls == [("orders", "head_sha")]


@pytest.mark.asyncio
async def test_sync_service_skips_on_head_sha_resolution_failure(monkeypatch):
    def fail_get_head(*a, **k):
        raise github_api.GitHubApiError("boom")

    monkeypatch.setattr(github_api, "get_remote_head_sha", fail_get_head)
    load_state_called = []
    monkeypatch.setattr(
        ingest.graph_writer, "load_service_state", lambda name: load_state_called.append(name)
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is None
    assert load_state_called == []


# --- run() manifest-driven ingestion (--service / service_filter) -----------

_MANIFEST = [
    ServiceSpec(name="orders", repo="org/orders-service"),
    ServiceSpec(name="shipping", repo="org/shipping-service"),
]


def _patch_run_dependencies(monkeypatch):
    calls = {
        "upsert_service_node": [],
        "update_service_summary": [],
    }

    monkeypatch.setattr(ingest, "load_services", lambda: list(_MANIFEST))

    async def fake_embed(text):
        return [0.1, 0.2]

    _patch_embed(monkeypatch, fake_embed)

    monkeypatch.setattr(
        ingest.graph_writer,
        "upsert_service_node",
        lambda name, repo: calls["upsert_service_node"].append((name, repo)),
    )
    monkeypatch.setattr(ingest.graph_writer, "create_vector_index", lambda: None)
    monkeypatch.setattr(
        ingest.graph_writer,
        "update_service_summary",
        lambda name, summary, embedding: calls["update_service_summary"].append((name, summary, embedding)),
    )

    async def fake_sync_service(spec, github_token, known_service_names):
        return None  # default: nothing changed / incremental, no fresh RepoGraph

    monkeypatch.setattr(ingest, "sync_service", fake_sync_service)
    monkeypatch.setattr(ingest.code_graph_writer, "create_code_indexes", lambda: None)
    monkeypatch.setattr(ingest.code_graph_writer, "create_chunk_vector_index", lambda: None)
    monkeypatch.setattr(ingest.code_graph_writer, "load_current_symbols", lambda name: ([], []))

    async def fake_generate_summaries(digests):
        return {}

    monkeypatch.setattr(ingest, "generate_summaries", fake_generate_summaries)

    return calls


@pytest.mark.asyncio
async def test_run_with_service_filter_only_writes_matching_service(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    await ingest.run(service_filter={"orders"})

    assert calls["upsert_service_node"] == [("orders", "org/orders-service")]


@pytest.mark.asyncio
async def test_run_with_service_filter_is_case_insensitive(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    await ingest.run(service_filter={"Orders"})

    assert calls["upsert_service_node"] == [("orders", "org/orders-service")]


@pytest.mark.asyncio
async def test_run_without_service_filter_writes_every_manifest_service(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    await ingest.run()

    assert set(calls["upsert_service_node"]) == {
        ("orders", "org/orders-service"),
        ("shipping", "org/shipping-service"),
    }


@pytest.mark.asyncio
async def test_run_with_no_matching_services_writes_nothing(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    await ingest.run(service_filter={"nonexistent"})

    assert calls["upsert_service_node"] == []


@pytest.mark.asyncio
async def test_run_scans_against_full_manifest_even_when_scoped(monkeypatch):
    """A scoped run for just "orders" must still know "shipping" exists as a
    valid dependency target — sync_service's known-service set comes from
    the FULL manifest, not the (possibly narrower) set of services this run
    is actually syncing."""
    _patch_run_dependencies(monkeypatch)
    seen_known_names = {}

    async def fake_sync_service(spec, github_token, known_service_names):
        seen_known_names[spec.name] = known_service_names
        return None

    monkeypatch.setattr(ingest, "sync_service", fake_sync_service)

    await ingest.run(service_filter={"orders"})

    assert seen_known_names["orders"] == {"orders", "shipping"}


@pytest.mark.asyncio
async def test_run_with_explicit_specs_never_reads_the_manifest(monkeypatch):
    """The published agent's own path: `specs` fully replaces
    services.yaml — load_services() must never even be called."""
    calls = _patch_run_dependencies(monkeypatch)

    def fail_load_services():
        raise AssertionError("must not read services.yaml when specs is given explicitly")

    monkeypatch.setattr(ingest, "load_services", fail_load_services)

    from src.ingestion.manifest import ServiceSpec

    specs = [ServiceSpec(name="payment", repo="org/payment-service", branch=None)]
    await ingest.run(specs=specs)

    assert calls["upsert_service_node"] == [("payment", "org/payment-service")]


@pytest.mark.asyncio
async def test_run_with_explicit_specs_and_service_filter_still_scopes(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    def fail_load_services():
        raise AssertionError("unused")

    monkeypatch.setattr(ingest, "load_services", fail_load_services)

    from src.ingestion.manifest import ServiceSpec

    specs = [
        ServiceSpec(name="orders", repo="org/orders-service", branch=None),
        ServiceSpec(name="payment", repo="org/payment-service", branch=None),
    ]
    await ingest.run(specs=specs, service_filter={"payment"})

    assert calls["upsert_service_node"] == [("payment", "org/payment-service")]


# --- code-derived summary generation (batched LLM call) ---------------------


def _repo_graph_with_one_class() -> RepoGraph:
    cls = ParsedSymbol(
        kind="class",
        name="OrderService",
        qualified_name="app.py::OrderService",
        file_path="app.py",
        line_start=1,
        line_end=10,
        signature="class OrderService:",
        docstring=None,
        language="python",
        source_snippet="class OrderService: ...",
    )
    return RepoGraph(classes=[cls], functions=[])


@pytest.mark.asyncio
async def test_run_writes_code_derived_summary_from_fresh_repo_graph(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    async def fake_sync_service(spec, github_token, known_service_names):
        return _repo_graph_with_one_class() if spec.name == "orders" else None

    monkeypatch.setattr(ingest, "sync_service", fake_sync_service)

    async def fake_generate_summaries(digests):
        assert {d["service"] for d in digests} == {"orders"}
        return {"orders": "Handles order placement."}

    monkeypatch.setattr(ingest, "generate_summaries", fake_generate_summaries)

    await ingest.run()

    assert calls["update_service_summary"] == [("orders", "Handles order placement.", [0.1, 0.2])]


@pytest.mark.asyncio
async def test_run_builds_digest_from_neo4j_when_no_fresh_repo_graph(monkeypatch):
    """An incremental sync (or an already-up-to-date service) never returns
    a fresh RepoGraph — the digest must come from Neo4j's current state
    instead, never a repo clone."""
    calls = _patch_run_dependencies(monkeypatch)

    async def fake_sync_service(spec, github_token, known_service_names):
        return None

    monkeypatch.setattr(ingest, "sync_service", fake_sync_service)

    def fake_load_current_symbols(name):
        if name == "orders":
            return (
                [{"qualified_name": "app.py::OrderService", "name": "OrderService", "file_path": "app.py", "language": "python", "docstring": None}],
                [],
            )
        return [], []

    monkeypatch.setattr(ingest.code_graph_writer, "load_current_symbols", fake_load_current_symbols)

    async def fake_generate_summaries(digests):
        assert {d["service"] for d in digests} == {"orders"}
        return {"orders": "Handles order placement."}

    monkeypatch.setattr(ingest, "generate_summaries", fake_generate_summaries)

    await ingest.run()

    assert calls["update_service_summary"] == [("orders", "Handles order placement.", [0.1, 0.2])]


@pytest.mark.asyncio
async def test_run_skips_summarizer_entirely_when_no_service_has_any_symbols(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    async def fake_generate_summaries(digests):
        raise AssertionError("generate_summaries must not be called when there are no symbols anywhere")

    monkeypatch.setattr(ingest, "generate_summaries", fake_generate_summaries)

    await ingest.run()

    assert calls["update_service_summary"] == []


@pytest.mark.asyncio
async def test_run_writes_no_summary_when_summarizer_call_fails(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch)

    async def fake_sync_service(spec, github_token, known_service_names):
        return _repo_graph_with_one_class() if spec.name == "orders" else None

    monkeypatch.setattr(ingest, "sync_service", fake_sync_service)

    async def failing_generate_summaries(digests):
        raise ingest.SummarizerError("provider unavailable")

    monkeypatch.setattr(ingest, "generate_summaries", failing_generate_summaries)

    await ingest.run()  # must not raise

    assert calls["update_service_summary"] == []


@pytest.mark.asyncio
async def test_sync_service_local_mode_full_ingests_changed_folder_and_stores_fingerprint(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path))

    def fail_github(*a, **k):
        raise AssertionError("must not call GitHub in local mode")

    monkeypatch.setattr(github_api, "get_remote_head_sha", fail_github)
    # even a service already "up to date" on GitHub gets fully re-ingested
    monkeypatch.setattr(
        ingest.graph_writer,
        "load_service_state",
        lambda name: {"commit_sha": "abc", "local_fingerprint": "old-fingerprint"},
    )

    graph = RepoGraph(classes=[], functions=[])

    async def fake_full(spec, token, names):
        return graph, "local"

    commits = []
    monkeypatch.setattr(ingest, "_ingest_full", fake_full)
    monkeypatch.setattr(
        ingest.graph_writer,
        "set_service_commit",
        lambda name, sha, local_fingerprint=None: commits.append((name, sha, local_fingerprint)),
    )

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    result = await ingest.sync_service(spec, None, {"orders"})

    assert result is graph
    expected = ingest.local_folder_fingerprint(tmp_path, {"orders"})
    assert commits == [("orders", None, expected)]


@pytest.mark.asyncio
async def test_sync_service_local_mode_skips_unchanged_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path))
    (tmp_path / "app.py").write_text("def f(): pass")
    fingerprint = ingest.local_folder_fingerprint(tmp_path, {"orders"})
    monkeypatch.setattr(
        ingest.graph_writer,
        "load_service_state",
        lambda name: {"commit_sha": None, "local_fingerprint": fingerprint},
    )

    async def fail_full(*a):
        raise AssertionError("an unchanged folder must not be re-ingested")

    monkeypatch.setattr(ingest, "_ingest_full", fail_full)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    assert await ingest.sync_service(spec, None, {"orders"}) is None


@pytest.mark.asyncio
async def test_sync_service_local_mode_bad_path_skips(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path / "missing"))

    async def fail_full(*a):
        raise AssertionError("must not ingest with a bad local path")

    monkeypatch.setattr(ingest, "_ingest_full", fail_full)

    spec = ServiceSpec(name="orders", repo="org/orders-service")
    assert await ingest.sync_service(spec, None, {"orders"}) is None
