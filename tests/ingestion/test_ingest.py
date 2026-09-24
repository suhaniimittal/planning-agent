import pytest

from src.ingestion import ingest
from src.ingestion.code_parser import CodeChunk, ParsedSymbol


def _fn_symbol(name="create_order", chunks=None) -> ParsedSymbol:
    return ParsedSymbol(
        kind="function",
        name=name,
        qualified_name=f"controller.py::{name}",
        file_path="controller.py",
        line_start=1,
        line_end=3,
        signature=f"def {name}():",
        docstring=None,
        language="python",
        source_snippet=f"def {name}():\n    pass",
        chunks=chunks if chunks is not None else [],
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

    monkeypatch.setattr(ingest, "embed", fake_embed)

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

    monkeypatch.setattr(ingest, "embed", flaky_embed)

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

    monkeypatch.setattr(ingest, "embed", fake_embed)

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
async def test_embed_chunks_empty_functions_returns_empty(monkeypatch):
    rows, edges = await ingest._embed_chunks("orders", [])
    assert rows == []
    assert edges == []


# --- run() scoped ingestion (--service / service_filter) -------------------

_ORDERS_MD = (
    "Generated from `org/orders`\n\n"
    "## Project Overview\n"
    "Handles order placement and checkout. It calls the payment service to authorize charges.\n\n"
    "## File Reference\n"
)


def _patch_run_dependencies(monkeypatch, *, existing_records):
    calls = {
        "fetch_doc_from_s3": [],
        "create_edge": [],
        "load_existing_called": False,
        "entity_index_input_names": None,
    }

    monkeypatch.setattr(ingest, "docs_bucket", lambda: "test-bucket")
    monkeypatch.setattr(
        ingest,
        "list_latest_docs",
        lambda bucket: {
            "orders": "RepoDocAI/documentation/orders_2026-09-24_documentation.md",
            "shipping": "RepoDocAI/documentation/shipping_2026-09-20_documentation.md",
        },
    )

    def fake_fetch_doc_from_s3(bucket, key):
        calls["fetch_doc_from_s3"].append(key)
        return _ORDERS_MD

    monkeypatch.setattr(ingest, "fetch_doc_from_s3", fake_fetch_doc_from_s3)

    async def fake_embed(text):
        return [0.1, 0.2]

    monkeypatch.setattr(ingest, "embed", fake_embed)

    def fake_load_existing_service_records():
        calls["load_existing_called"] = True
        return existing_records

    monkeypatch.setattr(ingest.graph_writer, "load_existing_service_records", fake_load_existing_service_records)
    monkeypatch.setattr(ingest.graph_writer, "upsert_service_node", lambda record: None)
    monkeypatch.setattr(ingest.graph_writer, "create_vector_index", lambda dimensions: None)
    monkeypatch.setattr(
        ingest.graph_writer,
        "create_edge",
        lambda source, target, rel_type="CALLS": calls["create_edge"].append((source, target)),
    )
    monkeypatch.setattr(ingest.graph_writer, "leaf_services_report", lambda: [])

    async def fake_build_code_graph_for_service(service_name, md_text, github_token):
        pass

    monkeypatch.setattr(ingest, "build_code_graph_for_service", fake_build_code_graph_for_service)
    monkeypatch.setattr(ingest.code_graph_writer, "create_code_indexes", lambda: None)
    monkeypatch.setattr(ingest.code_graph_writer, "create_chunk_vector_index", lambda dimensions: None)

    return calls


@pytest.mark.asyncio
async def test_run_with_service_filter_only_fetches_matching_docs(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch, existing_records=[])

    await ingest.run(service_filter={"orders"})

    assert calls["fetch_doc_from_s3"] == ["RepoDocAI/documentation/orders_2026-09-24_documentation.md"]


@pytest.mark.asyncio
async def test_run_with_service_filter_is_case_insensitive(monkeypatch):
    calls = _patch_run_dependencies(monkeypatch, existing_records=[])

    await ingest.run(service_filter={"Orders"})

    assert calls["fetch_doc_from_s3"] == ["RepoDocAI/documentation/orders_2026-09-24_documentation.md"]


@pytest.mark.asyncio
async def test_run_with_service_filter_resolves_mentions_against_existing_graph(monkeypatch):
    """A scoped run for "orders" must still resolve its "calls the payment
    service" mention against "payment" even though payment isn't part of
    this run's own records — it comes from the existing graph via
    graph_writer.load_existing_service_records, never re-fetched/re-embedded."""
    existing = [{"name": "payment", "full_doc": "Generated from `org/payment`", "key_symbols": []}]
    calls = _patch_run_dependencies(monkeypatch, existing_records=existing)

    await ingest.run(service_filter={"orders"})

    assert calls["load_existing_called"] is True
    assert calls["create_edge"] == [("orders", "payment")]


@pytest.mark.asyncio
async def test_run_without_service_filter_never_reads_existing_graph_records(monkeypatch):
    """Regression guard: an unscoped (default) run's behavior must not
    change — it never needs load_existing_service_records, since every
    service is already part of `records` this run."""
    calls = _patch_run_dependencies(monkeypatch, existing_records=[])

    await ingest.run()

    assert calls["load_existing_called"] is False
    # Unscoped: both docs in the bucket are fetched, nothing filtered out.
    assert set(calls["fetch_doc_from_s3"]) == {
        "RepoDocAI/documentation/orders_2026-09-24_documentation.md",
        "RepoDocAI/documentation/shipping_2026-09-20_documentation.md",
    }
