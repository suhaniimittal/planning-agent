"""Orchestrator: git-diff incremental sync per manifest service into Neo4j.

Rule: a repo is cloned exactly once per service — on its first ever
ingestion. Every subsequent run only compares the service's stored
commit_sha against the current remote HEAD and, for whatever files GitHub
reports as changed, fetches just those files' content via the API — an
already-ingested service is never cloned again, no matter how much of it
changed.

The set of services to ingest comes entirely from services.yaml (see
manifest.py) — no S3 bucket, no markdown doc, no RepoDocAI dependency.
Adding a new service means adding an entry there; nothing here needs
editing.

Per service, in order (see docs/INGESTION_V2_PLAN.md):

  Stage 0  every file gets a node with its full text (file_inventory.py)
  Stage 1  symbols, imports and facts are parsed (code_parser.py)
  Stage 2  chunks are embedded, reusing any embedding of identical text
  Stage 3  links inside the service are rebuilt (links.py)

and once every service is in, Stage 4 links services to each other
(service_links.py). Stage 5, the coverage report, is printed per service
and kept on its Service node.

Run directly: python -m src.ingestion.ingest
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path, PurePosixPath

from . import code_graph_writer, github_api, graph_writer, links, service_doc, service_links
from .code_parser import (
    LANGUAGE_BY_EXTENSION,
    CodeChunk,
    ParsedSymbol,
    RepoGraph,
    add_file,
    embedding_text,
    language_of,
    parse_repo,
)
from .edge_detector import find_service_references, find_service_references_in_text
from .embedder import embed, embed_many
from .file_inventory import build_record, file_row
from .manifest import ServiceSpec, load_services
from .repo_fetcher import (
    RepoFetchError,
    blob_shas_of,
    cloned_repo,
    local_folder_fingerprint,
    local_repo_root,
    service_list_fingerprint,
)
from .test_paths import is_test_path

# Bump this whenever a service's graph would come out DIFFERENT from the
# same source — a change to parsing, chunking, what a chunk is embedded as,
# or what is stored on a node. Not for a comment, a log line or a refactor
# that produces identical output.
#
# Without it, "has this service changed?" only ever asked whether the FILES
# changed. Improving how code is embedded therefore had no effect on any
# already-ingested service, silently, and the only way to pick it up was to
# know to go and clear the stored fingerprints by hand.
#
# 1 — the first version to record this at all. Its predecessors embedded a
#     chunk as its code alone and stored no imports.
# 2 — every test file now stores its own text on its File node, so the
#     planner can be shown a real test instead of only its name.
# 3 — EVERY file gets a node, whatever its extension, and a file that holds
#     no class and no function has its text chunked and embedded. Before
#     this, a File node existed only for a path some symbol reported, so a
#     third of ui — 3,369 of 10,285 parseable files, every messages*.ts
#     among them — was absent from the graph entirely.
# 4 — every file's full text is stored; type aliases, enums, records,
#     objects, constants, test blocks and useCallback handlers are symbols,
#     nested symbols are named after their parent; chunks are cut by line
#     ownership with no repeated signature; test files are no longer
#     embedded; and imports, references, tests, tables, operations, routes
#     and Kafka messages are linked. On 13 services: code lines inside a
#     symbol or chunk went from 80.4% to ~100%, and no file is missing.
PIPELINE_VERSION = 4

# How many repos to sync concurrently — serially syncing 40-50 repos would
# otherwise dominate ingestion runtime; unbounded concurrency risks GitHub
# rate limits and disk/CPU contention (for first-ingestion clones), so a
# small fixed pool instead.
_CODE_GRAPH_CONCURRENCY = 4

# How many chunk-embedding requests to have in flight at once, per service — a
# single oversized function can produce several chunks and a repo can have
# thousands of functions, so this needs its own (higher) cap than the
# per-repo one above, separate from GitHub's rate limits entirely (this is
# the embedding API, not GitHub).
_EMBEDDING_CONCURRENCY = 10

# Chunks are embedded in batches, one request per batch, instead of one
# request per chunk — a large repo (~40k chunks) otherwise spends an hour on
# round trips alone. A batch is closed at whichever limit it hits first; the
# char cap keeps a batch of unusually large chunks well under the embedding
# API's per-request token limit.
_EMBEDDING_BATCH_SIZE = 100
_EMBEDDING_BATCH_MAX_CHARS = 200_000

# Batching makes a big repo fast enough to hit the embedding API's rate limit,
# so a failed batch is retried after each of these waits (seconds) before
# falling back to one call per chunk — which would otherwise just hit the
# same limit and drop every chunk in the batch.
_EMBEDDING_RETRY_DELAYS = (5, 15, 30, 60)

# (text hashes) -> {text hash: embedding already stored for that text}
CacheLookup = Callable[[list[str]], dict[str, list[float]]]


def _is_retryable(error: Exception) -> bool:
    """A 4xx other than 429 (e.g. an input the API rejects) fails the same
    way every time, so waiting and retrying it only wastes minutes."""
    status = getattr(error, "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status != 429)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _symbol_row(symbol: ParsedSymbol) -> dict:
    return {
        "qualified_name": symbol.qualified_name,
        "name": symbol.name,
        "file_path": symbol.file_path,
        "line_start": symbol.line_start,
        "line_end": symbol.line_end,
        "signature": symbol.signature,
        "docstring": symbol.docstring,
        "source_snippet": symbol.source_snippet,
        "language": symbol.language,
        "is_test": is_test_path(symbol.file_path),
        "symbol_kind": symbol.symbol_kind,
        "parent_symbol": symbol.parent_symbol,
        "identifiers": symbol.identifiers,
        "supertypes": symbol.supertypes,
        # Neo4j properties cannot be maps; the raw facts travel as JSON and
        # what other stages query (kafka_*, avro_*) is written by them.
        "facts": json.dumps(symbol.facts, sort_keys=True) if symbol.facts else None,
    }


def _class_row(symbol: ParsedSymbol) -> dict:
    return _symbol_row(symbol)


def _function_row(symbol: ParsedSymbol) -> dict:
    return {
        **_symbol_row(symbol),
        "is_method": symbol.is_method,
        "parent_class": symbol.parent_class,
    }


def _variable_row(symbol: ParsedSymbol) -> dict:
    return _symbol_row(symbol)


def _owner_label(symbol: ParsedSymbol) -> str | None:
    """The name a chunk's embedding is anchored with besides its own: the
    method's class, or the symbol a nested function is declared in."""
    if symbol.parent_class:
        return symbol.parent_class
    if symbol.parent_symbol:
        return symbol.parent_symbol.split("::", 1)[-1]
    return None


def _chunk_embedding_text(symbol: ParsedSymbol, chunk: CodeChunk) -> str:
    return embedding_text(symbol.file_path, symbol.name, _owner_label(symbol), chunk.text)


def _chunk_row(
    function_symbol: ParsedSymbol,
    chunk: CodeChunk,
    embedding: list[float],
    text_hash: str | None = None,
) -> dict:
    return {
        "qualified_name": f"{function_symbol.qualified_name}::chunk{chunk.index}",
        # The owning symbol — a function, or since version 4 any symbol
        # that owns lines. Name kept for the readers that already use it.
        "function_qualified_name": function_symbol.qualified_name,
        "chunk_index": chunk.index,
        "file_path": function_symbol.file_path,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "text": chunk.text,
        "header": chunk.header,
        "text_hash": text_hash or _text_hash(_chunk_embedding_text(function_symbol, chunk)),
        "embedding": embedding,
        "language": function_symbol.language,
        "is_test": is_test_path(function_symbol.file_path),
    }


def _file_chunk_embedding_text(file_path: str, chunk: CodeChunk) -> str:
    return embedding_text(file_path, Path(file_path).name, None, chunk.text)


def _file_chunk_row(
    file_path: str, language: str, chunk: CodeChunk, embedding: list[float]
) -> dict:
    """A chunk of the lines in a file that no symbol owns. Shaped exactly
    like a symbol's chunk except that `function_qualified_name` is null —
    the file itself is the parent, reached by a File-[:HAS_CHUNK]->Chunk
    edge."""
    return {
        "qualified_name": f"{file_path}::filechunk{chunk.index}",
        "function_qualified_name": None,
        "chunk_index": chunk.index,
        "file_path": file_path,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "text": chunk.text,
        "header": chunk.header,
        "text_hash": _text_hash(_file_chunk_embedding_text(file_path, chunk)),
        "embedding": embedding,
        "language": language,
        "is_test": is_test_path(file_path),
    }


async def _vectors_for(
    service_name: str,
    texts: list[str],
    cache_lookup: CacheLookup | None,
    embed_batch: Callable,
) -> list[list[float] | None]:
    """One vector per text: from the cache where the same text was embedded
    before, from `embed_batch` for the rest. None where embedding failed."""
    hashes = [_text_hash(t) for t in texts]
    cached: dict[str, list[float]] = {}
    if cache_lookup is not None:
        try:
            cached = await asyncio.to_thread(cache_lookup, hashes)
        except Exception as e:  # noqa: BLE001 - a cache miss only costs an embed
            print(f"    [EMBED CACHE] {service_name}: lookup failed ({e}); embedding all")
    missing = [i for i, h in enumerate(hashes) if h not in cached]
    vectors: list[list[float] | None] = [cached.get(h) for h in hashes]
    if missing:
        fresh = await embed_batch([texts[i] for i in missing])
        for i, vector in zip(missing, fresh, strict=True):
            vectors[i] = vector
    return vectors


async def _embed_file_chunks(
    service_name: str,
    file_chunks: dict[str, list[CodeChunk]],
    language_by_file: dict[str, str],
    cache_lookup: CacheLookup | None = None,
) -> list[tuple[str, str]]:
    """Embed and write every file-owned chunk, returning the (file_path,
    chunk_qname) pairs to link.

    Reuses the function path's batching by pretending each file is a
    one-symbol unit; the embedding text is the file path followed by the
    chunk, the same shape embedding_text() gives a function's chunk, so a
    search matches on the path as well as the content.
    """
    pairs = [(fp, chunk) for fp, chunks in file_chunks.items() for chunk in chunks]
    if not pairs:
        return []
    print(f"  [FILE CHUNKS] {service_name}: {len(pairs)} chunk(s) from "
          f"{len(file_chunks)} file(s) outside any symbol")

    async def embed_batch(texts: list[str]) -> list[list[float] | None]:
        try:
            return list(await embed_many(texts))
        except Exception as e:  # noqa: BLE001 - one bad batch must not cost the service
            print(f"    [FILE CHUNKS] {service_name}: batch failed ({e}); skipped")
            return [None] * len(texts)

    edges: list[tuple[str, str]] = []
    for start in range(0, len(pairs), _EMBEDDING_BATCH_SIZE):
        batch = pairs[start : start + _EMBEDDING_BATCH_SIZE]
        texts = [_file_chunk_embedding_text(fp, chunk) for fp, chunk in batch]
        vectors = await _vectors_for(service_name, texts, cache_lookup, embed_batch)
        rows = [
            _file_chunk_row(fp, language_by_file.get(fp, "text"), chunk, vector)
            for (fp, chunk), vector in zip(batch, vectors, strict=True)
            if vector is not None
        ]
        if rows:
            await asyncio.to_thread(code_graph_writer.upsert_chunks, service_name, rows)
            edges.extend((row["file_path"], row["qualified_name"]) for row in rows)
    return edges


async def _embed_chunks(
    service_name: str,
    functions: list[ParsedSymbol],
    write_batch: Callable[[list[dict], list[tuple[str, str]]], None] | None = None,
    cache_lookup: CacheLookup | None = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Embed every chunk across every symbol that owns chunks, in batches
    (one embedding request per batch), with bounded concurrency across
    batches.

    Returns (chunk_rows, (owner_qname, chunk_qname) edge pairs). If a whole
    batch request fails (rate limit, one chunk the API rejects), that batch
    is retried one chunk at a time, so only a chunk that fails on its own is
    logged and skipped — never lets one bad call drop the whole service's
    chunks, same "one failure never aborts the run" principle used
    everywhere else in this pipeline.

    With `write_batch`, each batch's rows and edge pairs are handed to it
    (in a worker thread) as soon as they're embedded and are NOT kept: the
    returned chunk_rows is then empty, and only the small edge pairs come
    back. A large repo's embeddings would otherwise all sit in memory at
    once (~40k chunks x 1536 numbers is several GB).

    With `cache_lookup`, a chunk whose exact embedding text was embedded
    before reuses that vector instead of calling the API.
    """
    pairs = [(fn, chunk) for fn in functions for chunk in fn.chunks]
    batches: list[list[tuple[ParsedSymbol, CodeChunk]]] = []
    batch_chars = 0
    for pair in pairs:
        size = len(pair[1].text)
        if batches and (
            len(batches[-1]) < _EMBEDDING_BATCH_SIZE
            and batch_chars + size <= _EMBEDDING_BATCH_MAX_CHARS
        ):
            batches[-1].append(pair)
            batch_chars += size
        else:
            batches.append([pair])
            batch_chars = size
    if pairs:
        print(f"  [EMBED] {service_name}: {len(pairs)} chunks in {len(batches)} batches")

    semaphore = asyncio.Semaphore(_EMBEDDING_CONCURRENCY)
    write_lock = asyncio.Lock()
    chunk_rows: list[dict] = []
    edge_pairs: list[tuple[str, str]] = []
    batches_done = 0

    async def _embed_many_with_retry(texts: list[str]) -> list[list[float]] | None:
        for delay in (*_EMBEDDING_RETRY_DELAYS, None):
            try:
                vectors = await embed_many(texts)
            except Exception as e:  # noqa: BLE001 - retried, then per-chunk fallback
                if delay is None or not _is_retryable(e):
                    return None
                print(f"    [EMBED RETRY] {service_name}: batch failed ({e}); retrying in {delay}s")
                await asyncio.sleep(delay)
                continue
            return vectors if len(vectors) == len(texts) else None
        return None

    async def _embed_batch(batch: list[tuple[ParsedSymbol, CodeChunk]]) -> None:
        nonlocal batches_done
        async with semaphore:
            texts = [_chunk_embedding_text(fn, chunk) for fn, chunk in batch]

            async def embed_texts(missing: list[str]) -> list[list[float] | None]:
                vectors = await _embed_many_with_retry(missing)
                if vectors is not None:
                    return list(vectors)
                out: list[list[float] | None] = []
                for text in missing:
                    try:
                        out.append(await embed(text))
                    except Exception as e:  # noqa: BLE001 - one bad embed call must never abort the run
                        where = text.splitlines()[0][:80]
                        print(f"    [CHUNK-EMBED SKIP] {service_name}: {where}: {e}")
                        out.append(None)
                return out

            vectors = await _vectors_for(service_name, texts, cache_lookup, embed_texts)
            rows = [
                _chunk_row(fn, chunk, vector, _text_hash(text))
                for (fn, chunk), text, vector in zip(batch, texts, vectors, strict=True)
                if vector is not None
            ]
            batch_pairs = [(row["function_qualified_name"], row["qualified_name"]) for row in rows]
            if write_batch is not None:
                # One write at a time: a large function's chunks can land in
                # two batches, and two concurrent writes both MERGE-ing a
                # HAS_CHUNK edge onto that same Function node deadlock in
                # Neo4j (seen for real on a ~40k-chunk repo). Embedding stays
                # concurrent; only the (fast) writes are serialized.
                async with write_lock:
                    await asyncio.to_thread(write_batch, rows, batch_pairs)
            else:
                chunk_rows.extend(rows)
            edge_pairs.extend(batch_pairs)

            batches_done += 1
            if batches_done % 50 == 0 and batches_done < len(batches):
                print(f"  [EMBED] {service_name}: {batches_done}/{len(batches)} batches done")

    await asyncio.gather(*(_embed_batch(batch) for batch in batches))
    return chunk_rows, edge_pairs


def _language_for(path: str) -> str | None:
    return LANGUAGE_BY_EXTENSION.get(PurePosixPath(path).suffix)


def _clone_and_parse_full(
    spec: ServiceSpec, github_token: str | None, known_service_names: set[str]
) -> tuple[RepoGraph, dict[str, set[str]], str, dict[str, str]]:
    """The blocking half of a FIRST ingestion (zip download + CPU-bound
    tree-sitter walk + per-file text scan + local blob hashing) — run via
    asyncio.to_thread so it doesn't block the event loop while other
    services' embedding calls are in flight.

    Everything here runs INSIDE the `with` block, while the clone still
    exists on disk — repo_fetcher deletes it the moment this returns, and
    this service is never cloned again after this.
    """
    with cloned_repo(
        spec.repo, github_token, branch=spec.branch, service_name=spec.name
    ) as checkout:
        repo_root = checkout.root
        graph = parse_repo(repo_root, spec.name)
        references_by_file = find_service_references(repo_root, spec.name, known_service_names)
        commit_sha = checkout.head_sha
        blob_shas = blob_shas_of(repo_root)
        return graph, references_by_file, commit_sha, blob_shas


def _file_rows(
    graph: RepoGraph, references_by_file: dict[str, set[str]], blob_shas: dict[str, str]
) -> tuple[list[dict], dict[str, str]]:
    """One row per file — every file in the inventory, plus (for a graph
    built without one) every path a symbol or tracked file names.
    Returns (rows, {file_path: language})."""
    language_by_file: dict[str, str] = {p: r.language for p, r in graph.files.items()}
    for path, language in graph.tracked_files.items():
        language_by_file.setdefault(path, language)
    for symbol in graph.symbols:
        language_by_file.setdefault(symbol.file_path, symbol.language)
    rows = []
    for path, language in language_by_file.items():
        record = graph.files.get(path)
        row = file_row(record) if record is not None else {
            "file_path": path,
            "language": language,
            "source": graph.file_source.get(path, ""),
        }
        facts = dict(graph.file_facts.get(path, {}))
        constants = facts.pop("kotlin_constants", None)
        row.update({
            "blob_sha": blob_shas.get(path),
            "references": sorted(references_by_file.get(path, set())),
            # "" for a file that imports nothing, so the property always
            # exists and a reader can tell "no imports" from "never parsed".
            "imports": graph.file_imports.get(path, ""),
        })
        specs = graph.file_import_specs.get(path)
        if specs:
            row["import_specs"] = json.dumps([s.to_dict() for s in specs])
        if path in graph.file_packages:
            row["package"] = graph.file_packages[path]
        if facts:
            row["facts"] = json.dumps(facts, sort_keys=True)
            # Lists the cross-service stage queries directly.
            for key in ("graphql_operations", "graphql_exposes", "http_paths"):
                if key in facts:
                    row[key] = facts[key]
        if constants:
            row["kotlin_constants"] = json.dumps(constants, sort_keys=True)
        rows.append(row)
    return rows, language_by_file


def _write_symbols(service: str, graph: RepoGraph, file_rows: list[dict]) -> None:
    """Files, symbols and the structural edges between them."""
    file_paths = [row["file_path"] for row in file_rows]
    code_graph_writer.upsert_files(service, file_rows)
    code_graph_writer.upsert_classes(service, [_class_row(c) for c in graph.classes])
    code_graph_writer.upsert_functions(service, [_function_row(f) for f in graph.functions])
    if graph.variables:
        code_graph_writer.upsert_variables(service, [_variable_row(v) for v in graph.variables])

    class_pairs = [(c.file_path, c.qualified_name) for c in graph.classes]
    top_level_fn_pairs = [
        (f.file_path, f.qualified_name) for f in graph.functions if not f.is_method
    ]
    method_pairs = [
        (f.parent_class, f.qualified_name)
        for f in graph.functions
        if f.is_method and f.parent_class
    ]
    code_graph_writer.create_has_file_edges(service, file_paths)
    code_graph_writer.create_defines_edges(service, class_pairs, "Class")
    code_graph_writer.create_defines_edges(service, top_level_fn_pairs, "Function")
    if graph.variables:
        code_graph_writer.create_defines_edges(
            service, [(v.file_path, v.qualified_name) for v in graph.variables], "Variable"
        )
    code_graph_writer.create_declares_edges(service, method_pairs)
    contains = [(s.parent_symbol, s.qualified_name) for s in graph.symbols if s.parent_symbol]
    if contains:
        code_graph_writer.create_contains_edges(service, contains)


def _link_service(service: str, index: links.ServiceIndex) -> links.ServiceLinks | None:
    """Stage 3: rebuild every link inside `service`. Never fatal — a bug
    here costs the links, not the ingested code."""
    try:
        result = links.compute_links(index)
        code_graph_writer.write_service_links(service, result)
    except Exception as e:  # noqa: BLE001 - links are rebuilt next run
        print(f"  [LINKS SKIP] {service}: {e}")
        return None
    unresolved = sum(len(v) for v in result.unresolved_imports.values())
    print(
        f"  [LINKS] {service}: {len(result.imports)} imports ({unresolved} unresolved), "
        f"{len(result.references)} references, {len(result.extends) + len(result.implements)} "
        f"extends/implements, {len(result.tests)} tests, {len(result.tables)} tables, "
        f"{len(result.operations)} operations, {len(result.endpoints)} routes, "
        f"{len(result.kafka)} kafka classes"
    )
    return result


def _link_services(names: list[str]) -> None:
    """Stage 4: links across every service in the run. Never fatal."""
    try:
        result = service_links.link_services(names)
    except Exception as e:  # noqa: BLE001 - recomputed next run
        print(f"  [SERVICE LINKS SKIP] {e}")
        return
    vias: dict[str, int] = {}
    for entry in result.service_links.values():
        for via in entry["via"]:
            vias[via] = vias.get(via, 0) + 1
    print(
        f"  {len(result.handles)} handlers and {len(result.publishes)} publishers over "
        f"{len({r['message'] for r in result.handles + result.publishes})} messages, "
        f"{len(result.uses_operation)} operation uses, {len(result.calls_endpoint)} route calls; "
        f"service links by kind: {vias or 'none'}"
    )
    if result.unresolved_topics:
        print(f"  [UNRESOLVED TOPICS] {sorted(set(result.unresolved_topics))[:10]}")


def _report_coverage(service: str, graph: RepoGraph, linked: links.ServiceLinks | None) -> None:
    """Stage 5: how much of the service the graph holds, printed and kept."""
    code = sum(c for c, _ in graph.file_coverage.values())
    covered = sum(v for _, v in graph.file_coverage.values())
    failed = sorted(p for p, r in graph.files.items() if r.parse_status == "failed")
    coverage = {
        "files": len(graph.files),
        "code_lines": code,
        "covered_lines": covered,
        "percent": round(100 * covered / code, 1) if code else 100.0,
        "failed_files": len(failed),
        "unresolved_imports": (
            sum(len(v) for v in linked.unresolved_imports.values()) if linked else 0
        ),
    }
    print(
        f"  [COVERAGE] {service}: {coverage['percent']}% of code lines in a symbol or chunk "
        f"({covered}/{code}), {coverage['files']} files, {len(failed)} parse failures"
        + (f" e.g. {failed[:3]}" if failed else "")
    )
    try:
        graph_writer.set_service_coverage(service, coverage)
    except Exception as e:  # noqa: BLE001 - the report is informational
        print(f"  [COVERAGE] {service}: could not store ({e})")


async def _ingest_full(
    spec: ServiceSpec, github_token: str | None, known_service_names: set[str]
) -> tuple[RepoGraph | None, str | None]:
    """Full clone + parse — used the first time a service is ingested, and
    whenever the pipeline version moves.

    Writes every File/Class/Function/Variable/Chunk node, every file's
    dependency references, every link inside the service, and aggregates
    the references into the service's CALLS edges.

    Returns (RepoGraph | None, new_commit_sha | None): the RepoGraph lets
    the caller build this service's summary digest without a second clone;
    it's None on any skip/failure, in which case commit_sha is also None so
    the caller never advances past a commit that was never actually
    ingested (next run retries a full ingest, not a silent permanent skip).
    """
    try:
        graph, references_by_file, commit_sha, blob_shas = await asyncio.to_thread(
            _clone_and_parse_full, spec, github_token, known_service_names
        )
    except RepoFetchError as e:
        print(f"  [SKIP] {spec.name}: {e}")
        return None, None
    except Exception as e:  # noqa: BLE001 - a parse bug must never abort ingestion
        print(f"  [SKIP] {spec.name}: unexpected error: {e}")
        return None, None

    if not graph.files and not graph.symbols and not graph.tracked_files:
        print(
            f"  [SKIP] {spec.name}: no files found "
            f"({graph.files_parsed} files parsed, {graph.files_failed} failed)"
        )
        return None, None

    file_rows, language_by_file = _file_rows(graph, references_by_file, blob_shas)
    file_paths = [row["file_path"] for row in file_rows]

    # Files/symbols are written BEFORE chunks are embedded, so each embedded
    # batch can be written (with its HAS_CHUNK edges) the moment it's ready
    # instead of every embedding being held until the end.
    code_graph_writer.prune_stale_files(spec.name, file_paths)
    _write_symbols(spec.name, graph, file_rows)

    def write_chunk_batch(rows: list[dict], pairs: list[tuple[str, str]]) -> None:
        code_graph_writer.upsert_chunks(spec.name, rows)
        code_graph_writer.create_has_chunk_edges(spec.name, pairs)

    def cache_lookup(hashes: list[str]) -> dict[str, list[float]]:
        return code_graph_writer.load_chunk_embeddings(spec.name, hashes)

    owners = [s for s in graph.symbols if s.chunks]
    total_chunks = sum(len(s.chunks) for s in owners)
    print(f"  [FULL INGEST] {spec.name}: embedding {total_chunks} chunk(s)...")
    _, chunk_edge_pairs = await _embed_chunks(
        spec.name, owners, write_batch=write_chunk_batch, cache_lookup=cache_lookup
    )
    file_chunk_edges = await _embed_file_chunks(
        spec.name, graph.file_chunks, language_by_file, cache_lookup=cache_lookup
    )
    code_graph_writer.create_file_has_chunk_edges(spec.name, file_chunk_edges)

    # Runs last, once every current node is written: anything of this
    # service's not in the latest parse (or a chunk that failed to embed)
    # is stale and removed.
    all_qualified_names = (
        [s.qualified_name for s in graph.symbols]
        + [chunk_qname for _, chunk_qname in chunk_edge_pairs]
        + [chunk_qname for _, chunk_qname in file_chunk_edges]
    )
    code_graph_writer.prune_stale_code_nodes(spec.name, all_qualified_names)

    linked = _link_service(spec.name, links.index_from_repo_graph(spec.name, graph))

    targets = code_graph_writer.get_aggregated_file_references(spec.name)
    graph_writer.set_service_dependencies(spec.name, sorted(targets))
    if targets:
        print(f"  [DEPENDENCIES] {spec.name} -> {sorted(targets)}")

    print(
        f"  [FULL INGEST] {spec.name}: {len(graph.classes)} classes, "
        f"{len(graph.functions)} functions, {len(graph.variables)} variables, "
        f"{len(chunk_edge_pairs) + len(file_chunk_edges)}/"
        f"{total_chunks + sum(len(c) for c in graph.file_chunks.values())} chunks embedded "
        f"({graph.files_parsed} files parsed, {graph.files_skipped_oversized} oversized, "
        f"{graph.files_failed} failed)"
    )
    if graph.files:
        _report_coverage(spec.name, graph, linked)
    return graph, commit_sha


async def _ingest_incremental(
    spec: ServiceSpec,
    github_token: str | None,
    known_service_names: set[str],
    base_sha: str,
    head_sha: str,
) -> None:
    """Only ever called for a service that's already been fully ingested —
    NO clone here, ever. Fetches the changed-files list from GitHub's
    Compare API, and for each entry: deletes a removed file's nodes,
    relabels a pure rename (content unchanged, confirmed via blob sha), or
    fetches+reparses ONLY that one file's new content — every kind of file,
    exactly as a full ingest would. Every untouched file's nodes are left
    completely alone — no reparse, no rewrite. The service's links are then
    rebuilt from the graph, since one file's change can move another's.
    """
    try:
        changed = await asyncio.to_thread(
            github_api.get_changed_files, spec.repo, base_sha, head_sha, github_token
        )
    except github_api.GitHubApiError as e:
        print(f"  [INCREMENTAL SKIP] {spec.name}: {e}")
        return

    if not changed:
        print(f"  [INCREMENTAL] {spec.name}: commit advanced, no files changed here")
        return

    existing_blob_shas = code_graph_writer.load_file_blob_shas(spec.name)
    candidates = {name for name in known_service_names if name.lower() != spec.name.lower()}

    def cache_lookup(hashes: list[str]) -> dict[str, list[float]]:
        return code_graph_writer.load_chunk_embeddings(spec.name, hashes)

    any_written = False
    for entry in changed:
        path, status, sha, previous_path = (
            entry["path"],
            entry["status"],
            entry["sha"],
            entry["previous_path"],
        )

        if status == "removed":
            code_graph_writer.delete_file_and_descendants(spec.name, path)
            print(f"  [INCREMENTAL] {spec.name}: removed {path}")
            any_written = True
            continue

        if (
            status == "renamed"
            and previous_path
            and sha
            and sha == existing_blob_shas.get(previous_path)
        ):
            code_graph_writer.rename_file(spec.name, previous_path, path)
            print(
                f"  [INCREMENTAL] {spec.name}: renamed {previous_path} -> {path} (content unchanged)"
            )
            any_written = True
            continue

        if sha and sha == existing_blob_shas.get(path):
            # GitHub said "changed" (e.g. a rename with edits, or a pure
            # mode change) but the content at this path is byte-identical
            # to what's already recorded — proof, not a guess, that
            # nothing here actually needs reparsing.
            continue

        try:
            text = await asyncio.to_thread(
                github_api.fetch_file_content, spec.repo, path, head_sha, github_token
            )
        except github_api.GitHubApiError as e:
            print(f"  [INCREMENTAL SKIP] {spec.name}: {path}: {e}")
            continue
        except UnicodeDecodeError:
            # A binary file: still a node, with no text.
            text = None

        references = find_service_references_in_text(text or "", candidates)
        raw = (text or "").encode("utf-8") if text is not None else b"\x00"
        record = build_record(path, raw, language_of(path))
        mini = RepoGraph(classes=[], functions=[])
        add_file(mini, record, raw)
        rows, language_by_file = _file_rows(mini, {path: references}, {path: sha})

        _write_symbols(spec.name, mini, rows)
        chunk_rows, chunk_edge_pairs = await _embed_chunks(
            spec.name, [s for s in mini.symbols if s.chunks], cache_lookup=cache_lookup
        )
        code_graph_writer.upsert_chunks(spec.name, chunk_rows)
        code_graph_writer.create_has_chunk_edges(spec.name, chunk_edge_pairs)
        file_chunk_edges = await _embed_file_chunks(
            spec.name, mini.file_chunks, language_by_file, cache_lookup=cache_lookup
        )
        code_graph_writer.create_file_has_chunk_edges(spec.name, file_chunk_edges)

        current_qnames = (
            [s.qualified_name for s in mini.symbols]
            + [row["qualified_name"] for row in chunk_rows]
            + [qname for _, qname in file_chunk_edges]
        )
        code_graph_writer.prune_stale_code_nodes_for_file(spec.name, path, current_qnames)
        any_written = True
        print(
            f"  [INCREMENTAL] {spec.name}: reparsed {path} "
            f"({len(mini.classes)} classes, {len(mini.functions)} functions, "
            f"{len(mini.variables)} variables)"
        )

    if any_written:
        try:
            file_rows, symbol_rows = await asyncio.to_thread(
                code_graph_writer.load_service_index_rows, spec.name
            )
            index = links.index_from_rows(
                spec.name, file_rows, symbol_rows,
                lambda paths: code_graph_writer.load_sources(spec.name, paths),
            )
            await asyncio.to_thread(_link_service, spec.name, index)
        except Exception as e:  # noqa: BLE001 - links are rebuilt next run
            print(f"  [LINKS SKIP] {spec.name}: {e}")

    targets = code_graph_writer.get_aggregated_file_references(spec.name)
    graph_writer.set_service_dependencies(spec.name, sorted(targets))
    if targets:
        print(f"  [DEPENDENCIES] {spec.name} -> {sorted(targets)}")


async def _rescan_dependencies_only(
    spec: ServiceSpec, root: Path, known_service_names: set[str]
) -> None:
    """Recompute this service's CALLS edges and nothing else.

    Runs when the folder is untouched but the service LIST changed — adding
    a service means every file must be re-checked for references to the new
    name, while not one line of this service's own code has moved. The
    expensive half of ingestion (parse, chunk, embed) is therefore skipped
    entirely: this is a text scan, no model calls, no cost.

    Reuses the same per-file storage the incremental GitHub path uses — each
    File node carries its own `references`, and the service's edges are the
    union of them, recomputed from the graph rather than from a second scan.
    """
    references_by_file = await asyncio.to_thread(
        find_service_references, root, spec.name, known_service_names
    )
    existing = await asyncio.to_thread(code_graph_writer.load_file_blob_shas, spec.name)
    rows = [
        {"file_path": path, "references": sorted(references_by_file.get(path, ()))}
        for path in existing
    ]
    if rows:
        await asyncio.to_thread(code_graph_writer.upsert_files, spec.name, rows)
    targets = await asyncio.to_thread(code_graph_writer.get_aggregated_file_references, spec.name)
    await asyncio.to_thread(graph_writer.set_service_dependencies, spec.name, sorted(targets))
    if targets:
        print(f"  [DEPENDENCIES] {spec.name} -> {sorted(targets)}")


async def sync_service(
    spec: ServiceSpec, github_token: str | None, known_service_names: set[str]
) -> RepoGraph | None:
    """Per-service entry point: compares the stored commit_sha against the
    remote HEAD and does the minimum necessary work — nothing, a first full
    ingest, or a per-file incremental sync.

    Returns the freshly-parsed RepoGraph only when a FULL ingest just
    happened (used to write this service's code graph without a
    second clone) — an incremental run or an unchanged/skip never returns
    one, since no full RepoGraph exists for those; the caller builds the
    digest from Neo4j directly in that case (see run() below).

    Local mode (USE_LOCAL_REPO=true): a folder being edited has no commit to
    diff against, so it's a full ingest from LOCAL_REPO_PATH — skipped
    entirely when the folder's fingerprint matches the one stored by the
    last successful local ingest.
    """
    try:
        local_root = local_repo_root(spec.name)
    except RepoFetchError as e:
        print(f"  [SKIP] {spec.name}: {e}")
        return None
    if local_root is not None:
        # Taken BEFORE ingesting: a file edited mid-ingest leaves the stored
        # fingerprint stale, so the next run re-ingests rather than skips.
        fingerprint = await asyncio.to_thread(local_folder_fingerprint, local_root)
        services_fp = service_list_fingerprint(known_service_names)
        state = graph_writer.load_service_state(spec.name)
        if (
            state
            and state.get("local_fingerprint") == fingerprint
            and state.get("pipeline_version") == PIPELINE_VERSION
        ):
            # The code is untouched. The only thing that can still be stale
            # is this service's dependency links, and only if the set of
            # service names changed since they were computed.
            if state.get("services_fingerprint") == services_fp:
                print(f"  [UP TO DATE] {spec.name}: local folder unchanged since last ingest")
                return None
            print(
                f"  [DEPENDENCIES ONLY] {spec.name}: code unchanged, "
                "service list changed — rescanning references"
            )
            await _rescan_dependencies_only(spec, local_root, known_service_names)
            graph_writer.set_service_commit(
                spec.name,
                None,
                local_fingerprint=fingerprint,
                services_fingerprint=services_fp,
                pipeline_version=PIPELINE_VERSION,
            )
            return None
        print(f"  [LOCAL] {spec.name}: reading code from {local_root}")
        graph, commit_sha = await _ingest_full(spec, github_token, known_service_names)
        if commit_sha:
            # Clear any stored GitHub commit: the graph now holds local code,
            # so the next GitHub-mode run must re-ingest fully, not report
            # "up to date" against a commit it no longer reflects.
            graph_writer.set_service_commit(
                spec.name,
                None,
                local_fingerprint=fingerprint,
                services_fingerprint=services_fp,
                pipeline_version=PIPELINE_VERSION,
            )
        return graph

    try:
        head_sha = await asyncio.to_thread(
            github_api.get_remote_head_sha, spec.repo, spec.branch, github_token
        )
    except github_api.GitHubApiError as e:
        print(f"  [SKIP] {spec.name}: could not resolve remote HEAD: {e}")
        return None

    state = graph_writer.load_service_state(spec.name)

    if (
        state
        and state.get("commit_sha") == head_sha
        and state.get("pipeline_version") == PIPELINE_VERSION
    ):
        print(f"  [UP TO DATE] {spec.name} @ {head_sha[:8]}")
        return None

    if state and state.get("commit_sha") == head_sha:
        # Same commit, older pipeline: the files are identical but the graph
        # built from them would not be, so a per-file diff has nothing to
        # work from and this has to be a full ingest.
        print(f"  [REBUILD] {spec.name}: same commit, built by an older pipeline")
        graph, commit_sha = await _ingest_full(spec, github_token, known_service_names)
        if commit_sha:
            graph_writer.set_service_commit(
                spec.name, commit_sha, pipeline_version=PIPELINE_VERSION
            )
        return graph

    if not state or not state.get("commit_sha"):
        graph, commit_sha = await _ingest_full(spec, github_token, known_service_names)
        if commit_sha:
            graph_writer.set_service_commit(
                spec.name, commit_sha, pipeline_version=PIPELINE_VERSION
            )
        return graph

    await _ingest_incremental(
        spec, github_token, known_service_names, state["commit_sha"], head_sha
    )
    graph_writer.set_service_commit(spec.name, head_sha, pipeline_version=PIPELINE_VERSION)
    return None


async def run(
    specs: list[ServiceSpec] | None = None, service_filter: set[str] | None = None
) -> None:
    """`specs`, when given, IS this run's complete service list — the
    published agent's own path, built from the `repos` text a user typed
    into the UI form (see manifest.parse_service_specs), never from
    services.yaml. `specs=None` falls back to services.yaml via
    load_services() — local CLI/dev convenience only.

    `service_filter`, if given, is a set of manifest service names
    (case-insensitive) — only those are synced this run. Every other
    service already in the graph is left completely untouched. Only
    meaningful together with the services.yaml fallback (`specs=None`);
    the published agent's `specs` already names exactly the services to
    sync, so it has no separate filter to apply."""
    all_specs = specs if specs is not None else load_services()
    all_names = {s.name for s in all_specs}  # full manifest — see sync_service() calls below
    specs = all_specs

    if service_filter is not None:
        wanted = {s.lower() for s in service_filter}
        specs = [s for s in specs if s.name.lower() in wanted]
        print(f"=== Scoped run: {len(specs)} service(s) match {sorted(service_filter)} ===")

    print(f"=== {len(specs)} service(s) in the manifest ===")
    if not specs:
        print("\nNo services to ingest — nothing to write.")
        return

    print("\n=== Writing Service nodes (name + repo, from the manifest) ===")
    for spec in specs:
        graph_writer.upsert_service_node(spec.name, spec.repo)
        print(f"  wrote node: {spec.name} ({spec.repo})")

    print("\n=== Creating vector index ===")
    graph_writer.create_vector_index()
    code_graph_writer.create_code_indexes()
    code_graph_writer.create_chunk_vector_index()

    print("\n=== Syncing services (full ingest on first run, git-diff incremental after) ===")
    github_token = os.environ.get("GITHUB_TOKEN")
    semaphore = asyncio.Semaphore(_CODE_GRAPH_CONCURRENCY)

    async def _one(spec: ServiceSpec) -> tuple[str, RepoGraph | None]:
        async with semaphore:
            try:
                # Scanned against the FULL manifest (all_names), not just
                # this run's (possibly scoped) `specs` — a service's real
                # dependency can be on ANY other service, not only ones
                # being synced this run.
                graph = await sync_service(spec, github_token, all_names)
            except Exception as e:  # noqa: BLE001 - one bad service must never abort the run
                print(f"  [SKIP] {spec.name}: unexpected error: {e}")
                graph = None
            return spec.name, graph

    results = await asyncio.gather(*(_one(spec) for spec in specs))

    # Across the FULL manifest, not only this run's services: a link from an
    # unchanged service to one that just changed must be rebuilt too.
    print("\n=== Linking services to each other (Kafka, GraphQL, REST) ===")
    await asyncio.to_thread(_link_services, sorted(all_names))

    print("\n=== Reading each service's SERVICE.md ===")
    await _sync_service_docs(specs, github_token)


def _read_service_doc(spec: ServiceSpec, github_token: str | None) -> str | None:
    """A service's committed SERVICE.md, from wherever this run can reach it.

    Local mode reads the folder. GitHub mode asks the API for that one file
    rather than the checkout, because an incremental run never clones — it
    fetches only the files the Compare API says changed, so there is no tree
    to look in. One small request per service covers both kinds of run.
    """
    try:
        local_root = local_repo_root(spec.name)
    except RepoFetchError:
        local_root = None
    if local_root is not None:
        return service_doc.read_local_service_doc(local_root)
    try:
        text = github_api.fetch_file_content(
            spec.repo, service_doc.SERVICE_DOC_FILENAME, spec.branch or "HEAD", github_token
        )
    except github_api.GitHubApiError:
        return None
    return text if text.strip() else None


async def _sync_service_docs(specs: list[ServiceSpec], github_token: str | None) -> None:
    """Embeds each service's SERVICE.md, skipping any whose text is unchanged.

    Keyed on the text's own hash rather than on the commit: a commit that
    touches only code must not move a service's embedding, and editing the
    doc must take effect even when no code changed.
    """
    stored = graph_writer.service_doc_hashes([spec.name for spec in specs])
    unchanged: list[str] = []
    for spec in specs:
        text = await asyncio.to_thread(_read_service_doc, spec, github_token)
        if text is None:
            # Deliberately not fatal, and deliberately loud. The service is
            # still ingested and still searched; it scores 0 on the doc half
            # and has to win on its code (see graph_reader.summary_scores).
            print(
                f"  [NO DOC] {spec.name}: no {service_doc.SERVICE_DOC_FILENAME} — "
                "this service will score 0 on the description half of its ranking"
            )
            continue
        # The whole file is stored on the node and shown to the planner; a
        # trimmed extract of it is what gets embedded (see
        # service_doc.embedding_text), so the size warning is about the
        # extract — that is the text the 8000-byte cap actually truncates.
        to_embed = service_doc.embedding_text(text)
        note = service_doc.oversize_note(spec.name, to_embed)
        if note:
            print(note)
        text_hash = service_doc.doc_hash(text)
        if stored.get(spec.name) == text_hash:
            unchanged.append(spec.name)
            continue
        embedding = await embed(to_embed)
        graph_writer.update_service_doc(spec.name, text, embedding, text_hash)
        first_line = text.strip().splitlines()[0][:70]
        dropped = len(text) - len(to_embed)
        trimmed = f", {dropped} chars of exclusions not embedded" if dropped > 0 else ""
        print(f"  embedded {spec.name}: {len(text)} chars{trimmed}, starting {first_line!r}")
    if unchanged:
        print(f"  unchanged, left alone: {', '.join(sorted(unchanged))}")


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    # Loaded here, not at module import time — an application module must
    # not have an import-time side effect that pulls real secrets from a
    # local .env into any process that merely imports it (e.g. the test
    # suite), only a real entry point invoked directly should do that.
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Git-diff incremental sync of each manifest service's code graph + summary into Neo4j."
    )
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        metavar="NAME",
        help=(
            "Only sync this service (matches services.yaml's 'name' field, "
            "case-insensitive). Repeatable. Every other already-ingested service is "
            "left untouched. Omit to sync every service in the manifest, the default."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(service_filter=set(args.services) if args.services else None))
