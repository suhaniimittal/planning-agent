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

Run directly: python -m src.ingestion.ingest
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import PurePosixPath

from . import code_graph_writer, github_api, graph_writer
from .code_parser import (
    LANGUAGE_BY_EXTENSION,
    TRACKED_ONLY_EXTENSIONS,
    CodeChunk,
    ParsedSymbol,
    RepoGraph,
    parse_repo,
    parse_source,
)
from .edge_detector import find_service_references, find_service_references_in_text
from .embedder import embed, embed_many
from .manifest import ServiceSpec, load_services
from .repo_fetcher import (
    RepoFetchError,
    blob_shas_of,
    cloned_repo,
    local_folder_fingerprint,
    local_repo_root,
)
from .summarizer import SummarizerError, build_digest, generate_summaries

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


def _is_retryable(error: Exception) -> bool:
    """A 4xx other than 429 (e.g. an input the API rejects) fails the same
    way every time, so waiting and retrying it only wastes minutes."""
    status = getattr(error, "status_code", None)
    return not (isinstance(status, int) and 400 <= status < 500 and status != 429)


def _class_row(symbol: ParsedSymbol) -> dict:
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
    }


def _function_row(symbol: ParsedSymbol) -> dict:
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
        "is_method": symbol.is_method,
        "parent_class": symbol.parent_class,
    }


def _chunk_row(function_symbol: ParsedSymbol, chunk: CodeChunk, embedding: list[float]) -> dict:
    return {
        "qualified_name": f"{function_symbol.qualified_name}::chunk{chunk.index}",
        "function_qualified_name": function_symbol.qualified_name,
        "chunk_index": chunk.index,
        "file_path": function_symbol.file_path,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "text": chunk.text,
        "embedding": embedding,
        "language": function_symbol.language,
    }


async def _embed_chunks(
    service_name: str,
    functions: list[ParsedSymbol],
    write_batch: Callable[[list[dict], list[tuple[str, str]]], None] | None = None,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Embed every chunk across every function, in batches (one embedding
    request per batch), with bounded concurrency across batches.

    Returns (chunk_rows, (function_qname, chunk_qname) edge pairs). If a
    whole batch request fails (rate limit, one chunk the API rejects), that
    batch is retried one chunk at a time, so only a chunk that fails on its
    own is logged and skipped — never lets one bad call drop the whole
    service's chunks, same "one failure never aborts the run" principle
    used everywhere else in this pipeline.

    With `write_batch`, each batch's rows and edge pairs are handed to it
    (in a worker thread) as soon as they're embedded and are NOT kept: the
    returned chunk_rows is then empty, and only the small edge pairs come
    back. A large repo's embeddings would otherwise all sit in memory at
    once (~40k chunks x 1536 numbers is several GB).
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
            embedded: list[tuple[ParsedSymbol, CodeChunk, list[float]]] = []
            vectors = await _embed_many_with_retry([chunk.text for _, chunk in batch])
            if vectors is not None:
                embedded = [
                    (fn, chunk, vector) for (fn, chunk), vector in zip(batch, vectors, strict=True)
                ]
            else:
                for fn, chunk in batch:
                    try:
                        vector = await embed(chunk.text)
                    except Exception as e:  # noqa: BLE001 - one bad embed call must never abort the run
                        print(
                            f"    [CHUNK-EMBED SKIP] {service_name}: "
                            f"{fn.qualified_name}::chunk{chunk.index}: {e}"
                        )
                        continue
                    embedded.append((fn, chunk, vector))

            rows = [_chunk_row(fn, chunk, vector) for fn, chunk, vector in embedded]
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


async def _ingest_full(
    spec: ServiceSpec, github_token: str | None, known_service_names: set[str]
) -> tuple[RepoGraph | None, str | None]:
    """Full clone + parse — used ONLY the first time a service is ingested.

    Writes every File/Class/Function/Chunk node, every file's dependency
    references, and aggregates them into the service's CALLS edges.

    Returns (RepoGraph | None, new_commit_sha | None): the RepoGraph lets
    the caller build this service's summary digest without a second clone;
    it's None on any skip/failure, in which case commit_sha is also None so
    the caller never advances past a commit that was never actually
    ingested (next run retries a full ingest, not a silent permanent skip).
    """
    try:
        graph, references_by_file, commit_sha, blob_shas = (
            await asyncio.to_thread(_clone_and_parse_full, spec, github_token, known_service_names)
        )
    except RepoFetchError as e:
        print(f"  [SKIP] {spec.name}: {e}")
        return None, None
    except Exception as e:  # noqa: BLE001 - a parse bug must never abort ingestion
        print(f"  [SKIP] {spec.name}: unexpected error: {e}")
        return None, None

    if not graph.classes and not graph.functions and not graph.tracked_files:
        print(
            f"  [SKIP] {spec.name}: 0 classes/functions parsed and no trackable files "
            f"({graph.files_parsed} files parsed, {graph.files_failed} failed)"
        )
        return None, None

    # One File node per distinct file_path seen across classes+functions,
    # PLUS every tracked-only file (.html/.css/.csv — no symbols, but still
    # worth a node so a coding agent knows the file exists). Language is
    # whichever symbol at that path reports it (uniform within a file,
    # since a file is parsed with a single language's adapter).
    language_by_file: dict[str, str] = dict(graph.tracked_files)
    for symbol in (*graph.classes, *graph.functions):
        language_by_file.setdefault(symbol.file_path, symbol.language)
    file_paths = list(language_by_file.keys())
    file_rows = [
        {
            "file_path": fp,
            "language": lang,
            "blob_sha": blob_shas.get(fp),
            "references": sorted(references_by_file.get(fp, set())),
        }
        for fp, lang in language_by_file.items()
    ]

    # Files/classes/functions are written BEFORE chunks are embedded, so each
    # embedded batch can be written (with its HAS_CHUNK edges) the moment it's
    # ready instead of every embedding being held until the end.
    code_graph_writer.prune_stale_files(spec.name, file_paths)
    code_graph_writer.upsert_files(spec.name, file_rows)
    code_graph_writer.upsert_classes(spec.name, [_class_row(c) for c in graph.classes])
    code_graph_writer.upsert_functions(spec.name, [_function_row(f) for f in graph.functions])

    class_pairs = [(c.file_path, c.qualified_name) for c in graph.classes]
    top_level_fn_pairs = [(f.file_path, f.qualified_name) for f in graph.functions if not f.is_method]
    method_pairs = [
        (f.parent_class, f.qualified_name) for f in graph.functions if f.is_method and f.parent_class
    ]

    code_graph_writer.create_has_file_edges(spec.name, file_paths)
    code_graph_writer.create_defines_edges(spec.name, class_pairs, "Class")
    code_graph_writer.create_defines_edges(spec.name, top_level_fn_pairs, "Function")
    code_graph_writer.create_declares_edges(spec.name, method_pairs)

    def write_chunk_batch(rows: list[dict], pairs: list[tuple[str, str]]) -> None:
        code_graph_writer.upsert_chunks(spec.name, rows)
        code_graph_writer.create_has_chunk_edges(spec.name, pairs)

    total_chunks = sum(len(f.chunks) for f in graph.functions)
    print(f"  [FULL INGEST] {spec.name}: embedding {total_chunks} chunk(s)...")
    _, chunk_edge_pairs = await _embed_chunks(
        spec.name, graph.functions, write_batch=write_chunk_batch
    )

    # Runs last, once every current node is written: anything of this
    # service's not in the latest parse (or a chunk that failed to embed)
    # is stale and removed.
    all_qualified_names = (
        [c.qualified_name for c in graph.classes]
        + [f.qualified_name for f in graph.functions]
        + [chunk_qname for _, chunk_qname in chunk_edge_pairs]
    )
    code_graph_writer.prune_stale_code_nodes(spec.name, all_qualified_names)

    targets = code_graph_writer.get_aggregated_file_references(spec.name)
    graph_writer.set_service_dependencies(spec.name, sorted(targets))
    if targets:
        print(f"  [DEPENDENCIES] {spec.name} -> {sorted(targets)}")

    print(
        f"  [FULL INGEST] {spec.name}: {len(graph.classes)} classes, "
        f"{len(graph.functions)} functions, {len(chunk_edge_pairs)}/{total_chunks} chunks embedded "
        f"({graph.files_parsed} files parsed, {graph.files_skipped_oversized} oversized-skipped, "
        f"{graph.files_failed} failed)"
    )
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
    fetches+reparses ONLY that one file's new content. Every untouched
    file's nodes are left completely alone — no reparse, no rewrite.
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
            continue

        if status == "renamed" and previous_path and sha and sha == existing_blob_shas.get(previous_path):
            code_graph_writer.rename_file(spec.name, previous_path, path)
            print(f"  [INCREMENTAL] {spec.name}: renamed {previous_path} -> {path} (content unchanged)")
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

        references = find_service_references_in_text(text, candidates)
        language = _language_for(path)

        if language is None:
            # Not a tree-sitter-parseable file — still worth a dependency
            # rescan (e.g. a changed gateway/federation config), but there
            # are no Class/Function/Chunk nodes to reparse. TRACKED_ONLY_
            # EXTENSIONS (.html/.css/.csv) get their real language label
            # here; anything else just gets None.
            display_language = TRACKED_ONLY_EXTENSIONS.get(PurePosixPath(path).suffix.lower())
            code_graph_writer.upsert_files(
                spec.name,
                [
                    {
                        "file_path": path,
                        "language": display_language,
                        "blob_sha": sha,
                        "references": sorted(references),
                    }
                ],
            )
            code_graph_writer.create_has_file_edges(spec.name, [path])
            print(f"  [INCREMENTAL] {spec.name}: rescanned {path} (not source, no reparse)")
            continue

        try:
            symbols = parse_source(text.encode("utf-8"), path, language)
        except Exception as e:  # noqa: BLE001 - a parse bug must never abort ingestion
            print(f"  [INCREMENTAL SKIP] {spec.name}: {path}: parse failed: {e}")
            continue

        classes = [s for s in symbols if s.kind == "class"]
        functions = [s for s in symbols if s.kind == "function"]
        class_rows = [_class_row(c) for c in classes]
        function_rows = [_function_row(f) for f in functions]
        chunk_rows, chunk_edge_pairs = await _embed_chunks(spec.name, functions)

        current_qnames = (
            [c.qualified_name for c in classes]
            + [f.qualified_name for f in functions]
            + [row["qualified_name"] for row in chunk_rows]
        )
        code_graph_writer.prune_stale_code_nodes_for_file(spec.name, path, current_qnames)

        code_graph_writer.upsert_files(
            spec.name,
            [
                {
                    "file_path": path,
                    "language": language,
                    "blob_sha": sha,
                    "references": sorted(references),
                }
            ],
        )
        code_graph_writer.upsert_classes(spec.name, class_rows)
        code_graph_writer.upsert_functions(spec.name, function_rows)
        code_graph_writer.upsert_chunks(spec.name, chunk_rows)

        class_pairs = [(c.file_path, c.qualified_name) for c in classes]
        top_level_fn_pairs = [(f.file_path, f.qualified_name) for f in functions if not f.is_method]
        method_pairs = [
            (f.parent_class, f.qualified_name) for f in functions if f.is_method and f.parent_class
        ]

        code_graph_writer.create_has_file_edges(spec.name, [path])
        code_graph_writer.create_defines_edges(spec.name, class_pairs, "Class")
        code_graph_writer.create_defines_edges(spec.name, top_level_fn_pairs, "Function")
        code_graph_writer.create_declares_edges(spec.name, method_pairs)
        code_graph_writer.create_has_chunk_edges(spec.name, chunk_edge_pairs)

        print(
            f"  [INCREMENTAL] {spec.name}: reparsed {path} "
            f"({len(classes)} classes, {len(functions)} functions)"
        )

    targets = code_graph_writer.get_aggregated_file_references(spec.name)
    graph_writer.set_service_dependencies(spec.name, sorted(targets))
    if targets:
        print(f"  [DEPENDENCIES] {spec.name} -> {sorted(targets)}")


async def sync_service(
    spec: ServiceSpec, github_token: str | None, known_service_names: set[str]
) -> RepoGraph | None:
    """Per-service entry point: compares the stored commit_sha against the
    remote HEAD and does the minimum necessary work — nothing, a first full
    ingest, or a per-file incremental sync.

    Returns the freshly-parsed RepoGraph only when a FULL ingest just
    happened (used to build this service's summarizer digest without a
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
        fingerprint = await asyncio.to_thread(
            local_folder_fingerprint, local_root, known_service_names
        )
        state = graph_writer.load_service_state(spec.name)
        if state and state.get("local_fingerprint") == fingerprint:
            print(f"  [UP TO DATE] {spec.name}: local folder unchanged since last ingest")
            return None
        print(f"  [LOCAL] {spec.name}: reading code from {local_root}")
        graph, commit_sha = await _ingest_full(spec, github_token, known_service_names)
        if commit_sha:
            # Clear any stored GitHub commit: the graph now holds local code,
            # so the next GitHub-mode run must re-ingest fully, not report
            # "up to date" against a commit it no longer reflects.
            graph_writer.set_service_commit(spec.name, None, local_fingerprint=fingerprint)
        return graph

    try:
        head_sha = await asyncio.to_thread(
            github_api.get_remote_head_sha, spec.repo, spec.branch, github_token
        )
    except github_api.GitHubApiError as e:
        print(f"  [SKIP] {spec.name}: could not resolve remote HEAD: {e}")
        return None

    state = graph_writer.load_service_state(spec.name)

    if state and state.get("commit_sha") == head_sha:
        print(f"  [UP TO DATE] {spec.name} @ {head_sha[:8]}")
        return None

    if not state or not state.get("commit_sha"):
        graph, commit_sha = await _ingest_full(spec, github_token, known_service_names)
        if commit_sha:
            graph_writer.set_service_commit(spec.name, commit_sha)
        return graph

    await _ingest_incremental(spec, github_token, known_service_names, state["commit_sha"], head_sha)
    graph_writer.set_service_commit(spec.name, head_sha)
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

    print("\n=== Generating code-derived service summaries (one batched LLM call) ===")
    digests = []
    for name, graph in results:
        if graph is not None:
            # A full ingest just happened — build the digest from the
            # RepoGraph already in memory, no extra read.
            classes = [_class_row(c) for c in graph.classes]
            functions = [_function_row(f) for f in graph.functions]
            digests.append(build_digest(name, classes, functions))
        else:
            # No fresh RepoGraph this run (incremental sync, or already
            # up to date) — rebuild the digest from the service's CURRENT
            # state in Neo4j instead, never from a repo clone.
            classes, functions = code_graph_writer.load_current_symbols(name)
            if classes or functions:
                digests.append(build_digest(name, classes, functions))

    if not digests:
        print("  no services have a code graph — nothing to summarize")
        return

    try:
        summaries = await generate_summaries(digests)
    except SummarizerError as e:
        print(f"  [SUMMARIZER SKIP] batched call failed, no summaries written this run: {e}")
        return

    for name, summary_text in summaries.items():
        embedding = await embed(summary_text)
        graph_writer.update_service_summary(name, summary_text, embedding)
        print(f"  updated summary: {name}: {summary_text[:80]!r}...")


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
