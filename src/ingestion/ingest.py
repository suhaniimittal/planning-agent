"""Orchestrator: fetch each service's latest doc from S3, build its node,
resolve cross-service mentions into edges, write everything to Neo4j.

No hardcoded service list — the set of services to ingest is discovered by
listing whatever documentation.md files actually exist in the bucket, so
adding a new service just means RepoDocAI uploading its doc; nothing here
needs editing.

Run directly: python -m src.ingestion.ingest
(bucket is resolved from TENANT_ID / STORAGE_AGENTS_BUCKET, see s3_fetcher.docs_bucket)
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from . import code_graph_writer, graph_writer
from .code_parser import CodeChunk, ParsedSymbol, RepoGraph, parse_repo
from .embedder import embed
from .extractors import (
    build_entity_index,
    build_fallback_summary,
    extract_config_uri_targets,
    extract_key_symbols,
    extract_narrative_list_targets,
    extract_narrative_targets,
    extract_overview_table,
    extract_project_overview,
    extract_repo_slug,
    extract_service_name,
    resolve_service_name,
)
from .repo_fetcher import RepoFetchError, cloned_repo
from .s3_fetcher import docs_bucket, fetch_doc_from_s3, list_latest_docs

# How many repos to clone+parse concurrently — serially cloning 40-50 repos
# would otherwise dominate ingestion runtime; unbounded concurrency risks
# GitHub rate limits and disk/CPU contention, so a small fixed pool instead.
_CODE_GRAPH_CONCURRENCY = 4

# How many chunk-embedding calls to have in flight at once, per service — a
# single oversized function can produce several chunks and a repo can have
# thousands of functions, so this needs its own (higher) cap than the
# per-repo one above, separate from GitHub's rate limits entirely (this is
# the embedding API, not GitHub).
_EMBEDDING_CONCURRENCY = 10


async def build_service_record(md_text: str) -> dict:
    """Step 1-4: turn raw markdown into everything the node needs, plus the raw
    mention lists edges get resolved from later."""
    name = extract_service_name(md_text)
    summary = extract_project_overview(md_text)
    summary_is_fallback = False
    if not summary and name:
        summary = build_fallback_summary(md_text, name)
        summary_is_fallback = summary is not None
    table = extract_overview_table(md_text)
    symbols = extract_key_symbols(md_text)

    return {
        "name": name,
        "summary": summary,
        "summary_is_fallback": summary_is_fallback,
        "embedding": (await embed(summary)) if summary else [],
        "full_doc": md_text,
        "domain": table.get("Domain"),
        "entry_points": table.get("Entry points"),
        "external_systems": table.get("External systems"),
        "key_symbols": symbols,
        # not written to the node — used only for edge resolution below
        "_raw_mentions": (
            extract_config_uri_targets(symbols)
            + extract_narrative_targets(md_text)
            + extract_narrative_list_targets(md_text)
        ),
    }


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
    service_name: str, functions: list[ParsedSymbol]
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Embed every chunk across every function, bounded concurrency.

    Returns (chunk_rows, (function_qname, chunk_qname) edge pairs). A single
    chunk's embedding failing (rate limit, transient API error) is logged
    and that one chunk is skipped — never lets one bad call drop the whole
    service's chunks, same "one failure never aborts the run" principle
    used everywhere else in this pipeline.
    """
    semaphore = asyncio.Semaphore(_EMBEDDING_CONCURRENCY)
    chunk_rows: list[dict] = []
    edge_pairs: list[tuple[str, str]] = []

    async def _embed_one(fn: ParsedSymbol, chunk: CodeChunk) -> None:
        async with semaphore:
            try:
                vector = await embed(chunk.text)
            except Exception as e:  # noqa: BLE001 - one bad embed call must never abort the run
                print(
                    f"    [CHUNK-EMBED SKIP] {service_name}: "
                    f"{fn.qualified_name}::chunk{chunk.index}: {e}"
                )
                return
            row = _chunk_row(fn, chunk, vector)
            chunk_rows.append(row)
            edge_pairs.append((fn.qualified_name, row["qualified_name"]))

    await asyncio.gather(*(_embed_one(fn, chunk) for fn in functions for chunk in fn.chunks))
    return chunk_rows, edge_pairs


def _clone_and_parse(slug: str, service_name: str, github_token: str | None) -> RepoGraph:
    """The blocking half (subprocess clone + CPU-bound tree-sitter walk) —
    run via asyncio.to_thread so it doesn't block the event loop while other
    services' embedding calls are in flight."""
    with cloned_repo(slug, github_token) as repo_root:
        return parse_repo(repo_root, service_name)


async def build_code_graph_for_service(service_name: str, md_text: str, github_token: str | None) -> None:
    """Best-effort: clone the service's repo, parse it, embed each
    function's chunks, and write Class/Function/Chunk nodes nested under
    its already-written Service node.

    Never raises — any failure here (missing slug, clone failure, parse
    failure, zero symbols) is logged and the service simply keeps relying on
    its doc-derived key_symbols, per the ingestion architecture's fallback
    design. This function must never be allowed to abort the whole run.
    """
    slug = extract_repo_slug(md_text)
    if not slug:
        print(f"  [CODE-GRAPH SKIP] {service_name}: no repo slug in doc")
        return

    try:
        graph: RepoGraph = await asyncio.to_thread(_clone_and_parse, slug, service_name, github_token)
    except RepoFetchError as e:
        print(f"  [CODE-GRAPH SKIP] {service_name}: {e}")
        return
    except Exception as e:  # noqa: BLE001 - a parse bug must never abort ingestion
        print(f"  [CODE-GRAPH SKIP] {service_name}: unexpected error: {e}")
        return

    if not graph.classes and not graph.functions:
        print(
            f"  [CODE-GRAPH SKIP] {service_name}: 0 classes/functions parsed "
            f"({graph.files_parsed} files parsed, {graph.files_failed} failed) "
            "— key_symbols fallback stands"
        )
        return

    class_rows = [_class_row(c) for c in graph.classes]
    function_rows = [_function_row(f) for f in graph.functions]

    total_chunks = sum(len(f.chunks) for f in graph.functions)
    print(f"  [CODE-GRAPH] {service_name}: embedding {total_chunks} chunk(s)...")
    chunk_rows, chunk_edge_pairs = await _embed_chunks(service_name, graph.functions)

    all_qualified_names = (
        [c.qualified_name for c in graph.classes]
        + [f.qualified_name for f in graph.functions]
        + [row["qualified_name"] for row in chunk_rows]
    )

    code_graph_writer.prune_stale_code_nodes(service_name, all_qualified_names)
    code_graph_writer.upsert_classes(service_name, class_rows)
    code_graph_writer.upsert_functions(service_name, function_rows)
    code_graph_writer.upsert_chunks(service_name, chunk_rows)

    class_qnames = [c.qualified_name for c in graph.classes]
    top_level_fn_qnames = [f.qualified_name for f in graph.functions if not f.is_method]
    method_pairs = [
        (f.parent_class, f.qualified_name) for f in graph.functions if f.is_method and f.parent_class
    ]

    code_graph_writer.create_contains_edges(service_name, class_qnames, "Class")
    code_graph_writer.create_contains_edges(service_name, top_level_fn_qnames, "Function")
    code_graph_writer.create_declares_edges(service_name, method_pairs)
    code_graph_writer.create_has_chunk_edges(service_name, chunk_edge_pairs)

    print(
        f"  [CODE-GRAPH] {service_name}: {len(graph.classes)} classes, "
        f"{len(graph.functions)} functions, {len(chunk_rows)}/{total_chunks} chunks embedded "
        f"({graph.files_parsed} files parsed, {graph.files_skipped_oversized} oversized-skipped, "
        f"{graph.files_failed} failed)"
    )


async def run(service_filter: set[str] | None = None) -> None:
    """`service_filter`, if given, is a set of repo slugs (case-insensitive,
    matching the S3 filename's `<repo-slug>_<date>_documentation.md` part) —
    only those docs are fetched and (re)written this run. Every other
    service already in the graph is left completely untouched: not
    reprocessed, not re-embedded, not deleted. Mention resolution and the
    entity index still see the full set of already-ingested services (loaded
    from Neo4j itself, via graph_writer.load_existing_service_records), so a
    scoped run doesn't regress CALLS-edge resolution for the services it
    isn't touching."""
    bucket = docs_bucket()
    latest_docs = list_latest_docs(bucket)  # {repo_slug: s3_key}, discovered, not hardcoded

    if service_filter is not None:
        wanted = {s.lower() for s in service_filter}
        latest_docs = {slug: key for slug, key in latest_docs.items() if slug.lower() in wanted}
        print(f"=== Scoped run: {len(latest_docs)} doc(s) match {sorted(service_filter)} ===")

    print(f"=== Found {len(latest_docs)} service doc(s) in s3://{bucket} ===")

    records = []
    print("=== Fetching + building nodes ===")
    for repo_slug, key in latest_docs.items():
        md_text = fetch_doc_from_s3(bucket, key)
        record = await build_service_record(md_text)
        if not record["name"]:
            print(f"  [SKIP] {repo_slug}: could not extract a service name from {key}")
            continue
        if not record["summary"]:
            print(f"  [SKIP] {record['name']}: no Project Overview AND no Technology "
                  f"Stack table — nothing usable to embed or search on")
            continue
        tag = " (fallback: tech-stack based)" if record["summary_is_fallback"] else ""
        print(f"  {record['name']}{tag}: summary={record['summary'][:60]!r}...")
        records.append(record)

    if not records:
        print("\nNo ingestible services found (all skipped) — nothing to write.")
        return

    # In a scoped run, the services NOT in `records` this time still need to
    # be valid CALLS-edge targets and still need to contribute their owned
    # entities — otherwise a scoped run would regress resolution for every
    # already-ingested service it isn't touching. Loading them from Neo4j
    # itself (rather than re-fetching their docs from S3) keeps a scoped run
    # cheap: no extra S3 reads, no extra embedding calls, no rewritten nodes.
    existing_records = graph_writer.load_existing_service_records() if service_filter is not None else []

    known_services = {r["name"] for r in records} | {r["name"] for r in existing_records}

    entity_index = build_entity_index(records + existing_records)
    print(f"\n=== Entity index: {len(entity_index)} owned entities resolved ===")
    for entity, owner in sorted(entity_index.items()):
        print(f"  {entity} -> {owner}")

    print("\n=== Writing nodes ===")
    for record in records:
        node_fields = {k: v for k, v in record.items() if not k.startswith("_")}
        graph_writer.upsert_service_node(node_fields)
        print(f"  wrote node: {record['name']}")

    print("\n=== Creating vector index ===")
    graph_writer.create_vector_index(dimensions=len(records[0]["embedding"]))

    print("\n=== Building code graphs (clone + tree-sitter parse + chunk embedding, best-effort) ===")
    code_graph_writer.create_code_indexes()
    code_graph_writer.create_chunk_vector_index(dimensions=len(records[0]["embedding"]))
    github_token = os.environ.get("GITHUB_TOKEN")
    semaphore = asyncio.Semaphore(_CODE_GRAPH_CONCURRENCY)

    async def _one(record: dict) -> None:
        async with semaphore:
            try:
                await build_code_graph_for_service(record["name"], record["full_doc"], github_token)
            except Exception as e:  # noqa: BLE001 - one bad repo must never abort the run
                print(f"  [CODE-GRAPH SKIP] {record['name']}: unexpected error: {e}")

    await asyncio.gather(*(_one(record) for record in records))

    print("\n=== Resolving + writing edges ===")
    for record in records:
        source = record["name"]
        unresolved = []
        resolved_targets = set()
        for mention in record["_raw_mentions"]:
            target = resolve_service_name(mention, known_services, entity_index)
            if target and target != source:
                resolved_targets.add(target)
            elif not target:
                unresolved.append(mention)
        for target in resolved_targets:
            graph_writer.create_edge(source, target)
            print(f"  {source} -[:CALLS]-> {target}")
        if unresolved:
            print(f"  [UNRESOLVED] {source}: {unresolved}")

    print("\n=== Leaf services (no outgoing CALLS) ===")
    print(graph_writer.leaf_services_report())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest service docs (and best-effort code) from S3 into Neo4j."
    )
    parser.add_argument(
        "--service",
        action="append",
        dest="services",
        metavar="REPO_SLUG",
        help=(
            "Only ingest this repo's doc (matches the S3 filename's repo-slug part, "
            "case-insensitive). Repeatable. Every other already-ingested service is "
            "left untouched — not reprocessed, not deleted. Omit to ingest everything "
            "in the bucket, the default."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(service_filter=set(args.services) if args.services else None))
