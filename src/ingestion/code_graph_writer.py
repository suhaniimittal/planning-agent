"""Neo4j writes for the per-repo code graph: File/Class/Function nodes
nested under their Service via HAS_FILE/DEFINES/DECLARES.

Kept separate from graph_writer.py (Service nodes) — one-module-per-
responsibility split. All writes here are batched via UNWIND: a repo parsed
with "parse everything" can easily produce hundreds to thousands of symbols,
so one round trip per node is not an option.

`File` is the unit a future git-diff-based incremental update keys off (one
node per source file, keyed by (service, file_path)) — a file rename/
deletion becomes a File-node-scoped operation instead of a whole-service
reconciliation.
"""

from __future__ import annotations

from .graph_writer import get_driver

# Rows per write transaction. One UNWIND over a whole large repo (tens of
# thousands of rows — and each Chunk row carries a 1536-number embedding)
# is one huge transaction that the worker and Neo4j must both hold in memory
# at once; slices keep each write small. Chunks get a smaller slice for that
# reason.
_WRITE_BATCH = 2000
_CHUNK_WRITE_BATCH = 500


def _run_batched(
    query: str, key: str, items: list, batch_size: int | None = None, **params
) -> None:
    """Runs `query` once per slice of `items`, passed as parameter `key`."""
    if not items:
        return
    batch_size = batch_size or _WRITE_BATCH
    with get_driver().session() as session:
        for start in range(0, len(items), batch_size):
            session.run(query, **params, **{key: items[start : start + batch_size]})


def create_code_indexes() -> None:
    with get_driver().session() as session:
        session.run(
            "CREATE INDEX file_service_idx IF NOT EXISTS FOR (f:File) ON (f.service)"
        )
        session.run(
            "CREATE INDEX class_service_idx IF NOT EXISTS FOR (c:Class) ON (c.service)"
        )
        session.run(
            "CREATE INDEX function_service_idx IF NOT EXISTS FOR (f:Function) ON (f.service)"
        )


def create_chunk_vector_index(dimensions: int = 1536) -> None:
    with get_driver().session() as session:
        session.run(
            """
            CREATE VECTOR INDEX chunk_embedding_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: $dimensions,
              `vector.similarity_function`: 'cosine'
            }}
            """,
            dimensions=dimensions,
        )


def upsert_files(service: str, records: list[dict]) -> None:
    """Each record: {"file_path": str, "language": str}."""
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (f:File {service: $service, file_path: row.file_path})
        SET f += row
        """,
        "rows",
        records,
        service=service,
    )


def upsert_classes(service: str, records: list[dict]) -> None:
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (c:Class {service: $service, qualified_name: row.qualified_name})
        SET c += row
        """,
        "rows",
        records,
        service=service,
    )


def upsert_functions(service: str, records: list[dict]) -> None:
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (f:Function {service: $service, qualified_name: row.qualified_name})
        SET f += row
        """,
        "rows",
        records,
        service=service,
    )


def upsert_chunks(service: str, records: list[dict]) -> None:
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (c:Chunk {service: $service, qualified_name: row.qualified_name})
        SET c += row
        """,
        "rows",
        records,
        batch_size=_CHUNK_WRITE_BATCH,
        service=service,
    )


def create_has_chunk_edges(service: str, function_to_chunk_qnames: list[tuple[str, str]]) -> None:
    """(Function)-[:HAS_CHUNK]->(Chunk) for every (function_qname, chunk_qname) pair."""
    rows = [{"function_qname": f, "chunk_qname": c} for f, c in function_to_chunk_qnames]
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:Function {service: $service, qualified_name: row.function_qname})
        MATCH (c:Chunk {service: $service, qualified_name: row.chunk_qname})
        MERGE (f)-[:HAS_CHUNK]->(c)
        """,
        "rows",
        rows,
        service=service,
    )


def create_has_file_edges(service: str, file_paths: list[str]) -> None:
    """(Service)-[:HAS_FILE]->(File) for every given file path."""
    _run_batched(
        """
        UNWIND $file_paths AS file_path
        MATCH (s:Service {name: $service})
        MATCH (f:File {service: $service, file_path: file_path})
        MERGE (s)-[:HAS_FILE]->(f)
        """,
        "file_paths",
        file_paths,
        service=service,
    )


def create_defines_edges(service: str, file_path_to_qname: list[tuple[str, str]], label: str) -> None:
    """(File)-[:DEFINES]->(label) for every (file_path, qualified_name) pair.

    `label` is "Class" for all classes, or "Function" for TOP-LEVEL
    functions only — methods are reached via their class's DECLARES edge,
    not directly DEFINES-ed by their file.
    """
    rows = [{"file_path": fp, "qname": q} for fp, q in file_path_to_qname]
    _run_batched(
        f"""
        UNWIND $rows AS row
        MATCH (file:File {{service: $service, file_path: row.file_path}})
        MATCH (n:{label} {{service: $service, qualified_name: row.qname}})
        MERGE (file)-[:DEFINES]->(n)
        """,
        "rows",
        rows,
        service=service,
    )


def create_declares_edges(service: str, class_to_method_qnames: list[tuple[str, str]]) -> None:
    """(Class)-[:DECLARES]->(Function) for every (class_qname, method_qname) pair."""
    rows = [{"class_qname": c, "method_qname": m} for c, m in class_to_method_qnames]
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (c:Class {service: $service, qualified_name: row.class_qname})
        MATCH (f:Function {service: $service, qualified_name: row.method_qname})
        MERGE (c)-[:DECLARES]->(f)
        """,
        "rows",
        rows,
        service=service,
    )


def prune_stale_files(service: str, current_file_paths: list[str]) -> None:
    """Delete File nodes for `service` no longer present in the latest
    parse (removed/renamed away) — their Class/Function/Chunk descendants
    are already handled by prune_stale_code_nodes, so this only needs
    DETACH DELETE on the File node itself, not a cascade."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (f:File)
            WHERE f.service = $service AND NOT f.file_path IN $current_file_paths
            DETACH DELETE f
            """,
            service=service,
            current_file_paths=current_file_paths,
        )


def prune_stale_code_nodes(service: str, current_qualified_names: list[str]) -> None:
    """Delete Class/Function/Chunk nodes for `service` no longer present in
    the latest parse — required here (unlike the Service/CALLS writer),
    since functions/classes/chunks get renamed/deleted/re-chunked across
    re-ingestion runs much more often than whole services do. `current_
    qualified_names` must include every class, function, AND chunk
    qualified_name from the latest parse — omitting chunks here would let
    stale chunks (from a function whose chunking changed) survive forever.
    Runs before each re-upsert.
    """
    with get_driver().session() as session:
        session.run(
            """
            MATCH (n)
            WHERE (n:Class OR n:Function OR n:Chunk) AND n.service = $service
              AND NOT n.qualified_name IN $current_qualified_names
            DETACH DELETE n
            """,
            service=service,
            current_qualified_names=current_qualified_names,
        )


def prune_stale_code_nodes_for_file(service: str, file_path: str, current_qualified_names: list[str]) -> None:
    """File-scoped version of prune_stale_code_nodes — used by the
    incremental sync, which only ever reparses ONE changed file at a time
    and must not touch any other file's Class/Function/Chunk nodes."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (n)
            WHERE (n:Class OR n:Function OR n:Chunk) AND n.service = $service
              AND n.file_path = $file_path
              AND NOT n.qualified_name IN $current_qualified_names
            DETACH DELETE n
            """,
            service=service,
            file_path=file_path,
            current_qualified_names=current_qualified_names,
        )


def load_file_blob_shas(service: str) -> dict[str, str]:
    """{file_path: blob_sha} for every File node already recorded for
    `service` — the incremental sync's baseline to confirm a file GitHub
    reports as "changed" (or "renamed") actually has different content,
    never trusting the status label alone."""
    with get_driver().session() as session:
        result = session.run(
            "MATCH (f:File {service: $service}) RETURN f.file_path AS file_path, f.blob_sha AS blob_sha",
            service=service,
        )
        return {r["file_path"]: r["blob_sha"] for r in result if r["blob_sha"] is not None}


def delete_file_and_descendants(service: str, file_path: str) -> None:
    """Deletes the File node for `file_path` plus every Class/Function/
    Chunk node scoped to it — used when GitHub reports a file as removed.
    Class/Function/Chunk all carry `file_path` as their own property (see
    _class_row/_function_row/_chunk_row in ingest.py), so this is one
    property-scoped delete, no graph traversal needed."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (n)
            WHERE (n:File OR n:Class OR n:Function OR n:Chunk)
              AND n.service = $service AND n.file_path = $file_path
            DETACH DELETE n
            """,
            service=service,
            file_path=file_path,
        )


def rename_file(service: str, old_path: str, new_path: str) -> None:
    """Relabels a File node (and its Class/Function/Chunk descendants) that
    moved to a new path with NO content change — GitHub confirms this via
    the blob sha staying identical, so this is a pure property update, no
    reparse. `qualified_name` (which embeds the OLD path textually) is left
    untouched: it's just a stored identifier, and nothing downstream
    recomputes a symbol's file_path from it — file_path is always read as
    its own explicit property, never derived from qualified_name."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (n)
            WHERE (n:File OR n:Class OR n:Function OR n:Chunk)
              AND n.service = $service AND n.file_path = $old_path
            SET n.file_path = $new_path
            """,
            service=service,
            old_path=old_path,
            new_path=new_path,
        )


def get_aggregated_file_references(service: str) -> list[str]:
    """The union of every File's stored `references` for `service` — how
    the service's CALLS edges get (re)computed after any file-level change,
    without ever needing a fresh full-repo text scan. Correct after an
    incremental run touching only a few files, since every OTHER file's
    `references` property is untouched and still contributes here."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service {name: $service})-[:HAS_FILE]->(f:File)
            UNWIND coalesce(f.references, []) AS target
            RETURN DISTINCT target
            """,
            service=service,
        )
        return [r["target"] for r in result]


def load_current_symbols(service: str) -> tuple[list[dict], list[dict]]:
    """(classes, functions) currently in the graph for `service`, each a
    plain dict shaped for summarizer.build_digest — used to rebuild a
    service's summary digest WITHOUT a fresh RepoGraph (an incremental run
    only reparses the files that changed, so it never has the service's
    full symbol set in memory; this reads it back from Neo4j instead,
    never from a repo clone)."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Class OR n:Function) AND n.service = $service
            RETURN n.qualified_name AS qualified_name, n.name AS name,
                   n.file_path AS file_path, n.language AS language,
                   n.docstring AS docstring, n.is_method AS is_method,
                   n.parent_class AS parent_class, labels(n)[0] AS kind
            """,
            service=service,
        )
        classes: list[dict] = []
        functions: list[dict] = []
        for record in result:
            row = dict(record)
            kind = row.pop("kind")
            (classes if kind == "Class" else functions).append(row)
        return classes, functions
