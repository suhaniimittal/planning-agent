"""Neo4j writes for the per-repo code graph: File/Class/Function/Variable
nodes nested under their Service via HAS_FILE/DEFINES/DECLARES/CONTAINS,
their chunks, and every link between them.

Kept separate from graph_writer.py (Service nodes) — one-module-per-
responsibility split. All writes here are batched via UNWIND: a repo parsed
with "parse everything" can easily produce hundreds to thousands of symbols,
so one round trip per node is not an option.

`File` is the unit a git-diff-based incremental update keys off (one node
per file, keyed by (service, file_path)) — a file rename/deletion becomes a
File-node-scoped operation instead of a whole-service reconciliation.

Every Class, Function and Variable node also carries the `Symbol` label, so
a query that does not care which kind a symbol is — a chunk's owner, a
reference's target — matches one label instead of three.
"""

from __future__ import annotations

import json

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
        for name, label, properties in (
            ("file_key_idx", "File", "f.service, f.file_path"),
            ("variable_service_idx", "Variable", "f.service"),
            ("symbol_key_idx", "Symbol", "f.service, f.qualified_name"),
            ("chunk_key_idx", "Chunk", "f.service, f.qualified_name"),
            # The embedding cache looks chunks up by the hash of their text.
            ("chunk_text_hash_idx", "Chunk", "f.service, f.text_hash"),
            ("table_service_idx", "Table", "f.service, f.name"),
            ("operation_service_idx", "Operation", "f.service, f.type, f.name"),
            ("endpoint_service_idx", "Endpoint", "f.service, f.method, f.path"),
            ("message_name_idx", "Message", "f.name"),
            ("topic_name_idx", "Topic", "f.name"),
        ):
            session.run(f"CREATE INDEX {name} IF NOT EXISTS FOR (f:{label}) ON ({properties})")


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
    """Each record: {"file_path": str, "language": str}, optionally with
    "imports" — the file's import lines verbatim. Stored on the File node
    rather than as a symbol because imports belong to the file, not to any
    class or function in it, and nothing searches them: they are shown to
    the planner so that code written for this file uses module paths that
    exist."""
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
        SET c += row, c:Symbol
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
        SET f += row, f:Symbol
        """,
        "rows",
        records,
        service=service,
    )


def upsert_variables(service: str, records: list[dict]) -> None:
    """Top-level constants and properties: a styled component, a config
    object, a `static final` field."""
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (v:Variable {service: $service, qualified_name: row.qualified_name})
        SET v += row, v:Symbol
        """,
        "rows",
        records,
        service=service,
    )


def create_contains_edges(service: str, parent_to_child_qnames: list[tuple[str, str]]) -> None:
    """(Symbol)-[:CONTAINS]->(Symbol) for every nested symbol: a method in
    its class, a handler in the component that declares it."""
    rows = [{"parent": p, "child": c} for p, c in parent_to_child_qnames]
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (p:Symbol {service: $service, qualified_name: row.parent})
        MATCH (c:Symbol {service: $service, qualified_name: row.child})
        MERGE (p)-[:CONTAINS]->(c)
        """,
        "rows",
        rows,
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


def create_file_has_chunk_edges(service: str, file_to_chunk_qnames: list[tuple[str, str]]) -> None:
    """(File)-[:HAS_CHUNK]->(Chunk), for a file that holds no class and no
    function — an object literal, a barrel, a constants table. Its chunks
    have no Function to hang off, and without this edge they would be
    unreachable from anything."""
    rows = [{"file_path": f, "chunk_qname": c} for f, c in file_to_chunk_qnames]
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:File {service: $service, file_path: row.file_path})
        MATCH (c:Chunk {service: $service, qualified_name: row.chunk_qname})
        MERGE (f)-[:HAS_CHUNK]->(c)
        """,
        "rows",
        rows,
        service=service,
    )


def create_has_chunk_edges(service: str, function_to_chunk_qnames: list[tuple[str, str]]) -> None:
    """(Symbol)-[:HAS_CHUNK]->(Chunk) for every (owner_qname, chunk_qname)
    pair. The owner is whichever symbol owns the chunk's lines — a
    function, or since version 4 also a class (its fields and header) or a
    large constant."""
    rows = [{"function_qname": f, "chunk_qname": c} for f, c in function_to_chunk_qnames]
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:Symbol {service: $service, qualified_name: row.function_qname})
        MATCH (c:Chunk {service: $service, qualified_name: row.chunk_qname})
        MERGE (f)-[:HAS_CHUNK]->(c)
        """,
        "rows",
        rows,
        service=service,
    )


def load_chunk_embeddings(service: str, text_hashes: list[str]) -> dict[str, list[float]]:
    """{text_hash: embedding} for chunks of `service` already embedded from
    exactly this text — the embedding cache. A re-ingest after a pipeline
    change re-parses everything but pays only for text that changed."""
    if not text_hashes:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $hashes AS h
            MATCH (c:Chunk {service: $service, text_hash: h})
            WHERE c.embedding IS NOT NULL
            WITH h, head(collect(c.embedding)) AS embedding
            RETURN h AS text_hash, embedding
            """,
            service=service,
            hashes=sorted(set(text_hashes)),
        )
        return {r["text_hash"]: r["embedding"] for r in result}


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
            WHERE (n:Class OR n:Function OR n:Variable OR n:Chunk) AND n.service = $service
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
            WHERE (n:Class OR n:Function OR n:Variable OR n:Chunk) AND n.service = $service
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
            WHERE (n:File OR n:Class OR n:Function OR n:Variable OR n:Chunk)
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
            WHERE (n:File OR n:Class OR n:Function OR n:Variable OR n:Chunk)
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


# --- reading a service back (incremental Stage 3) ---------------------------


def load_service_index_rows(service: str) -> tuple[list[dict], list[dict]]:
    """(file rows, symbol rows) for every File and Symbol of `service` — what
    links.index_from_rows() rebuilds a ServiceIndex from after an
    incremental sync, which never holds the whole service in memory."""
    with get_driver().session() as session:
        files = [
            dict(r)
            for r in session.run(
                """
                MATCH (f:File {service: $service})
                RETURN f.file_path AS file_path, f.language AS language, f.kind AS kind,
                       f.import_specs AS import_specs, f.package AS package, f.facts AS facts
                """,
                service=service,
            )
        ]
        symbols = [
            dict(r)
            for r in session.run(
                """
                MATCH (n:Symbol {service: $service})
                RETURN n.qualified_name AS qualified_name, n.name AS name,
                       [l IN labels(n) WHERE l <> 'Symbol'][0] AS label,
                       n.symbol_kind AS symbol_kind, n.file_path AS file_path,
                       n.parent_symbol AS parent_symbol, n.identifiers AS identifiers,
                       n.supertypes AS supertypes, n.facts AS facts, n.language AS language
                """,
                service=service,
            )
        ]
    return files, symbols


def load_sources(service: str, file_paths: list[str]) -> dict[str, str]:
    """{file_path: source} for exactly these files."""
    if not file_paths:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $paths AS path
            MATCH (f:File {service: $service, file_path: path})
            WHERE f.source IS NOT NULL AND f.source <> ''
            RETURN f.file_path AS file_path, f.source AS source
            """,
            service=service,
            paths=file_paths,
        )
        return {r["file_path"]: r["source"] for r in result}


# --- Stage 3: links inside one service --------------------------------------

# Edges between a service's own nodes, rebuilt whole on every sync of it.
_FILE_LINK_TYPES = "IMPORTS|TESTS"
_SYMBOL_LINK_TYPES = "REFERENCES|EXTENDS|IMPLEMENTS|MAPS_TO|READS|WRITES|RESOLVES|SERVES"


def _pairs(rows: list[tuple], keys: tuple[str, ...]) -> list[dict]:
    return [dict(zip(keys, row, strict=True)) for row in rows]


def write_service_links(service: str, links) -> None:
    """Replace every link inside `service` with `links` (a links.ServiceLinks).

    Deleted and rewritten whole rather than diffed: the links of one file
    depend on other files (a renamed export changes its importers' edges),
    so recomputing the service is both simpler and correct, and it is pure
    graph work — no model calls."""
    with get_driver().session() as session:
        session.run(
            f"MATCH (:File {{service: $service}})-[r:{_FILE_LINK_TYPES}]->() DELETE r",
            service=service,
        )
        session.run(
            f"MATCH (:Symbol {{service: $service}})-[r:{_SYMBOL_LINK_TYPES}]->() DELETE r",
            service=service,
        )
        session.run(
            """
            MATCH (n) WHERE (n:Table OR n:Operation OR n:Endpoint) AND n.service = $service
            DETACH DELETE n
            """,
            service=service,
        )
        session.run(
            """
            MATCH (f:File {service: $service}) WHERE f.unresolved_imports IS NOT NULL
            REMOVE f.unresolved_imports
            """,
            service=service,
        )
        session.run(
            """
            MATCH (c:Class {service: $service})
            WHERE c.kafka_handles IS NOT NULL OR c.kafka_publishes IS NOT NULL
               OR c.kafka_publishes_built IS NOT NULL OR c.avro_fqn IS NOT NULL
            REMOVE c.kafka_handles, c.kafka_publishes, c.kafka_publishes_built,
                   c.kafka_input_topics, c.kafka_output_topics, c.avro_fqn, c.avro_is_message
            """,
            service=service,
        )

    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (a:File {service: $service, file_path: row.source})
        MATCH (b:File {service: $service, file_path: row.target})
        MERGE (a)-[r:IMPORTS]->(b)
        SET r.names = row.names, r.via = row.via, r.reexport = row.reexport
        """,
        "rows",
        [
            {"source": e.source, "target": e.target, "names": e.names, "via": e.via,
             "reexport": e.reexport}
            for e in links.imports
        ],
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:File {service: $service, file_path: row.file_path})
        SET f.unresolved_imports = row.modules
        """,
        "rows",
        [{"file_path": p, "modules": sorted(set(m))} for p, m in links.unresolved_imports.items()],
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (a:File {service: $service, file_path: row.test})
        MATCH (b:File {service: $service, file_path: row.tested})
        MERGE (a)-[:TESTS]->(b)
        """,
        "rows",
        _pairs(links.tests, ("test", "tested")),
        service=service,
    )
    for rel, rows in (
        ("REFERENCES", links.references),
        ("EXTENDS", links.extends),
        ("IMPLEMENTS", links.implements),
    ):
        _run_batched(
            f"""
            UNWIND $rows AS row
            MATCH (a:Symbol {{service: $service, qualified_name: row.source}})
            MATCH (b:Symbol {{service: $service, qualified_name: row.target}})
            MERGE (a)-[:{rel}]->(b)
            """,
            "rows",
            _pairs(rows, ("source", "target")),
            service=service,
        )

    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: $service})
        CREATE (t:Table {service: $service, name: row.name})
        SET t.columns = row.columns, t.defined_in = row.defined_in,
            t.foreign_keys = row.references, t.kind = row.kind
        MERGE (s)-[:OWNS]->(t)
        WITH t, row
        UNWIND row.defined_in AS path
        MATCH (f:File {service: $service, file_path: path})
        MERGE (f)-[:DEFINES]->(t)
        """,
        "rows",
        [
            {"name": t.name, "columns": [c.render() for c in t.columns.values()],
             "defined_in": t.defined_in, "references": t.references, "kind": t.kind}
            for t in links.tables
        ],
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (a:Table {service: $service, name: row.source})
        MATCH (b:Table {service: $service, name: row.target})
        MERGE (a)-[:FOREIGN_KEY]->(b)
        """,
        "rows",
        [{"source": t.name, "target": ref} for t in links.tables for ref in t.references],
        service=service,
    )
    for rel, rows in (("MAPS_TO", links.maps_to), ("READS", links.reads), ("WRITES", links.writes)):
        _run_batched(
            f"""
            UNWIND $rows AS row
            MATCH (a:Symbol {{service: $service, qualified_name: row.source}})
            MATCH (t:Table {{service: $service, name: row.target}})
            MERGE (a)-[:{rel}]->(t)
            """,
            "rows",
            _pairs(rows, ("source", "target")),
            service=service,
        )

    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: $service})
        MERGE (o:Operation {service: $service, type: row.type, name: row.name})
        SET o.file_path = row.file_path
        MERGE (s)-[:EXPOSES]->(o)
        """,
        "rows",
        [{"type": o.type, "name": o.name, "file_path": o.file_path} for o in links.operations],
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (a:Symbol {service: $service, qualified_name: row.source})
        MATCH (o:Operation {service: $service, type: row.type, name: row.name})
        MERGE (a)-[:RESOLVES]->(o)
        """,
        "rows",
        _pairs(links.resolves, ("source", "type", "name")),
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: $service})
        MATCH (a:Symbol {service: $service, qualified_name: row.source})
        MERGE (e:Endpoint {service: $service, method: row.method, path: row.path})
        MERGE (s)-[:EXPOSES]->(e)
        MERGE (a)-[:SERVES]->(e)
        """,
        "rows",
        _pairs(links.endpoints, ("source", "method", "path")),
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (c:Class {service: $service, qualified_name: row.qualified_name})
        SET c.kafka_handles = row.handles, c.kafka_publishes = row.publishes,
            c.kafka_publishes_built = row.publishes_built,
            c.kafka_input_topics = row.input_topics, c.kafka_output_topics = row.output_topics
        """,
        "rows",
        [
            {
                "qualified_name": q,
                **{
                    key: k.get(key, [])
                    for key in ("handles", "publishes", "publishes_built", "input_topics")
                },
                "output_topics": k.get("output_topics", []),
            }
            for q, k in links.kafka.items()
        ],
        service=service,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (c:Class {service: $service, qualified_name: row.qualified_name})
        SET c.avro_fqn = row.fqn, c.avro_is_message = row.is_message
        """,
        "rows",
        links.avro_messages,
        service=service,
    )


# --- Stage 4: links across services -----------------------------------------


def load_cross_service_facts(services: list[str]) -> dict:
    """Everything service_links.py matches across services, in one read."""
    queries = {
        "avro": """
            MATCH (c:Class) WHERE c.service IN $services AND c.avro_fqn IS NOT NULL
            RETURN c.service AS service, c.qualified_name AS qualified_name,
                   c.avro_fqn AS fqn, c.avro_is_message AS is_message, c.file_path AS file_path
        """,
        "kafka": """
            MATCH (c:Class) WHERE c.service IN $services
              AND (c.kafka_handles IS NOT NULL OR c.kafka_publishes IS NOT NULL
                   OR c.kafka_publishes_built IS NOT NULL)
            RETURN c.service AS service, c.qualified_name AS qualified_name,
                   coalesce(c.kafka_handles, []) AS handles,
                   coalesce(c.kafka_publishes, []) AS publishes,
                   coalesce(c.kafka_publishes_built, []) AS publishes_built,
                   coalesce(c.kafka_input_topics, []) AS input_topics,
                   coalesce(c.kafka_output_topics, []) AS output_topics
        """,
        "constants": """
            MATCH (f:File) WHERE f.service IN $services AND f.kotlin_constants IS NOT NULL
            RETURN f.service AS service, f.file_path AS file_path, f.kotlin_constants AS constants
        """,
        "operations": """
            MATCH (o:Operation) WHERE o.service IN $services
            RETURN o.service AS service, o.type AS type, o.name AS name
        """,
        "client_operations": """
            MATCH (f:File) WHERE f.service IN $services AND f.graphql_operations IS NOT NULL
              AND coalesce(f.kind, '') <> 'test'
            RETURN f.service AS service, f.file_path AS file_path,
                   f.graphql_operations AS operations
        """,
        "endpoints": """
            MATCH (e:Endpoint) WHERE e.service IN $services
            RETURN e.service AS service, e.method AS method, e.path AS path
        """,
        "http_calls": """
            MATCH (f:File) WHERE f.service IN $services AND f.http_paths IS NOT NULL
              AND coalesce(f.kind, '') <> 'test'
            RETURN f.service AS service, f.file_path AS file_path, f.http_paths AS paths
        """,
        "text_scan": """
            MATCH (s:Service)-[:HAS_FILE]->(f:File) WHERE s.name IN $services
            UNWIND coalesce(f.references, []) AS target
            RETURN s.name AS service, target, count(*) AS files
        """,
    }
    data: dict[str, list[dict]] = {}
    with get_driver().session() as session:
        for key, query in queries.items():
            data[key] = [dict(r) for r in session.run(query, services=services)]
    for row in data["constants"]:
        try:
            row["constants"] = json.loads(row["constants"])
        except (TypeError, ValueError):
            row["constants"] = {}
    return data


def write_cross_service_links(services: list[str], result) -> None:
    """Replace every cross-service link (a service_links.CrossServiceLinks)
    owned by `services`."""
    with get_driver().session() as session:
        # Only the topic links of messages these services send or handle —
        # a run over a few services must not wipe every other service's.
        # Deleted before HANDLES/PUBLISHES, which are how they are found.
        session.run(
            """
            MATCH (c:Class)-[:HANDLES|PUBLISHES]->(m:Message)-[r:ON_TOPIC]->(:Topic)
            WHERE c.service IN $services
            DELETE r
            """,
            services=services,
        )
        session.run(
            """
            MATCH (a)-[r:HANDLES|PUBLISHES|USES_OPERATION|CALLS_ENDPOINT]->()
            WHERE a.service IN $services
            DELETE r
            """,
            services=services,
        )
    _run_batched(
        """
        UNWIND $rows AS row
        MERGE (m:Message {name: row.name})
        SET m.defined_in_service = row.service, m.defined_in = row.file_path,
            m.is_message = row.is_message, m.short_name = row.short_name
        """,
        "rows",
        result.messages,
    )
    for rel, rows in (("HANDLES", result.handles), ("PUBLISHES", result.publishes)):
        _run_batched(
            f"""
            UNWIND $rows AS row
            MATCH (c:Class {{service: row.service, qualified_name: row.qualified_name}})
            MERGE (m:Message {{name: row.message}})
            MERGE (c)-[:{rel}]->(m)
            """,
            "rows",
            rows,
        )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (m:Message {name: row.message})
        MERGE (t:Topic {name: row.topic})
        SET t.constant = row.constant
        MERGE (m)-[:ON_TOPIC]->(t)
        """,
        "rows",
        result.on_topic,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:File {service: row.service, file_path: row.file_path})
        MATCH (o:Operation {service: row.target_service, type: row.type, name: row.name})
        MERGE (f)-[r:USES_OPERATION]->(o)
        SET r.ambiguous = row.ambiguous
        """,
        "rows",
        result.uses_operation,
    )
    _run_batched(
        """
        UNWIND $rows AS row
        MATCH (f:File {service: row.service, file_path: row.file_path})
        MATCH (e:Endpoint {service: row.target_service, method: row.method, path: row.path})
        MERGE (f)-[:CALLS_ENDPOINT]->(e)
        """,
        "rows",
        result.calls_endpoint,
    )
    with get_driver().session() as session:
        session.run("MATCH (m:Message) WHERE NOT (m)--() DELETE m")
        session.run("MATCH (t:Topic) WHERE NOT (t)--() DELETE t")
