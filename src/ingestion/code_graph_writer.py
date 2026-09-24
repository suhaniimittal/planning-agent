"""Neo4j writes for the per-repo code graph: Class/Function nodes nested
under their Service via CONTAINS/DECLARES.

Kept separate from graph_writer.py (Service nodes + inter-service CALLS) —
same one-module-per-responsibility split already used for github_fetcher.py
vs s3_fetcher.py. All writes here are batched via UNWIND: a repo parsed with
"parse everything" can easily produce hundreds to thousands of symbols, so
one round trip per node is not an option.
"""

from __future__ import annotations

from .graph_writer import get_driver


def create_code_indexes() -> None:
    with get_driver().session() as session:
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


def upsert_classes(service: str, records: list[dict]) -> None:
    if not records:
        return
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Class {service: $service, qualified_name: row.qualified_name})
            SET c += row
            """,
            service=service,
            rows=records,
        )


def upsert_functions(service: str, records: list[dict]) -> None:
    if not records:
        return
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (f:Function {service: $service, qualified_name: row.qualified_name})
            SET f += row
            """,
            service=service,
            rows=records,
        )


def upsert_chunks(service: str, records: list[dict]) -> None:
    if not records:
        return
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {service: $service, qualified_name: row.qualified_name})
            SET c += row
            """,
            service=service,
            rows=records,
        )


def create_has_chunk_edges(service: str, function_to_chunk_qnames: list[tuple[str, str]]) -> None:
    """(Function)-[:HAS_CHUNK]->(Chunk) for every (function_qname, chunk_qname) pair."""
    if not function_to_chunk_qnames:
        return
    rows = [{"function_qname": f, "chunk_qname": c} for f, c in function_to_chunk_qnames]
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (f:Function {service: $service, qualified_name: row.function_qname})
            MATCH (c:Chunk {service: $service, qualified_name: row.chunk_qname})
            MERGE (f)-[:HAS_CHUNK]->(c)
            """,
            service=service,
            rows=rows,
        )


def create_contains_edges(service: str, qualified_names: list[str], label: str) -> None:
    """(Service)-[:CONTAINS]->(label) for every given node.

    `label` is "Class" for all classes, or "Function" for TOP-LEVEL
    functions only — methods are reached via their class's DECLARES edge,
    not directly CONTAINS-ed by the service.
    """
    if not qualified_names:
        return
    with get_driver().session() as session:
        session.run(
            f"""
            UNWIND $qnames AS qname
            MATCH (s:Service {{name: $service}})
            MATCH (n:{label} {{service: $service, qualified_name: qname}})
            MERGE (s)-[:CONTAINS]->(n)
            """,
            service=service,
            qnames=qualified_names,
        )


def create_declares_edges(service: str, class_to_method_qnames: list[tuple[str, str]]) -> None:
    """(Class)-[:DECLARES]->(Function) for every (class_qname, method_qname) pair."""
    if not class_to_method_qnames:
        return
    rows = [{"class_qname": c, "method_qname": m} for c, m in class_to_method_qnames]
    with get_driver().session() as session:
        session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Class {service: $service, qualified_name: row.class_qname})
            MATCH (f:Function {service: $service, qualified_name: row.method_qname})
            MERGE (c)-[:DECLARES]->(f)
            """,
            service=service,
            rows=rows,
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
