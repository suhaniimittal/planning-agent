"""Neo4j writes: one Service node per service, CALLS edges between them."""

from __future__ import annotations

import os

from neo4j import GraphDatabase

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ.get("NEO4J_USER", "neo4j")
        password = os.environ["NEO4J_PASSWORD"]
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def create_vector_index(dimensions: int = 1536) -> None:
    with get_driver().session() as session:
        session.run(
            """
            CREATE VECTOR INDEX service_summary_index IF NOT EXISTS
            FOR (s:Service) ON (s.summary_embedding)
            OPTIONS {indexConfig: {
              `vector.dimensions`: $dimensions,
              `vector.similarity_function`: 'cosine'
            }}
            """,
            dimensions=dimensions,
        )


def upsert_service_node(record: dict) -> None:
    with get_driver().session() as session:
        session.run(
            """
            MERGE (s:Service {name: $name})
            SET s.summary = $summary,
                s.summary_is_fallback = $summary_is_fallback,
                s.summary_embedding = $embedding,
                s.full_doc = $full_doc,
                s.domain = $domain,
                s.entry_points = $entry_points,
                s.external_systems = $external_systems,
                s.key_symbols = $key_symbols
            """,
            **record,
        )


def create_edge(source: str, target: str, rel_type: str = "CALLS") -> None:
    with get_driver().session() as session:
        session.run(
            f"""
            MATCH (a:Service {{name: $source}})
            MATCH (b:Service {{name: $target}})
            MERGE (a)-[:{rel_type}]->(b)
            """,
            source=source,
            target=target,
        )


def load_existing_service_records() -> list[dict]:
    """Name/full_doc/key_symbols for every Service node already in the
    graph — used for a scoped (single-service) ingestion run, so mention
    resolution and the entity index still see every already-ingested
    service, not just the one(s) being (re)written this run."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            RETURN s.name AS name, s.full_doc AS full_doc, s.key_symbols AS key_symbols
            """
        )
        return [
            {
                "name": r["name"],
                "full_doc": r["full_doc"] or "",
                "key_symbols": r["key_symbols"] or [],
            }
            for r in result
        ]


def leaf_services_report() -> list[str]:
    """Services with no outgoing CALLS edge — sanity-check these are genuine leaves."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE NOT (s)-[:CALLS]->()
            RETURN s.name AS name
            """
        )
        return [r["name"] for r in result]
