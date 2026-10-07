"""Neo4j writes for Service nodes: one per service, keyed by name, with a
code-derived summary (see summarizer.py) attached after its code graph is
built."""

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


def upsert_service_node(name: str, repo: str) -> None:
    """Creates/updates the Service node from its manifest entry alone —
    name and repo are all that's known before its code graph is built;
    `summary`/`summary_embedding` are set afterward by
    update_service_summary().

    Also strips any leftover properties from the old RepoDocAI/md-based
    pipeline (`full_doc`/`domain`/`entry_points`/`external_systems`/
    `key_symbols`/`summary_is_fallback`) — a Service node ingested before
    this redesign otherwise keeps that stale data forever, since a plain
    `SET` never removes a property it doesn't mention. This makes
    re-ingesting an old node self-healing instead of needing a manual
    Cypher cleanup.
    """
    with get_driver().session() as session:
        session.run(
            """
            MERGE (s:Service {name: $name})
            SET s.repo = $repo
            REMOVE s.full_doc, s.domain, s.entry_points, s.external_systems,
                   s.key_symbols, s.summary_is_fallback
            """,
            name=name,
            repo=repo,
        )


def load_service_state(name: str) -> dict | None:
    """{"commit_sha": str | None, "local_fingerprint": str | None} for this
    service's Service node, or None if the node doesn't exist at all yet
    (never ingested, vs. ingested but somehow missing a commit_sha — a
    genuinely different case: the former means "do a full first ingest,"
    the latter shouldn't normally happen but is treated the same way
    defensively)."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service {name: $name})
            RETURN s.commit_sha AS commit_sha, s.local_fingerprint AS local_fingerprint
            """,
            name=name,
        )
        record = result.single()
        if record is None:
            return None
        return {
            "commit_sha": record["commit_sha"],
            "local_fingerprint": record["local_fingerprint"],
        }


def set_service_commit(
    name: str, commit_sha: str | None, local_fingerprint: str | None = None
) -> None:
    """Records which commit `name` is currently ingested at — the anchor
    the git-diff incremental sync compares against next run. None clears
    it, so the next run does a full ingest.

    `local_fingerprint` is set only after a local-mode ingest (see
    repo_fetcher.local_folder_fingerprint); every other write clears it, so
    a graph rebuilt from GitHub is never mistaken for an unchanged local
    folder."""
    from datetime import UTC, datetime

    with get_driver().session() as session:
        session.run(
            """
            MATCH (s:Service {name: $name})
            SET s.commit_sha = $commit_sha, s.last_ingested_at = $last_ingested_at,
                s.local_fingerprint = $local_fingerprint
            """,
            name=name,
            commit_sha=commit_sha,
            last_ingested_at=datetime.now(UTC).isoformat(),
            local_fingerprint=local_fingerprint,
        )


def set_service_dependencies(source: str, targets: list[str]) -> None:
    """Replaces every outgoing CALLS edge from `source` with exactly
    `targets` — a dependency no longer detected in the code (removed,
    renamed, refactored away) has its stale edge deleted, not left behind
    forever, matching the reconciliation approach code_graph_writer's
    prune_stale_code_nodes already uses for Class/Function/Chunk nodes."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (a:Service {name: $source})
            OPTIONAL MATCH (a)-[r:CALLS]->(stale:Service)
            WHERE NOT stale.name IN $targets
            DELETE r
            WITH a
            UNWIND $targets AS target_name
            MATCH (b:Service {name: target_name})
            MERGE (a)-[:CALLS]->(b)
            """,
            source=source,
            targets=targets,
        )


def update_service_summary(name: str, summary: str, embedding: list[float]) -> None:
    """Overwrite the summary/embedding on an already-written Service node —
    used by the code-derived batched summarizer, which runs after every
    service's code graph has been built this run."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (s:Service {name: $name})
            SET s.summary = $summary,
                s.summary_embedding = $embedding
            """,
            name=name,
            summary=summary,
            embedding=embedding,
        )
