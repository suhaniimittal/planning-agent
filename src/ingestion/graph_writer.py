"""Neo4j writes for Service nodes: one per service, keyed by name, with a
committed description (see service_doc.py) attached after its code graph is
built."""

from __future__ import annotations

import os

from neo4j import GraphDatabase

from src.ingestion import code_crypto
from src.ingestion.secret_store import secret

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        uri = os.environ["NEO4J_URI"]
        user = os.environ.get("NEO4J_USER", "neo4j")
        # A mounted secret in production, the variable on a laptop (2026-10-15).
        password = secret("NEO4J_PASSWORD") or os.environ["NEO4J_PASSWORD"]
        _driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            # Two notice types Neo4j would otherwise log on nearly every
            # query: a property not present yet on older data (e.g. `is_test`
            # on nodes ingested before it existed — the queries fall back for
            # that), and db.index.vector.queryNodes being deprecated.
            notifications_disabled_classifications=["UNRECOGNIZED", "DEPRECATION"],
        )
    return _driver


def create_vector_index(dimensions: int = 1536) -> None:
    with get_driver().session() as session:
        session.run(
            """
            CREATE VECTOR INDEX service_doc_index IF NOT EXISTS
            FOR (s:Service) ON (s.doc_embedding)
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
    `doc`/`doc_embedding`/`doc_hash` are set afterward by
    update_service_doc().

    Also strips any leftover properties from the old RepoDocAI/md-based
    pipeline (`full_doc`/`domain`/`entry_points`/`external_systems`/
    `key_symbols`/`summary_is_fallback`) and from the generated-summary
    pipeline that preceded SERVICE.md (`summary`/`summary_embedding`) — a
    Service node ingested before either redesign otherwise keeps that stale
    data forever, since a plain `SET` never removes a property it doesn't
    mention. This makes re-ingesting an old node self-healing instead of
    needing a manual Cypher cleanup.

    `summary` in particular is not merely unused but misleading: it is an
    LLM's description of parsed code, sitting next to the repository's own
    committed `doc`, and the two disagree. Eleven services still carried
    one after the switch.
    """
    with get_driver().session() as session:
        session.run(
            """
            MERGE (s:Service {name: $name})
            SET s.repo = $repo
            REMOVE s.full_doc, s.domain, s.entry_points, s.external_systems,
                   s.key_symbols, s.summary_is_fallback,
                   s.summary, s.summary_embedding
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
            RETURN s.commit_sha AS commit_sha, s.local_fingerprint AS local_fingerprint,
                   s.services_fingerprint AS services_fingerprint,
                   s.pipeline_version AS pipeline_version
            """,
            name=name,
        )
        record = result.single()
        if record is None:
            return None
        return {
            "commit_sha": record["commit_sha"],
            "local_fingerprint": record["local_fingerprint"],
            "services_fingerprint": record["services_fingerprint"],
            # None for a service ingested before this was recorded, which
            # compares unequal to any version and so forces a rebuild.
            "pipeline_version": record["pipeline_version"],
        }


def set_service_commit(
    name: str,
    commit_sha: str | None,
    local_fingerprint: str | None = None,
    services_fingerprint: str | None = None,
    pipeline_version: int | None = None,
) -> None:
    """Records which commit `name` is currently ingested at — the anchor
    the git-diff incremental sync compares against next run. None clears
    it, so the next run does a full ingest.

    `local_fingerprint` is set only after a local-mode ingest (see
    repo_fetcher.local_folder_fingerprint); every other write clears it, so
    a graph rebuilt from GitHub is never mistaken for an unchanged local
    folder.

    `services_fingerprint` records WHICH services existed when this
    service's dependency links were last computed. It moves when a
    service is added or removed, and nothing else does — which is what
    lets ingest.py re-run the cheap edge scan on its own instead of
    re-ingesting a service whose own code never changed.

    `pipeline_version` records which version of the parsing/chunking/
    embedding code built this graph, so improving that code invalidates
    every service by itself rather than needing someone to remember."""
    from datetime import UTC, datetime

    with get_driver().session() as session:
        session.run(
            """
            MATCH (s:Service {name: $name})
            SET s.commit_sha = $commit_sha, s.last_ingested_at = $last_ingested_at,
                s.local_fingerprint = $local_fingerprint,
                s.services_fingerprint = $services_fingerprint,
                s.pipeline_version = $pipeline_version
            """,
            name=name,
            commit_sha=commit_sha,
            last_ingested_at=datetime.now(UTC).isoformat(),
            local_fingerprint=local_fingerprint,
            services_fingerprint=services_fingerprint,
            pipeline_version=pipeline_version,
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


def update_service_doc(name: str, doc: str, embedding: list[float], doc_hash: str) -> None:
    """Overwrite a Service node's committed description and its embedding.

    `doc_hash` is stored alongside so the next run can tell an edited
    SERVICE.md from an unchanged one and skip re-embedding — the file, not
    the code, is what decides whether this moves."""
    with get_driver().session() as session:
        session.run(
            """
            MATCH (s:Service {name: $name})
            SET s.doc = $doc,
                s.doc_embedding = $embedding,
                s.doc_hash = $doc_hash
            """,
            name=name,
            # SERVICE.md describes the code, so it is stored encrypted too.
            doc=code_crypto.encrypt(
                doc, label="Service", service=name, qualified_name=name, field="doc"
            ),
            embedding=embedding,
            doc_hash=doc_hash,
        )


def service_doc_hashes(names: list[str]) -> dict[str, str]:
    """{service: stored doc hash} for those of `names` that already have a
    description AND its embedding.

    Lets a run re-embed only the services whose SERVICE.md actually changed.
    Re-embedding an unchanged description would be wasted work, and when the
    text came from a model rather than a file it was worse than wasted: the
    rewording moved the embedding and silently reshuffled which services a
    TDD considered (observed: a 0.03 swing on an unchanged service, larger
    than the gap between 3rd and 4th place).
    """
    if not names:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE s.name IN $names AND s.doc IS NOT NULL
              AND s.doc <> '' AND s.doc_embedding IS NOT NULL
            RETURN s.name AS name, s.doc_hash AS doc_hash
            """,
            names=names,
        )
        return {r["name"]: r["doc_hash"] for r in result}
