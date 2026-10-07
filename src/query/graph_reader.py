"""Neo4j reads for the query-time flow: vector search, CALLS-graph
traversal, candidate-doc loading. Counterpart to graph_writer.py's writes —
kept separate since the two paths have different callers and failure modes
(a write failure breaks ingestion; a read failure breaks a live request).
"""

from __future__ import annotations

from src.ingestion.graph_writer import get_driver


def vector_search_services(embedding: list[float], k: int = 5) -> list[dict]:
    """Top-k services by cosine similarity on summary_embedding.

    Returns [{"name": str, "score": float}, ...], descending score.
    """
    with get_driver().session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('service_summary_index', $k, $embedding)
            YIELD node, score
            RETURN node.name AS name, score
            ORDER BY score DESC
            """,
            k=k,
            embedding=embedding,
        )
        return [{"name": r["name"], "score": r["score"]} for r in result]


def expand_neighbors(seed_names: list[str], hops: int = 2) -> list[dict]:
    """Services within `hops` of any seed, via CALLS in either direction.

    Returns [{"name": str, "hop_distance": int}, ...], seeds excluded.
    A vector-matched service alone doesn't tell you whether the real change
    belongs in something it calls (root cause) or something that calls it
    (blast radius), so both directions are included — this only widens the
    candidate set for the LLM to judge, it doesn't decide anything itself.
    """
    if not seed_names:
        return []
    assert 1 <= hops <= 2, "hops must be 1 or 2 — traversal range is not a Cypher parameter"
    with get_driver().session() as session:
        result = session.run(
            f"""
            MATCH (seed:Service)
            WHERE seed.name IN $seed_names
            MATCH path = (seed)-[:CALLS*1..{hops}]-(neighbor:Service)
            WHERE NOT neighbor.name IN $seed_names
            WITH neighbor.name AS name, min(length(path)) AS hop_distance
            RETURN name, hop_distance
            ORDER BY hop_distance ASC
            """,
            seed_names=seed_names,
        )
        return [{"name": r["name"], "hop_distance": r["hop_distance"]} for r in result]


def get_edges_among(names: list[str]) -> list[dict]:
    """Real CALLS edges where both endpoints are in `names`.

    Returns [{"source": str, "target": str}, ...] — used to draw an honest
    dependency diagram, since it's read straight from the graph rather than
    guessed by the LLM.
    """
    if not names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (a:Service)-[:CALLS]->(b:Service)
            WHERE a.name IN $names AND b.name IN $names
            RETURN a.name AS source, b.name AS target
            """,
            names=names,
        )
        return [{"source": r["source"], "target": r["target"]} for r in result]


def load_candidate_docs(names: list[str]) -> list[dict]:
    """name + summary for exactly these service names — the short,
    code-derived overview generated at ingestion time (see
    src/ingestion/summarizer.py). There's no separate "full doc" tier
    anymore: real per-symbol detail comes from load_code_symbols_for_services/
    load_source_snippets instead, not a markdown document."""
    if not names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE s.name IN $names
            RETURN s.name AS name, s.summary AS summary
            """,
            names=names,
        )
        return [dict(r) for r in result]


def load_code_symbols_for_services(names: list[str]) -> list[dict]:
    """Class/Function metadata for these services — everything EXCEPT
    source_snippet, which is kept in a separate, narrower read
    (load_source_snippets) since a service can have hundreds of symbols and
    most won't survive candidates.select_relevant_symbols' ranking/cap.
    """
    if not names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (n)
            WHERE (n:Class OR n:Function) AND n.service IN $names
            RETURN n.service AS service, n.qualified_name AS qualified_name,
                   n.name AS name, n.file_path AS file_path,
                   n.line_start AS line_start, n.line_end AS line_end,
                   n.signature AS signature, n.docstring AS docstring,
                   n.language AS language,
                   n.is_method AS is_method, n.parent_class AS parent_class,
                   labels(n)[0] AS kind
            """,
            names=names,
        )
        return [dict(r) for r in result]


def vector_search_chunks(embedding: list[float], service_names: list[str], k: int = 50) -> list[dict]:
    """Top-k Chunk nodes by cosine similarity to `embedding`, rolled up to
    the Function each one belongs to, restricted to `service_names`.

    Returns [{"service", "function_qualified_name", "text", "score"}, ...],
    descending score. The service filter is applied AFTER the vector index
    lookup (Neo4j's simple vector-index query procedure doesn't support
    pre-filtering by property) — with only a handful of services in the
    graph today this is a non-issue, but at real scale a global top-k could
    theoretically miss a relevant chunk that belongs to a candidate service
    if enough higher-scoring chunks from OTHER services crowd it out; same
    class of caveat already noted for `vector_search_services` and the
    `db.index.vector.queryNodes` deprecation in docs/INGESTION_PIPELINE.md.
    """
    if not service_names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $embedding)
            YIELD node, score
            MATCH (f:Function)-[:HAS_CHUNK]->(node)
            WHERE node.service IN $service_names
            RETURN node.service AS service, f.qualified_name AS function_qualified_name,
                   node.text AS text, score
            ORDER BY score DESC
            """,
            k=k,
            embedding=embedding,
            service_names=service_names,
        )
        return [dict(r) for r in result]


def load_source_snippets(
    service_qualified_name_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """source_snippet for exactly these (service, qualified_name) symbols,
    keyed by that same (service, qualified_name) pair — called only for the
    final, ranked+capped shortlist, never for every symbol a service has.

    Keyed by the pair, not qualified_name alone: qualified_name is only
    unique WITHIN a service (it's derived from file_path, which two
    different repos can coincidentally share), so a bare-string key could
    collide across services in the result.
    """
    if not service_qualified_name_pairs:
        return {}
    rows = [{"service": s, "qualified_name": q} for s, q in service_qualified_name_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE (n:Class OR n:Function)
              AND n.service = row.service AND n.qualified_name = row.qualified_name
            RETURN n.service AS service, n.qualified_name AS qualified_name,
                   n.source_snippet AS source_snippet
            """,
            rows=rows,
        )
        return {(r["service"], r["qualified_name"]): r["source_snippet"] for r in result}


def file_exists(service: str, file_path: str) -> bool:
    """Whether `service` has a File node at exactly `file_path`."""
    with get_driver().session() as session:
        result = session.run(
            "MATCH (f:File {service: $service, file_path: $file_path}) RETURN count(f) AS n",
            service=service,
            file_path=file_path,
        )
        return result.single()["n"] > 0


def folder_exists(service: str, folder: str) -> bool:
    """Whether `service` has at least one File under `folder/` — i.e.
    `folder` is a real directory of that service, not a file."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (f:File {service: $service})
            WHERE f.file_path STARTS WITH $prefix
            RETURN f.file_path AS file_path LIMIT 1
            """,
            service=service,
            prefix=folder.rstrip("/") + "/",
        )
        return result.single() is not None


def find_symbol_files(service: str, symbol: str) -> list[str]:
    """Every file in `service` defining a Class/Function named `symbol` —
    matched by plain name ("view") or by its qualified_name's tail
    ("PreAdverseActionModal.view"), the two forms the LLM is shown."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (n:Class {service: $service})
            WHERE n.name = $symbol OR n.qualified_name ENDS WITH $by_file
               OR n.qualified_name ENDS WITH $by_class
            RETURN n.file_path AS file_path
            UNION
            MATCH (n:Function {service: $service})
            WHERE n.name = $symbol OR n.qualified_name ENDS WITH $by_file
               OR n.qualified_name ENDS WITH $by_class
            RETURN n.file_path AS file_path
            """,
            service=service,
            symbol=symbol,
            by_file="::" + symbol,
            by_class="." + symbol,
        )
        return sorted({r["file_path"] for r in result})
