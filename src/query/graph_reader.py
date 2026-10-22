"""Neo4j reads for the query-time flow: vector search, CALLS-graph
traversal, candidate-doc loading. Counterpart to graph_writer.py's writes —
kept separate since the two paths have different callers and failure modes
(a write failure breaks ingestion; a read failure breaks a live request).

Code, signatures, docstrings, chunk text and summaries are stored encrypted
(see src/ingestion/code_crypto.py); each read below decrypts only the rows it
returns, as it returns them.
"""

from __future__ import annotations

from src.ingestion import code_crypto
from src.ingestion.graph_writer import get_driver
from src.ingestion.test_paths import CYPHER_TEST_PATH_RE


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
    anymore: real per-symbol detail comes from search_service_chunks/
    load_symbols instead, not a markdown document."""
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
        return [_open_summary(dict(r)) for r in result]


def _open_summary(row: dict) -> dict:
    return code_crypto.decrypt_row(
        row, label="Service", service=row["name"], qualified_name=row["name"]
    )


def _open_symbol(row: dict) -> dict:
    """A Class/Function row with its code fields decrypted; the row must
    carry service, qualified_name and kind (the node's label)."""
    return code_crypto.decrypt_row(
        row, label=row["kind"], service=row["service"], qualified_name=row["qualified_name"]
    )


def _open_chunk(row: dict, service: str) -> dict:
    """A chunk row with its text decrypted; the chunk's own qualified_name,
    needed to decrypt, is dropped from what callers see."""
    qualified_name = row.pop("chunk_qualified_name")
    return code_crypto.decrypt_row(
        row, label="Chunk", service=service, qualified_name=qualified_name
    )


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
                   node.qualified_name AS chunk_qualified_name, node.text AS text, score
            ORDER BY score DESC
            """,
            k=k,
            embedding=embedding,
            service_names=service_names,
        )
        return [_open_chunk(dict(r), r["service"]) for r in result]


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


# Real code only: `is_test` is stamped at ingestion; nodes ingested before it
# existed fall back to the same path rule, evaluated here.
_NOT_TEST = "NOT coalesce(c.is_test, c.file_path =~ $test_re)"


def summary_scores(embedding: list[float], names: list[str] | None = None) -> dict[str, float]:
    """{service: cosine(issue embedding, summary embedding)} for `names`
    (every summarized service when None). Exact, not the vector index's
    approximate top-k — a run never has more than a few dozen services."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE s.summary_embedding IS NOT NULL AND ($names IS NULL OR s.name IN $names)
            RETURN s.name AS name,
                   vector.similarity.cosine(s.summary_embedding, $embedding) AS score
            """,
            names=names,
            embedding=embedding,
        )
        return {r["name"]: r["score"] for r in result}


def search_service_chunks(
    service: str, vectors: list[list[float]], identifiers: list[str], k: int = 20
) -> list[dict]:
    """One service's best-matching NON-TEST chunks, searched within that
    service only — so a large service can never crowd a small one out.

    A chunk's score is its BEST cosine against any of `vectors` (the whole
    ticket plus each of its parts) — matching one part of a multi-topic
    ticket strongly is enough, so a specific detail is not averaged away.
    Two reads, merged: the top `k` by that score, plus every chunk
    containing one of `identifiers` (exact code names from the issue text),
    whatever its score. The text is encrypted, so the second read matches
    the hashes of each identifier's words (`text_terms`), then keeps only
    the chunks whose decrypted text really contains it.

    Returns [{"function_qualified_name", "text", "score", "identifiers"}].
    """
    score = (
        "reduce(best = 0.0, v IN $vectors | "
        "CASE WHEN vector.similarity.cosine(c.embedding, v) > best "
        "THEN vector.similarity.cosine(c.embedding, v) ELSE best END)"
    )
    params = {"service": service, "vectors": vectors, "test_re": CYPHER_TEST_PATH_RE}
    with get_driver().session() as session:
        semantic = session.run(
            f"""
            MATCH (c:Chunk {{service: $service}})
            WHERE {_NOT_TEST}
            WITH c, {score} AS score
            ORDER BY score DESC LIMIT $k
            MATCH (f:Function)-[:HAS_CHUNK]->(c)
            RETURN f.qualified_name AS function_qualified_name,
                   c.qualified_name AS chunk_qualified_name, c.text AS text, score,
                   [] AS identifiers
            """,
            k=k,
            **params,
        )
        rows = [_open_chunk(dict(r), service) for r in semantic]
        if identifiers:
            groups = [code_crypto.query_terms(service, w) for w in identifiers]
            exact = session.run(
                f"""
                MATCH (c:Chunk {{service: $service}})
                WHERE {_NOT_TEST} AND (
                  any(g IN $groups WHERE size(g) > 0
                      AND all(t IN g WHERE t IN coalesce(c.text_terms, [])))
                  OR (c.text_terms IS NULL AND any(w IN $identifiers WHERE c.text CONTAINS w)))
                WITH c, {score} AS score
                MATCH (f:Function)-[:HAS_CHUNK]->(c)
                RETURN f.qualified_name AS function_qualified_name,
                       c.qualified_name AS chunk_qualified_name, c.text AS text, score
                LIMIT 200
                """,
                groups=groups,
                identifiers=identifiers,
                **params,
            )
            for record in exact:
                row = _open_chunk(dict(record), service)
                row["identifiers"] = [w for w in identifiers if w in (row["text"] or "")]
                if row["identifiers"]:  # a hash match is checked against the real text
                    rows.append(row)
        return rows


def load_symbols(service_qualified_name_pairs: list[tuple[str, str]]) -> list[dict]:
    """Full metadata AND source for exactly these (service, qualified_name)
    Class/Function symbols — the final shortlist only."""
    if not service_qualified_name_pairs:
        return []
    rows = [{"service": s, "qualified_name": q} for s, q in service_qualified_name_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE (n:Class OR n:Function)
              AND n.service = row.service AND n.qualified_name = row.qualified_name
            RETURN n.service AS service, n.qualified_name AS qualified_name,
                   n.name AS name, n.file_path AS file_path,
                   n.line_start AS line_start, n.line_end AS line_end,
                   n.signature AS signature, n.docstring AS docstring,
                   n.language AS language, n.source_snippet AS source_snippet,
                   n.is_method AS is_method, n.parent_class AS parent_class,
                   labels(n)[0] AS kind
            """,
            rows=rows,
        )
        return [_open_symbol(dict(r)) for r in result]


def load_file_context(
    service_file_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """{(service, file_path): names of everything defined in that file} —
    classes, top-level functions, and each class's methods as
    "Class.method" — so the planner sees what a matched function sits next
    to and can call, without being sent all of that code."""
    if not service_file_pairs:
        return {}
    rows = [{"service": s, "file_path": f} for s, f in service_file_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (file:File {service: row.service, file_path: row.file_path})-[:DEFINES]->(n)
            OPTIONAL MATCH (n)-[:DECLARES]->(m:Function)
            WITH row, n, collect(n.name + "." + m.name) AS methods
            RETURN row.service AS service, row.file_path AS file_path,
                   collect(n.name) + reduce(acc = [], ms IN collect(methods) | acc + ms) AS names
            """,
            rows=rows,
        )
        return {(r["service"], r["file_path"]): sorted(set(r["names"])) for r in result}
