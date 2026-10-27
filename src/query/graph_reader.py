"""Neo4j reads for the query-time flow: vector search, CALLS-graph
traversal, candidate-doc loading. Counterpart to graph_writer.py's writes —
kept separate since the two paths have different callers and failure modes
(a write failure breaks ingestion; a read failure breaks a live request).
"""

from __future__ import annotations

from src.ingestion.code_parser import MAX_TEST_SOURCE_CHARS
from src.ingestion.graph_writer import get_driver
from src.ingestion.test_paths import CYPHER_SPEC_FILE_RE, CYPHER_TEST_PATH_RE, is_test_path


def vector_search_services(embedding: list[float], k: int = 5) -> list[dict]:
    """Top-k services by cosine similarity on doc_embedding.

    Returns [{"name": str, "score": float}, ...], descending score.
    """
    with get_driver().session() as session:
        result = session.run(
            """
            CALL db.index.vector.queryNodes('service_doc_index', $k, $embedding)
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
    """name + description for exactly these service names — the SERVICE.md
    each repository commits (see src/ingestion/service_doc.py). Per-symbol
    detail comes from load_symbols/load_file_context instead; this is only
    what the service is FOR."""
    if not names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE s.name IN $names
            RETURN s.name AS name, s.doc AS summary
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
            WHERE (n:Class OR n:Function OR n:Variable) AND n.service IN $names
            RETURN n.service AS service, n.qualified_name AS qualified_name,
                   n.name AS name, n.file_path AS file_path,
                   n.line_start AS line_start, n.line_end AS line_end,
                   n.signature AS signature, n.docstring AS docstring,
                   n.language AS language,
                   n.is_method AS is_method, n.parent_class AS parent_class,
                   [l IN labels(n) WHERE l <> 'Symbol'][0] AS kind
            """,
            names=names,
        )
        return [dict(r) for r in result]


def vector_search_chunks(
    embedding: list[float], service_names: list[str], k: int = 50
) -> list[dict]:
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
            WHERE (n:Class OR n:Function OR n:Variable)
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
    """Every file in `service` defining a Class/Function/Variable named
    `symbol` — matched by plain name ("view") or by its qualified_name's tail
    ("PreAdverseActionModal.view"), the two forms the LLM is shown.

    Variable is included because since pipeline version 4 a top-level
    constant is its own node: a styled component like `SearchAndButtonWrap`
    was otherwise reported as "not in the file today" while it was."""
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
            UNION
            MATCH (n:Variable {service: $service})
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
    """{service: cosine(issue embedding, its SERVICE.md embedding)} for
    `names` (every service when None). Exact, not the vector index's
    approximate top-k — a run never has more than a few dozen services.

    A service with no committed SERVICE.md scores 0.0 rather than being left
    out. Omitting it would delete it from the run: callers build the search
    shortlist from these names, so a service missing here is never searched
    and never appears in the design, with nothing said about it. Scoring 0
    instead costs it half its combined score and leaves it to win on its
    code, which it can.
    """
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (s:Service)
            WHERE $names IS NULL OR s.name IN $names
            RETURN s.name AS name,
                   CASE WHEN s.doc_embedding IS NULL THEN 0.0
                        ELSE vector.similarity.cosine(s.doc_embedding, $embedding) END AS score
            """,
            names=names,
            embedding=embedding,
        )
        return {r["name"]: r["score"] for r in result}


def search_service_chunks(
    service: str, vectors: list[list[float]], identifiers: list[str], k: int = 50
) -> list[dict]:
    """One service's best-matching NON-TEST chunks, searched within that
    service only — so a large service can never crowd a small one out.

    A chunk's score is its BEST cosine against any of `vectors` (the whole
    ticket plus each of its parts) — matching one part of a multi-topic
    ticket strongly is enough, so a specific detail is not averaged away.
    Two reads, merged: the top `k` by that score, plus every chunk
    containing one of `identifiers` (exact code names from the issue text),
    whatever its score.

    `k` is close to a count of candidate FUNCTIONS, not of chunks: 93% of
    functions are short enough to be a single chunk, so measured on this
    graph 20 chunks yielded 17-22 distinct functions and 50 yielded ~43.
    It is also nearly free to raise — the scan already computes a score for
    every chunk in the service, and LIMIT only truncates the sorted output
    (measured: k=20 and k=300 both ~3.5s on 36k chunks). What costs is
    code_search.rank_functions' cap, which decides how much source reaches
    the prompt.

    A chunk reports the symbol that owns its lines (stored on the chunk as
    `function_qualified_name` — a function, or since pipeline version 4 any
    symbol that owns lines), and a chunk of lines no symbol owns reports its
    FILE PATH instead. Without that, a third of ui — every messages.ts among
    them — would score and then be silently dropped from the results.

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
            RETURN coalesce(c.function_qualified_name, c.file_path) AS function_qualified_name,
                   c.text AS text, score, [] AS identifiers
            """,
            k=k,
            **params,
        )
        rows = [dict(r) for r in semantic]
        if identifiers:
            exact = session.run(
                f"""
                MATCH (c:Chunk {{service: $service}})
                WHERE any(w IN $identifiers WHERE c.text CONTAINS w) AND {_NOT_TEST}
                WITH c, {score} AS score
                RETURN coalesce(c.function_qualified_name, c.file_path) AS function_qualified_name,
                       c.text AS text, score,
                       [w IN $identifiers WHERE c.text CONTAINS w] AS identifiers
                LIMIT 200
                """,
                identifiers=identifiers,
                **params,
            )
            rows += [dict(r) for r in exact]
        return rows


def load_symbols(service_qualified_name_pairs: list[tuple[str, str]]) -> list[dict]:
    """Full metadata AND source for exactly these (service, qualified_name)
    Class/Function symbols — the final shortlist only.

    A pair whose "qualified_name" is a FILE PATH resolves to that File
    instead, shaped like a symbol so every caller downstream treats it the
    same way. That is how a file with no class and no function reaches the
    planner at all: search_service_chunks reports such a chunk by its file
    path, and without this it would resolve to nothing and be dropped.
    """
    if not service_qualified_name_pairs:
        return []
    rows = [{"service": s, "qualified_name": q} for s, q in service_qualified_name_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (n)
            WHERE (n:Class OR n:Function OR n:Variable)
              AND n.service = row.service AND n.qualified_name = row.qualified_name
            RETURN n.service AS service, n.qualified_name AS qualified_name,
                   n.name AS name, n.file_path AS file_path,
                   n.line_start AS line_start, n.line_end AS line_end,
                   n.signature AS signature, n.docstring AS docstring,
                   n.language AS language, n.source_snippet AS source_snippet,
                   n.is_method AS is_method, n.parent_class AS parent_class,
                   [l IN labels(n) WHERE l <> 'Symbol'][0] AS kind
            """,
            rows=rows,
        )
        found = [dict(r) for r in result]
        seen = {(r["service"], r["qualified_name"]) for r in found}
        unresolved = [r for r in rows if (r["service"], r["qualified_name"]) not in seen]
        if not unresolved:
            return found
        files = session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {service: row.service, file_path: row.qualified_name})
            RETURN f.service AS service, f.file_path AS qualified_name,
                   f.file_path AS file_path,
                   split(f.file_path, '/')[-1] AS name,
                   f.language AS language
            """,
            rows=unresolved,
        )
        for row in files:
            record = dict(row)
            # Shaped like a symbol, with the fields a file has no answer
            # for left empty rather than invented. `kind` says File so the
            # prompt can say what this is.
            record.update(
                line_start=1, line_end=None, signature=None, docstring=None,
                source_snippet=None, is_method=False, parent_class=None, kind="File",
            )
            found.append(record)
        return found


def load_file_imports(service_file_pairs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """{(service, file_path): that file's import lines}, empty entries dropped.

    The planner otherwise has no idea how this project imports anything:
    the parser extracts classes and functions, so the lines above them were
    never stored. Code written without them invents module paths — a
    component importing "@client/hooks", which does not exist, and a new
    test file importing a props type that is not exported.
    """
    if not service_file_pairs:
        return {}
    rows = [{"service": s, "file_path": f} for s, f in service_file_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {service: row.service, file_path: row.file_path})
            WHERE f.imports IS NOT NULL AND f.imports <> ''
            RETURN row.service AS service, row.file_path AS file_path, f.imports AS imports
            """,
            rows=rows,
        )
        return {(r["service"], r["file_path"]): r["imports"] for r in result}


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


# A sibling-test convention counts only when it is clearly the project's
# habit, not an accident of a few files — below these the service is left
# with no convention and the planner asks where tests live instead.
_CONVENTION_MIN_FILES = 5
_CONVENTION_MIN_SHARE = 0.3
_SIBLING_TEST_INFIXES = (".test.", ".spec.")


def test_naming_conventions(services: list[str]) -> dict[str, str]:
    """{service: ".test." | ".spec."} for services whose tests demonstrably
    sit BESIDE the code they cover, named for it.

    Measured, never assumed: a test file counts only when removing the infix
    from its own path names a real non-test file in the same folder. That is
    what makes a derived path ("employmentVerificationForm.test.ts" beside
    "employmentVerificationForm.ts") evidence rather than a guess — the
    planner is forbidden from inventing test paths, and without this it can
    only put "where do tests for this live?" into open_questions, which
    leaves the test unwritten.

    Services whose tests live in a mirrored tree instead (src/test/kotlin
    beside src/main/kotlin) are deliberately absent: no infix pairs there,
    so nothing is claimed about them.
    """
    if not services:
        return {}
    with get_driver().session() as session:
        rows = session.run(
            """
            UNWIND $services AS service
            MATCH (t:File {service: service})
            WHERE t.file_path =~ $test_re
            WITH service, t, count(t) AS _
            WITH service, collect(t.file_path) AS test_paths
            UNWIND $infixes AS infix
            UNWIND test_paths AS test_path
            WITH service, infix, test_paths, test_path
            WHERE test_path CONTAINS infix
            OPTIONAL MATCH (s:File {service: service, file_path: replace(test_path, infix, '.')})
            RETURN service, infix, size(test_paths) AS total,
                   count(s) AS paired
            """,
            services=list(services),
            test_re=CYPHER_TEST_PATH_RE,
            infixes=list(_SIBLING_TEST_INFIXES),
        )
        best: dict[str, tuple[int, str]] = {}
        for row in rows:
            paired, total = row["paired"], row["total"]
            if paired < _CONVENTION_MIN_FILES or paired < total * _CONVENTION_MIN_SHARE:
                continue
            if paired > best.get(row["service"], (0, ""))[0]:
                best[row["service"]] = (paired, row["infix"])
    return {service: infix for service, (_, infix) in best.items()}


def new_test_path(file_path: str, infix: str) -> str | None:
    """Where a new test for `file_path` belongs under `infix`, or None for a
    path with no extension to put the infix in front of."""
    folder, _, name = file_path.rpartition("/")
    stem, dot, ext = name.rpartition(".")
    if not dot or not stem:
        return None
    return f"{folder}/{stem}{infix}{ext}" if folder else f"{stem}{infix}{ext}"


# A term in more spec files than this is describing an area, not a symbol.
_MAX_SPEC_FILES_PER_TERM = 10


def tests_naming(
    service_symbol_terms: list[tuple[str, str, list[str]]],
    limit: int = 4,
    max_files: int = _MAX_SPEC_FILES_PER_TERM,
) -> dict[tuple[str, str], list[str]]:
    """{(service, symbol): [test file paths naming that symbol or one of its
    terms]}.

    load_sibling_tests only looks in the SAME FOLDER, which misses the test
    on any project that separates them. Observed on a real ticket: the file
    to change was under src/common/, five test files covering it were under
    src/apps/, the sibling lookup found none, and the design said no test
    existed and had a brand-new one written — which then has to invent every
    import, since imports are not ingested.

    `terms` are the identifiers THIS symbol's own source mentions, and they
    exist because the symbol name alone finds nothing on a UI project. No
    test file in the repository contains "AtsIntegrationSectionView" or
    "handleCcEmailAddressesBlur" — a component renders through a wrapper and
    a handler is never named — while three contain
    `sendReportRequestedLinkCcEmailAddresses`, the field the handler reads,
    and those three hold the working fixture.

    Taking the terms from the SYMBOL rather than from the ticket is the
    whole point. A real ticket is written in English: "ATS Link CC email
    addresses changes semi-colon to comma" yields no identifiers at all, so
    a lookup keyed on the ticket had nothing to search with and fell back to
    reporting that no test existed.

    A term matching more than `max_files` specs in its service is dropped as
    too common to identify anything — the same idea as
    rank_functions' `max_identifier_hits`. Measured over ui's 1,105 spec
    files: sendReportRequestedLinkCcEmailAddresses 3, atsIntegration 5,
    candidateId 11, handleSubmit 60, reportNumber 117, SetupComponentBuilder
    260, React 1,013. Real signal sits at or below a handful and noise
    starts around sixty, so anything in that gap works; the default sits
    near the signal end of it.

    A plain CONTAINS, not a vector search: this runs AFTER the files to
    change are chosen, so it cannot take a result slot from real code
    (search_service_chunks excludes test chunks for exactly that reason —
    test prose echoes a ticket's wording and outscores the code it tests).

    Matched against the test file's OWN STORED TEXT, not its chunks. Chunks
    fail here for the same reason that made this lookup necessary at all: a
    test file's chunks are a fraction of it. Measured against the live
    graph, searching chunks for the ATS ticket's field found one file — a
    29,652-char suite — and missed the three focused specs (1,605 / 2,235 /
    3,920 chars) that hold the fixture worth copying, because their chunks
    are 172-331 characters and do not contain it.

    Chunks were tried as a second arm and dropped: they added 17.4s per
    lookup, against 0.11s for the stored text alone, to contribute only
    that one long suite — which smallest-first would never have shown.
    MAX_TEST_SOURCE_CHARS therefore bounds what is FINDABLE as well as what
    is shown; a term buried past it in a very long test is not reachable,
    and the planner falls back to writing a new test as it did before.

    Restricted to files whose NAME says they are a spec. A generated type
    declaration under cypress/support/@types/ is test-path code but is not
    a test, and eight of them match this ticket's field while being the
    shortest matches of all — smallest-first would have shown the planner
    nothing but type stubs.
    """
    if not service_symbol_terms:
        return {}
    lookups = [
        {"service": svc, "symbol": sym, "terms": sorted({sym, *terms})}
        for svc, sym, terms in service_symbol_terms
        if sym
    ]
    if not lookups:
        return {}
    with get_driver().session() as session:
        rows = session.run(
            """
            UNWIND $lookups AS lookup
            UNWIND lookup.terms AS term
            MATCH (f:File {service: lookup.service})
            WHERE f.file_path =~ $spec_re
              AND coalesce(f.source, '') CONTAINS term
            WITH lookup, term, collect(f.file_path) AS paths
            WHERE size(paths) <= $max_files
            UNWIND paths AS file_path
            RETURN lookup.service AS service, lookup.symbol AS symbol, file_path
            """,
            lookups=lookups,
            spec_re=CYPHER_SPEC_FILE_RE,
            max_files=max_files,
        )
        found: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            found.setdefault((row["service"], row["symbol"]), set()).add(row["file_path"])
    return {key: sorted(paths)[:limit] for key, paths in found.items()}


def load_test_sources(
    service_file_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """{(service, file_path): that test file's stored text}, empty dropped.

    Capped at MAX_TEST_SOURCE_CHARS here, at read time: since pipeline
    version 4 every file's text is stored whole. Everything a
    planner gets wrong when writing a test lives here and nowhere else: the
    complete fixture object, which props must be set for the component to
    render at all, the shape of the payload a callback receives, and whether
    the component is driven by its own state or by its parent's. A path
    alone taught it none of that, and a test written without it asserted
    against a value the component never reads.
    """
    if not service_file_pairs:
        return {}
    rows = [{"service": s, "file_path": f} for s, f in service_file_pairs]
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {service: row.service, file_path: row.file_path})
            WHERE f.source IS NOT NULL AND f.source <> ''
            RETURN row.service AS service, row.file_path AS file_path,
                   substring(f.source, 0, $cap) AS source
            """,
            rows=rows,
            cap=MAX_TEST_SOURCE_CHARS,
        )
        return {(r["service"], r["file_path"]): r["source"] for r in result}


def tests_of_sibling_components(
    service_file_pairs: list[tuple[str, str]], limit: int = 3
) -> dict[tuple[str, str], list[str]]:
    """{(service, file_path): [spec files testing components BESIDE it]}.

    The third place a test can live, and the one the other two lookups
    cannot reach. load_sibling_tests sees only the file's own folder, which
    is empty for a component that has no test yet. tests_naming matches the
    identifiers a symbol touches, which a sibling component's test never
    contains — it tests a different component.

    What a sibling test carries is the one thing neither of those gives:
    how components in THIS part of the codebase are wired up for a test.
    Observed on the ATS CC-email ticket, where the design was told to copy
    a spec from another app that renders its component through props, while
    the component being changed reads a React context. Twenty-one specs sit
    beside it under the same parent, every one of them wrapping the
    component in the context provider it actually needs.

    "Beside" means sharing the file's grandparent directory — the parent
    holds the component itself, the grandparent holds its peers.
    """
    if not service_file_pairs:
        return {}
    rows = []
    for service, path in service_file_pairs:
        parts = path.split("/")
        if len(parts) < 3:
            continue
        rows.append({"service": service, "file_path": path, "near": "/".join(parts[:-2]) + "/"})
    if not rows:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {service: row.service})
            WHERE f.file_path STARTS WITH row.near
              AND f.file_path <> row.file_path
              AND f.file_path =~ $spec_re
              AND f.source IS NOT NULL AND f.source <> ''
            RETURN row.service AS service, row.file_path AS file_path,
                   f.file_path AS test_path, size(f.source) AS size
            ORDER BY size
            """,
            rows=rows,
            spec_re=CYPHER_SPEC_FILE_RE,
        )
        found: dict[tuple[str, str], list[str]] = {}
        for r in result:
            key = (r["service"], r["file_path"])
            paths = found.setdefault(key, [])
            if len(paths) < limit:
                paths.append(r["test_path"])
    return found


def load_sibling_tests(
    service_file_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], list[str]]:
    """{(service, file_path): [test file paths in the SAME folder]}.

    Test code is deliberately absent from search_service_chunks — test
    functions echo an issue's wording closely enough to crowd real code out
    of the results (ui alone holds 1,097 test files; reports-api holds more
    test chunks than real ones). This does not relax that: it runs AFTER the
    files to change have been chosen, and only looks beside them.

    The planner otherwise proposes behaviour changes with no idea that a
    test asserts the current behaviour — observed for real, when a TDD said
    to replace ReportForm's console logging and the existing
    ReportForm/view.test.tsx failed on the next CI run.

    Only the paths are returned, never the test source: naming the file a
    coding agent must update is the actionable part, and the body would
    reintroduce exactly the crowding this avoids.
    """
    by_folder: dict[tuple[str, str], list[str]] = {}
    for service, path in service_file_pairs:
        if "/" not in path or is_test_path(path):
            continue
        by_folder.setdefault((service, path.rsplit("/", 1)[0]), []).append(path)
    if not by_folder:
        return {}

    lookups = [{"service": s, "folder": f} for s, f in by_folder]
    with get_driver().session() as session:
        rows = session.run(
            """
            UNWIND $lookups AS lookup
            MATCH (f:File {service: lookup.service})
            WHERE f.file_path STARTS WITH lookup.folder + '/'
              AND NOT f.file_path CONTAINS lookup.folder + '//'
              AND f.file_path =~ $test_re
            RETURN lookup.service AS service, lookup.folder AS folder,
                   f.file_path AS file_path
            """,
            lookups=lookups,
            test_re=CYPHER_TEST_PATH_RE,
        )
        found: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            folder = row["folder"]
            rest = row["file_path"][len(folder) + 1 :]
            if "/" in rest:  # a deeper folder, not a sibling
                continue
            found.setdefault((row["service"], folder), []).append(row["file_path"])

    siblings: dict[tuple[str, str], list[str]] = {}
    for (service, folder), paths in by_folder.items():
        tests = sorted(found.get((service, folder), []))
        for path in paths:
            siblings[(service, path)] = [t for t in tests if t != path]
    return siblings


# --- planner v2: whole files and the links between them -----------------------
#
# Everything below reads what ingestion pipeline version 4 writes: File.source,
# the IMPORTS/REFERENCES/TESTS edges, tables, operations, routes and Kafka
# messages. On an older graph these return nothing, which is why
# query_flow checks pipeline_versions() before taking the v2 path.


def _pair_rows(pairs: list[tuple[str, str]]) -> list[dict]:
    return [{"service": s, "file_path": p} for s, p in pairs]


def pipeline_versions(names: list[str]) -> dict[str, int | None]:
    with get_driver().session() as session:
        result = session.run(
            "MATCH (s:Service) WHERE s.name IN $names "
            "RETURN s.name AS name, s.pipeline_version AS version",
            names=names,
        )
        return {r["name"]: r["version"] for r in result}


def load_file_sources(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], dict]:
    """{(service, path): {"source", "line_count", "kind", "truncated"}}."""
    if not pairs:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (f:File {service: row.service, file_path: row.file_path})
            RETURN row.service AS service, row.file_path AS file_path,
                   coalesce(f.source, '') AS source, f.line_count AS line_count,
                   f.kind AS kind, coalesce(f.truncated, false) AS truncated
            """,
            rows=_pair_rows(pairs),
        )
        return {(r["service"], r["file_path"]): dict(r) for r in result}


def load_file_symbols(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """Every symbol declared in each file, in line order — the outline the
    planner picks `function_or_symbol` from."""
    if not pairs:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (n:Symbol)
            WHERE n.service = row.service AND n.file_path = row.file_path
            RETURN row.service AS service, row.file_path AS file_path,
                   n.name AS name, n.qualified_name AS qualified_name,
                   [l IN labels(n) WHERE l <> 'Symbol'][0] AS label,
                   n.symbol_kind AS symbol_kind, n.line_start AS line_start,
                   n.line_end AS line_end, n.parent_symbol AS parent_symbol
            ORDER BY n.line_start
            """,
            rows=_pair_rows(pairs),
        )
        found: dict[tuple[str, str], list[dict]] = {}
        for r in result:
            found.setdefault((r["service"], r["file_path"]), []).append(dict(r))
        return found


def imports_of(pairs: list[tuple[str, str]]) -> list[dict]:
    """The local files each file imports, with the names it takes."""
    if not pairs:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (a:File {service: row.service, file_path: row.file_path})-[r:IMPORTS]->(b:File)
            RETURN row.service AS service, a.file_path AS source, b.file_path AS target,
                   coalesce(r.names, []) AS names, b.kind AS kind
            """,
            rows=_pair_rows(pairs),
        )
        return [dict(r) for r in result]


def tests_of(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """{(service, path): test files with a TESTS edge to it}."""
    if not pairs:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (t:File)-[:TESTS]->(f:File {service: row.service, file_path: row.file_path})
            RETURN row.service AS service, row.file_path AS file_path,
                   collect(t.file_path) AS tests
            """,
            rows=_pair_rows(pairs),
        )
        return {(r["service"], r["file_path"]): sorted(r["tests"]) for r in result}


def references_from(pairs: list[tuple[str, str]]) -> list[dict]:
    """Other files of the same service whose symbols these files' symbols
    reference, most-referenced first. Reaches what an import cannot: a
    Kotlin class used from the same package needs no import at all."""
    if not pairs:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (a:Symbol)-[:REFERENCES]->(b:Symbol)
            WHERE a.service = row.service AND a.file_path = row.file_path
              AND b.file_path <> row.file_path
            RETURN row.service AS service, row.file_path AS source, b.file_path AS target,
                   count(*) AS refs, collect(DISTINCT b.name)[..8] AS names
            ORDER BY refs DESC
            """,
            rows=_pair_rows(pairs),
        )
        return [dict(r) for r in result]


def cross_service_links(pairs: list[tuple[str, str]]) -> list[dict]:
    """The files at the other end of every concrete link out of these files:
    the handlers and publishers of the Kafka messages their classes send or
    receive (and the message's Avro schema), the resolver of a GraphQL
    operation they use (or the clients of one they resolve), and the
    controller of a route they call.

    Returns [{"from_service", "from_file", "service", "file_path", "reason"}]."""
    if not pairs:
        return []
    rows = _pair_rows(pairs)
    queries = [
        # Kafka: the other side of every message, and its schema.
        """
        UNWIND $rows AS row
        MATCH (c:Class {service: row.service, file_path: row.file_path})-[r1:HANDLES|PUBLISHES]->(m:Message)
        MATCH (o:Class)-[r2:HANDLES|PUBLISHES]->(m)
        WHERE NOT (o.service = c.service AND o.file_path = c.file_path)
        RETURN row.service AS from_service, row.file_path AS from_file,
               o.service AS service, o.file_path AS file_path,
               toLower(type(r2)) + ' ' + coalesce(m.short_name, m.name) + ' (' + c.name + ' ' +
               toLower(type(r1)) + ' it)' AS reason
        """,
        """
        UNWIND $rows AS row
        MATCH (c:Class {service: row.service, file_path: row.file_path})-[:HANDLES|PUBLISHES]->(m:Message)
        WHERE m.defined_in IS NOT NULL
        RETURN row.service AS from_service, row.file_path AS from_file,
               m.defined_in_service AS service, m.defined_in AS file_path,
               'Avro schema of ' + coalesce(m.short_name, m.name) AS reason
        """,
        # GraphQL: client file -> the resolver and the schema of what it uses.
        """
        UNWIND $rows AS row
        MATCH (f:File {service: row.service, file_path: row.file_path})-[:USES_OPERATION]->(o:Operation)
        OPTIONAL MATCH (r:Symbol)-[:RESOLVES]->(o)
        WITH row, o, collect(DISTINCT r.file_path) AS resolvers
        UNWIND (resolvers + [o.file_path]) AS target
        WITH row, o, target WHERE target IS NOT NULL
        RETURN row.service AS from_service, row.file_path AS from_file,
               o.service AS service, target AS file_path,
               'serves ' + o.type + '.' + o.name + ' used here' AS reason
        """,
        # GraphQL: resolver -> the client files that use what it serves.
        """
        UNWIND $rows AS row
        MATCH (s:Symbol)-[:RESOLVES]->(o:Operation)<-[:USES_OPERATION]-(f:File)
        WHERE s.service = row.service AND s.file_path = row.file_path
        WITH row, o, collect(DISTINCT f)[..3] AS clients
        UNWIND clients AS f
        RETURN row.service AS from_service, row.file_path AS from_file,
               f.service AS service, f.file_path AS file_path,
               'uses ' + o.type + '.' + o.name + ' resolved here' AS reason
        """,
        # REST: caller -> controller.
        """
        UNWIND $rows AS row
        MATCH (f:File {service: row.service, file_path: row.file_path})-[:CALLS_ENDPOINT]->(e:Endpoint)<-[:SERVES]-(m:Symbol)
        RETURN row.service AS from_service, row.file_path AS from_file,
               m.service AS service, m.file_path AS file_path,
               'serves ' + e.method + ' ' + e.path + ' called here' AS reason
        """,
    ]
    found: list[dict] = []
    with get_driver().session() as session:
        for query in queries:
            found += [dict(r) for r in session.run(query, rows=rows)]
    return found


def tables_for(pairs: list[tuple[str, str]]) -> list[dict]:
    """Tables the symbols in these files map to, read or write, with the
    entity classes mapped to them (whose files show the column mapping)."""
    if not pairs:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (s:Symbol)-[r:READS|WRITES|MAPS_TO]->(t:Table)
            WHERE s.service = row.service AND s.file_path = row.file_path
            WITH t, collect(DISTINCT row.file_path + ' ' + toLower(type(r))) AS used_by
            OPTIONAL MATCH (e:Class)-[:MAPS_TO]->(t)
            RETURN t.service AS service, t.name AS name, t.columns AS columns,
                   t.foreign_keys AS foreign_keys, t.defined_in AS defined_in, used_by,
                   collect(DISTINCT e.file_path) AS entity_files
            """,
            rows=_pair_rows(pairs),
        )
        return [dict(r) for r in result]


def get_edges_with_evidence(names: list[str]) -> list[dict]:
    """CALLS edges among `names`, with how the services talk (`via`) and
    what proves it (`evidence`) where pipeline version 4 recorded it."""
    if not names:
        return []
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (a:Service)-[r:CALLS]->(b:Service)
            WHERE a.name IN $names AND b.name IN $names
            RETURN a.name AS source, b.name AS target,
                   coalesce(r.via, []) AS via, coalesce(r.evidence, []) AS evidence
            ORDER BY source, target
            """,
            names=names,
        )
        return [dict(r) for r in result]


def file_relevance(
    pairs: list[tuple[str, str]], vectors: list[list[float]]
) -> dict[tuple[str, str], float]:
    """{(service, path): how closely the file's best chunk matches the
    ticket} — the same best-cosine score search_service_chunks uses. Ranks
    the files a focus file imports, so the one that matters to this ticket
    (the grid rendering the button) is not cut behind thirty utilities."""
    if not pairs or not vectors:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk)
            WHERE c.service = row.service AND c.qualified_name STARTS WITH row.file_path + '::'
            WITH row, max(reduce(best = 0.0, v IN $vectors |
                 CASE WHEN vector.similarity.cosine(c.embedding, v) > best
                 THEN vector.similarity.cosine(c.embedding, v) ELSE best END)) AS score
            RETURN row.service AS service, row.file_path AS file_path, score
            """,
            rows=_pair_rows(pairs),
            vectors=vectors,
        )
        return {(r["service"], r["file_path"]): r["score"] or 0.0 for r in result}


def file_relevance_by_vector(
    pairs: list[tuple[str, str]], vectors: list[list[float]]
) -> dict[tuple[str, str], list[float]]:
    """{(service, path): [best chunk cosine against each vector]} — one score
    per part of the ticket, so each part can find its own best file. A file
    with no chunks (a test, a tiny barrel, generated code) is absent."""
    if not pairs or not vectors:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (c:Chunk)
            WHERE c.service = row.service AND c.qualified_name STARTS WITH row.file_path + '::'
            WITH row, [v IN $vectors | vector.similarity.cosine(c.embedding, v)] AS sims
            WITH row, collect(sims) AS all_sims
            RETURN row.service AS service, row.file_path AS file_path,
                   [i IN range(0, size($vectors) - 1) |
                    reduce(best = 0.0, s IN all_sims | CASE WHEN s[i] > best THEN s[i] ELSE best END)
                   ] AS scores
            """,
            rows=_pair_rows(pairs),
            vectors=vectors,
        )
        return {(r["service"], r["file_path"]): list(r["scores"]) for r in result}


def importers_of(pairs: list[tuple[str, str]], limit: int = 10) -> dict[tuple[str, str], dict]:
    """{(service, path): {"count", "files"}} — every file that imports each
    one, so the planner can see who a change to a shared file reaches
    instead of guessing (a TDD once warned that other screens pass a prop
    to a grid component, when none did)."""
    if not pairs:
        return {}
    with get_driver().session() as session:
        result = session.run(
            """
            UNWIND $rows AS row
            MATCH (a:File)-[:IMPORTS]->(b:File {service: row.service, file_path: row.file_path})
            WITH row, collect(DISTINCT a.file_path) AS importers
            RETURN row.service AS service, row.file_path AS file_path,
                   size(importers) AS count, importers[..$limit] AS files
            """,
            rows=_pair_rows(pairs),
            limit=limit,
        )
        return {
            (r["service"], r["file_path"]): {"count": r["count"], "files": sorted(r["files"])}
            for r in result
        }


# --- reads behind the planner's lookup tools (tool_loop.py) -------------------


def find_files(service: str, fragment: str, limit: int = 10) -> list[str]:
    """Files of `service` whose path contains `fragment`, shortest first —
    what a lookup of a path that does not exist (an import alias such as
    "@client/components/InfiniteScrollGrid") is answered with."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (f:File {service: $service}) WHERE f.file_path CONTAINS $fragment
            RETURN f.file_path AS file_path ORDER BY size(f.file_path), f.file_path LIMIT $limit
            """,
            service=service, fragment=fragment, limit=limit,
        )
        return [r["file_path"] for r in result]


def usages_of(service: str, symbol: str, limit: int = 30) -> list[dict]:
    """Where `symbol` of `service` is used: symbols that reference it, and
    files that import it by name. [{"service", "file_path", "by"}]."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (a:Symbol)-[:REFERENCES]->(b:Symbol {service: $service})
            WHERE b.name = $symbol OR b.qualified_name ENDS WITH '::' + $symbol
               OR b.qualified_name ENDS WITH '.' + $symbol
            RETURN DISTINCT a.service AS service, a.file_path AS file_path, a.name AS by
            UNION
            MATCH (a:File)-[r:IMPORTS]->(:File {service: $service})
            WHERE $symbol IN coalesce(r.names, [])
            RETURN DISTINCT a.service AS service, a.file_path AS file_path, 'import' AS by
            """,
            service=service, symbol=symbol,
        )
        return [dict(r) for r in result][:limit]


def tables_named(name: str, services: list[str]) -> list[dict]:
    """Tables of `services` whose name is `name` or ends with ".name", with
    the symbols that read, write or map to them."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (t:Table) WHERE t.service IN $services
              AND (toLower(t.name) = toLower($name) OR toLower(t.name) ENDS WITH '.' + toLower($name))
            OPTIONAL MATCH (s:Symbol)-[r:READS|WRITES|MAPS_TO]->(t)
            RETURN t.service AS service, t.name AS name, t.columns AS columns,
                   t.foreign_keys AS foreign_keys, t.defined_in AS defined_in,
                   collect(DISTINCT s.file_path + ' ' + toLower(type(r)))[..20] AS used_by
            """,
            name=name, services=services,
        )
        return [dict(r) for r in result]


def messages_named(name: str) -> list[dict]:
    """Kafka messages named `name` (full or short name): their schema, topic,
    and the classes that publish or handle them."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (m:Message) WHERE m.name = $name OR m.short_name = $name
            OPTIONAL MATCH (m)-[:ON_TOPIC]->(t:Topic)
            OPTIONAL MATCH (c:Class)-[r:HANDLES|PUBLISHES]->(m)
            RETURN m.name AS name, m.defined_in_service AS schema_service,
                   m.defined_in AS schema_file, collect(DISTINCT t.name) AS topics,
                   collect(DISTINCT {service: c.service, file_path: c.file_path,
                                     class: c.name, how: toLower(type(r))}) AS users
            """,
            name=name,
        )
        return [dict(r) for r in result]


def find_text(services: list[str], text: str, limit: int = 20) -> list[dict]:
    """Files of `services` whose source contains `text` exactly, with up to
    five of the lines it is on. [{"service", "file_path", "hits": [{"line",
    "text"}]}]."""
    with get_driver().session() as session:
        result = session.run(
            """
            MATCH (f:File) WHERE f.service IN $services AND f.source CONTAINS $text
            WITH f, split(f.source, '\n') AS lines
            ORDER BY f.service, f.file_path LIMIT $limit
            RETURN f.service AS service, f.file_path AS file_path,
                   [i IN range(0, size(lines) - 1) WHERE lines[i] CONTAINS $text |
                    {line: i + 1, text: lines[i]}][..5] AS hits
            """,
            services=services, text=text, limit=limit,
        )
        return [dict(r) for r in result]
