"""Neo4j reads for the query-time flow: vector search, CALLS-graph
traversal, candidate-doc loading. Counterpart to graph_writer.py's writes —
kept separate since the two paths have different callers and failure modes
(a write failure breaks ingestion; a read failure breaks a live request).

Code, signatures, docstrings, chunk text, SERVICE.md text, import lines and
test file text are stored encrypted (see src/ingestion/code_crypto.py); each
read below decrypts only the rows it returns, as it returns them.
"""

from __future__ import annotations

from src.ingestion import code_crypto
from src.ingestion.graph_writer import get_driver
from src.ingestion.test_paths import CYPHER_SPEC_FILE_RE, CYPHER_TEST_PATH_RE, is_test_path


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
            RETURN s.name AS name, s.doc AS doc
            """,
            names=names,
        )
        return [{"name": r["name"], "summary": _open_doc(dict(r))["doc"]} for r in result]


def _open_doc(row: dict) -> dict:
    return code_crypto.decrypt_row(
        row, label="Service", service=row["name"], qualified_name=row["name"]
    )


def _open_file(row: dict) -> dict:
    """A File row (service, file_path, ...) with its encrypted fields opened."""
    return code_crypto.decrypt_row(
        row, label="File", service=row["service"], qualified_name=row["file_path"]
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
    class of caveat as the
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
    whatever its score. The text is encrypted, so the second read matches
    the hashes of each identifier's words (`text_terms`), then keeps only
    the chunks whose decrypted text really contains it.

    `k` is close to a count of candidate FUNCTIONS, not of chunks: 93% of
    functions are short enough to be a single chunk, so measured on this
    graph 20 chunks yielded 17-22 distinct functions and 50 yielded ~43.
    It is also nearly free to raise — the scan already computes a score for
    every chunk in the service, and LIMIT only truncates the sorted output
    (measured: k=20 and k=300 both ~3.5s on 36k chunks). What costs is
    code_search.rank_functions' cap, which decides how much source reaches
    the prompt.

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
        opened = [_open_file(dict(r)) for r in result]
        return {(r["service"], r["file_path"]): r["imports"] for r in opened}


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

    A plain substring match, not a vector search: this runs AFTER the files to
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
    # The stored text is encrypted, so Neo4j cannot search inside it: each
    # service's spec files are read once, opened here, and matched exactly
    # as the CONTAINS in Cypher did. Bounded: a spec file stores at most
    # MAX_TEST_SOURCE_CHARS (6,000), so even ui's 1,105 specs are ~6 MB.
    services = sorted({lookup["service"] for lookup in lookups})
    with get_driver().session() as session:
        rows = session.run(
            """
            UNWIND $services AS service
            MATCH (f:File {service: service})
            WHERE f.file_path =~ $spec_re AND f.source IS NOT NULL AND f.source <> ''
            RETURN service, f.file_path AS file_path, f.source AS source
            """,
            services=services,
            spec_re=CYPHER_SPEC_FILE_RE,
        )
        specs: dict[str, list[tuple[str, str]]] = {}
        for row in rows:
            opened = _open_file(dict(row))
            specs.setdefault(opened["service"], []).append(
                (opened["file_path"], opened["source"] or "")
            )

    found: dict[tuple[str, str], set[str]] = {}
    for lookup in lookups:
        files = specs.get(lookup["service"], [])
        for term in lookup["terms"]:
            paths = [path for path, source in files if term in source]
            if paths and len(paths) <= max_files:
                found.setdefault((lookup["service"], lookup["symbol"]), set()).update(paths)
    return {key: sorted(paths)[:limit] for key, paths in found.items()}


def load_test_sources(
    service_file_pairs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """{(service, file_path): that test file's stored text}, empty dropped.

    Only test files carry `source` (see RepoGraph.file_source). Everything a
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
            RETURN row.service AS service, row.file_path AS file_path, f.source AS source
            """,
            rows=rows,
        )
        opened = [_open_file(dict(r)) for r in result]
        return {(r["service"], r["file_path"]): r["source"] for r in opened}


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
