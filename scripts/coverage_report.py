"""Stage 5 of ingestion: how much of every service the graph holds.

Two modes:

  python scripts/coverage_report.py
      Reads what the last ingest stored in Neo4j: per service, its files,
      code-line coverage, parse failures and unresolved imports, plus every
      kind of link inside and between services.

  python scripts/coverage_report.py --local
      Parses each folder in LOCAL_REPO_PATH in memory and runs Stages 0-4
      there — the same code the ingest runs — then reports the same table.
      Nothing is written and nothing is embedded, so it is a free dry run of
      the next ingest.

Both exit non-zero when a service falls below --min-coverage or has a file
with no record. Run from the repo root so .env is found. Never writes.
Never prints credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_env() -> None:
    env = Path(__file__).resolve().parents[1] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _row(name: str, s: dict) -> str:
    return (
        f"{name:24}{s['files']:>7}{s['percent']:>8.1f}%{s['failed']:>7}{s['missing']:>8}"
        f"{s['unresolved']:>11}{s['imports']:>9}{s['references']:>8}{s['tests']:>7}"
        f"{s['tables']:>7}{s['operations']:>5}{s['routes']:>7}{s['kafka']:>7}"
    )


_HEADER = (
    f"{'service':24}{'files':>7}{'covered':>9}{'failed':>7}{'missing':>8}{'unresolved':>11}"
    f"{'imports':>9}{'refs':>8}{'tests':>7}{'tables':>7}{'ops':>5}{'routes':>7}{'kafka':>7}"
)


def _local(services: list[str] | None) -> tuple[dict[str, dict], object]:
    from src.ingestion import service_links
    from src.ingestion.code_parser import DEFAULT_DENYLIST, parse_repo
    from src.ingestion.edge_detector import find_service_references
    from src.ingestion.links import compute_links, index_from_repo_graph
    from src.ingestion.repo_fetcher import _local_repo_paths

    paths = _local_repo_paths()
    if isinstance(paths, Path):
        paths = {paths.name: paths}
    names = sorted(paths)
    wanted = [n for n in names if not services or n in services]
    stats: dict[str, dict] = {}
    data = {
        k: []
        for k in (
            "avro",
            "kafka",
            "constants",
            "operations",
            "client_operations",
            "endpoints",
            "http_calls",
            "text_scan",
        )
    }
    for name in wanted:
        root = paths[name]
        started = time.time()
        graph = parse_repo(root, name)
        links = compute_links(index_from_repo_graph(name, graph))
        on_disk = set()
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in DEFAULT_DENYLIST]
            on_disk |= {(Path(dirpath) / f).relative_to(root).as_posix() for f in filenames}
        code = sum(c for c, _ in graph.file_coverage.values())
        covered = sum(v for _, v in graph.file_coverage.values())
        stats[name] = {
            "files": len(graph.files),
            "percent": 100 * covered / code if code else 100.0,
            "failed": graph.files_failed,
            "missing": len(on_disk - set(graph.files)),
            "unresolved": sum(len(v) for v in links.unresolved_imports.values()),
            "imports": len(links.imports),
            "references": len(links.references),
            "tests": len(links.tests),
            "tables": len(links.tables),
            "operations": len(links.operations),
            "routes": len(links.endpoints),
            "kafka": len(links.kafka),
            "symbols": Counter(s.symbol_kind for s in graph.symbols),
            "chunks": sum(len(s.chunks) for s in graph.symbols)
            + sum(len(c) for c in graph.file_chunks.values()),
            "seconds": time.time() - started,
            "unresolved_examples": Counter(
                m for v in links.unresolved_imports.values() for m in v
            ).most_common(5),
        }
        # The same rows Stage 4 reads back from Neo4j, built in memory.
        data["avro"] += [{**m, "service": name} for m in links.avro_messages]
        data["kafka"] += [
            {
                "service": name,
                "qualified_name": q,
                "handles": k.get("handles", []),
                "publishes": k.get("publishes", []),
                "publishes_built": k.get("publishes_built", []),
                "input_topics": k.get("input_topics", []),
                "output_topics": k.get("output_topics", []),
            }
            for q, k in links.kafka.items()
        ]
        for path, facts in graph.file_facts.items():
            kind = graph.files[path].kind if path in graph.files else ""
            if "kotlin_constants" in facts:
                data["constants"].append(
                    {"service": name, "file_path": path, "constants": facts["kotlin_constants"]}
                )
            if kind != "test" and facts.get("graphql_operations"):
                data["client_operations"].append(
                    {"service": name, "file_path": path, "operations": facts["graphql_operations"]}
                )
            if kind != "test" and facts.get("http_paths"):
                data["http_calls"].append(
                    {"service": name, "file_path": path, "paths": facts["http_paths"]}
                )
        data["operations"] += [
            {"service": name, "type": o.type, "name": o.name} for o in links.operations
        ]
        data["endpoints"] += [
            {"service": name, "method": m, "path": p} for _, m, p in links.endpoints
        ]
        targets = Counter(
            t for refs in find_service_references(root, name, set(names)).values() for t in refs
        )
        data["text_scan"] += [
            {"service": name, "target": t, "files": c} for t, c in targets.items()
        ]
        print(f"  parsed {name} in {stats[name]['seconds']:.0f}s", file=sys.stderr)
    return stats, service_links.compute(data)


def _graph(services: list[str] | None) -> tuple[dict[str, dict], dict]:
    from src.ingestion.graph_writer import get_driver

    stats: dict[str, dict] = {}
    with get_driver().session() as session:
        names = [
            r["name"] for r in session.run("MATCH (s:Service) RETURN s.name AS name ORDER BY name")
        ]
        for name in names:
            if services and name not in services:
                continue
            service = session.run(
                "MATCH (s:Service {name: $n}) RETURN properties(s) AS p", n=name
            ).single()["p"]
            files = session.run(
                """
                MATCH (f:File {service: $n})
                RETURN count(f) AS files,
                       sum(CASE WHEN f.parse_status = 'failed' THEN 1 ELSE 0 END) AS failed,
                       sum(size(coalesce(f.unresolved_imports, []))) AS unresolved
                """,
                n=name,
            ).single()
            edges = {
                r["t"]: r["c"]
                for r in session.run(
                    """
                    MATCH (a {service: $n})-[r]->()
                    WHERE type(r) IN ['IMPORTS', 'REFERENCES', 'TESTS', 'RESOLVES', 'SERVES',
                                      'HANDLES', 'PUBLISHES']
                    RETURN type(r) AS t, count(*) AS c
                    """,
                    n=name,
                )
            }
            tables = session.run(
                "MATCH (t:Table {service: $n}) RETURN count(t) AS c", n=name
            ).single()["c"]
            operations = session.run(
                "MATCH (o:Operation {service: $n}) RETURN count(o) AS c", n=name
            ).single()["c"]
            stats[name] = {
                "files": files["files"],
                "percent": service.get("coverage_percent") or 0.0,
                "failed": files["failed"],
                "missing": 0,
                "unresolved": files["unresolved"],
                "imports": edges.get("IMPORTS", 0),
                "references": edges.get("REFERENCES", 0),
                "tests": edges.get("TESTS", 0),
                "tables": tables,
                "operations": operations,
                "routes": edges.get("SERVES", 0),
                "kafka": edges.get("HANDLES", 0) + edges.get("PUBLISHES", 0),
                "version": service.get("pipeline_version"),
            }
        calls = [
            dict(r)
            for r in session.run(
                """
                MATCH (a:Service)-[r:CALLS]->(b:Service)
                RETURN a.name AS source, b.name AS target, r.via AS via, r.evidence AS evidence
                ORDER BY source, target
                """
            )
        ]
    return stats, {"calls": calls}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--local", action="store_true", help="dry run over LOCAL_REPO_PATH")
    parser.add_argument("--service", action="append", help="limit to this service (repeatable)")
    parser.add_argument("--min-coverage", type=float, default=99.0)
    args = parser.parse_args()
    _load_env()

    if args.local:
        stats, cross = _local(args.service)
        calls = [
            {"source": s, "target": t, "via": sorted(e["via"]), "evidence": e["evidence"]}
            for (s, t), e in sorted(cross.service_links.items())
        ]
    else:
        stats, cross = _graph(args.service)
        calls = cross["calls"]

    print(_HEADER)
    failing = []
    for name, s in stats.items():
        print(_row(name, s))
        if s["percent"] < args.min_coverage or s["missing"]:
            failing.append(name)
    if args.local:
        print("\nSymbol kinds and chunks per service:")
        for name, s in stats.items():
            top = ", ".join(f"{k} {v}" for k, v in s["symbols"].most_common(8))
            print(f"  {name}: {s['chunks']} chunks; {top}")
            if s["unresolved_examples"]:
                print(f"    unresolved imports: {s['unresolved_examples']}")
        print(
            f"\nKafka: {len(cross.handles)} handlers, {len(cross.publishes)} publishers, "
            f"{len(cross.on_topic)} message-topic links, unresolved topics: "
            f"{sorted(set(cross.unresolved_topics))[:8]}"
        )
        print(
            f"GraphQL: {len(cross.uses_operation)} operation uses, "
            f"{cross.ambiguous_operations} ambiguous"
        )
        print(f"REST: {len(cross.calls_endpoint)} route calls")
    print("\nService links (CALLS):")
    for call in calls:
        evidence = ", ".join((call.get("evidence") or [])[:4])
        print(
            f"  {call['source']:24} -> {call['target']:24} "
            f"via {call.get('via') or ['(none)']}  {evidence}"
        )
    if failing:
        print(f"\nBelow {args.min_coverage}% coverage or with files missing: {failing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
