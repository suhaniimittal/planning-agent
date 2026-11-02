"""Read-only health check on the Neo4j code graph.

Answers the questions that come up when a TDD looks wrong and it isn't
obvious whether the cause is ingestion, retrieval, or the prompt:
  - is every service actually in the graph, and how much of it
  - does a specific symbol exist as a parsed Function/Class node
  - does any chunk's text contain a given string (what the exact-identifier
    Cypher in graph_reader.search_service_chunks matches on)

Run from the repo root so .env is found:
    python scripts/check_graph.py
    python scripts/check_graph.py --symbol toAvro --service candidate-service
    python scripts/check_graph.py --text ".setPhone(" --service candidate-service

Never writes. Never prints credentials.
"""

from __future__ import annotations

import argparse
import os
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--service", help="limit symbol/text lookups to one service")
    parser.add_argument("--symbol", help="exact symbol name to look for, e.g. toAvro")
    parser.add_argument("--text", help="substring to look for in chunk text, e.g. .setPhone(")
    args = parser.parse_args()

    _load_env()
    missing = [v for v in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD") if not os.environ.get(v)]
    if missing:
        print(f"not set: {', '.join(missing)} — run from the repo root so .env is read")
        return 1

    from src.ingestion.graph_writer import get_driver

    print(f"connecting to {os.environ['NEO4J_URI']}\n")
    with get_driver().session() as session:
        rows = session.run(
            """
            MATCH (s:Service)
            OPTIONAL MATCH (s)<-[:BELONGS_TO]-(f:File)
            WITH s, count(DISTINCT f) AS files
            OPTIONAL MATCH (c:Chunk {service: s.name})
            WITH s, files, count(c) AS chunks
            OPTIONAL MATCH (fn:Function {service: s.name})
            RETURN s.name AS name, files, chunks, count(fn) AS functions,
                   s.summary IS NOT NULL AS has_summary
            ORDER BY name
            """
        )
        print(f"{'service':22} {'files':>7} {'funcs':>7} {'chunks':>8}  summary")
        for r in rows:
            flag = "yes" if r["has_summary"] else "NO"
            print(
                f"{r['name']:22} {r['files']:>7} {r['functions']:>7} "
                f"{r['chunks']:>8}  {flag}"
            )

        print()
        edges = session.run(
            "MATCH (a:Service)-[:CALLS]->(b:Service) RETURN a.name AS a, b.name AS b ORDER BY a, b"
        )
        found = [f"{r['a']} -> {r['b']}" for r in edges]
        print("CALLS edges:", ", ".join(found) if found else "(none)")

        if args.symbol:
            print(f"\nsymbols named {args.symbol!r}:")
            res = session.run(
                """
                MATCH (n) WHERE (n:Function OR n:Class) AND n.name = $name
                  AND ($service IS NULL OR n.service = $service)
                RETURN n.service AS service, labels(n)[0] AS kind,
                       n.qualified_name AS qualified_name, n.file_path AS file_path,
                       n.line_start AS line_start, n.line_end AS line_end
                ORDER BY service, qualified_name
                """,
                name=args.symbol,
                service=args.service,
            )
            hits = list(res)
            for r in hits:
                print(
                    f"  [{r['service']}] {r['kind']:8} {r['qualified_name']}"
                    f"  {r['file_path']}:{r['line_start']}-{r['line_end']}"
                )
            if not hits:
                print("  (none — not parsed into the graph)")

        if args.text:
            # Chunk text is encrypted, so this matches the same hashed words
            # graph_reader.search_service_chunks does (or plain text on nodes
            # written before encryption), then checks the decrypted text.
            from src.ingestion import code_crypto

            print(f"\nchunks containing {args.text!r}:")
            services = [args.service] if args.service else [
                r["name"] for r in session.run("MATCH (s:Service) RETURN s.name AS name")
            ]
            hits = []
            for service in services:
                res = session.run(
                    """
                    MATCH (c:Chunk {service: $service})
                    WHERE (size($terms) > 0
                           AND all(t IN $terms WHERE t IN coalesce(c.text_terms, [])))
                       OR (c.text_terms IS NULL AND c.text CONTAINS $text)
                    OPTIONAL MATCH (f:Function)-[:HAS_CHUNK]->(c)
                    RETURN c.qualified_name AS qualified_name, f.qualified_name AS fn,
                           c.text AS text
                    """,
                    service=service,
                    terms=code_crypto.query_terms(service, args.text),
                    text=args.text,
                )
                for r in res:
                    text = code_crypto.decrypt(
                        r["text"],
                        label="Chunk",
                        service=service,
                        qualified_name=r["qualified_name"],
                        field="text",
                    )
                    if args.text in (text or ""):
                        hits.append((service, r["fn"], len(text)))
            for service, fn, length in sorted(hits, key=lambda h: (h[0], h[1] or ""))[:25]:
                print(f"  [{service}] {fn}  ({length} chars)")
            if not hits:
                print("  (none — the exact-identifier query would find nothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
