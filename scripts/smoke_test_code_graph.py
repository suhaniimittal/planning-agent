"""Manual smoke test for the code-graph feature — bypasses S3/OpenAI entirely.

Creates a minimal Service node (so CONTAINS has a target), then runs the real
clone -> tree-sitter parse -> Neo4j write pipeline against one real repo.

Usage:
    uv run python scripts/smoke_test_code_graph.py <org/repo> [service_name]

Example:
    uv run python scripts/smoke_test_code_graph.py pallets/flask flask
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from src.ingestion import graph_writer
from src.ingestion.ingest import build_code_graph_for_service


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <org/repo> [service_name]")
        sys.exit(1)

    slug = sys.argv[1]
    service_name = sys.argv[2] if len(sys.argv) > 2 else slug.split("/")[-1].lower()

    print(f"=== Creating minimal Service node: {service_name} ===")
    graph_writer.upsert_service_node(
        {
            "name": service_name,
            "summary": f"Smoke-test placeholder for {slug}",
            "summary_is_fallback": True,
            "embedding": [],
            "full_doc": f"Generated from `{slug}` at `smoke-test` — 0 source files, 0 symbols.",
            "domain": None,
            "entry_points": None,
            "external_systems": None,
            "key_symbols": [],
        }
    )

    print(f"=== Building code graph for {service_name} ({slug}) ===")
    build_code_graph_for_service(
        service_name,
        f"Generated from `{slug}` at `smoke-test` — 0 source files, 0 symbols.",
        None,  # GITHUB_TOKEN picked up from env automatically if set
    )

    print("\nDone. In Neo4j Browser (http://localhost:7474), try:")
    print(f"  MATCH (s:Service {{name: '{service_name}'}})-[r]->(n) RETURN s, r, n LIMIT 200")


if __name__ == "__main__":
    main()
