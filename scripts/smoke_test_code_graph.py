"""Manual smoke test for the ingestion pipeline — bypasses the services.yaml
manifest entirely, runs the real clone -> tree-sitter parse -> Neo4j write
pipeline (or a git-diff incremental sync, on a second run) against one real
repo.

Usage:
    uv run python scripts/smoke_test_code_graph.py <org/repo> [service_name]

Example:
    uv run python scripts/smoke_test_code_graph.py pallets/flask flask
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.ingestion import graph_writer
from src.ingestion.ingest import sync_service
from src.ingestion.manifest import ServiceSpec


async def _main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <org/repo> [service_name]")
        sys.exit(1)

    slug = sys.argv[1]
    service_name = sys.argv[2] if len(sys.argv) > 2 else slug.split("/")[-1].lower()
    spec = ServiceSpec(name=service_name, repo=slug)

    print(f"=== Creating Service node: {service_name} ({slug}) ===")
    graph_writer.upsert_service_node(spec.name, spec.repo)

    print(f"=== Syncing {service_name} (full ingest on first run, incremental after) ===")
    await sync_service(spec, os.environ.get("GITHUB_TOKEN"), {service_name})

    print("\nDone. In Neo4j Browser (http://localhost:7474), try:")
    print(f"  MATCH (s:Service {{name: '{service_name}'}})-[r]->(n) RETURN s, r, n LIMIT 200")


if __name__ == "__main__":
    asyncio.run(_main())
