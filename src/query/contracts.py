"""The services' API contracts, read from their repositories (2026-10-15).

A change between services must keep working for the services that call it:
the refund endpoint gaining a *required* ``amount`` breaks every caller still
sending none, the moment it is deployed. So the design is given each service's
current contracts — OpenAPI / Swagger, AsyncAPI (events) and protobuf (gRPC) —
and must change them backward-compatibly, with an order to roll the change out.

Pure: which paths are contracts, and how they read in the prompt. Fetching
lives in ``src/tools/query_tools.py``.
"""

from __future__ import annotations

import re

_CONTRACT_NAME = re.compile(r"^(openapi|swagger|asyncapi)[\w.-]*\.(ya?ml|json)$", re.IGNORECASE)
# Generated or vendored copies are not the service's own contract.
_SKIP_DIRS = ("node_modules/", "target/", "build/", "dist/", "vendor/", ".git/")

MAX_FILES_PER_SERVICE = 5
MAX_CHARS_PER_FILE = 15_000
MAX_CHARS_TOTAL = 40_000


def contract_paths(paths: list[str]) -> list[str]:
    """The contract files among a repository's paths, shortest (most central) first."""
    found = [
        p
        for p in paths
        if not any(f"/{d}" in f"/{p}" for d in _SKIP_DIRS)
        and (_CONTRACT_NAME.match(p.rsplit("/", 1)[-1]) or p.endswith(".proto"))
    ]
    return sorted(found, key=lambda p: (p.count("/"), p))[:MAX_FILES_PER_SERVICE]


def contracts_section(contracts: dict[str, list[tuple[str, str]]]) -> str:
    """The contracts as the design's prompt reads them, within the size budget."""
    parts: list[str] = []
    used = 0
    for service, files in sorted(contracts.items()):
        for path, text in files:
            body = text if len(text) <= MAX_CHARS_PER_FILE else text[:MAX_CHARS_PER_FILE] + "\n…"
            if used + len(body) > MAX_CHARS_TOTAL:
                parts.append(f"### {service}: {path}\n(not shown: over the size budget)")
                continue
            used += len(body)
            parts.append(f"### {service}: {path}\n```\n{body}\n```")
    return "\n\n".join(parts)
