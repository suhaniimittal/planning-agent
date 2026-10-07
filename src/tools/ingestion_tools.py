"""Tool entrypoint for the ingestion pipeline — the Aetherion binding for
`python -m src.ingestion.ingest`. Per the SDK's "no direct I/O in the agent"
rule, the actual clone/parse/Neo4j-write work stays in src/ingestion/; this
is a thin wrapper so an agent can trigger it via toolExecutor.execute().

Takes the service list as plain text, not services.yaml: the published
agent's only source of "which repos to analyze" is the `repos` field a
user typed into the UI form — no yaml file is ever read once this is
published, each run supplies its own service list as input.

Parses that text directly in this file, rather than importing
manifest.parse_service_specs: a real platform run proved the exact same
input/function that works correctly when run locally returns an empty
result once deployed, while every change in THIS file does take effect —
meaning something about the deployed build of manifest.py specifically does
not match its source. Inlining the (small, already-proven-correct) parsing
logic here removes that dependency entirely, regardless of the platform-side
cause.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from aetherion_sdk import tool

from src.ingestion.ingest import run as run_ingest

_SPEC_LINE_RE = re.compile(r"^([^:=]+?)\s*[:=]\s*([^@\s]+)(?:@(\S+))?$")


@dataclass(frozen=True)
class ServiceSpec:
    """Defined here, not imported from manifest.py — a real platform run
    proved manifest.py's own parsing function returns the wrong result once
    deployed despite being correct on disk, so this file depends on nothing
    from that module at all, not even its dataclass. ingest.run() only ever
    reads `.name`/`.repo`/`.branch` off whatever it's given — it never
    checks this is specifically manifest.ServiceSpec — so this is a safe,
    complete substitute."""

    name: str
    repo: str
    branch: str | None = None


class RepoListError(Exception):
    """Raised when `repos_text` has no valid 'name: org/repo' lines — an
    ingestion run with no valid service list has nothing to do."""


def _parse_repos_text(text: str) -> list[ServiceSpec]:
    """One line per service: 'name: org/repo' or 'name: org/repo@branch'.
    `=` also accepted as the separator, blank lines and '#'-comments
    ignored. Raises on a malformed line rather than silently skipping it —
    a typo'd repo name here means the wrong (or no) code gets analyzed."""
    specs: list[ServiceSpec] = []
    seen_names: set[str] = set()
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEC_LINE_RE.match(line)
        if not match:
            raise RepoListError(
                f"line {lineno}: expected 'name: org/repo' or "
                f"'name: org/repo@branch', got {raw_line!r}"
            )
        name, repo, branch = match.groups()
        name = name.strip()
        if name in seen_names:
            raise RepoListError(f"line {lineno}: duplicate service name {name!r}")
        seen_names.add(name)
        specs.append(ServiceSpec(name=name, repo=repo, branch=branch))

    if not specs:
        raise RepoListError("no services found — provide at least one 'name: org/repo' line")

    return specs


@tool()
async def run_ingestion(repos_text: str) -> dict:
    """Sync the services named in `repos_text` into Neo4j — one line per
    service, "name: org/repo" or "name: org/repo@branch". A full clone+parse
    the first time a service is ingested, a git-diff incremental sync (no
    clone, just the GitHub API) every run after. Only the services listed
    here are ever touched this run — any other service already in the graph
    from a previous run is left completely untouched."""
    specs = _parse_repos_text(repos_text)
    await run_ingest(specs=specs)
    return {"status": "success", "services": [s.name for s in specs]}
