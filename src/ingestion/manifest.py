"""Two independent sources of "which services exist and which repo to
clone for each," both producing the same ServiceSpec shape:

- load_services(): reads services.yaml — a LOCAL/DEV convenience only
  (the CLI entry point at the bottom of ingest.py, ad-hoc scripts). The
  published agent never calls this.
- parse_service_specs(): reads the `repos` text a user typed into the
  Aetherion UI form at run time — this is the published agent's own,
  only source of truth. No yaml file is ever read once this project is
  published; each run supplies its own service list as input.

Pure I/O + parsing either way, no Neo4j/GitHub calls here.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# src/ingestion/manifest.py -> src/ingestion -> src -> repo root
_DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "services.yaml"


class ManifestError(Exception):
    """Raised when services.yaml is missing, unreadable, or malformed —
    callers should let this abort the run: an ingestion run with no valid
    service list has nothing to do."""


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    repo: str
    branch: str | None = None


def _manifest_path() -> Path:
    override = os.environ.get("SERVICES_MANIFEST_PATH")
    return Path(override) if override else _DEFAULT_MANIFEST_PATH


def load_services(path: str | Path | None = None) -> list[ServiceSpec]:
    """Parses services.yaml into a list of ServiceSpec, in file order.

    `path` overrides the default (repo root's services.yaml, or
    SERVICES_MANIFEST_PATH if set) — mainly for tests.
    """
    manifest_path = Path(path) if path is not None else _manifest_path()

    try:
        raw_text = manifest_path.read_text()
    except OSError as e:
        raise ManifestError(f"could not read manifest at {manifest_path}: {e}") from e

    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as e:
        raise ManifestError(f"{manifest_path} is not valid YAML: {e}") from e

    if not isinstance(data, dict) or "services" not in data:
        raise ManifestError(f"{manifest_path} must have a top-level 'services' list")

    entries = data["services"]
    if not isinstance(entries, list):
        raise ManifestError(f"{manifest_path}: 'services' must be a list")

    specs: list[ServiceSpec] = []
    seen_names: set[str] = set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ManifestError(f"{manifest_path}: services[{i}] must be a mapping")
        name = entry.get("name")
        repo = entry.get("repo")
        if not name or not isinstance(name, str):
            raise ManifestError(f"{manifest_path}: services[{i}] is missing a string 'name'")
        if not repo or not isinstance(repo, str):
            raise ManifestError(f"{manifest_path}: services[{i}] ({name!r}) is missing a string 'repo'")
        if name in seen_names:
            raise ManifestError(f"{manifest_path}: duplicate service name {name!r}")
        seen_names.add(name)
        specs.append(ServiceSpec(name=name, repo=repo, branch=entry.get("branch")))

    return specs


# One service per line: "name: org/repo" or "name: org/repo@branch".
# `=` also accepted as the separator, blank lines and "#"-comments ignored —
# forgiving of the small formatting differences a person typing into a UI
# textarea will actually produce.
_SPEC_LINE_RE = re.compile(r"^([^:=]+?)\s*[:=]\s*([^@\s]+)(?:@(\S+))?$")


def parse_service_specs(text: str) -> list[ServiceSpec]:
    """Parses the `repos` field a user typed into the Aetherion UI form —
    the published agent's ENTIRE source of "which services exist" for this
    run. Raises ManifestError on a malformed line rather than silently
    skipping it: a typo'd repo name here means the wrong (or no) code gets
    analyzed, which must fail loudly, not quietly ingest nothing.
    """
    specs: list[ServiceSpec] = []
    seen_names: set[str] = set()
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SPEC_LINE_RE.match(line)
        if not match:
            raise ManifestError(
                f"line {lineno}: expected 'name: org/repo' or "
                f"'name: org/repo@branch', got {raw_line!r}"
            )
        name, repo, branch = match.groups()
        name = name.strip()
        if name in seen_names:
            raise ManifestError(f"line {lineno}: duplicate service name {name!r}")
        seen_names.add(name)
        specs.append(ServiceSpec(name=name, repo=repo, branch=branch))

    if not specs:
        raise ManifestError("no services found — provide at least one 'name: org/repo' line")

    return specs
