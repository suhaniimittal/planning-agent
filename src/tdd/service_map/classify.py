"""Which files of a repo show how its service connects to others.

No LLM: only path patterns. From the recursive tree it picks four kinds of file:

  contract : openapi / swagger / asyncapi, .proto, .graphqls, .graphql, .avdl
  config   : application*.yml|properties, bootstrap*, docker-compose*, gateway/routes
  build    : build.gradle(.kts), pom.xml, package.json  (the libraries it uses)
  code     : controllers, resolvers, handlers, listeners, producers, clients, ...

Tests, generated code and vendored folders are skipped. The number of files is
capped, best first, so one run never reads a whole repo.
"""

from __future__ import annotations

import re

SKIP_DIRS = (
    "node_modules/",
    "target/",
    "build/",
    "dist/",
    ".git/",
    "vendor/",
    ".gradle/",
    "generated/",
    "__generated__/",
    "coverage/",
    ".next/",
    "out/",
)
TEST_DIRS = (
    "/test/",
    "/tests/",
    "/__tests__/",
    "/androidtest/",
    "/e2e/",
    "/cypress/",
    "/__mocks__/",
    "/mocks/",
)
TEST_NAME = re.compile(
    r"(\.|_)(spec|test|stories)\.[jt]sx?$|(Test|Tests|IT|Spec)\.(kt|java)$", re.IGNORECASE
)
CODE_EXT = (".kt", ".java", ".ts", ".tsx", ".js", ".jsx")

_CONTRACT_NAME = re.compile(r"^(openapi|swagger|asyncapi)[\w.-]*\.(ya?ml|json)$", re.IGNORECASE)
_CONTRACT_EXT = (".proto", ".graphqls", ".graphql", ".avdl")
_CONFIG_NAME = re.compile(
    r"^((application|bootstrap)[\w.-]*\.(ya?ml|properties)"
    r"|docker-compose[\w.-]*\.ya?ml"
    r"|(gateway|routes|nginx)[\w.-]*\.(ya?ml|conf|json))$",
    re.IGNORECASE,
)
_BUILD_NAME = ("build.gradle", "build.gradle.kts", "pom.xml", "package.json")
# A file whose name says it sends, receives or serves something.
_CODE_HIGH = re.compile(
    r"(controller|resolver|handler|listener|consumer|producer|client|gateway|route|router)", re.I
)
# A file in a folder like graphql/ or handlers/ matters whatever its name is.
_CODE_DIRS = re.compile(
    r"/(graphql|queries|mutations|api|apis|clients?|gateway|kafka|events|handlers?|listeners?"
    r"|consumers?|producers?|controllers?|resolvers?|routes?)/",
    re.I,
)
_CODE_LOW = re.compile(r"(api|service|queries|mutations|graphql|saga|sender|publisher)", re.I)

# Contracts, config and build files are few and always useful, so each kind has a small cap.
# Code files get whatever room is left under max_files, so --max-files really controls it.
CAPS = {"contract": 30, "config": 10, "build": 6}  # a schema is often split over many files


def _skipped(path: str) -> bool:
    padded = f"/{path.lower()}"
    return any(f"/{d}" in padded for d in SKIP_DIRS) or any(d in padded for d in TEST_DIRS)


def _kind(path: str) -> tuple[str, int] | None:
    """(kind, priority) of a path, or None when it does not matter."""
    name = path.rsplit("/", 1)[-1]
    if _skipped(path) or TEST_NAME.search(name):
        return None
    if _CONTRACT_NAME.match(name) or name.endswith(_CONTRACT_EXT):
        return "contract", 0
    if _CONFIG_NAME.match(name) and "test" not in name.lower():
        return "config", 1
    if name in _BUILD_NAME:
        return "build", 2
    if name.endswith(CODE_EXT):
        stem = name.rsplit(".", 1)[0]
        if _CODE_HIGH.search(stem) or _CODE_DIRS.search(f"/{path}"):
            return "code", 3
        if _CODE_LOW.search(stem):
            return "code", 4
    return None


def classify_with_cut(
    paths: list[str], prefix: str = "", max_files: int = 80
) -> tuple[list[tuple[str, str]], int]:
    """([(kind, path)] of the files to read, most useful first; how many candidate
    files were left out by the limit).

    ``prefix`` is the service's folder inside its repo ("" for the whole repo)."""
    prefix = prefix.strip("/")
    prefix = f"{prefix}/" if prefix else ""
    found: dict[str, list[tuple[int, int, str]]] = {k: [] for k in (*CAPS, "code")}
    for path in paths:
        if prefix and not path.startswith(prefix):
            continue
        hit = _kind(path)
        if hit:
            kind, priority = hit
            found[kind].append((priority, path.count("/"), path))
    chosen: list[tuple[str, str]] = []
    for kind in CAPS:
        for _, _, path in sorted(found[kind])[: CAPS[kind]]:
            chosen.append((kind, path))
    room = max(max_files - len(chosen), 0)
    chosen += [("code", path) for _, _, path in sorted(found["code"])[:room]]
    chosen = chosen[:max_files]
    candidates = sum(len(v) for v in found.values())
    return chosen, candidates - len(chosen)


def classify(paths: list[str], prefix: str = "", max_files: int = 80) -> list[tuple[str, str]]:
    """The files to read, most useful first, within the limit."""
    return classify_with_cut(paths, prefix, max_files)[0]
