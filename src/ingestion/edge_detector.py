"""Detects real inter-service dependencies by scanning a cloned repo's own
text for OTHER known service names — the code-derived replacement for the
old md-narrative-sentence edge resolution ("calls the payment service").

Deliberately a plain text scan, not tree-sitter parsing: the signal for a
dependency (a GraphQL client's `uri`, a federation gateway's subgraph list,
a REST client's base URL) can live in ANY file — a .py/.js/.ts client call,
but just as often a .json/.yaml gateway/federation config that
code_parser.py's LANGUAGE_BY_EXTENSION never touches at all. Scanning raw
text catches all of these with one mechanism instead of needing a separate
parser per communication style.

Never guesses: a match only counts when a KNOWN service name (from the
manifest, not inferred) appears near a networking-relevant keyword — a bare
substring match on an English word like "orders" appearing incidentally in
a comment or string is not enough evidence of a real dependency.
"""

from __future__ import annotations

from pathlib import Path

# Directories never scanned — same rationale as code_parser's DEFAULT_DENYLIST:
# build/dependency artifacts, not first-party code or config.
_DENYLIST = {
    "node_modules", ".venv", "venv", "dist", "build", "target",
    "__pycache__", ".git", ".idea", ".gradle", ".mvn", "vendor",
    "bower_components",
}

# Extensions worth scanning: source files (where a client call lives) PLUS
# plain config formats (where a gateway/federation subgraph list or a base
# URL env default often lives) — deliberately broader than
# code_parser.LANGUAGE_BY_EXTENSION, which only covers tree-sitter-parseable
# source.
_SCANNABLE_EXTENSIONS = {
    ".py", ".java", ".ts", ".tsx", ".js", ".jsx", ".kt",
    ".json", ".yaml", ".yml", ".env", ".properties", ".toml", ".xml",
    ".graphqls", ".bpmn", ".avdl",  # schema/process definitions can reference other services too
    ".html", ".css", ".csv",  # no code-shaped symbols, but still worth a dependency scan
}

_MAX_FILE_SIZE_BYTES = 500_000  # a lightweight text scan, not full parsing

# A match only counts when a known service name appears within this many
# characters of one of these keywords — the networking context that turns a
# bare name mention into real evidence of a dependency.
#
# Deliberately NOT "service", "client", or "host": those false-positive
# constantly in ordinary OOP code — Spring's own `XxxService` class-naming
# convention and `...service.auth`-style package names put the literal word
# "service" next to almost every class in a typical Java app, regardless of
# any real network call. Observed in practice: a service name as generic as
# "user" matched against nearly every file in an auth service purely because
# of `UserService`/`import ...service.auth` — zero real dependencies. The
# word keywords below are all specific to describing a network address/
# protocol, which plain OOP naming conventions don't otherwise produce, and
# are token-bounded like the service name itself (so "uri" doesn't match
# inside "security"). The scheme keywords are checked as plain substrings —
# a URL scheme is inherently followed by a domain, so requiring a non-
# alphanumeric character right after "http://" would reject every real URL.
_CONTEXT_WINDOW_CHARS = 60
_CONTEXT_WORD_KEYWORDS = ("url", "uri", "endpoint", "graphql", "gateway")
_CONTEXT_SCHEME_KEYWORDS = ("http://", "https://")


def _iter_scannable_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _DENYLIST for part in path.parts):
            continue
        if path.suffix.lower() not in _SCANNABLE_EXTENSIONS:
            continue
        try:
            if path.stat().st_size > _MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _is_token_boundary(text_lower: str, start: int, length: int) -> bool:
    """True when the match at [start, start+length) isn't glued to another
    letter/digit on either side — rejects "user" inside "UsernamePassword..."
    or "userRepository" (a compound identifier, not a distinct token) while
    still accepting "user" inside "PAYMENT_SERVICE_URL" or "user-service"
    (underscore/hyphen/dot/quote/etc. are valid separators, not just
    non-alphanumerics in the strict regex \\b sense — plain \\b would itself
    reject "payment" in "PAYMENT_SERVICE_URL" since underscore counts as a
    word character in regex).
    """
    before = text_lower[start - 1] if start > 0 else ""
    after = text_lower[start + length] if start + length < len(text_lower) else ""
    return not before.isalnum() and not after.isalnum()


_URL_HOST_STOP_CHARS = set("/:'\"\\ \t\n\r?#")


def _host_of(window: str, scheme_idx: int, scheme_len: int) -> str:
    """The host component right after a scheme match — up to the next `/`,
    `:`, quote, whitespace, `?`, or `#`. `https://codeforces.com/api/user.info`
    has host `codeforces.com`; a service name appearing later in the PATH
    (like "user" here) is not evidence that repo talks to a "user" service —
    it's just a coincidental word in some unrelated third-party API's path."""
    start = scheme_idx + scheme_len
    end = start
    while end < len(window) and window[end] not in _URL_HOST_STOP_CHARS:
        end += 1
    return window[start:end]


def find_service_references_in_text(text: str, candidates: set[str]) -> set[str]:
    """Core matcher — scans one already-loaded text blob for any of
    `candidates` appearing as a standalone token near a networking keyword
    (also token-bounded, so "uri" doesn't match inside "security"). When the
    only signal is a literal URL scheme, the name must be in the URL's HOST,
    not just anywhere in it — otherwise any external API whose path happens
    to contain the name (e.g. a third-party `/api/user.info` endpoint) would
    look like a call to our own "user" service. `candidates` should already
    exclude the service doing the scanning.

    This is the entry point the INCREMENTAL update path uses directly, one
    changed file at a time (fetched via the GitHub API, no clone) — the
    per-repo scan below is just this same matcher applied to every file in
    a full clone.
    """
    if not candidates:
        return set()
    lower = text.lower()
    found: set[str] = set()
    for name in candidates:
        needle = name.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx == -1:
                break
            if not _is_token_boundary(lower, idx, len(needle)):
                start = idx + len(needle)
                continue
            window_start = max(0, idx - _CONTEXT_WINDOW_CHARS)
            window = lower[window_start : idx + len(needle) + _CONTEXT_WINDOW_CHARS]
            match_in_window = idx - window_start

            has_scheme_hit = any(
                _in_host(window, scheme, match_in_window, len(needle))
                for scheme in _CONTEXT_SCHEME_KEYWORDS
            )
            has_word_keyword = any(
                _is_token_boundary(window, m, len(keyword))
                for keyword in _CONTEXT_WORD_KEYWORDS
                for m in _find_all(window, keyword)
            )
            if has_scheme_hit or has_word_keyword:
                found.add(name)
                break
            start = idx + len(needle)
    return found


def _in_host(window: str, scheme: str, match_pos: int, match_len: int) -> bool:
    """Does the service-name match at [match_pos, match_pos+match_len) in
    `window` fall inside the host of a `scheme` occurrence in that window?"""
    for scheme_idx in _find_all(window, scheme):
        host = _host_of(window, scheme_idx, len(scheme))
        host_start = scheme_idx + len(scheme)
        host_end = host_start + len(host)
        if host_start <= match_pos and match_pos + match_len <= host_end:
            return True
    return False


def _find_all(text: str, needle: str):
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            return
        yield idx
        start = idx + len(needle)


def find_service_references(
    repo_root: Path, this_service: str, known_service_names: set[str]
) -> dict[str, set[str]]:
    """Walks every scannable file under `repo_root`, returning
    {file_path (posix, relative to repo_root): referenced service names} —
    PER FILE, not just a whole-repo aggregate, so each file's own
    dependency contribution can be stored on its File node. That's what
    lets an incremental run correctly update just the changed files'
    contributions without ever re-scanning the whole repo: the service's
    aggregate CALLS edges are the union of every File's stored references,
    recomputed from the graph itself, not from a fresh full-text scan.

    Files with zero references are simply absent from the returned dict —
    still worth iterating over all of them (no early exit), since even
    after every OTHER candidate has already matched somewhere, this file's
    OWN contribution must still be recorded correctly.
    """
    candidates = {name for name in known_service_names if name.lower() != this_service.lower()}
    if not candidates:
        return {}

    results: dict[str, set[str]] = {}
    for path in _iter_scannable_files(repo_root):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        found = find_service_references_in_text(text, candidates)
        if found:
            rel = str(path.relative_to(repo_root).as_posix())
            results[rel] = found
    return results
