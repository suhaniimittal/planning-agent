"""Step 4 of the TDD flow (2026-11-02): the evidence a TDD is written from.

Gathered by code, within hard caps (files per service, files in total,
characters in total, GitHub calls, code searches), in this order:

1. Reference changes. Every pull request listed for a service: its title,
   description, state and changed files; with code reads on, its patches,
   which count against the character cap. A branch listed without a pull
   request is compared against the service's branch instead.
2. Candidate files, each with a score and the reasons it was picked:
   - GitHub code search for the ticket's most distinctive terms (paths
     only), the main source: it finds a file by what it CONTAINS, which a
     path cannot ("Archive" lives in a toolbar whose name says nothing);
   - tree paths whose names match the ticket's terms;
   - files the service map points to (a route or topic named in the ticket);
   - files a reference pull request changed.
   Search covers the default branch only, so every hit is checked against
   the pinned tree; one not there is noted, never read.
3. Reading (code reads on only), best first, up to the caps, keeping a
   third of each service's files for step 4 and a few for the scout's
   picks (`fill()` reads the best leads into that room when the scout is
   skipped or fails).
4. One step up: for the best files read, a search for their names finds the
   files that USE them — the grid or toolbar that lays a button out — and
   those are read next. The tests that go with each file are listed by
   path, never read.

Every candidate not read is listed with why (a cap, an error, code reads
off, not at the pinned commit), and the TDD shows that list as "not read".
`drop()` clears every file text and patch at the end of the run.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Settings
from .github import ChangedFile, GitHubError, GitHubReader, GitHubStopped, Tree
from .inputs import PlanInput
from .prepare import ReadyService
from .service_map.classify import SKIP_DIRS, TEST_DIRS, TEST_NAME
from .service_map.slice import MapSlice

# A candidate at least this strong is a real lead: the scout call is skipped
# when every service has one.
STRONG = 3.0
# Scores per kind of evidence; a file found several ways adds them up.
_SEARCH_HIT = 3.0
_NAME_MATCH = 2.0
_FOLDER_MATCH = 1.0
_MAP_POINTER = 2.0
_PR_FILE = 2.0
_USES_READ_FILE = 2.5
# Above one search hit: the scout runs because the leads were weak.
_SCOUT_PICK = 3.5
# Unread leads listed under "not read", per service.
_NOT_READ_PER_SERVICE = 25
# A term matching more of a service's paths than this says nothing about any
# one of them ("report" in a reporting app).
_COMMON_SHARE = 0.05
_COMMON_MIN = 30
# Searches: at most this many terms per service in step 2, and files used
# for the step up in step 4.
_TERMS_PER_SERVICE = 4
_STEP_UP_FILES = 3
# Files per service kept free for the scout's picks (BTS-40: steps 3 and 4
# used the whole cap, so every pick was left unread).
_SCOUT_ROOM = 5
# A file larger than this is generated or data, not the code to change.
_MAX_FILE_BYTES = 300_000
# One patch of a reference pull request, and a file read when little of the
# character cap is left.
_MAX_PATCH_CHARS = 6_000
_MIN_USEFUL_CHARS = 2_000
_SKIP_EXT = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".svg", ".pdf", ".zip", ".jar",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mp3", ".lock", ".map", ".min.js", ".snap",
)
_SKIP_NAMES = ("package-lock.json", "yarn.lock", "pnpm-lock.yaml", "gradle.lockfile")
_GENERIC_STEMS = {
    "index", "main", "app", "utils", "util", "types", "type", "constants", "styles", "style",
    "helpers", "helper", "common", "config", "init", "__init__", "module", "setup", "base",
}
_STOPWORDS = {
    "the", "and", "for", "with", "when", "that", "this", "from", "into", "have", "been",
    "will", "would", "should", "could", "also", "only", "then", "than", "them", "they",
    "there", "their", "what", "which", "while", "where", "whatever", "being", "about",
    "after", "before", "above", "below", "under", "over", "same", "each", "every", "some",
    "more", "most", "such", "very", "just", "still", "does", "done", "doing", "make",
    "made", "added", "cause", "causes", "page", "screen", "user", "users", "issue", "ticket",
    "please", "check", "shown", "show", "shows", "displayed", "display", "appears", "appear",
    "including", "always", "until", "both", "happens", "happen", "first", "second",
    "latest", "like", "need", "needs", "want", "able", "case", "cases", "example",
    # Words every screen's code is full of: a search for them finds everything
    # (BTS-40: "selected" pulled in files that only track a selection).
    "selected", "select", "selection", "click", "clicked", "button", "buttons", "shift",
    "shifted", "space", "white", "control", "controls", "view", "views", "table", "list",
    "field", "fields", "value", "values", "data", "hide", "hidden", "open", "close", "item",
    "items", "icon", "text", "link", "row", "rows", "column", "columns", "move", "moves",
}
_IDENTIFIER = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:_[a-z0-9]+)+)\b"
)
_QUOTED = re.compile(r"[\"'`“”‘’]([^\"'`“”‘’\n]{3,40})[\"'`“”‘’]")
_TITLE_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b")
_CAPS_WORD = re.compile(r"\b[A-Z]{4,}\b")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]{3,}")


def norm(text: str) -> str:
    """Lower case, letters and digits only: "Candidate Status" -> "candidatestatus"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


@dataclass
class Candidate:
    service: str
    path: str
    score: float = 0.0
    why: list[str] = field(default_factory=list)
    # x1.5 under a folder the ticket's title names ("Client UI" -> apps/client/):
    # ranking only, never what counts as a strong lead.
    boost: float = 1.0

    def add(self, score: float, why: str) -> None:
        self.score += score
        if why not in self.why:
            self.why.append(why)

    @property
    def rank(self) -> float:
        return self.score * self.boost


@dataclass
class ReadFile:
    service: str
    path: str
    text: str
    cut: bool
    why: str


@dataclass
class ReferenceChange:
    """A pull request, or a branch with no pull request, listed as a pattern."""

    service: str
    ref: str  # "#142" or "branch fix/toolbar"
    card: str
    title: str
    body: str
    state: str
    url: str
    files: list[ChangedFile]
    files_cut: bool
    commit_lines: tuple[str, ...] = ()
    patches: dict[str, str] = field(default_factory=dict)  # path -> patch, code reads on only


@dataclass
class Evidence:
    terms: list[str] = field(default_factory=list)
    term_scores: dict[str, float] = field(default_factory=dict)
    files: list[ReadFile] = field(default_factory=list)
    references: list[ReferenceChange] = field(default_factory=list)
    candidates: dict[str, dict[str, Candidate]] = field(default_factory=dict)
    trees: dict[str, Tree] = field(default_factory=dict)
    folders: dict[str, str] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)
    not_read: list[str] = field(default_factory=list)  # errors, cut lists, searches left out
    unread_leads: list[str] = field(default_factory=list)  # strong leads the caps left unread
    warnings: list[str] = field(default_factory=list)

    def known(self, service: str, path: str) -> bool:
        """Is ``path`` a file of ``service`` at its pinned commit?"""
        tree = self.trees.get(service)
        return tree is not None and path in tree.sizes

    def was_read(self, service: str, path: str) -> bool:
        return any(f.service == service and f.path == path for f in self.files)

    def ranked(self, service: str) -> list[Candidate]:
        return sorted(
            self.candidates.get(service, {}).values(), key=lambda c: (-c.rank, c.path)
        )

    def has_strong_lead(self, service: str) -> bool:
        return any(c.score >= STRONG for c in self.candidates.get(service, {}).values())

    def chars(self) -> int:
        return sum(len(f.text) for f in self.files) + sum(
            len(p) for r in self.references for p in r.patches.values()
        )

    def drop(self) -> None:
        """Clears every file text and patch held for this run."""
        for f in self.files:
            f.text = ""
        self.files.clear()
        for ref in self.references:
            ref.patches.clear()


@dataclass
class Budget:
    max_files: int
    max_per_service: int
    max_chars: int
    files: int = 0
    chars: int = 0
    per_service: Counter = field(default_factory=Counter)

    def why_not(self, service: str, per_service_limit: int | None = None) -> str | None:
        """Why one more file of ``service`` cannot be read, or None when it can."""
        limit = per_service_limit or self.max_per_service
        if self.files >= self.max_files:
            return f"the run's cap of {self.max_files} files (TDD_MAX_FILES)"
        if self.per_service[service] >= limit:
            return (
                f"the cap of {self.max_per_service} files per service "
                "(TDD_MAX_FILES_PER_SERVICE)"
            )
        if self.max_chars - self.chars < _MIN_USEFUL_CHARS:
            return f"the run's cap of {self.max_chars:,} characters (TDD_MAX_CHARS)"
        return None

    def room(self) -> int:
        return max(self.max_chars - self.chars, 0)


# --- the ticket's terms -----------------------------------------------------------------


def clue_terms(plan: PlanInput, limit: int = 20) -> list[tuple[str, float]]:
    """[(term, score)], the ticket's most telling terms first. Names and
    quoted phrases count most, then capitalised phrases ("Candidate Status"),
    then words; each weighted by where it appears (title > description >
    attachments and answers) and how often."""
    card = plan.card
    sources = [(card.title, 3.0), (card.description, 2.0)]
    sources += [(f"{f.description}\n{f.content}", 1.0) for f in card.files]
    sources += [(" ".join([q.question, *q.answers]), 1.0) for q in plan.questions if q.answered]
    scores: Counter = Counter()
    shown: dict[str, str] = {}

    def add(term: str, weight: float) -> None:
        key = norm(term)
        if len(key) < 4 or key in _STOPWORDS:
            return
        scores[key] += weight
        shown.setdefault(key, term.strip())

    for text, weight in sources:
        for m in _IDENTIFIER.finditer(text):
            add(m.group(0), 3 * weight)
        for m in _QUOTED.finditer(text):
            add(m.group(1), 3 * weight)
        for m in _TITLE_PHRASE.finditer(text):
            words = m.group(1).split()
            if not all(w.lower() in _STOPWORDS for w in words):
                add(m.group(1), 2 * weight)
        for m in _CAPS_WORD.finditer(text):
            add(m.group(0).capitalize(), weight)
        for m in _WORD.finditer(text):
            if m.group(0).lower() not in _STOPWORDS:
                add(m.group(0).lower(), weight)
    return [(shown[key], score) for key, score in scores.most_common(limit)]


# --- paths ------------------------------------------------------------------------------


def readable(path: str) -> bool:
    lower = path.lower()
    name = lower.rsplit("/", 1)[-1]
    padded = f"/{lower}"
    return not (
        lower.endswith(_SKIP_EXT)
        or name in _SKIP_NAMES
        or any(f"/{d}" in padded for d in SKIP_DIRS)
    )


def is_test(path: str) -> bool:
    padded = f"/{path.lower()}"
    return any(d in padded for d in TEST_DIRS) or bool(TEST_NAME.search(path.rsplit("/", 1)[-1]))


def _stem(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    for ext in (".test", ".spec", ".stories"):
        name = name.replace(ext, "")
    return name.split(".", 1)[0]


def _in_folder(paths: tuple[str, ...], folder: str) -> list[str]:
    if not folder:
        return list(paths)
    return [p for p in paths if p.startswith(folder + "/")]


# --- the gatherer -----------------------------------------------------------------------


class Gatherer:
    """Holds one run's evidence and budget; gather() once, read_more() for the scout."""

    def __init__(
        self,
        reader: GitHubReader,
        settings: Settings,
        services: list[ReadyService],
        map_slice: MapSlice,
    ):
        self.reader = reader
        self.settings = settings
        self.services = services
        self.map = map_slice
        self.ev = Evidence()
        self.budget = Budget(settings.max_files, settings.max_files_per_service, settings.max_chars)
        self.stopped = ""  # set by a rate limit or the run's call cap: no more GitHub
        self.scope: set[str] = set()  # the title's words, to rank files under a named area
        self.failed: set[tuple[str, str]] = set()  # (service, path) whose read failed, noted

    # --- the steps ----------------------------------------------------------------------

    def gather(self, plan: PlanInput) -> Evidence:
        self.scope = {
            w for w in (norm(m.group(0)) for m in _WORD.finditer(plan.card.title))
            if w not in _STOPWORDS
        }
        self.ev.term_scores = dict(clue_terms(plan))
        self.ev.terms = list(self.ev.term_scores)
        print(f"  [TDD] ticket terms: {', '.join(self.ev.terms[:12])}")
        usable = []
        for svc in self.services:
            try:
                self.ev.trees[svc.name] = self.reader.tree(svc.repo, svc.sha)
            except GitHubError as error:
                self._stop_if(error)
                self.ev.warnings.append(f"{svc.name}: its file list could not be read: {error}")
                continue
            self.ev.folders[svc.name] = self.map.folders.get(svc.name, svc.path)
            usable.append(svc)
        for svc in usable:
            self._references(svc)
        searches_each = self._search_share(len(usable))
        for svc in usable:
            self._candidates(svc, searches_each)
        limit = self.settings.max_files_per_service - self._scout_room()
        for svc in usable:
            reserve = max(1, limit // 3)
            self._read_best(svc, limit - reserve)
        for svc in usable:
            self._step_up(svc)
        for svc in usable:
            self._read_best(svc, limit)
            self._list_tests(svc)
        self._note_unread()
        self.ev.warnings += [w for w in self.reader.warnings if w not in self.ev.warnings]
        print(
            f"  [TDD] evidence: {len(self.ev.files)} files read, {self.budget.chars:,} chars, "
            f"{len(self.ev.references)} reference changes, {len(self.ev.not_read)} not read; "
            f"{self.reader.calls} GitHub calls, {self.reader.searches} searches"
        )
        return self.ev

    def _scout_room(self) -> int:
        """Files per service left for the scout: none when it cannot run."""
        if not (self.settings.scout and self.settings.allow_code_reads):
            return 0
        return min(_SCOUT_ROOM, self.settings.max_files_per_service // 3)

    def fill(self) -> int:
        """Reads the best unread leads into the room kept for the scout, up to
        the full caps; after the scout's own reads there is usually none left.
        Returns how many files it read."""
        before = len(self.ev.files)
        for svc in self.services:
            if svc.name in self.ev.trees:
                self._read_best(svc, self.settings.max_files_per_service)
        self._note_unread()
        return len(self.ev.files) - before

    def good_enough(self) -> bool:
        """Every service has a strong lead (and, with code reads on, read it):
        the scout call would add little."""
        for svc in self.services:
            if svc.name not in self.ev.trees:
                continue
            if not self.ev.has_strong_lead(svc.name):
                return False
            if self.settings.allow_code_reads and not any(
                f.service == svc.name for f in self.ev.files
            ):
                return False
        return True

    def read_more(self, picks: list[dict]) -> list[str]:
        """The scout's picks ({service, path, why}): checked against the pinned
        tree, added as candidates, and read within the same caps. Returns a
        note per pick that was refused."""
        refused = []
        by_name = {svc.name: svc for svc in self.services}
        touched = set()
        for pick in picks:
            service = str(pick.get("service") or "").strip()
            path = str(pick.get("path") or "").strip().lstrip("/")
            if not self.ev.known(service, path) and path.startswith(service + "/"):
                path = path[len(service) + 1 :]  # written as the prompt shows it: ui/src/...
            svc = by_name.get(service)
            if svc is None or not self.ev.known(service, path):
                refused.append(f"scout picked {service}/{path}, which is not a file of this run")
                continue
            self._candidate(service, path).add(
                _SCOUT_PICK, f"picked by the scout: {str(pick.get('why') or '')[:120]}"
            )
            touched.add(service)
        for svc in self.services:
            if svc.name in touched:
                self._read_best(svc, self.settings.max_files_per_service)
        self._note_unread()
        return refused

    # --- 1. reference changes -----------------------------------------------------------

    def _references(self, svc: ReadyService) -> None:
        for ref in svc.prs:
            if self.stopped:
                self.ev.not_read.append(
                    f"{svc.name}: reference changes not fetched: {self.stopped}"
                )
                return
            try:
                if ref.pr:
                    info = self.reader.pull(svc.repo, ref.pr)
                    files, cut = self.reader.pull_files(svc.repo, ref.pr)
                    change = ReferenceChange(
                        service=svc.name, ref=f"#{ref.pr}", card=ref.card,
                        title=info.title, body=info.body,
                        state="merged" if info.merged else info.state,
                        url=info.url, files=files, files_cut=cut,
                    )
                elif ref.branch:
                    cmp = self.reader.compare(svc.repo, svc.branch, ref.branch)
                    change = ReferenceChange(
                        service=svc.name, ref=f"branch {ref.branch}", card=ref.card,
                        title=f"{cmp.ahead_by} commit(s) on {ref.branch} not on {svc.branch}",
                        body="", state="branch", url="", files=list(cmp.files),
                        files_cut=cmp.files_cut, commit_lines=cmp.commit_lines,
                    )
                else:
                    self.ev.warnings.append(
                        f"{svc.name}: a reference with neither a pull request number nor a "
                        "branch was ignored"
                    )
                    continue
            except GitHubError as error:
                self._stop_if(error)
                self.ev.warnings.append(f"{svc.name}: reference skipped: {error}")
                continue
            if change.files_cut:
                self.ev.not_read.append(
                    f"{svc.name}: {change.ref} changes more files than GitHub lists; "
                    "only the first ones are known"
                )
            self._keep_patches(change)
            for f in change.files:
                if f.status != "removed" and self.ev.known(svc.name, f.path):
                    self._candidate(svc.name, f.path).add(_PR_FILE, f"changed by {change.ref}")
            self.ev.references.append(change)
            print(f"  [TDD] {svc.name}: reference {change.ref} ({change.state}), "
                  f"{len(change.files)} files")

    def _keep_patches(self, change: ReferenceChange) -> None:
        """With code reads on, the patches of a reference change, within the
        character cap: the service's own folder first, tests last."""
        if not self.settings.allow_code_reads:
            return
        folder = self.ev.folders.get(change.service, "")
        ordered = sorted(
            (f for f in change.files if f.patch),
            key=lambda f: (bool(folder) and not f.path.startswith(folder + "/"),
                           is_test(f.path), f.path),
        )
        for f in ordered:
            if not readable(f.path):
                continue
            if self.budget.room() < _MIN_USEFUL_CHARS:
                self.ev.not_read.append(
                    f"{change.service}: patch of {f.path} in {change.ref}: the run's character "
                    "cap (TDD_MAX_CHARS)"
                )
                continue
            patch = f.patch[: min(_MAX_PATCH_CHARS, self.budget.room())]
            change.patches[f.path] = patch
            self.budget.chars += len(patch)

    # --- 2. candidates ------------------------------------------------------------------

    def _search_share(self, services: int) -> int:
        """Code searches each service may use in step 2; the rest is kept for the step up."""
        if not services:
            return 0
        first_round = max(1, (self.settings.max_code_searches * 3) // 5)
        return max(1, min(_TERMS_PER_SERVICE, first_round // services))

    def _candidates(self, svc: ReadyService, searches: int) -> None:
        folder = self.ev.folders.get(svc.name, "")
        paths = [p for p in _in_folder(self.ev.trees[svc.name].paths, folder) if readable(p)]
        terms = self.ev.terms
        # Path names.
        common = max(_COMMON_MIN, int(len(paths) * _COMMON_SHARE))
        too_common = set()
        for term in terms:
            key = norm(term)
            name_hits = [p for p in paths if key in norm(_stem(p))]
            folder_hits = [p for p in paths if key in norm(p.rsplit("/", 1)[0])]
            if len(name_hits) + len(folder_hits) > common:
                too_common.add(term)  # says nothing about any one file
                continue
            for p in name_hits:
                self._candidate(svc.name, p).add(_NAME_MATCH, f"its name matches '{term}'")
            for p in folder_hits:
                if p not in name_hits:
                    self._candidate(svc.name, p).add(_FOLDER_MATCH, f"its folder matches '{term}'")
        # The service map.
        for what, path in self.map.pointers(svc.name):
            hits = [t for t in terms if norm(t) in norm(what)]
            if hits and self.ev.known(svc.name, path):
                self._candidate(svc.name, path).add(
                    _MAP_POINTER, f"the service map has '{what.strip()}' there"
                )
        # Code search: the best-scored terms, a name or phrase counting half
        # again; a term all over the paths is all over the code too.
        def priority(term: str) -> float:
            named = " " in term or bool(_IDENTIFIER.fullmatch(term))
            return self.ev.term_scores.get(term, 0.0) * (1.5 if named else 1.0)

        ordered = sorted((t for t in terms if t not in too_common), key=priority, reverse=True)
        for term in ordered[:searches]:
            self._search(svc, term, _SEARCH_HIT, f"contains '{term}'")

    def _search(self, svc: ReadyService, term: str, score: float, why: str,
                exclude: str = "") -> int:
        if self.stopped:
            return 0
        folder = self.ev.folders.get(svc.name, "")
        try:
            found = self.reader.search_code(svc.repo, term, folder)
        except GitHubError as error:
            self._stop_if(error)
            self.ev.warnings.append(f"{svc.name}: code search for '{term}' failed: {error}")
            return 0
        kept = missing = 0
        for path in found:
            if path == exclude:
                continue
            if not self.ev.known(svc.name, path):
                missing += 1
            elif readable(path):
                self._candidate(svc.name, path).add(score, why)
                kept += 1
        if missing:
            self.ev.not_read.append(
                f"{svc.name}: {missing} file(s) code search found for '{term}' (it searches the "
                f"default branch) are not at the pinned commit {svc.sha[:12]}"
            )
        self.ev.searches.append(f"{svc.name}: '{term}' -> {kept} files")
        return kept

    def _candidate(self, service: str, path: str) -> Candidate:
        per_service = self.ev.candidates.setdefault(service, {})
        if path not in per_service:
            per_service[path] = Candidate(service, path, boost=self._boost(path))
        return per_service[path]

    def _boost(self, path: str) -> float:
        """x1.5 when a folder on the path is a word of the ticket's title, singular
        or plural: "Client UI ... report" -> apps/client/, modules/Reports/."""
        folders = {norm(part) for part in path.split("/")[:-1]}
        named = any(f in self.scope or f.removesuffix("s") in self.scope for f in folders)
        return 1.5 if named else 1.0

    # --- 3. reading ---------------------------------------------------------------------

    def _read_best(self, svc: ReadyService, limit: int) -> None:
        """Reads the best unread candidates (tests aside) until ``limit`` files
        of this service are read or a cap is reached. Code reads off: nothing."""
        if not self.settings.allow_code_reads or self.stopped:
            return
        sizes = self.ev.trees[svc.name].sizes
        for candidate in self.ev.ranked(svc.name):
            if self.ev.was_read(svc.name, candidate.path) or is_test(candidate.path):
                continue
            if (svc.name, candidate.path) in self.failed:
                continue
            if sizes.get(candidate.path, 0) > _MAX_FILE_BYTES:
                continue  # noted as too large in _note_unread
            if self.budget.why_not(svc.name, limit):
                return
            try:
                got = self.reader.read(svc.repo, candidate.path, svc.sha)
            except GitHubError as error:
                self._stop_if(error)
                if self.stopped:
                    return  # not noted as failed: _note_unread says why it was not read
                self.failed.add((svc.name, candidate.path))
                self.ev.not_read.append(f"{svc.name}/{candidate.path}: {error}")
                continue
            text = got.text[: self.budget.room()]
            self.ev.files.append(
                ReadFile(
                    service=svc.name, path=candidate.path, text=text,
                    cut=got.cut or len(text) < len(got.text), why="; ".join(candidate.why),
                )
            )
            self.budget.files += 1
            self.budget.per_service[svc.name] += 1
            self.budget.chars += len(text)

    # --- 4. one step up, and the tests --------------------------------------------------

    def _step_up(self, svc: ReadyService) -> None:
        """Who uses the best files found: a search for each one's name."""
        if self.settings.allow_code_reads:
            base = [f.path for f in self.ev.files if f.service == svc.name]
        else:
            base = [
                c.path for c in self.ev.ranked(svc.name)
                if c.score >= _NAME_MATCH and not is_test(c.path)
            ]
        done = 0
        for path in base:
            if done == _STEP_UP_FILES or self.stopped:
                break
            stem = _stem(path)
            if len(stem) < 5 or stem.lower() in _GENERIC_STEMS:
                continue
            if self.reader.searches >= self.settings.max_code_searches:
                self.ev.not_read.append(
                    f"{svc.name}: files that use {stem} not searched: the run's search cap "
                    "(TDD_MAX_CODE_SEARCHES)"
                )
                break
            self._search(svc, stem, _USES_READ_FILE, f"uses {stem}", exclude=path)
            done += 1

    def _list_tests(self, svc: ReadyService) -> None:
        """The tests that go with each file read (or, code reads off, each lead):
        by path only, never read."""
        if self.settings.allow_code_reads:
            base = [f.path for f in self.ev.files if f.service == svc.name]
        else:
            base = [c.path for c in self.ev.ranked(svc.name) if c.score >= STRONG]
        stems = {norm(_stem(p)): _stem(p) for p in base if len(_stem(p)) >= 4}
        folder = self.ev.folders.get(svc.name, "")
        for path in _in_folder(self.ev.trees[svc.name].paths, folder):
            if is_test(path):
                key = norm(_stem(path))
                if key in stems:
                    self._candidate(svc.name, path).add(0.5, f"test of {stems[key]}")

    # --- the not-read list --------------------------------------------------------------

    def _note_unread(self) -> None:
        """Every strong candidate not read, with why. Rebuilt each time so a
        later read (the scout's) takes a file off it."""
        self.ev.unread_leads = []
        for svc in self.services:
            if svc.name not in self.ev.trees:
                continue
            sizes = self.ev.trees[svc.name].sizes
            unread = [
                c for c in self.ev.ranked(svc.name)
                if c.score >= _NAME_MATCH
                and not self.ev.was_read(svc.name, c.path)
                and not is_test(c.path)  # tests are listed, never read
                and (svc.name, c.path) not in self.failed  # already noted with its error
            ]
            if len(unread) > _NOT_READ_PER_SERVICE:
                self.ev.unread_leads.append(
                    f"{svc.name}: {len(unread) - _NOT_READ_PER_SERVICE} weaker "
                    "leads not listed"
                )
            for c in unread[:_NOT_READ_PER_SERVICE]:
                if not self.settings.allow_code_reads:
                    why = "code reads are off (ALLOW_CODE_READS)"
                elif sizes.get(c.path, 0) > _MAX_FILE_BYTES:
                    why = f"over {_MAX_FILE_BYTES // 1000} KB, likely generated"
                elif self.stopped:
                    why = self.stopped
                else:
                    why = self.budget.why_not(svc.name) or "ranked below the files read"
                self.ev.unread_leads.append(f"{svc.name}/{c.path}: {why}")

    def _stop_if(self, error: GitHubError) -> None:
        if isinstance(error, GitHubStopped) and not self.stopped:
            self.stopped = str(error)
            self.ev.warnings.append(f"GitHub steps stopped: {error}")
