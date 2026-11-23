"""Step 4 of the TDD flow (2026-11-02): the evidence a TDD is written from.

Gathered by code, by path only (2026-11-03: no file's contents are read),
within the run's caps on GitHub calls and code searches, in this order:

1. Reference changes. Every pull request listed for a service: its title,
   description, state and changed files, no patches. A branch listed without
   a pull request is compared against the service's branch instead.
2. Candidate files, each with a score and the reasons it was picked:
   - GitHub code search for the ticket's most distinctive terms (paths
     only), the main source: it finds a file by what it CONTAINS, which a
     path cannot ("Archive" lives in a toolbar whose name says nothing);
   - tree paths whose names match the ticket's terms;
   - files a reference pull request changed.
   Search covers the default branch only, so every hit is checked against
   the pinned tree; one not there is noted.
3. One step up: for the best leads by name, a search for their names finds
   the files that USE them — the grid or toolbar that lays a button out.
   The tests that go with each lead are listed by path.
4. Contract files (2026-11-15; before, a separate service map): each
   service's schemas and API definitions, those matching the ticket's terms
   first, and the folders that hold them. A contract's name often has none of
   the ticket's words, yet it is where a "confirm the data" step looks.

The leads go to the writer as places to choose folders from, each with why
it was found; the coding agent reads the files.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from .config import Settings
from .github import ChangedFile, GitHubError, GitHubReader, GitHubStopped, Tree
from .inputs import PlanInput
from .prepare import ReadyService

# Not source: dependencies, build output, generated code.
SKIP_DIRS = (
    "node_modules/", "target/", "build/", "dist/", ".git/", "vendor/", ".gradle/",
    "generated/", "__generated__/", "coverage/", ".next/", "out/",
)
# Tests and test data (KS-16908: client/fixtures/GetClient.ts as a step's file).
TEST_DIRS = (
    "/test/", "/tests/", "/__tests__/", "/androidtest/", "/e2e/", "/cypress/", "/__mocks__/",
    "/mocks/", "/fixtures/", "/__fixtures__/",
)
TEST_NAME = re.compile(
    r"(\.|_)(spec|test|stories)\.[jt]sx?$|(Test|Tests|IT|Spec)\.(kt|java)$", re.IGNORECASE
)
# A contract: a schema or API definition another service or the UI relies on.
_CONTRACT_NAME = re.compile(r"^(openapi|swagger|asyncapi)[\w.-]*\.(ya?ml|json)$", re.IGNORECASE)
_CONTRACT_EXT = (".proto", ".graphqls", ".graphql", ".avdl", ".avsc")
# Contract files shown per service: the matching ones, best first, and the
# folders holding contracts, most contracts first. A prompt-size guard only:
# a ticket rarely matches more.
_CONTRACTS_SHOWN = 15

# A candidate at least this strong is a real lead: its tests are listed.
STRONG = 3.0
# Scores per kind of evidence; a file found several ways adds them up.
_SEARCH_HIT = 3.0
_NAME_MATCH = 2.0
_FOLDER_MATCH = 1.0
_PR_FILE = 2.0
_USES_LEAD = 2.5
# Per extra distinct ticket term a path matches, up to _PAIR_MAX in all: a
# file matching "package" AND "template" outranks one matching only
# "template" (KS-16908: form-rendering templates outranked the Packages module).
_PAIR_BONUS = 1.0
_PAIR_MAX = 2.0
# Leads listed for the log and Orchestration, per service.
_LEADS_PER_SERVICE = 25
# A term matching more of a service's paths than this says nothing about any
# one of them ("report" in a reporting app).
_COMMON_SHARE = 0.05
_COMMON_MIN = 30
# Generated code: a type file per query, written by a tool, never the place
# of a change (BTS-40: utils/graphql/@types/ArchiveReports.ts as a lead).
_GENERATED = re.compile(r"(?:^|/)(?:@types|__generated__|generated)/|\.generated\.|\.d\.ts$", re.I)
# A monorepo's app folders sit right under one of these (apps/client, apps/internal).
_APP_ROOTS = ("apps", "packages")
# Another app's files and generated files are named to the writer only while
# the app the title names, and shared code, give fewer leads than this: ten
# is enough to plan from, and below it a lead elsewhere is better than none.
_FEW_LEADS = 10
# Searches: at most this many terms per service in step 2, and leads used
# for the step up in step 3.
_TERMS_PER_SERVICE = 4
_STEP_UP_FILES = 3
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
    # Words of how a bug is told, not of the code (BTS-40: "down", "clicks", "Process").
    "down", "clicks", "process", "processes", "status", "misclick", "misclicks", "shifts",
    "moved", "moving", "happened", "selecting", "unselect", "unselecting",
}
_IDENTIFIER = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]+)+|[a-z]+(?:_[a-z0-9]+)+)\b"
)
_QUOTED = re.compile(r"[\"'`“”‘’]([^\"'`“”‘’\n]{3,40})[\"'`“”‘’]")
# Two or three capitalised words; a hyphenated one stays whole ("Initiate
# Pre-Adverse Action", not "Initiate Pre").
_TITLE_PHRASE = re.compile(
    r"\b([A-Z][a-z]+(?:-[A-Z][a-z]+)*(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)*){1,2})\b"
)
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
    # 0 in the app the title names, 1 shared code or no app named, 2 another
    # app, 3 generated: ranked in this order before the score (2026-11-03).
    tier: int = 1
    terms: set[str] = field(default_factory=set)  # the distinct ticket terms it matched

    def add(self, score: float, why: str, term: str = "") -> None:
        self.score += score
        if why not in self.why:
            self.why.append(why)
        if term:
            self.terms.add(norm(term))

    @property
    def rank(self) -> float:
        return self.score * self.boost

    @property
    def note(self) -> str:
        return {2: "in another app than the one the title names", 3: "a generated file"}.get(
            self.tier, ""
        )


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


@dataclass
class Evidence:
    terms: list[str] = field(default_factory=list)
    term_scores: dict[str, float] = field(default_factory=dict)
    references: list[ReferenceChange] = field(default_factory=list)
    candidates: dict[str, dict[str, Candidate]] = field(default_factory=dict)
    trees: dict[str, Tree] = field(default_factory=dict)
    folders: dict[str, str] = field(default_factory=dict)
    contracts: dict[str, list[str]] = field(default_factory=dict)  # matching contract files
    contract_folders: dict[str, list[str]] = field(default_factory=dict)
    searches: list[str] = field(default_factory=list)
    not_read: list[str] = field(default_factory=list)  # errors, cut lists, searches left out
    unread_leads: list[str] = field(default_factory=list)  # the strong leads, with why each
    warnings: list[str] = field(default_factory=list)
    _folders: dict[str, set[str]] = field(default_factory=dict)

    def known(self, service: str, path: str) -> bool:
        """Is ``path`` a file of ``service`` at its pinned commit?"""
        tree = self.trees.get(service)
        return tree is not None and path in tree.sizes

    def known_folder(self, service: str, folder: str) -> bool:
        """Is ``folder`` a folder of ``service`` at its pinned commit?"""
        tree = self.trees.get(service)
        if tree is None:
            return False
        if service not in self._folders:
            folders = set()
            for path in tree.paths:
                parts = path.split("/")[:-1]
                folders.update("/".join(parts[: i + 1]) for i in range(len(parts)))
            self._folders[service] = folders
        return folder in self._folders[service]

    def ranked(self, service: str) -> list[Candidate]:
        """Best first: the app the title names, then shared code, then other
        apps, then generated files; by score within each."""
        return sorted(
            self.candidates.get(service, {}).values(), key=lambda c: (c.tier, -c.rank, c.path)
        )

    def leads(self, service: str, limit: int) -> list[Candidate]:
        """Candidates worth naming, best first. Another app's files and
        generated files only while the others are fewer than _FEW_LEADS."""
        good = [c for c in self.ranked(service) if c.score >= 1 and not is_test(c.path)]
        near = [c for c in good if c.tier < 2]
        return (near if len(near) >= _FEW_LEADS else good)[:limit]

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
        if norm(term) in _STOPWORDS:
            return
        term = _singular(term.strip())
        key = norm(term)
        if len(key) < 4 or key in _STOPWORDS:
            return
        scores[key] += weight
        shown.setdefault(key, term)

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
    ranked = [(shown[key], score) for key, score in scores.most_common(limit * 2)]
    texts = [text for text, _ in sources]
    kept = [(t, sc) for t, sc in ranked if not _only_in_phrase(t, ranked, texts)]
    # A phrase that is two terms side by side is no clue of its own:
    # "Spanish Pre-Adverse Action" is "spanish" + "Pre-Adverse Action". One
    # that adds a word ("Pre-Adverse Action" beside "Adverse Action") stays.
    keys = {norm(t) for t, _ in kept}
    kept = [
        (t, sc) for t, sc in kept
        if not any(o != t and norm(o) in norm(t)
                   and norm(t).replace(norm(o), "", 1) in keys for o, _ in kept)
    ]
    return kept[:limit]


def _only_in_phrase(term: str, ranked: list[tuple[str, float]], texts: list[str]) -> bool:
    """Is a one-word term only ever part of a phrase that is a term itself?
    Then it is the same clue counted twice: "adverse" and "action" from
    "Adverse Action", which also matched every actions/ folder (2026-11-15)."""
    if " " in term or "-" in term:
        return False
    phrases = [t for t, _ in ranked if (" " in t or "-" in t) and norm(term) in norm(t)]
    if not phrases:
        return False
    alone = re.compile(rf"\b{re.escape(term)}s?\b", re.I)
    for text in texts:
        for phrase in sorted(phrases, key=len, reverse=True):
            text = re.sub(re.escape(phrase), " ", text, flags=re.I)
        if alone.search(text):
            return False
    return True


def _singular(term: str) -> str:
    """One plain word in the singular, so "templates" and "template" are one
    term (KS-16908: each was scored, and matched, twice). Names, phrases and
    words ending in "ss", "us" or "is" ("address", "status", "analysis") are
    kept as they are."""
    if not term.isalpha() or not (term.islower() or term[1:].islower()):
        return term
    if len(term) > 4 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return term


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


def _is_contract(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return bool(_CONTRACT_NAME.match(name)) or name.endswith(_CONTRACT_EXT)


def _app_of(path: str) -> str | None:
    """The monorepo app a path is in: "src/apps/client/x.tsx" -> "client"."""
    parts = path.split("/")
    for i, part in enumerate(parts[:-2]):
        if part in _APP_ROOTS:
            return parts[i + 1]
    return None


def _named_apps(paths: tuple[str, ...], scope: set[str]) -> set[str]:
    """The repo's apps whose folder name is a word of the title ("Client UI" ->
    client), singular or plural; empty when the title names none."""
    apps = {a for a in (_app_of(p) for p in paths) if a}
    return {a for a in apps if norm(a) in scope or norm(a).removesuffix("s") in scope}


def _in_folder(paths: tuple[str, ...], folder: str) -> list[str]:
    if not folder:
        return list(paths)
    return [p for p in paths if p.startswith(folder + "/")]


# --- the gatherer -----------------------------------------------------------------------


class Gatherer:
    """Holds one run's evidence; gather() once."""

    def __init__(
        self,
        reader: GitHubReader,
        settings: Settings,
        services: list[ReadyService],
    ):
        self.reader = reader
        self.settings = settings
        self.services = services
        self.ev = Evidence()
        self.stopped = ""  # set by a rate limit or the run's call cap: no more GitHub
        self.scope: set[str] = set()  # the title's words, to rank files under a named area
        self.named_apps: dict[str, set[str]] = {}  # service -> app folders the title names

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
            self.ev.folders[svc.name] = self._folder(svc)
            usable.append(svc)
        for svc in usable:
            named = _named_apps(self.ev.trees[svc.name].paths, self.scope)
            self.named_apps[svc.name] = named
            if named:
                print(f"  [TDD] {svc.name}: the title names the app(s) {', '.join(sorted(named))}; "
                      "their files rank first")
        for svc in usable:
            self._references(svc)
        searches_each = self._search_share(len(usable))
        for svc in usable:
            self._candidates(svc, searches_each)
        for svc in usable:
            self._step_up(svc)
            self._list_tests(svc)
            self._contracts(svc)
        self._pair_bonus()
        self._note_leads()
        self.ev.warnings += [w for w in self.reader.warnings if w not in self.ev.warnings]
        leads = sum(len(self.ev.leads(svc.name, 10_000)) for svc in usable)
        print(
            f"  [TDD] evidence: {leads} leads, {len(self.ev.references)} reference changes, "
            f"{len(self.ev.not_read)} notes; {self.reader.calls} GitHub calls, "
            f"{self.reader.searches} searches"
        )
        return self.ev

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
            for f in change.files:
                if f.status != "removed" and self.ev.known(svc.name, f.path):
                    self._candidate(svc.name, f.path).add(_PR_FILE, f"changed by {change.ref}")
            self.ev.references.append(change)
            print(f"  [TDD] {svc.name}: reference {change.ref} ({change.state}), "
                  f"{len(change.files)} files")

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
                self._candidate(svc.name, p).add(_NAME_MATCH, f"its name matches '{term}'", term)
            for p in folder_hits:
                if p not in name_hits:
                    self._candidate(svc.name, p).add(
                        _FOLDER_MATCH, f"its folder matches '{term}'", term
                    )
        # Code search: the best-scored terms, a name or phrase counting half
        # again; a term all over the paths is all over the code too.
        def priority(term: str) -> float:
            named = " " in term or bool(_IDENTIFIER.fullmatch(term))
            return self.ev.term_scores.get(term, 0.0) * (1.5 if named else 1.0)

        ordered = sorted((t for t in terms if t not in too_common), key=priority, reverse=True)
        for term in ordered[:searches]:
            self._search(svc, term, _SEARCH_HIT, f"contains '{term}'", ticket_term=True)

    def _search(self, svc: ReadyService, term: str, score: float, why: str,
                exclude: str = "", ticket_term: bool = False) -> int:
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
                self._candidate(svc.name, path).add(score, why, term if ticket_term else "")
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
            per_service[path] = Candidate(
                service, path, boost=self._boost(path), tier=self._tier(service, path)
            )
        return per_service[path]

    def _boost(self, path: str) -> float:
        """x1.5 when a folder on the path is a word of the ticket's title, singular
        or plural: "Client UI ... report" -> apps/client/, modules/Reports/.
        Not for an index, main or app file: an app's root (KS-16908:
        apps/client/index.tsx is the router, not a screen)."""
        if _stem(path).lower() in _GENERIC_STEMS:
            return 1.0
        folders = {norm(part) for part in path.split("/")[:-1]}
        named = any(f in self.scope or f.removesuffix("s") in self.scope for f in folders)
        return 1.5 if named else 1.0

    def _tier(self, service: str, path: str) -> int:
        if _GENERATED.search(path):
            return 3
        named = self.named_apps.get(service)
        app = _app_of(path)
        if not named or app is None:
            return 1
        return 0 if app in named else 2

    # --- 3. one step up, and the tests --------------------------------------------------

    def _step_up(self, svc: ReadyService) -> None:
        """Who uses the best leads found: a search for each one's name."""
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
            self._search(svc, stem, _USES_LEAD, f"uses {stem}", exclude=path)
            done += 1

    def _list_tests(self, svc: ReadyService) -> None:
        """The tests that go with each strong lead: by path only."""
        base = [c.path for c in self.ev.ranked(svc.name) if c.score >= STRONG]
        stems = {norm(_stem(p)): _stem(p) for p in base if len(_stem(p)) >= 4}
        folder = self.ev.folders.get(svc.name, "")
        for path in _in_folder(self.ev.trees[svc.name].paths, folder):
            if is_test(path):
                key = norm(_stem(path))
                if key in stems:
                    self._candidate(svc.name, path).add(0.5, f"test of {stems[key]}")

    def _pair_bonus(self) -> None:
        """More for a path that matches several distinct ticket terms. A term
        inside another one it also matched ("action" in "adverseaction") is
        the same clue, not a second one."""
        for per_service in self.ev.candidates.values():
            for c in per_service.values():
                distinct = {t for t in c.terms if not any(t != o and t in o for o in c.terms)}
                if len(distinct) > 1:
                    bonus = min(_PAIR_BONUS * (len(distinct) - 1), _PAIR_MAX)
                    c.add(bonus, f"matches {len(distinct)} of the ticket's terms")

    def _folder(self, svc: ReadyService) -> str:
        """The service's folder inside its repo; the whole repo, with a
        warning, when the folder the input gives is not there."""
        folder = svc.path
        paths = self.ev.trees[svc.name].paths
        if folder and not any(p.startswith(folder + "/") for p in paths):
            self.ev.warnings.append(
                f"{svc.name}: the folder '{folder}' is not in {svc.repo} at {svc.sha[:12]}; "
                "the whole repository is used"
            )
            return ""
        return folder

    def _contracts(self, svc: ReadyService) -> None:
        """The service's contract files that match the ticket's terms, best
        first, and the folders that hold its contracts, most first."""
        folder = self.ev.folders.get(svc.name, "")
        keys = [norm(t) for t in self.ev.terms]
        found = [
            p for p in _in_folder(self.ev.trees[svc.name].paths, folder)
            if readable(p) and not is_test(p) and _is_contract(p)
        ]
        hits = sorted(
            ((sum(k in norm(p) for k in keys), p) for p in found), key=lambda h: (-h[0], h[1])
        )
        self.ev.contracts[svc.name] = [p for n, p in hits if n][:_CONTRACTS_SHOWN]
        held = Counter(p.rsplit("/", 1)[0] if "/" in p else "" for p in found)
        self.ev.contract_folders[svc.name] = [
            f for f, _ in held.most_common(_CONTRACTS_SHOWN) if f
        ]

    # --- the leads list ---------------------------------------------------------------

    def _note_leads(self) -> None:
        """Every strong lead (tests aside), with why it was found: for the log
        and Orchestration."""
        self.ev.unread_leads = []
        for svc in self.services:
            if svc.name not in self.ev.trees:
                continue
            strong = [
                c for c in self.ev.ranked(svc.name)
                if c.score >= _NAME_MATCH and not is_test(c.path)
            ]
            for c in strong[:_LEADS_PER_SERVICE]:
                why = "; ".join([*c.why, c.note] if c.note else c.why)
                self.ev.unread_leads.append(f"{svc.name}/{c.path}: {why}")
            if len(strong) > _LEADS_PER_SERVICE:
                self.ev.unread_leads.append(
                    f"{svc.name}: {len(strong) - _LEADS_PER_SERVICE} weaker leads not listed"
                )

    def _stop_if(self, error: GitHubError) -> None:
        if isinstance(error, GitHubStopped) and not self.stopped:
            self.stopped = str(error)
            self.ev.warnings.append(f"GitHub steps stopped: {error}")
