"""GitHub's REST API: the planner's only view of the code (2026-11-02).

No clone and no archive. Every fact about a repository comes from one of the
calls below, and every call after the first pins to a commit sha, so a push
in the middle of a run cannot mix two versions of the code. Built on the
service-map POC's reader, with what planning adds: pull requests, compare,
code search, the rate limit, pagination, a cap on calls per run and the
ALLOW_CODE_READS switch.

With ALLOW_CODE_READS off, `read()` refuses before any request is sent and
pull request / compare results carry no patch text, only paths, statuses and
counts. Code search returns file paths only in either mode: no text matches
are asked for.

Results live in this object for one run and are written nowhere. Error
messages name the repository, branch or file — never its contents, and never
the token. Synchronous (httpx.Client): callers run it in a thread.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from src.ingestion.secret_store import secret

from .config import Settings

_API_VERSION = "2022-11-28"
_TIMEOUT = 30.0
# One file's text is cut here: past it, a file is too large to be one
# change's context anyway, and the run's character cap is what really limits.
# A longer file keeps its start AND its end, the middle left out: a screen's
# render section is at its end (BTS-40: the Archive button's placement was in
# the part a start-only cut dropped).
MAX_FILE_CHARS = 60_000
_HEAD_SHARE = 0.6
_PAGE = 100
# 3,000 entries: GitHub's own maximum for the files of one pull request.
_MAX_PAGES = 30
# A pull request's description, and how many commit headlines a compare keeps.
_MAX_BODY_CHARS = 4_000
_MAX_COMMIT_LINES = 20
# GitHub's compare lists at most 300 files.
_COMPARE_MAX_FILES = 300
_SEARCH_RESULTS = 30
# A fenced block in a pull request's description: code, left out when code
# reads are off.
_FENCED = re.compile(r"```.*?(?:```|\Z)", re.S)


class GitHubError(Exception):
    """A GitHub call failed; the message says why in plain words."""


class GitHubStopped(GitHubError):
    """GitHub's rate limit, or this run's own call or search cap: the step stops."""


class CodeReadsOff(GitHubError):
    """ALLOW_CODE_READS is off, so no file text is fetched."""


@dataclass(frozen=True)
class RepoInfo:
    full_name: str
    default_branch: str
    private: bool


@dataclass(frozen=True)
class Tree:
    paths: tuple[str, ...]
    sizes: dict[str, int]  # bytes per path, from the tree itself: no file is read for it
    truncated: bool  # GitHub cut the list short: some files are missing from it


@dataclass(frozen=True)
class FileText:
    path: str
    text: str
    cut: bool  # longer than MAX_FILE_CHARS; its start and end kept, the middle left out


@dataclass(frozen=True)
class ChangedFile:
    path: str
    status: str  # added | removed | modified | renamed | copied | changed
    additions: int
    deletions: int
    previous_path: str | None
    patch: str | None  # None whenever code reads are off


@dataclass(frozen=True)
class PullInfo:
    number: int
    title: str
    body: str
    state: str  # open | closed
    merged: bool
    head_branch: str
    base_branch: str
    url: str


@dataclass(frozen=True)
class Comparison:
    base: str
    head: str
    ahead_by: int
    commit_lines: tuple[str, ...]  # first line of each commit message, oldest first
    files: tuple[ChangedFile, ...]
    files_cut: bool  # GitHub lists at most 300 files


class GitHubReader:
    def __init__(
        self,
        settings: Settings,
        token: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.settings = settings
        self._token = secret("GITHUB_TOKEN") if token is None else token
        self.base = settings.github_api_base
        self.calls = 0
        self.searches = 0
        self.warnings: list[str] = []
        self._client = client or httpx.Client(timeout=_TIMEOUT, follow_redirects=True)
        self._repos: dict[str, RepoInfo] = {}
        self._heads: dict[tuple[str, str], str] = {}
        self._trees: dict[tuple[str, str], Tree] = {}

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GitHubReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- one request -------------------------------------------------------------------

    def _request(
        self, path: str, what: str, params: dict | None = None, *,
        accept: str = "application/vnd.github+json", counted: bool = True,
    ) -> httpx.Response:
        """One GET. ``path`` is under the API root, or a next-page URL GitHub
        gave. ``what`` names the thing fetched, for the error message."""
        if path.startswith(("http://", "https://")):
            # A next-page link: followed only when it points back at the same
            # API, so the token is never sent anywhere else.
            if not path.startswith(self.base + "/"):
                raise GitHubError(f"{what}: GitHub pointed the next page at another server")
            url = path
        else:
            url = f"{self.base}{path}"
        if counted:
            if self.calls >= self.settings.max_github_calls:
                raise GitHubStopped(
                    f"{what}: not fetched, the run's limit of "
                    f"{self.settings.max_github_calls} GitHub calls is reached "
                    "(TDD_MAX_GITHUB_CALLS)"
                )
            self.calls += 1
        headers = {"Accept": accept, "X-GitHub-Api-Version": _API_VERSION}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        try:
            response = self._client.get(url, headers=headers, params=params)
        except httpx.HTTPError as error:
            raise GitHubError(
                f"{what}: GitHub could not be reached ({type(error).__name__})"
            ) from None
        if response.status_code >= 400:
            raise _error(response, what)
        return response

    def _get(self, path: str, what: str, params: dict | None = None, **kwargs):
        return self._request(path, what, params, **kwargs).json()

    def _pages(self, path: str, what: str) -> tuple[list[dict], bool]:
        """Every item of a paged list, and whether it was cut at _MAX_PAGES."""
        items: list[dict] = []
        url: str | None = path
        params: dict | None = {"per_page": str(_PAGE)}
        for _ in range(_MAX_PAGES):
            response = self._request(url, what, params)
            items += response.json()
            url = response.links.get("next", {}).get("url")
            params = None  # the next-page link carries its own
            if not url:
                return items, False
        return items, True

    # --- the repository and its commit ---------------------------------------------------

    def repo(self, repo: str) -> RepoInfo:
        """The repository, if the token can see it: its default branch."""
        if repo not in self._repos:
            data = self._get(f"/repos/{repo}", f"the repository {repo}")
            self._repos[repo] = RepoInfo(
                full_name=data.get("full_name") or repo,
                default_branch=data.get("default_branch") or "main",
                private=bool(data.get("private")),
            )
        return self._repos[repo]

    def head_sha(self, repo: str, branch: str) -> str:
        """The commit ``branch`` is at now; asked once per (repo, branch).

        The sha media type returns the sha alone: the full commit would also
        bring its files' patches, which is code."""
        key = (repo, branch)
        if key not in self._heads:
            response = self._request(
                f"/repos/{repo}/commits/{_q(branch)}",
                f"the branch '{branch}' of {repo}",
                accept="application/vnd.github.sha",
            )
            self._heads[key] = response.text.strip()
        return self._heads[key]

    def tree(self, repo: str, sha: str) -> Tree:
        """Every file path in the repository at ``sha``: paths only, no contents."""
        key = (repo, sha)
        if key not in self._trees:
            data = self._get(
                f"/repos/{repo}/git/trees/{sha}",
                f"the file list of {repo} at {sha[:12]}",
                {"recursive": "1"},
            )
            truncated = bool(data.get("truncated"))
            if truncated:
                self.warnings.append(
                    f"{repo}@{sha[:12]}: GitHub cut the file list short (over 100,000 "
                    "entries or 7 MB); some files are missing from it."
                )
            blobs = [i for i in data.get("tree") or [] if i.get("type") == "blob"]
            self._trees[key] = Tree(
                paths=tuple(i["path"] for i in blobs),
                sizes={i["path"]: int(i.get("size") or 0) for i in blobs},
                truncated=truncated,
            )
        return self._trees[key]

    # --- file text: only with ALLOW_CODE_READS -------------------------------------------

    def read(self, repo: str, path: str, sha: str) -> FileText:
        """One file's text at ``sha``, held in memory for this run only."""
        if not self.settings.allow_code_reads:
            raise CodeReadsOff(f"{path} in {repo}: not read (ALLOW_CODE_READS is off)")
        data = self._get(
            f"/repos/{repo}/contents/{_q(path)}", f"the file {path} in {repo}", {"ref": sha}
        )
        if isinstance(data, list) or data.get("type") != "file":
            raise GitHubError(f"{path} in {repo} is not a file")
        if data.get("encoding") != "base64":
            raise GitHubError(f"{path} in {repo} is over 1 MB, too large for GitHub's contents API")
        raw = base64.b64decode(data.get("content") or "")
        if b"\x00" in raw[:8000]:
            raise GitHubError(f"{path} in {repo} is a binary file")
        text = raw.decode("utf-8", errors="replace")
        if len(text) <= MAX_FILE_CHARS:
            return FileText(path=path, text=text, cut=False)
        head = int(MAX_FILE_CHARS * _HEAD_SHARE)
        tail = MAX_FILE_CHARS - head
        left_out = len(text) - head - tail
        kept = f"{text[:head]}\n... [{left_out:,} characters in the middle left out] ...\n"
        return FileText(path=path, text=kept + text[-tail:], cut=True)

    # --- pull requests and branches ------------------------------------------------------

    def pull(self, repo: str, number: int) -> PullInfo:
        """A pull request's title, description, state and branches."""
        data = self._get(f"/repos/{repo}/pulls/{number}", f"pull request #{number} of {repo}")
        body = data.get("body") or ""
        if not self.settings.allow_code_reads:
            body = _FENCED.sub("(code left out)", body)
        return PullInfo(
            number=number,
            title=data.get("title") or "",
            body=body[:_MAX_BODY_CHARS],
            state=data.get("state") or "",
            merged=bool(data.get("merged_at")),
            head_branch=(data.get("head") or {}).get("ref") or "",
            base_branch=(data.get("base") or {}).get("ref") or "",
            url=data.get("html_url") or "",
        )

    def pull_files(self, repo: str, number: int) -> tuple[list[ChangedFile], bool]:
        """The files a pull request changes, and whether the list was cut."""
        items, cut = self._pages(
            f"/repos/{repo}/pulls/{number}/files", f"the files of pull request #{number} of {repo}"
        )
        return [self._changed(item) for item in items], cut

    def compare(self, repo: str, base: str, head: str) -> Comparison:
        """What ``head`` changed against ``base``: for a branch with no pull request."""
        data = self._get(
            f"/repos/{repo}/compare/{_q(base)}...{_q(head)}",
            f"the changes of '{head}' against '{base}' in {repo}",
        )
        commits = data.get("commits") or []
        files = data.get("files") or []
        return Comparison(
            base=base,
            head=head,
            ahead_by=int(data.get("ahead_by") or 0),
            commit_lines=tuple(
                ((c.get("commit") or {}).get("message") or "").split("\n", 1)[0]
                for c in commits[:_MAX_COMMIT_LINES]
            ),
            files=tuple(self._changed(item) for item in files),
            files_cut=len(files) >= _COMPARE_MAX_FILES,
        )

    def _changed(self, item: dict) -> ChangedFile:
        return ChangedFile(
            path=item.get("filename") or "",
            status=item.get("status") or "",
            additions=int(item.get("additions") or 0),
            deletions=int(item.get("deletions") or 0),
            previous_path=item.get("previous_filename"),
            patch=item.get("patch") if self.settings.allow_code_reads else None,
        )

    # --- search and limits ---------------------------------------------------------------

    def search_code(self, repo: str, term: str, folder: str = "") -> list[str]:
        """Paths of the files in ``repo`` whose text contains ``term``.

        Paths only: no text matches are requested. GitHub searches the
        DEFAULT branch, so a caller checks each path against its pinned tree."""
        if self.searches >= self.settings.max_code_searches:
            raise GitHubStopped(
                f"code search for '{term}' in {repo}: not run, the run's limit of "
                f"{self.settings.max_code_searches} searches is reached (TDD_MAX_CODE_SEARCHES)"
            )
        self.searches += 1
        phrase = term.replace('"', "")
        query = f'"{phrase}" repo:{repo}'
        if folder:
            query += f" path:{folder}"
        data = self._get(
            "/search/code",
            f"code search for '{term}' in {repo}",
            {"q": query, "per_page": str(_SEARCH_RESULTS)},
        )
        return list(dict.fromkeys(i["path"] for i in data.get("items") or [] if i.get("path")))

    def rate_limit(self) -> dict[str, tuple[int, str]]:
        """{"core" | "search": (calls left, reset time)}. Free: GitHub does not
        count this call, and neither does the run's cap."""
        data = self._get("/rate_limit", "the rate limit", counted=False)
        out = {}
        for name in ("core", "search"):
            res = (data.get("resources") or {}).get(name) or {}
            out[name] = (int(res.get("remaining") or 0), _clock(res.get("reset")))
        return out


def _q(path: str) -> str:
    return quote(path, safe="/")


def _clock(epoch: object) -> str:
    try:
        return datetime.fromtimestamp(int(epoch), UTC).strftime("%H:%M UTC")
    except (TypeError, ValueError):
        return "unknown"


def _message(response: httpx.Response) -> str:
    """GitHub's own short error message, when it gave one."""
    try:
        return str(response.json().get("message") or "")[:200]
    except ValueError:
        return ""


def _error(response: httpx.Response, what: str) -> GitHubError:
    code = response.status_code
    message = _message(response)
    limited = (
        response.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in message.lower()
    )
    if code == 401:
        return GitHubError(f"{what}: GitHub rejected the token (401). Check GITHUB_TOKEN.")
    if code in (403, 429) and limited:
        retry = response.headers.get("retry-after")
        when = (
            f"try again in {retry} seconds"
            if retry
            else f"it resets at {_clock(response.headers.get('x-ratelimit-reset'))}"
        )
        return GitHubStopped(f"{what}: GitHub's rate limit is reached ({code}); {when}.")
    if code == 403:
        return GitHubError(
            f"{what}: GitHub refused access (403). "
            "The token may lack permission for this repository."
        )
    if code == 404:
        return GitHubError(
            f"{what}: not found (404). The name may be wrong, or the token cannot see "
            "this private repository."
        )
    if code == 409:
        return GitHubError(f"{what}: the repository is empty (409).")
    if code == 422:
        return GitHubError(f"{what}: GitHub could not process the request (422). {message}".strip())
    if code >= 500:
        return GitHubError(f"{what}: GitHub had a server error ({code}); try again later.")
    return GitHubError(f"{what}: GitHub answered {code}.")
