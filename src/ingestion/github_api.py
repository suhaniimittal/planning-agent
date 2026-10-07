"""Talks to the GitHub REST API for the git-diff incremental update path:
which commit a repo's branch is at right now, which files changed between
two commits, and one changed file's content — all WITHOUT a clone, so an
already-ingested service is never cloned again.

Same auth convention as repo_fetcher.py's clone URL: GITHUB_TOKEN, optional
for public repos.
"""

from __future__ import annotations

import base64
import os

import httpx

_API_BASE = "https://api.github.com"
_TIMEOUT = 30


class GitHubApiError(Exception):
    """Raised on any GitHub API failure — callers must catch this and skip
    this service's sync for the run, never let it abort the whole run."""


def _headers(token: str | None) -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = token if token is not None else os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def get_remote_head_sha(repo: str, branch: str | None = None, token: str | None = None) -> str:
    """The current HEAD commit sha of `repo`'s `branch`, or its default
    branch when `branch` is None — a single lightweight API call, no clone."""
    ref = branch or "HEAD"
    url = f"{_API_BASE}/repos/{repo}/commits/{ref}"
    try:
        resp = httpx.get(url, headers=_headers(token), timeout=_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["sha"]
    except httpx.HTTPError as e:
        raise GitHubApiError(f"could not resolve HEAD for {repo}@{ref}: {e}") from e


def get_changed_files(repo: str, base: str, head: str, token: str | None = None) -> list[dict]:
    """GitHub's Compare API: exactly which files differ between `base` and
    `head` — the "ask GitHub directly which files changed" step, no local
    diff/clone needed.

    Returns [{"path", "status", "sha", "previous_path"}, ...]. `status` is
    normalized to one of "added"/"removed"/"renamed"/"modified" (GitHub's
    own "copied"/"changed"/"unchanged" statuses collapse to "modified" here
    — this pipeline only needs to decide "reparse" vs "delete" vs "just
    relabel"). `sha` is the file's blob sha AT `head` (None for a removed
    file). `previous_path` is set only for a rename.

    Known limitation: GitHub's compare API paginates at 300 changed files
    per response; a diff larger than that returns only the first page here.
    """
    url = f"{_API_BASE}/repos/{repo}/compare/{base}...{head}"
    try:
        resp = httpx.get(url, headers=_headers(token), timeout=_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GitHubApiError(f"could not compare {base}...{head} for {repo}: {e}") from e

    data = resp.json()
    files = []
    for f in data.get("files", []):
        status = f["status"]
        if status not in ("added", "removed", "renamed"):
            status = "modified"
        files.append(
            {
                "path": f["filename"],
                "status": status,
                "sha": f.get("sha"),
                "previous_path": f.get("previous_filename"),
            }
        )
    return files


def fetch_file_content(repo: str, path: str, ref: str, token: str | None = None) -> str:
    """Raw text content of one file at `ref`, via GitHub's Contents API —
    fetches exactly this one file, never the whole repo."""
    url = f"{_API_BASE}/repos/{repo}/contents/{path}"
    try:
        resp = httpx.get(url, headers=_headers(token), params={"ref": ref}, timeout=_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GitHubApiError(f"could not fetch {repo}/{path}@{ref}: {e}") from e
    content_b64 = resp.json()["content"]
    return base64.b64decode(content_b64).decode("utf-8")


def create_pull_request(
    repo: str,
    head: str,
    base: str,
    title: str,
    body: str,
    token: str | None = None,
) -> str:
    """Opens a PR from `head` (the branch the coding agent just pushed) into
    `base` (the service's default/manifest branch) — never a merge, always a
    PR for human review. Returns the PR's html_url."""
    url = f"{_API_BASE}/repos/{repo}/pulls"
    payload = {"title": title, "body": body, "head": head, "base": base}
    try:
        resp = httpx.post(url, headers=_headers(token), json=payload, timeout=_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GitHubApiError(f"could not open PR for {repo} {head}->{base}: {e}") from e
    return resp.json()["html_url"]


def download_zipball(repo: str, ref: str, token: str | None = None, timeout: int = 120) -> bytes:
    """The repo's tracked files at `ref` (a commit sha or branch) as zip
    bytes — replaces `git clone`, since the platform container has no `git`
    binary. GitHub answers with a redirect to the archive."""
    url = f"{_API_BASE}/repos/{repo}/zipball/{ref}"
    try:
        resp = httpx.get(url, headers=_headers(token), timeout=timeout, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise GitHubApiError(f"could not download {repo}@{ref}: {e}") from e
    return resp.content
