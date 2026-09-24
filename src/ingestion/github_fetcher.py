"""Fetches a service's documentation.md straight from its GitHub repo.

No local file is ever written — the content is read into memory and handed
directly to the extraction functions.
"""

from __future__ import annotations

import base64
import os

import httpx


def fetch_doc(owner: str, repo: str, path: str = "documentation.md", branch: str = "master") -> str:
    """Fetch one file's raw text content via the GitHub Contents API."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"  # optional — public repos work without it

    resp = httpx.get(url, headers=headers, params={"ref": branch}, timeout=30)
    resp.raise_for_status()
    content_b64 = resp.json()["content"]
    return base64.b64decode(content_b64).decode("utf-8")
