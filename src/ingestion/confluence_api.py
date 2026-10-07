"""Talks to Confluence Cloud's REST API to publish a TDD as a real page.

Same auth convention as the rest of this pipeline's external calls
(repo_fetcher.py/github_api.py): credentials come from the environment,
never hardcoded. Confluence Cloud uses HTTP Basic auth with the account
email + an API token (the same kind of token Jira uses — same Atlassian
account, same token, different product).
"""

from __future__ import annotations

import os

import httpx

_TIMEOUT = 30


class ConfluenceApiError(Exception):
    """Raised on any Confluence API failure — callers must catch this and
    skip publishing for this run, never let it abort TDD generation."""


def _base_url() -> str:
    base = os.environ.get("CONFLUENCE_BASE_URL")
    if not base:
        raise ConfluenceApiError("CONFLUENCE_BASE_URL is not set")
    return base.rstrip("/")


def _auth() -> tuple[str, str]:
    email = os.environ.get("CONFLUENCE_EMAIL")
    token = os.environ.get("CONFLUENCE_API_TOKEN")
    if not email or not token:
        raise ConfluenceApiError("CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN are not set")
    return email, token


def create_page(
    space_key: str,
    title: str,
    body_html: str,
    parent_id: str | None = None,
) -> str:
    """Creates a new Confluence page in `space_key` with `body_html` as its
    content, in Confluence's "storage format" (its own XHTML-like markup —
    see render.py's render_confluence()). Returns the page's real,
    clickable URL.

    Never updates an existing page — always creates a new one. A TDD is a
    point-in-time snapshot; overwriting an earlier one would silently lose
    history a reader might still want.
    """
    url = f"{_base_url()}/wiki/rest/api/content"
    payload = {
        "type": "page",
        "title": title,
        "space": {"key": space_key},
        "body": {"storage": {"value": body_html, "representation": "storage"}},
    }
    if parent_id:
        payload["ancestors"] = [{"id": parent_id}]

    try:
        resp = httpx.post(url, json=payload, auth=_auth(), timeout=_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise ConfluenceApiError(f"could not create Confluence page {title!r} in space {space_key!r}: {e}") from e

    data = resp.json()
    links = data.get("_links", {})
    webui_path = links.get("webui")
    if not webui_path:
        raise ConfluenceApiError(f"Confluence response for {title!r} had no page link: {data}")
    # Confluence's own "base" is the authoritative prefix for "webui" — NOT
    # CONFLUENCE_BASE_URL reconstructed by hand. A real run showed the two
    # can genuinely differ (this site's webui link needs a "/wiki" segment
    # that CONFLUENCE_BASE_URL alone doesn't carry), producing a dead link
    # when the wrong one was used. Falls back to CONFLUENCE_BASE_URL only if
    # Confluence's response is ever missing "base" outright.
    base = links.get("base") or _base_url()
    return f"{base}{webui_path}"
