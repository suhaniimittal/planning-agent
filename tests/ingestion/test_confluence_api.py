import httpx
import pytest

from src.ingestion import confluence_api


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


def _set_env(monkeypatch, **overrides):
    defaults = dict(
        CONFLUENCE_BASE_URL="https://example.atlassian.net",
        CONFLUENCE_EMAIL="a@example.com",
        CONFLUENCE_API_TOKEN="tok",
    )
    defaults.update(overrides)
    for key, value in defaults.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def test_create_page_returns_full_url_using_confluences_own_base(monkeypatch):
    """Regression test: a real run showed Confluence's "webui" link can need
    a "/wiki" segment that plain CONFLUENCE_BASE_URL concatenation doesn't
    supply — producing a dead link. The correct prefix is always
    Confluence's own "_links.base", even when it differs from
    CONFLUENCE_BASE_URL (as it genuinely does here)."""
    _set_env(monkeypatch)
    captured = {}

    def fake_post(url, json=None, auth=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["auth"] = auth
        return FakeResponse(
            {
                "_links": {
                    "webui": "/spaces/TDD/pages/123/My+Title",
                    "base": "https://example.atlassian.net/wiki",
                }
            }
        )

    monkeypatch.setattr(httpx, "post", fake_post)

    url = confluence_api.create_page("TDD", "My Title", "<p>body</p>")

    assert url == "https://example.atlassian.net/wiki/spaces/TDD/pages/123/My+Title"
    assert captured["url"] == "https://example.atlassian.net/wiki/rest/api/content"
    assert captured["auth"] == ("a@example.com", "tok")
    assert captured["json"]["title"] == "My Title"
    assert captured["json"]["space"] == {"key": "TDD"}
    assert captured["json"]["body"]["storage"]["value"] == "<p>body</p>"
    assert captured["json"]["body"]["storage"]["representation"] == "storage"
    assert "ancestors" not in captured["json"]


def test_create_page_falls_back_to_base_url_when_response_has_no_base(monkeypatch):
    _set_env(monkeypatch)

    monkeypatch.setattr(
        httpx, "post", lambda *a, **kw: FakeResponse({"_links": {"webui": "/wiki/spaces/TDD/pages/123"}})
    )

    url = confluence_api.create_page("TDD", "My Title", "<p>body</p>")

    assert url == "https://example.atlassian.net/wiki/spaces/TDD/pages/123"


def test_create_page_includes_parent_id_when_given(monkeypatch):
    _set_env(monkeypatch)

    def fake_post(url, json=None, auth=None, timeout=None, **kwargs):
        assert json["ancestors"] == [{"id": "999"}]
        return FakeResponse({"_links": {"webui": "/wiki/spaces/TDD/pages/1"}})

    monkeypatch.setattr(httpx, "post", fake_post)

    confluence_api.create_page("TDD", "Title", "<p>x</p>", parent_id="999")


def test_create_page_raises_on_http_failure(monkeypatch):
    _set_env(monkeypatch)

    def fake_post(url, json=None, auth=None, timeout=None, **kwargs):
        return FakeResponse({}, status_code=403)

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(confluence_api.ConfluenceApiError, match="could not create Confluence page"):
        confluence_api.create_page("TDD", "Title", "<p>x</p>")


def test_create_page_raises_when_response_has_no_link(monkeypatch):
    _set_env(monkeypatch)

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: FakeResponse({"_links": {}}))

    with pytest.raises(confluence_api.ConfluenceApiError, match="no page link"):
        confluence_api.create_page("TDD", "Title", "<p>x</p>")


def test_create_page_raises_when_base_url_missing(monkeypatch):
    _set_env(monkeypatch, CONFLUENCE_BASE_URL=None)

    with pytest.raises(confluence_api.ConfluenceApiError, match="CONFLUENCE_BASE_URL"):
        confluence_api.create_page("TDD", "Title", "<p>x</p>")


def test_create_page_raises_when_credentials_missing(monkeypatch):
    _set_env(monkeypatch, CONFLUENCE_EMAIL=None)

    with pytest.raises(confluence_api.ConfluenceApiError, match="CONFLUENCE_EMAIL"):
        confluence_api.create_page("TDD", "Title", "<p>x</p>")
