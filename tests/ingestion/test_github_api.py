import base64

import httpx
import pytest

from src.ingestion import github_api


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)

    def json(self):
        return self._json


def test_get_remote_head_sha_returns_sha(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse({"sha": "abc123"})

    monkeypatch.setattr(httpx, "get", fake_get)

    sha = github_api.get_remote_head_sha("org/orders-service", branch="main")

    assert sha == "abc123"
    assert captured["url"] == "https://api.github.com/repos/org/orders-service/commits/main"


def test_get_remote_head_sha_defaults_to_head_ref(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        return FakeResponse({"sha": "abc123"})

    monkeypatch.setattr(httpx, "get", fake_get)
    github_api.get_remote_head_sha("org/orders-service")

    assert captured["url"].endswith("/commits/HEAD")


def test_get_remote_head_sha_adds_auth_header_when_token_given(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["headers"] = headers
        return FakeResponse({"sha": "abc"})

    monkeypatch.setattr(httpx, "get", fake_get)
    github_api.get_remote_head_sha("org/orders-service", token="sekret")

    assert captured["headers"]["Authorization"] == "token sekret"


def test_get_remote_head_sha_raises_github_api_error_on_http_failure(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(github_api.GitHubApiError):
        github_api.get_remote_head_sha("org/orders-service")


def test_get_changed_files_normalizes_statuses(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **kwargs):
        return FakeResponse(
            {
                "files": [
                    {"filename": "a.py", "status": "added", "sha": "sha_a"},
                    {"filename": "b.py", "status": "removed"},
                    {
                        "filename": "c.py",
                        "status": "renamed",
                        "sha": "sha_c",
                        "previous_filename": "old_c.py",
                    },
                    {"filename": "d.py", "status": "changed", "sha": "sha_d"},
                ]
            }
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    files = github_api.get_changed_files("org/orders-service", "base_sha", "head_sha")

    assert files == [
        {"path": "a.py", "status": "added", "sha": "sha_a", "previous_path": None},
        {"path": "b.py", "status": "removed", "sha": None, "previous_path": None},
        {"path": "c.py", "status": "renamed", "sha": "sha_c", "previous_path": "old_c.py"},
        {"path": "d.py", "status": "modified", "sha": "sha_d", "previous_path": None},
    ]


def test_get_changed_files_builds_compare_url(monkeypatch):
    captured = {}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        captured["url"] = url
        return FakeResponse({"files": []})

    monkeypatch.setattr(httpx, "get", fake_get)
    github_api.get_changed_files("org/orders-service", "base_sha", "head_sha")

    assert captured["url"] == "https://api.github.com/repos/org/orders-service/compare/base_sha...head_sha"


def test_get_changed_files_raises_on_http_failure(monkeypatch):
    def fake_get(url, headers=None, timeout=None, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(github_api.GitHubApiError):
        github_api.get_changed_files("org/orders-service", "a", "b")


def test_fetch_file_content_decodes_base64(monkeypatch):
    encoded = base64.b64encode(b"print('hello')").decode()

    def fake_get(url, headers=None, params=None, timeout=None, **kwargs):
        return FakeResponse({"content": encoded})

    monkeypatch.setattr(httpx, "get", fake_get)
    text = github_api.fetch_file_content("org/orders-service", "app.py", "abc123")

    assert text == "print('hello')"


def test_fetch_file_content_passes_ref_param(monkeypatch):
    captured = {}
    encoded = base64.b64encode(b"content").decode()

    def fake_get(url, headers=None, params=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse({"content": encoded})

    monkeypatch.setattr(httpx, "get", fake_get)
    github_api.fetch_file_content("org/orders-service", "app.py", "abc123")

    assert captured["url"] == "https://api.github.com/repos/org/orders-service/contents/app.py"
    assert captured["params"] == {"ref": "abc123"}


def test_fetch_file_content_raises_on_http_failure(monkeypatch):
    def fake_get(url, headers=None, params=None, timeout=None, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "get", fake_get)

    with pytest.raises(github_api.GitHubApiError):
        github_api.fetch_file_content("org/orders-service", "app.py", "abc123")


def test_create_pull_request_returns_html_url(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse({"html_url": "https://github.com/org/orders-service/pull/7"})

    monkeypatch.setattr(httpx, "post", fake_post)

    html_url = github_api.create_pull_request(
        "org/orders-service", head="fix/branch", base="main", title="t", body="b"
    )

    assert html_url == "https://github.com/org/orders-service/pull/7"
    assert captured["url"] == "https://api.github.com/repos/org/orders-service/pulls"
    assert captured["json"] == {"title": "t", "body": "b", "head": "fix/branch", "base": "main"}


def test_create_pull_request_raises_on_http_failure(monkeypatch):
    def fake_post(url, headers=None, json=None, timeout=None, **kwargs):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(httpx, "post", fake_post)

    with pytest.raises(github_api.GitHubApiError):
        github_api.create_pull_request("org/orders-service", head="fix/branch", base="main", title="t", body="b")


def test_download_zipball_returns_bytes_and_follows_redirects(monkeypatch):
    seen = {}

    def fake_get(url, headers=None, timeout=None, **kwargs):
        seen.update(url=url, follow=kwargs.get("follow_redirects"))
        resp = FakeResponse({})
        resp.content = b"zipbytes"
        return resp

    monkeypatch.setattr(github_api.httpx, "get", fake_get)

    assert github_api.download_zipball("org/orders", "sha1", token="t") == b"zipbytes"
    assert seen["url"].endswith("/repos/org/orders/zipball/sha1")
    assert seen["follow"] is True


def test_download_zipball_wraps_http_errors(monkeypatch):
    monkeypatch.setattr(
        github_api.httpx, "get", lambda *a, **kw: FakeResponse({}, status_code=404)
    )

    with pytest.raises(github_api.GitHubApiError):
        github_api.download_zipball("org/orders", "sha1", token="t")
