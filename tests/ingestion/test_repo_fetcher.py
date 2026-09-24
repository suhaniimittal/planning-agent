import subprocess

import pytest

from src.ingestion.repo_fetcher import RepoFetchError, _clone_url, cloned_repo


def test_clone_url_uses_plain_https_without_token():
    assert _clone_url("org/orders", None) == "https://github.com/org/orders.git"


def test_clone_url_embeds_token_when_present():
    url = _clone_url("org/orders", "sekret")
    assert url == "https://x-access-token:sekret@github.com/org/orders.git"


def test_cloned_repo_invokes_git_clone_with_depth_1(monkeypatch):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with cloned_repo("org/orders", github_token=None) as path:
        assert path.exists()

    cmd = captured["cmd"]
    assert cmd[:2] == ["git", "clone"]
    assert "--depth" in cmd and "1" in cmd
    assert "https://github.com/org/orders.git" in cmd
    assert "--branch" not in cmd


def test_cloned_repo_raises_repo_fetch_error_on_clone_failure(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="fatal: repository not found")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RepoFetchError):
        with cloned_repo("org/does-not-exist"):
            pass


def test_cloned_repo_raises_repo_fetch_error_on_timeout(monkeypatch):
    def fake_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1))

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RepoFetchError):
        with cloned_repo("org/slow-repo", timeout=1):
            pass


def test_cloned_repo_directory_removed_after_context_exits(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    with cloned_repo("org/orders") as path:
        held_path = path
    assert not held_path.exists()
