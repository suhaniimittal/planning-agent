import hashlib
import io
import subprocess
import zipfile

import pytest

from src.ingestion import repo_fetcher
from src.ingestion.github_api import GitHubApiError
from src.ingestion.repo_fetcher import RepoFetchError, blob_shas_of, cloned_repo


def _zip(entries: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in entries.items():
            info = zipfile.ZipInfo(name)
            info.external_attr = 0o100644 << 16
            zf.writestr(info, content)
    return buf.getvalue()


def _patch_github(monkeypatch, entries=None, head="sha9"):
    calls = {}

    def fake_head(slug, branch, token):
        calls["head"] = (slug, branch)
        return head

    def fake_zip(slug, ref, token, timeout):
        calls["zip"] = (slug, ref)
        return _zip(entries or {"top/a.py": "print('a')\n"})

    monkeypatch.setattr(repo_fetcher.github_api, "get_remote_head_sha", fake_head)
    monkeypatch.setattr(repo_fetcher.github_api, "download_zipball", fake_zip)
    return calls


def test_cloned_repo_downloads_head_commit_and_yields_checkout(monkeypatch):
    calls = _patch_github(monkeypatch, {"org-orders-sha9/src/A.py": "x = 1\n"})

    with cloned_repo("org/orders", github_token="t") as checkout:
        assert (checkout.root / "src" / "A.py").read_text() == "x = 1\n"
        assert checkout.head_sha == "sha9"

    assert calls["head"] == ("org/orders", None)  # no branch -> repo default
    assert calls["zip"] == ("org/orders", "sha9")  # downloaded at the exact commit


def test_cloned_repo_passes_branch_when_given(monkeypatch):
    calls = _patch_github(monkeypatch)

    with cloned_repo("org/orders", branch="develop", github_token="t"):
        pass

    assert calls["head"] == ("org/orders", "develop")


def test_cloned_repo_raises_repo_fetch_error_on_api_failure(monkeypatch):
    def boom(slug, branch, token):
        raise GitHubApiError("404 not found")

    monkeypatch.setattr(repo_fetcher.github_api, "get_remote_head_sha", boom)

    with pytest.raises(RepoFetchError, match="404"):
        with cloned_repo("org/does-not-exist", github_token="t"):
            pass


def test_cloned_repo_raises_repo_fetch_error_on_bad_zip(monkeypatch):
    monkeypatch.setattr(repo_fetcher.github_api, "get_remote_head_sha", lambda *a: "s")
    monkeypatch.setattr(repo_fetcher.github_api, "download_zipball", lambda *a, **k: b"not a zip")

    with pytest.raises(RepoFetchError, match="valid zip"):
        with cloned_repo("org/orders", github_token="t"):
            pass


def test_cloned_repo_directory_removed_after_context_exits(monkeypatch):
    _patch_github(monkeypatch)

    with cloned_repo("org/orders", github_token="t") as checkout:
        root = checkout.root
        assert root.exists()

    assert not root.exists()


def test_extract_zip_rejects_path_escaping_destination(tmp_path):
    with pytest.raises(RepoFetchError, match="unsafe"):
        repo_fetcher._extract_zip(_zip({"top/../../evil.txt": "x"}), tmp_path / "dest")


def test_blob_shas_of_returns_one_entry_per_file_with_posix_paths(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("print('b')\n")

    shas = blob_shas_of(tmp_path)

    assert set(shas) == {"a.py", "pkg/b.py"}
    assert all(len(sha) == 40 for sha in shas.values())
    assert shas["a.py"] != shas["pkg/b.py"]


def test_blob_shas_of_matches_git_hash_object(tmp_path):
    (tmp_path / "a.py").write_text("print('a')\n")
    expected = subprocess.run(
        ["git", "hash-object", "a.py"], capture_output=True, text=True, check=True, cwd=tmp_path
    ).stdout.strip()

    assert blob_shas_of(tmp_path)["a.py"] == expected
    content = b"print('a')\n"
    assert expected == hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


# --- local repo mode ---------------------------------------------------------


def test_local_mode_off_by_default(monkeypatch):
    monkeypatch.delenv("USE_LOCAL_REPO", raising=False)
    assert repo_fetcher.local_repo_root("orders") is None


def test_local_mode_yields_the_folder_without_touching_github(monkeypatch, tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path))

    def fail(*a, **k):
        raise AssertionError("must not call GitHub in local mode")

    monkeypatch.setattr(repo_fetcher.github_api, "get_remote_head_sha", fail)
    monkeypatch.setattr(repo_fetcher.github_api, "download_zipball", fail)

    with cloned_repo("org/repo") as checkout:
        assert checkout.root == tmp_path.resolve()
        assert checkout.head_sha == repo_fetcher.LOCAL_HEAD_SHA

    # the user's own folder is never deleted
    assert (tmp_path / "app.py").exists()


def test_local_mode_without_path_raises(monkeypatch):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.delenv("LOCAL_REPO_PATH", raising=False)
    with pytest.raises(RepoFetchError, match="LOCAL_REPO_PATH is not set"):
        repo_fetcher.local_repo_root("orders")


def test_local_mode_with_missing_folder_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path / "does-not-exist"))
    with pytest.raises(RepoFetchError, match="not an existing folder"):
        repo_fetcher.local_repo_root("orders")


def test_blob_shas_of_skips_denylisted_folders(tmp_path):
    (tmp_path / "app.py").write_text("x = 1\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tmp_path / "node_modules" / "pkg").mkdir(parents=True)
    (tmp_path / "node_modules" / "pkg" / "index.js").write_text("module.exports = 1\n")

    assert set(blob_shas_of(tmp_path)) == {"app.py"}


def test_local_mode_several_folders_maps_each_service_to_its_own(monkeypatch, tmp_path):
    api = tmp_path / "reports-api"
    ui = tmp_path / "ui"
    api.mkdir()
    ui.mkdir()
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", f"{api}, {ui}")

    assert repo_fetcher.local_repo_root("reports-api") == api.resolve()
    assert repo_fetcher.local_repo_root("ui") == ui.resolve()


def test_local_mode_name_equals_path_entries(monkeypatch, tmp_path):
    folder = tmp_path / "some-folder"
    other = tmp_path / "other"
    folder.mkdir()
    other.mkdir()
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", f"orders={folder},shipping={other}")

    assert repo_fetcher.local_repo_root("orders") == folder.resolve()
    assert repo_fetcher.local_repo_root("shipping") == other.resolve()


def test_local_mode_service_missing_from_list_raises(monkeypatch, tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", f"{a},{b}")

    with pytest.raises(RepoFetchError, match="no folder for service 'orders'"):
        repo_fetcher.local_repo_root("orders")


def test_local_mode_single_path_serves_every_service(monkeypatch, tmp_path):
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", str(tmp_path))

    assert repo_fetcher.local_repo_root("orders") == tmp_path.resolve()
    assert repo_fetcher.local_repo_root("anything") == tmp_path.resolve()


def test_local_mode_cloned_repo_picks_the_named_service_folder(monkeypatch, tmp_path):
    api = tmp_path / "reports-api"
    ui = tmp_path / "ui"
    api.mkdir()
    ui.mkdir()
    monkeypatch.setenv("USE_LOCAL_REPO", "true")
    monkeypatch.setenv("LOCAL_REPO_PATH", f"{api},{ui}")

    with cloned_repo("org/ui", service_name="ui") as checkout:
        assert checkout.root == ui.resolve()


def test_local_folder_fingerprint_changes_when_a_file_changes(tmp_path):
    import os

    (tmp_path / "app.py").write_text("one")
    first = repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"})
    assert repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"}) == first

    (tmp_path / "app.py").write_text("two!")
    edited = repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"})
    assert edited != first

    (tmp_path / "new.py").write_text("x")
    assert repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"}) != edited

    os.remove(tmp_path / "new.py")
    assert repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"}) == edited


def test_local_folder_fingerprint_ignores_denylisted_folders(tmp_path):
    (tmp_path / "app.py").write_text("one")
    before = repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"})
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("noise")

    assert repo_fetcher.local_folder_fingerprint(tmp_path, {"orders"}) == before


def test_local_folder_fingerprint_changes_when_service_list_changes(tmp_path):
    (tmp_path / "app.py").write_text("one")

    assert repo_fetcher.local_folder_fingerprint(
        tmp_path, {"orders"}
    ) != repo_fetcher.local_folder_fingerprint(tmp_path, {"orders", "payments"})
