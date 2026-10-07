"""Snapshot of a service's source repo, for the code-graph build.

This needs the actual working tree on disk so code_parser.py can walk it
with tree-sitter, not one file's text in memory.

No `git` binary is used — the platform container has none. The repo is
downloaded through the GitHub API as a zip and extracted into a temp dir.

The snapshot is always discarded after use: callers get a RepoCheckout via
the `cloned_repo` context manager and everything under its root is deleted
on exit, whether parsing succeeded or not.

Local mode (USE_LOCAL_REPO=true + LOCAL_REPO_PATH): for local testing only.
Each service's code is read straight from a folder on this machine instead
of from GitHub, and the folder is never modified or deleted.
"""

from __future__ import annotations

import hashlib
import io
import os
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from . import github_api
from .code_parser import DEFAULT_DENYLIST

# Stand-in commit sha for a local folder, which has no GitHub commit.
LOCAL_HEAD_SHA = "local"


class RepoFetchError(Exception):
    """Raised when a repo can't be downloaded — caller must catch this and
    fall back to key_symbols, never let it abort the whole ingestion run."""

#the repo downloaded with its exact commit downloaded
@dataclass(frozen=True)
class RepoCheckout:
    """`root` is the extracted working tree; `head_sha` the exact commit it
    was taken from (what ingestion records as the service's commit_sha)."""

    root: Path
    head_sha: str


def _extract_zip(data: bytes, dest: Path) -> None:
    """Extracts GitHub's zipball into `dest`, dropping its single top-level
    `owner-repo-sha/` folder. Skips symlinks (a zip stores them as small
    text files, which would corrupt them if written as regular files) and
    refuses any path escaping `dest`."""
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue

            #It essentially removes the first directory component.
            relative = Path(*Path(info.filename).parts[1:])

            if not relative.parts:
                continue
            mode = info.external_attr >> 16

            #skip symlinks, which are stored as small text files in a zip
            if stat.S_ISLNK(mode):
                continue

            #security check for path traversal / Zip Slip (it could write files outside the temporary repository directory)
            target = (root / relative).resolve()
            if root not in target.parents:
                raise RepoFetchError(f"unsafe path in repo archive: {info.filename!r}")
            
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))


def _local_repo_paths() -> dict[str, Path] | Path:
    """Parses LOCAL_REPO_PATH. Comma-separated entries, each either
    `name=/path` or a bare `/path` (its folder name is the service name).
    A single bare path is returned as a plain Path: it serves every
    service, as LOCAL_REPO_PATH did before it accepted several entries."""
    raw = os.environ.get("LOCAL_REPO_PATH", "").strip()
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if not entries:
        raise RepoFetchError("USE_LOCAL_REPO=true but LOCAL_REPO_PATH is not set")

    def as_dir(path_text: str) -> Path:
        root = Path(path_text).expanduser().resolve()
        if not root.is_dir():
            raise RepoFetchError(f"LOCAL_REPO_PATH entry is not an existing folder: {root}")
        return root

    if len(entries) == 1 and "=" not in entries[0]:
        return as_dir(entries[0])

    paths: dict[str, Path] = {}
    for entry in entries:
        name, sep, path_text = entry.partition("=")
        root = as_dir(path_text if sep else entry)
        name = name.strip() if sep else root.name
        if name in paths:
            raise RepoFetchError(f"LOCAL_REPO_PATH names service {name!r} twice")
        paths[name] = root
    return paths


def local_repo_root(service_name: str) -> Path | None:
    """The folder to read `service_name`'s code from when
    USE_LOCAL_REPO=true, else None. Raises RepoFetchError if local mode is
    on but the setting is missing, names a folder that doesn't exist, or
    has no entry for this service — failing loudly beats silently falling
    back to GitHub and ingesting different code than the user meant to test."""
    if os.environ.get("USE_LOCAL_REPO", "").strip().lower() != "true":
        return None
    paths = _local_repo_paths()
    if isinstance(paths, Path):
        return paths
    if service_name not in paths:
        raise RepoFetchError(
            f"LOCAL_REPO_PATH has no folder for service {service_name!r} "
            f"(it has: {', '.join(sorted(paths))})"
        )
    return paths[service_name]


def local_folder_fingerprint(root: Path, service_names: set[str]) -> str:
    """A cheap fingerprint of a local folder: every file's path, size and
    modification time (no file is read), skipping the same build/dependency
    folders the parser skips — so walking a UI folder never descends into
    node_modules/. Editing, adding, removing or renaming any file changes it.

    `service_names` (every service in this run) is folded in too: each file's
    dependency references are matched against that list, so a change to it
    must re-ingest even an untouched folder."""
    digest = hashlib.sha1()
    digest.update(("\0".join(sorted(service_names)) + "\n").encode())
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in DEFAULT_DENYLIST)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            try:
                st = path.stat()
            except OSError:
                continue
            rel = path.relative_to(root).as_posix()
            digest.update(f"{rel}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return digest.hexdigest()


@contextmanager
def cloned_repo(
    slug: str,
    github_token: str | None = None,
    branch: str | None = None,
    timeout: int = 120,
    service_name: str = "",
) -> Iterator[RepoCheckout]:
    """Downloads `slug` ("org/repo") at `branch` (None = the repo's default
    branch) into a temp dir, yields a RepoCheckout. Only pass `branch` when
    a service's manifest entry explicitly names one.

    In local mode, yields `service_name`'s folder from LOCAL_REPO_PATH
    instead — nothing is downloaded, and the folder is left untouched."""
    local_root = local_repo_root(service_name)
    if local_root is not None:
        yield RepoCheckout(local_root, LOCAL_HEAD_SHA)
        return

    token = github_token if github_token is not None else os.environ.get("GITHUB_TOKEN")
    try:
        head_sha = github_api.get_remote_head_sha(slug, branch, token)
        data = github_api.download_zipball(slug, head_sha, token, timeout=timeout)
    except github_api.GitHubApiError as e:
        raise RepoFetchError(f"could not download {slug}: {e}") from e


    #a temporary directory is created and repo is extracted there
    with tempfile.TemporaryDirectory(prefix="repo_fetcher_") as tmpdir:
        try:
            _extract_zip(data, Path(tmpdir))
        except zipfile.BadZipFile as e:
            raise RepoFetchError(f"downloaded archive for {slug} is not a valid zip") from e
        
        #gives directory to the caller, which is deleted on exit
        yield RepoCheckout(Path(tmpdir), head_sha)


#calculates the blob sha of every file in the repo, which is used to check if a file has changed
def blob_shas_of(repo_root: Path) -> dict[str, str]:
    """{file_path: blob_sha} for every file under `repo_root`, computed the
    way git does (sha1 of "blob <size>\\0" + content) — so it equals what
    GitHub reports for the same file, with no git binary or extra network
    call. Used by later incremental runs to confirm a file GitHub reports
    as "changed" actually has different content."""
    shas: dict[str, str] = {}
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        # A local checkout (unlike a GitHub zip) carries .git/, node_modules/
        # etc. — skip them, same as the parser does.
        if any(part in DEFAULT_DENYLIST for part in path.relative_to(repo_root).parts):
            continue
        content = path.read_bytes()
        digest = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
        shas[path.relative_to(repo_root).as_posix()] = digest
    return shas
