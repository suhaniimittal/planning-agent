"""Shallow git clone of a service's source repo, for the code-graph build.

Separate from github_fetcher.py (which fetches a single file's text via the
GitHub Contents API for the doc pipeline) — this needs the actual working
tree on disk so code_parser.py can walk it with tree-sitter, not one file's
text in memory.

The clone is always discarded after use: callers get a temp directory via
the `cloned_repo` context manager and everything under it is deleted on
exit, whether parsing succeeded or not.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


class RepoFetchError(Exception):
    """Raised when a repo can't be cloned — caller must catch this and fall
    back to key_symbols, never let it abort the whole ingestion run."""


def _clone_url(slug: str, github_token: str | None) -> str:
    if github_token:
        return f"https://x-access-token:{github_token}@github.com/{slug}.git"
    return f"https://github.com/{slug}.git"


@contextmanager
def cloned_repo(slug: str, github_token: str | None = None, timeout: int = 120) -> Iterator[Path]:
    """Shallow-clones `slug` ("org/repo") into a temp dir, yields its path.

    No --branch is passed — a depth-1 clone with none specified already
    checks out the remote's default branch, avoiding a second API call just
    to look that up.
    """
    github_token = github_token if github_token is not None else os.environ.get("GITHUB_TOKEN")
    url = _clone_url(slug, github_token)

    with tempfile.TemporaryDirectory(prefix="repo_fetcher_") as tmpdir:
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", url, tmpdir],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RepoFetchError(f"git clone failed for {slug}: {e.stderr.strip()}") from e
        except subprocess.TimeoutExpired as e:
            raise RepoFetchError(f"git clone timed out for {slug} after {timeout}s") from e
        except OSError as e:
            raise RepoFetchError(f"git clone could not run for {slug}: {e}") from e

        yield Path(tmpdir)
