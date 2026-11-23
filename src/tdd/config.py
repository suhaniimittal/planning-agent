"""The TDD flow's switches and limits (2026-11-02).

Read from the environment once per run — not at import — so a changed value
takes effect on the next run without restarting the worker.

The planner never reads a source file's text (2026-11-03): no contents
call, no pull request or compare patches. The TDD is built from file paths,
the files a pull request changed, the contract files and the ticket; the
coding agent reads the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # GitHub calls per run, and code searches (GitHub allows ~10 a minute).
    max_github_calls: int = 150
    max_code_searches: int = 10
    # github.com, or a GitHub Enterprise Server's https://<host>/api/v3.
    github_api_base: str = "https://api.github.com"


def _count(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, not {raw!r}") from None
    if value < 1:
        raise ValueError(f"{name} must be at least 1, not {value}")
    return value


def load() -> Settings:
    """The settings for this run, from the environment; defaults where unset."""
    default = Settings()
    return Settings(
        max_github_calls=_count("TDD_MAX_GITHUB_CALLS", default.max_github_calls),
        max_code_searches=_count("TDD_MAX_CODE_SEARCHES", default.max_code_searches),
        github_api_base=(
            os.environ.get("GITHUB_API_BASE", "").strip() or default.github_api_base
        ).rstrip("/"),
    )


def describe(settings: Settings) -> str:
    """One log line: what this run may ask GitHub for."""
    return (
        f"paths only, no file contents; caps: {settings.max_github_calls} GitHub calls, "
        f"{settings.max_code_searches} searches"
    )
