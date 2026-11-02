"""The TDD flow's switches and limits (2026-11-02).

Read from the environment once per run — not at import — so a changed value
takes effect on the next run without restarting the worker.

ALLOW_CODE_READS (default on) is the policy switch: may the planner read a
source file's text into memory for one run? It is not yet confirmed that it
may. Off, the contents endpoint is never called and no pull request or
compare patch text is kept; the TDD is built from file paths, the files a
pull request changed, the service map's file lists and the ticket, and says
it is high-level.

A value that is not clearly on or off stops the run instead of guessing: a
typo in ALLOW_CODE_READS must never turn code reads on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ON = {"1", "true", "yes", "on"}
_OFF = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class Settings:
    # The policy switch: may a file's text be read into memory for one run.
    allow_code_reads: bool = True
    # LLM call 1, which picks extra files from the path list (TDD_SCOUT).
    scout: bool = True
    # Files read per service and in total, characters read in total: hard
    # caps on how much code one run ever holds.
    # Up to 5 of each service's files are kept for the scout's picks.
    max_files_per_service: int = 20
    max_files: int = 40
    max_chars: int = 300_000
    # Connection files (contracts, config, controllers, clients) the service
    # map reads per service. Their text is dropped once their facts are out.
    map_files_per_service: int = 15
    # GitHub calls per run, and code searches (GitHub allows ~10 a minute).
    max_github_calls: int = 150
    max_code_searches: int = 10
    # github.com, or a GitHub Enterprise Server's https://<host>/api/v3.
    github_api_base: str = "https://api.github.com"


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _ON:
        return True
    if raw in _OFF:
        return False
    raise ValueError(f"{name} must be true or false, not {raw!r}")


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
        allow_code_reads=_flag("ALLOW_CODE_READS", default.allow_code_reads),
        scout=_flag("TDD_SCOUT", default.scout),
        max_files_per_service=_count("TDD_MAX_FILES_PER_SERVICE", default.max_files_per_service),
        max_files=_count("TDD_MAX_FILES", default.max_files),
        max_chars=_count("TDD_MAX_CHARS", default.max_chars),
        map_files_per_service=_count("TDD_MAP_FILES_PER_SERVICE", default.map_files_per_service),
        max_github_calls=_count("TDD_MAX_GITHUB_CALLS", default.max_github_calls),
        max_code_searches=_count("TDD_MAX_CODE_SEARCHES", default.max_code_searches),
        github_api_base=(
            os.environ.get("GITHUB_API_BASE", "").strip() or default.github_api_base
        ).rstrip("/"),
    )


def describe(settings: Settings) -> str:
    """One log line: what this run may read, and how much."""
    return (
        f"code reads {'on' if settings.allow_code_reads else 'OFF (paths only)'}; "
        f"scout {'on' if settings.scout else 'off'}; caps: "
        f"{settings.max_files_per_service} files/service, {settings.max_files} files, "
        f"{settings.max_chars:,} chars, {settings.map_files_per_service} map files/service, "
        f"{settings.max_github_calls} GitHub calls, "
        f"{settings.max_code_searches} searches"
    )
