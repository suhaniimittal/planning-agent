"""Checks every file/symbol an LLM-written TDD names against the real code
graph, after the LLM call — the model is told not to guess paths or symbol
names, but it sometimes does anyway (a folder given as a "file", a function
that doesn't exist marked "modify"), and a downstream coding agent would
otherwise act on that as fact.

Each FileChange ends up with `verification` set, and with `file_path`
rewritten only when the graph proves where the symbol really is. Nothing is
ever silently dropped: anything that can't be confirmed stays in the TDD,
flagged "unverified" with a note saying why.
"""

from __future__ import annotations

import posixpath
import re
from typing import NamedTuple

from . import graph_reader
from .models import FileChange, TechnicalDesignDoc

# "save()", "save(order)", "`save`" -> "save"
_CALL_SUFFIX = re.compile(r"\(.*\)$")


def _normalize_symbol(symbol: str | None) -> str:
    text = (symbol or "").strip().strip("`").strip()
    text = _CALL_SUFFIX.sub("", text)
    # A fully qualified "path::Class.method" — the graph matches the tail.
    return text.rsplit("::", 1)[-1].strip()


def _normalize_path(path: str | None) -> str | None:
    text = (path or "").strip().strip("`").strip()
    if text.startswith("./"):
        text = text[2:]
    return text.lstrip("/") or None


def _list(paths: list[str], limit: int = 3) -> str:
    shown = ", ".join(f"`{p}`" for p in paths[:limit])
    return shown + (f" (+{len(paths) - limit} more)" if len(paths) > limit else "")


def _verify_create(service: str, path: str | None) -> tuple[str, str]:
    if path is None:
        return "unverified", "No file was given for the new code."
    if graph_reader.file_exists(service, path):
        return "new", f"New code in the existing file `{path}`."
    if graph_reader.folder_exists(service, path):
        return (
            "unverified",
            f"`{path}` is a folder, not a file — the new file's name was not given.",
        )
    parent = posixpath.dirname(path)
    if parent and graph_reader.folder_exists(service, parent):
        return "new", f"New file in the existing folder `{parent}`."
    return "unverified", f"Neither `{path}` nor its folder exists in `{service}`."


class _Result(NamedTuple):
    status: str
    note: str
    path: str | None
    # Set only when the symbol name itself was corrected.
    symbol: str | None = None


def _dotted_part_in_file(service: str, path: str, symbol: str) -> str | None:
    """For a dotted name that doesn't exist as written — the LLM often glues
    a component to its file's name ("PreAdverseActionModal.view" for the
    `PreAdverseActionModal` defined in view.tsx) — the one part of it that
    IS a symbol in `path`, if exactly one is; otherwise None."""
    parts = [p for p in symbol.split(".") if p]
    if len(parts) < 2:
        return None
    matches = [p for p in parts if path in graph_reader.find_symbol_files(service, p)]
    return matches[0] if len(matches) == 1 else None


def _verify_existing(
    service: str, path: str | None, symbol: str, shown_files: set[str]
) -> _Result:
    """For "modify"/"delete": the symbol must already exist."""
    found_in = graph_reader.find_symbol_files(service, symbol) if symbol else []

    if path is not None and path in found_in:
        return _Result("verified", f"`{symbol}` found in `{path}`.", path)

    if len(found_in) > 1 and path is not None:
        # A folder path can still pick out one of several same-named symbols.
        prefix = path.rstrip("/") + "/"
        in_folder = [f for f in found_in if f.startswith(prefix)]
        if len(in_folder) == 1:
            found_in = in_folder

    if len(found_in) == 1:
        real = found_in[0]
        if path is None:
            note = f"File filled in from the code graph: `{symbol}` is in `{real}`."
        else:
            note = f"Path corrected: `{symbol}` is in `{real}`, not `{path}`."
        return _Result("corrected", note, real)

    if found_in:
        return _Result(
            "unverified",
            f"`{symbol}` exists in {len(found_in)} files ({_list(found_in)}) — "
            "which one is meant is unclear.",
            path,
        )

    # The symbol, as written, exists nowhere in this service.
    if path is not None and graph_reader.file_exists(service, path):
        real_symbol = _dotted_part_in_file(service, path, symbol)
        if real_symbol is not None:
            return _Result(
                "corrected",
                f"Symbol corrected: `{path}` has no `{symbol}`, but has `{real_symbol}`.",
                path,
                real_symbol,
            )
        return _Result(
            "unverified",
            f"`{path}` exists, but `{symbol}` was not found in it — "
            "it may need to be created, not modified.",
            path,
        )
    if path is not None and graph_reader.folder_exists(service, path):
        note = f"`{path}` is a folder, not a file, and `{symbol}` was not found in `{service}`."
        prefix = path.rstrip("/") + "/"
        related = sorted(f for f in shown_files if f.startswith(prefix))
        if related:
            note += f" Files in that folder matched to this issue: {_list(related)}."
        return _Result("unverified", note, path)
    if path is not None:
        return _Result(
            "unverified", f"Neither `{path}` nor `{symbol}` was found in `{service}`.", path
        )
    return _Result(
        "unverified", f"`{symbol}` was not found in `{service}`, and no file was given.", path
    )


def verify_change(service: str, change: FileChange, shown_files: set[str]) -> None:
    path = _normalize_path(change.file_path)
    symbol = _normalize_symbol(change.function_or_symbol)
    if change.change_type == "create":
        status, note = _verify_create(service, path)
        result = _Result(status, note, path)
    else:
        result = _verify_existing(service, path, symbol, shown_files)
    change.verification = result.status
    change.verification_note = result.note
    change.file_path = result.path
    if result.symbol is not None:
        change.function_or_symbol = result.symbol


def verify_tdd(
    tdd: TechnicalDesignDoc, shown_files_by_service: dict[str, set[str]] | None = None
) -> None:
    """Sets `verification`/`verification_note` on every change in place.
    `shown_files_by_service`: the files whose code was shown to the LLM for
    this issue — used only to point a reader at likely files when a path
    turns out to be a folder."""
    shown = shown_files_by_service or {}
    for service_plan in tdd.services:
        for change in service_plan.changes:
            verify_change(service_plan.service, change, shown.get(service_plan.service, set()))
