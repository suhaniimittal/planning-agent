"""Data shapes shared across the code parser: the parsed symbols/graph, and
the LanguageAdapter contract every per-language adapter (see adapters/)
implements.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from tree_sitter import Node


@dataclass
class CodeChunk:
    index: int
    line_start: int
    line_end: int
    text: str  # for multi-chunk functions, prefixed with the function's own
    # signature as an anchor, so the chunk never loses track of what
    # function it's part of — a cheaper substitute for line-overlap between
    # neighboring chunks (see design discussion: structure-aware cuts don't
    # slice mid-thought, so raw content overlap isn't needed, just an anchor)


@dataclass
class ParsedSymbol:
    kind: str  # "class" | "function"
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None
    language: str
    source_snippet: str
    is_method: bool = False
    parent_class: str | None = None
    chunks: list[CodeChunk] = field(default_factory=list)  # only ever populated for functions


@dataclass
class RepoGraph:
    classes: list[ParsedSymbol]
    functions: list[ParsedSymbol]
    files_parsed: int = 0
    files_skipped_unsupported: int = 0
    files_skipped_oversized: int = 0
    files_failed: int = 0


@dataclass
class LanguageAdapter:
    """Everything specific to one language/grammar — the only place its
    quirks are allowed to live. `name_of`/`body_of`/`docstring_of` all
    receive a tree-sitter node and must return plain data (a string, or
    another node), never raise, and return None when nothing is found
    (never guess)."""

    language: str
    class_types: set[str]
    function_types: set[str]
    name_of: Callable[[Node], str | None]
    body_of: Callable[[Node], Node | None]
    docstring_of: Callable[[Node, "Node | None", bytes], str | None]
