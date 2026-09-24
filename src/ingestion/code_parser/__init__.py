"""tree-sitter based extraction of classes/functions from a cloned repo.

Structured as one small "adapter" per language (see adapters/): each
LanguageAdapter knows its own grammar's quirks (which node types are
class-like/function-like, how to find a node's name/body, how to find its
docstring) and hands back symbols in the SAME shape regardless of language —
everything downstream (chunking, embeddings, Neo4j writing) never needs to
know which language a symbol came from. Adding a new language means writing
one more adapter (`adapters/<language>_adapter.py`) and registering it in
`LANGUAGE_BY_EXTENSION` below and `adapters.ADAPTERS`;
tree_sitter_language_pack already bundles the grammar, so no new dependency
is needed.

No I/O beyond reading the files under `repo_root` — parse_repo() never
touches Neo4j or the network, matching extractors.py's "pure functions,
easy to test" charter (see docs/INGESTION_PIPELINE.md).
"""

from __future__ import annotations

import os
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from .adapters import ADAPTERS
from .chunking import _CHUNK_TARGET_CHARS, chunk_function
from .models import CodeChunk, LanguageAdapter, ParsedSymbol, RepoGraph
from .text_utils import node_text, qualified_name, signature_of

__all__ = [
    "CodeChunk",
    "LanguageAdapter",
    "ParsedSymbol",
    "RepoGraph",
    "DEFAULT_DENYLIST",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "LANGUAGE_BY_EXTENSION",
    "parse_file",
    "parse_repo",
]

# Directories never descended into — build/dependency artifacts, not
# first-party code. Test directories are deliberately NOT here: a TDD often
# needs to point at an existing test file, so excluding them would blind the
# tool to exactly the symbols a coding agent needs to locate.
DEFAULT_DENYLIST: set[str] = {
    "node_modules", ".venv", "venv", "dist", "build", "target",
    "__pycache__", ".git", ".idea", ".gradle", ".mvn", "vendor",
    "bower_components",
}

DEFAULT_MAX_FILE_SIZE_BYTES = 2_000_000

LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",  # plain javascript's own grammar already parses JSX
    ".kt": "kotlin",
}


def _walk(
    node: Node,
    *,
    file_path: str,
    source: bytes,
    adapter: LanguageAdapter,
    ctx_class_qname: str | None,
    classes: list[ParsedSymbol],
    functions: list[ParsedSymbol],
) -> None:
    for child in node.children:
        if child.type in adapter.class_types:
            name = adapter.name_of(child)
            if name is None:
                continue
            qname = qualified_name(file_path, name, None)
            body = adapter.body_of(child)
            classes.append(
                ParsedSymbol(
                    kind="class",
                    name=name,
                    qualified_name=qname,
                    file_path=file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=signature_of(child, body, source),
                    docstring=adapter.docstring_of(child, body, source),
                    language=adapter.language,
                    source_snippet=node_text(child, source),
                )
            )
            if body is not None:
                _walk(
                    body,
                    file_path=file_path,
                    source=source,
                    adapter=adapter,
                    ctx_class_qname=qname,
                    classes=classes,
                    functions=functions,
                )
        elif child.type in adapter.function_types:
            name = adapter.name_of(child)
            if name is None:
                continue
            is_method = ctx_class_qname is not None
            qname = qualified_name(file_path, name, ctx_class_qname if is_method else None)
            body = adapter.body_of(child)
            signature = signature_of(child, body, source)
            functions.append(
                ParsedSymbol(
                    kind="function",
                    name=name,
                    qualified_name=qname,
                    file_path=file_path,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                    signature=signature,
                    docstring=adapter.docstring_of(child, body, source),
                    language=adapter.language,
                    source_snippet=node_text(child, source),
                    is_method=is_method,
                    parent_class=ctx_class_qname if is_method else None,
                    chunks=chunk_function(child, body, signature, source),
                )
            )
            if body is not None:
                # Nested defs (closures/local classes) reset context: only a
                # function's DIRECT membership in a class body makes it a
                # method, matching DECLARES' intended meaning.
                _walk(
                    body,
                    file_path=file_path,
                    source=source,
                    adapter=adapter,
                    ctx_class_qname=None,
                    classes=classes,
                    functions=functions,
                )
        else:
            _walk(
                child,
                file_path=file_path,
                source=source,
                adapter=adapter,
                ctx_class_qname=ctx_class_qname,
                classes=classes,
                functions=functions,
            )


def parse_file(path: Path, repo_root: Path, language: str) -> list[ParsedSymbol]:
    """Parse one file, returning its classes + functions (kind-tagged, flat).

    Raises on a genuine parse failure (tree-sitter itself doesn't raise on
    malformed syntax — it produces error nodes — so this mainly covers I/O
    and decode errors); callers are expected to catch and count, not let one
    bad file abort a repo scan.
    """
    source = path.read_bytes()
    parser = get_parser(language)
    tree = parser.parse(source)
    adapter = ADAPTERS[language]

    file_path = str(path.relative_to(repo_root).as_posix())
    classes: list[ParsedSymbol] = []
    functions: list[ParsedSymbol] = []
    _walk(
        tree.root_node,
        file_path=file_path,
        source=source,
        adapter=adapter,
        ctx_class_qname=None,
        classes=classes,
        functions=functions,
    )
    return classes + functions


def parse_repo(
    repo_root: Path,
    service_name: str,
    denylist: set[str] | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> RepoGraph:
    """Walk `repo_root`, parsing every supported source file.

    "Parse everything" by design (per the ingestion architecture decision) —
    the denylist only prunes build/dependency noise, and the size guardrail
    only protects against pathological single files (generated/minified/
    vendored) that slipped past it; neither is a node-count cap.
    """
    denylist = denylist if denylist is not None else DEFAULT_DENYLIST
    graph = RepoGraph(classes=[], functions=[])

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in denylist]
        for filename in filenames:
            path = Path(dirpath) / filename
            language = LANGUAGE_BY_EXTENSION.get(path.suffix)
            if language is None:
                graph.files_skipped_unsupported += 1
                continue
            try:
                if path.stat().st_size > max_file_size_bytes:
                    graph.files_skipped_oversized += 1
                    continue
            except OSError:
                graph.files_failed += 1
                continue
            try:
                symbols = parse_file(path, repo_root, language)
            except Exception:
                graph.files_failed += 1
                continue
            graph.files_parsed += 1
            for symbol in symbols:
                if symbol.kind == "class":
                    graph.classes.append(symbol)
                else:
                    graph.functions.append(symbol)

    return graph
