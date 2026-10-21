"""tree-sitter based extraction of classes/functions from a cloned repo.

A single flat module (deliberately — not a package/subfolder) containing:
data shapes, pure text helpers, one small "adapter" per language (each
LanguageAdapter knows its own grammar's quirks — which node types are
class-like/function-like, how to find a node's name/body, how to find its
docstring — and hands back symbols in the SAME shape regardless of
language), chunking, two hand-written parsers for formats with no
tree-sitter grammar (BPMN, Avro IDL), and the core walker/entry points.
Adding a new language means adding one more LanguageAdapter below and
registering it in LANGUAGE_BY_EXTENSION/ADAPTERS; tree_sitter_language_pack
already bundles the grammar, so no new dependency is needed.

No I/O beyond reading the files under `repo_root` — parse_repo() never
touches Neo4j or the network, keeping it a pure function that's easy to
test in isolation.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from .test_paths import is_test_path

__all__ = [
    "CodeChunk",
    "LanguageAdapter",
    "ParsedSymbol",
    "RepoGraph",
    "DEFAULT_DENYLIST",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "LANGUAGE_BY_EXTENSION",
    "TRACKED_ONLY_EXTENSIONS",
    "parse_file",
    "parse_repo",
    "parse_source",
]


# --- data shapes ----------------------------------------------------------


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
    # file_path -> language, for files worth a File node even though they
    # have no Class/Function symbols (e.g. .html/.css/.csv — no code-shaped
    # structure to extract, but still worth tracking so a coding agent knows
    # the file exists and can be told to edit it as a whole, and so it's
    # still covered by the cross-service dependency scan).
    tracked_files: dict[str, str] = field(default_factory=dict)
    # file_path -> its import lines, verbatim. Nothing else in the pipeline
    # records them: parse_file() extracts only classes and functions, so the
    # lines above them are lost. That gap is why generated code invents
    # module paths — a new test file importing a props type that is not
    # exported, a component importing "@client/hooks" which does not exist.
    file_imports: dict[str, str] = field(default_factory=dict)
    # file_path -> its whole text, TEST FILES ONLY, capped. A test file is
    # mostly top-level describe()/it() calls and object literals, none of
    # which parse_file() extracts, so a 3,920-char test file reached the
    # graph as 192 characters — under 10% of itself, and never the part
    # that matters. What the planner needs from a test is precisely what is
    # missing: the fixture, which props must be set for the component to
    # render at all, and the shape of the payload its callbacks receive.
    # Production files are not stored this way; their symbols are already
    # extracted, and storing both would double the graph for nothing.
    file_source: dict[str, str] = field(default_factory=dict)
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
    docstring_of: Callable[[Node, Node | None, bytes], str | None]


# --- pure text helpers ------------------------------------------------------
#
# Shared across chunking, the language adapters, and the core walker — kept
# dependency-free (no reference to anything else in this file) since they're
# the lowest-level building block everything else uses.


def text_of(node: Node | None) -> str | None:
    if node is None:
        return None
    return node.text.decode("utf-8", errors="replace")


def signature_of(node: Node, body: Node | None, source: bytes) -> str:
    end = body.start_byte if body is not None else node.end_byte
    return source[node.start_byte : end].decode("utf-8", errors="replace").strip()


def node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def qualified_name(file_path: str, name: str, parent_class: str | None) -> str:
    if parent_class is None:
        return f"{file_path}::{name}"
    parent_simple = parent_class.split("::", 1)[1]
    return f"{file_path}::{parent_simple}.{name}"


# --- adapter building blocks ------------------------------------------------
#
# Reused by more than one LanguageAdapter: field-based name/body lookup (for
# grammars that expose named fields — Python, Java, TypeScript/TSX/
# JavaScript), and the "doc comment sitting immediately above the
# declaration" convention (Javadoc/JSDoc/KDoc — Java, TypeScript-family,
# Kotlin all follow this idiom, just with different comment node-type names).


def field_name_of(node: Node) -> str | None:
    return text_of(node.child_by_field_name("name"))


def field_body_of(node: Node) -> Node | None:
    return node.child_by_field_name("body")


def adjacent_comment_docstring(node: Node, comment_types: set[str]) -> str | None:
    prev = node.prev_sibling
    if prev is None or prev.type not in comment_types:
        return None
    if node.start_point[0] - prev.end_point[0] > 1:
        return None  # not immediately adjacent, likely unrelated
    text = prev.text.decode("utf-8", errors="replace").strip()
    text = text.removeprefix("/**").removeprefix("/*").removesuffix("*/").strip()
    lines = [line.strip().lstrip("*").strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line) or None


# --- Python -----------------------------------------------------------------


def _python_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    if body is None or not body.children:
        return None
    first = body.children[0]
    if first.type != "string":
        return None
    text = first.text.decode("utf-8", errors="replace").strip()
    return text.strip("\"'").strip() or None


PYTHON_ADAPTER = LanguageAdapter(
    language="python",
    class_types={"class_definition"},
    function_types={"function_definition"},
    name_of=field_name_of,
    body_of=field_body_of,
    docstring_of=_python_docstring_of,
)


# --- Java --------------------------------------------------------------------


def _java_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    return adjacent_comment_docstring(node, {"block_comment"})


JAVA_ADAPTER = LanguageAdapter(
    language="java",
    class_types={"class_declaration", "interface_declaration"},
    function_types={"method_declaration", "constructor_declaration"},
    name_of=field_name_of,
    body_of=field_body_of,
    docstring_of=_java_docstring_of,
)


# --- TypeScript / TSX / JavaScript -------------------------------------------
#
# Same grammar family, JS is the base that TS/TSX extend, so all three share
# identical node shapes (confirmed against real grammars, not guessed — see
# the language-support design discussion).


def _typescript_name_of(node: Node) -> str | None:
    name = field_name_of(node)
    if name is not None:
        return name
    # An arrow function has no name of its own (it's anonymous) — if it's
    # the direct value of `const x = (...) => {...}`, borrow x's name.
    # Anything else (an inline callback like `.map(x => x+1)`) stays
    # unnamed and gets skipped by the caller, rather than guessed.
    if (
        node.type == "arrow_function"
        and node.parent is not None
        and node.parent.type == "variable_declarator"
    ):
        return field_name_of(node.parent)
    return None


_TS_STATEMENT_WRAPPER_TYPES = {"variable_declarator", "lexical_declaration", "export_statement"}


def _ts_statement_root(node: Node) -> Node:
    """Walk up through the wrapper nodes a declaration sits inside.

    A class is only wrapped once (`export_statement`), but `const x = () =>
    {}` wraps its arrow function twice more first (`variable_declarator`,
    `lexical_declaration`) — walking up generically, instead of checking one
    fixed hop, handles both shapes with the same code.
    """
    current = node
    while current.parent is not None and current.parent.type in _TS_STATEMENT_WRAPPER_TYPES:
        current = current.parent
    return current


def _typescript_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    doc = adjacent_comment_docstring(node, {"comment"})
    if doc is not None:
        return doc
    # A top-level exported class/function (and any decorator on it, e.g.
    # NestJS's @Controller()) is wrapped in export_statement — and a
    # `const`-assigned arrow function is wrapped twice more on top of that —
    # so its own prev_sibling is never the doc comment; check the outermost
    # wrapper instead.
    root = _ts_statement_root(node)
    if root is not node:
        return adjacent_comment_docstring(root, {"comment"})
    return None


def _make_js_family_adapter(language: str) -> LanguageAdapter:
    return LanguageAdapter(
        language=language,
        class_types={"class_declaration", "interface_declaration"},
        function_types={"method_definition", "function_declaration", "arrow_function"},
        name_of=_typescript_name_of,
        body_of=field_body_of,
        docstring_of=_typescript_docstring_of,
    )


TYPESCRIPT_ADAPTER = _make_js_family_adapter("typescript")
TSX_ADAPTER = _make_js_family_adapter("tsx")  # TypeScript + JSX (React)
JAVASCRIPT_ADAPTER = _make_js_family_adapter("javascript")  # also parses plain JSX


# --- Kotlin -------------------------------------------------------------------
#
# Kotlin's grammar exposes NO named fields at all (unlike the others above),
# so name/body have to be found positionally — by node TYPE, not field name.
# It also reuses the same node type ("function_declaration") for both a
# top-level function and a class method, same as Python — DECLARES vs.
# CONTAINS is still decided purely by nesting context, not the node's type.

_KOTLIN_NAME_CHILD_TYPE = {
    "class_declaration": "type_identifier",
    "function_declaration": "simple_identifier",
}


def _kotlin_name_of(node: Node) -> str | None:
    target_type = _KOTLIN_NAME_CHILD_TYPE.get(node.type)
    if target_type is None:
        return None
    for child in node.children:
        if child.type == target_type:
            return text_of(child)
    return None


def _kotlin_body_of(node: Node) -> Node | None:
    for child in node.children:
        if child.type in ("class_body", "function_body"):
            return child
    return None


def _kotlin_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    return adjacent_comment_docstring(node, {"multiline_comment"})


KOTLIN_ADAPTER = LanguageAdapter(
    language="kotlin",
    class_types={"class_declaration"},
    function_types={"function_declaration"},
    name_of=_kotlin_name_of,
    body_of=_kotlin_body_of,
    docstring_of=_kotlin_docstring_of,
)


# --- GraphQL SDL ----------------------------------------------------------
#
# Object/interface/input/enum type definitions as classes, their fields as
# functions (methods). Unlike Java/TypeScript/Python/Kotlin, this grammar
# exposes NO named tree-sitter fields (`child_by_field_name` returns None
# for everything) — the tree is purely type-tagged, so name/body/docstring
# all do a type-based scan of direct children instead of the field-based
# lookup used above.
#
# Field ARGUMENTS (`user(id: ID!): User`) are also `input_value_definition`
# nodes, same as a real input-type field — but since a `field_definition`'s
# own "body" (per `_graphql_body_of` below) only recognizes
# fields_definition/input_fields_definition/enum_values_definition, the core
# walker never recurses into a field's `arguments_definition`, so arguments
# are never mistaken for top-level symbols.


def _graphql_child_of_type(node: Node, type_name: str) -> Node | None:
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def _graphql_name_of(node: Node) -> str | None:
    return text_of(_graphql_child_of_type(node, "name"))


def _graphql_body_of(node: Node) -> Node | None:
    for body_type in ("fields_definition", "input_fields_definition", "enum_values_definition"):
        body = _graphql_child_of_type(node, body_type)
        if body is not None:
            return body
    return None


def _graphql_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    """GraphQL SDL's own description string (a block string or a plain
    quoted string) directly preceding a type/field — a real child of the
    definition node itself, not an adjacent sibling comment like Java/JS's
    doc-comment convention."""
    description = _graphql_child_of_type(node, "description")
    text = text_of(description)
    return text.strip('"').strip() or None if text else None


GRAPHQL_ADAPTER = LanguageAdapter(
    language="graphql",
    class_types={
        "object_type_definition",
        "interface_type_definition",
        "input_object_type_definition",
        "enum_type_definition",
    },
    function_types={"field_definition", "input_value_definition"},
    name_of=_graphql_name_of,
    body_of=_graphql_body_of,
    docstring_of=_graphql_docstring_of,
)


ADAPTERS: dict[str, LanguageAdapter] = {
    "python": PYTHON_ADAPTER,
    "java": JAVA_ADAPTER,
    "typescript": TYPESCRIPT_ADAPTER,
    "tsx": TSX_ADAPTER,
    "javascript": JAVASCRIPT_ADAPTER,
    "kotlin": KOTLIN_ADAPTER,
    "graphql": GRAPHQL_ADAPTER,
}


# --- chunking ----------------------------------------------------------------
#
# Splitting an oversized function's body into embeddable pieces.

# The ~500-800 char chunk-size target agreed in design: small enough to keep
# one embedding focused on one idea, large enough not to be pointlessly tiny.
_CHUNK_TARGET_CHARS = 700


def embedding_text(file_path: str, name: str, parent_class: str | None, chunk_text: str) -> str:
    """What a chunk is embedded AS — its path and symbol name, then its code.

    Not what is stored or shown: `Chunk.text` stays the code alone, so the
    planner still reads exactly what is in the file.

    Measured on this graph, 22% of single-chunk functions do not contain
    their own name anywhere in their text — an arrow function's chunk
    begins at its parameter list, so the name it was assigned to is lost.
    Only multi-chunk functions were anchored, and 93% of functions are a
    single chunk. So for roughly a fifth of the graph the most descriptive
    thing about a symbol was never embedded.

    That matters because a ticket is written in words and code is written
    in symbols. `multipleEmailFieldValidation` contains the semicolon-to-
    comma conversion a real ticket was about, as `.replace(/;/g, ",")` —
    characters, not the words "semicolon" and "comma" — and scored 0.292
    against that ticket while unrelated code scored 0.78. Its NAME holds
    "email" and "validation", both of which the ticket says, and its path
    holds more. Embedding those costs nothing and is often the only English
    a chunk has.
    """
    head = file_path if not parent_class else f"{file_path}\n{parent_class}"
    return f"{head}\n{name}\n{chunk_text}"


def chunk_function(node: Node, body: Node | None, signature: str, source: bytes) -> list[CodeChunk]:
    """Split a function's body into embeddable pieces.

    Small functions (the common case) stay as ONE chunk covering the whole
    function — no special-casing needed downstream between "chunked" and
    "not chunked" (Option B from the design discussion). Oversized functions
    are split at natural boundaries (each top-level statement in the body),
    greedily grouped so each chunk stays close to the target size, never by
    blind line/character counting — this reuses the same syntax tree
    already built for parsing, no extra parse pass.
    """
    whole_text = node_text(node, source)
    if len(whole_text) <= _CHUNK_TARGET_CHARS or body is None or not body.children:
        return [
            CodeChunk(
                index=0,
                line_start=node.start_point[0] + 1,
                line_end=node.end_point[0] + 1,
                text=whole_text,
            )
        ]

    chunks: list[CodeChunk] = []

    def flush(pieces: list[Node]) -> None:
        if not pieces:
            return
        piece_text = source[pieces[0].start_byte : pieces[-1].end_byte].decode(
            "utf-8", errors="replace"
        )
        chunks.append(
            CodeChunk(
                index=len(chunks),
                line_start=pieces[0].start_point[0] + 1,
                line_end=pieces[-1].end_point[0] + 1,
                # Anchored with the function's own signature — these pieces
                # are body-interior only, so without this they'd have no
                # indication of which function they belong to.
                text=f"{signature}\n{piece_text}",
            )
        )

    current: list[Node] = []
    current_len = 0
    for statement in body.children:
        statement_len = statement.end_byte - statement.start_byte
        if current and current_len + statement_len > _CHUNK_TARGET_CHARS:
            flush(current)
            current, current_len = [], 0
        current.append(statement)
        current_len += statement_len
    flush(current)

    return chunks


# --- custom (non-tree-sitter) parsers ----------------------------------------
#
# Hand-written parsers for formats with no tree-sitter grammar available:
# BPMN (an XML dialect) and Avro IDL. Deliberately lightweight scans — not a
# full grammar/schema validator — matching this project's existing "cheap
# text-based signal, not a full compiler" philosophy (see edge_detector.py).
# Each returns plain ParsedSymbol objects directly, bypassing the tree-sitter
# walker entirely (there's no tree-sitter Node to walk). Both are dispatched
# from parse_source() below, before it ever reaches the tree-sitter path.

# --- BPMN ---
#
# A <process> element is treated as a class; its flow nodes (tasks, events,
# gateways — anything whose local tag ends in one of these) become methods.
# xml.etree.ElementTree doesn't track source line numbers, so line numbers
# here are recovered by a second, text-based pass: find each element's own
# `id="..."` attribute in the raw source and count newlines up to it. This
# is approximate (an id reused elsewhere would resolve to the wrong
# occurrence) but good enough for "here's roughly where this task lives",
# and never crashes on a well-formed BPMN file.

_FLOW_NODE_SUFFIXES = ("Task", "Event", "Gateway")


def _local_tag(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _line_and_index_of(text: str, needle: str, start: int = 0) -> tuple[int, int]:
    idx = text.find(needle, start)
    if idx == -1:
        return 1, start
    return text.count("\n", 0, idx) + 1, idx


def _tag_snippet(text: str, idx: int) -> str:
    tag_start = text.rfind("<", 0, idx + 1)
    tag_end = text.find(">", idx)
    if tag_start == -1 or tag_end == -1:
        return text[idx : idx + 80]
    return text[tag_start : tag_end + 1]


def parse_bpmn_source(source: bytes, file_path: str) -> list[ParsedSymbol]:
    text = source.decode("utf-8", errors="replace")
    try:
        root = ET.fromstring(source)
    except ET.ParseError:
        return []

    classes: list[ParsedSymbol] = []
    functions: list[ParsedSymbol] = []
    search_pos = 0

    for process in root.iter():
        if _local_tag(process.tag) != "process":
            continue
        proc_id = process.get("id") or process.get("name") or "process"
        proc_name = process.get("name") or proc_id
        line, idx = _line_and_index_of(text, f'id="{proc_id}"', search_pos)
        qname = f"{file_path}::{proc_id}"
        classes.append(
            ParsedSymbol(
                kind="class",
                name=proc_name,
                qualified_name=qname,
                file_path=file_path,
                line_start=line,
                line_end=line,
                signature=f"process {proc_id}",
                docstring=None,
                language="bpmn",
                source_snippet=_tag_snippet(text, idx),
            )
        )

        node_search_pos = idx
        for child in process:
            local = _local_tag(child.tag)
            if not local.lower().endswith(tuple(s.lower() for s in _FLOW_NODE_SUFFIXES)):
                continue
            child_id = child.get("id")
            if not child_id:
                continue
            child_name = child.get("name") or child_id
            c_line, c_idx = _line_and_index_of(text, f'id="{child_id}"', node_search_pos)
            functions.append(
                ParsedSymbol(
                    kind="function",
                    name=child_name,
                    qualified_name=f"{qname}.{child_id}",
                    file_path=file_path,
                    line_start=c_line,
                    line_end=c_line,
                    signature=f"{local} {child_id}",
                    docstring=None,
                    language="bpmn",
                    source_snippet=_tag_snippet(text, c_idx),
                    is_method=True,
                    parent_class=qname,
                )
            )
        search_pos = idx + 1

    return classes + functions


# --- Avro IDL (.avdl) ---
#
# `record`/`error`/`enum`/`protocol NAME { ... }` blocks become classes;
# a record/error's `;`-terminated field statements and a protocol's
# `;`-terminated message declarations become methods. Brace depth is
# tracked manually (not a real grammar) so a field like
# `union { null, string } email = null;` doesn't get split mid-union.

_BLOCK_RE = re.compile(r"\b(protocol|record|error|enum)\s+(\w+)\s*\{")
_NESTED_BLOCK_RE = re.compile(r"\b(record|error|enum)\s+(\w+)\s*\{")


def _strip_comments(text: str) -> str:
    """Blanks out comments while preserving line count, so line numbers
    computed on the result still line up with the original source."""
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text) - 1


def _split_top_level_statements(body: str) -> list[tuple[int, str]]:
    """[(offset_into_body, statement_text), ...] — split on `;` only at
    brace depth 0, so a nested `{ ... }` (a union type, an inline enum)
    never gets torn in half."""
    statements: list[tuple[int, str]] = []
    depth = 0
    start = 0
    for i, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == ";" and depth == 0:
            statements.append((start, body[start:i]))
            start = i + 1
    return statements


def _strip_nested_blocks(body: str) -> str:
    """Blanks out nested record/error/enum blocks inside a protocol body
    (each already gets its own top-level match from _BLOCK_RE, so they'd
    otherwise be double-counted as protocol "messages" too) — preserves
    line count via newline substitution, same trick as _strip_comments."""
    result = []
    pos = 0
    for m in _NESTED_BLOCK_RE.finditer(body):
        if m.start() < pos:
            continue
        result.append(body[pos : m.start()])
        close_idx = _find_matching_brace(body, m.end() - 1)
        removed = body[m.start() : close_idx + 1]
        result.append("\n" * removed.count("\n"))
        pos = close_idx + 1
    result.append(body[pos:])
    return "".join(result)


_MESSAGE_NAME_RE = re.compile(r"\b(\w+)\s*\(")
# search(), not match(): the field name is often preceded by a multi-word
# type (`union { null, string } email = null`) — search() tries every
# starting position until one satisfies the whole pattern through to `$`,
# which naturally lands on the LAST identifier (the actual field name), not
# the first word of the type. match() would only ever try position 0 and
# fail on anything but a single-word type with no default value.
_FIELD_NAME_RE = re.compile(r"\b(\w+)\s*(?:=.*)?$", re.DOTALL)


def parse_avdl_source(source: bytes, file_path: str) -> list[ParsedSymbol]:
    raw = source.decode("utf-8", errors="replace")
    text = _strip_comments(raw)
    classes: list[ParsedSymbol] = []
    functions: list[ParsedSymbol] = []

    for m in _BLOCK_RE.finditer(text):
        keyword, name = m.group(1), m.group(2)
        open_idx = m.end() - 1
        close_idx = _find_matching_brace(text, open_idx)
        line_start = text.count("\n", 0, m.start()) + 1
        line_end = text.count("\n", 0, close_idx) + 1
        qname = f"{file_path}::{name}"
        classes.append(
            ParsedSymbol(
                kind="class",
                name=name,
                qualified_name=qname,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                signature=f"{keyword} {name}",
                docstring=None,
                language="avdl",
                source_snippet=raw[m.start() : close_idx + 1][:2000],
            )
        )

        body = text[open_idx + 1 : close_idx]
        body_offset = open_idx + 1
        member_area = _strip_nested_blocks(body) if keyword == "protocol" else body
        name_pattern = _MESSAGE_NAME_RE if keyword == "protocol" else _FIELD_NAME_RE

        for stmt_offset, stmt in _split_top_level_statements(member_area):
            stmt_stripped = stmt.strip()
            if not stmt_stripped:
                continue
            match = name_pattern.search(stmt_stripped)
            if not match:
                continue
            member_name = match.group(1)
            member_line = text.count("\n", 0, body_offset + stmt_offset) + 1
            functions.append(
                ParsedSymbol(
                    kind="function",
                    name=member_name,
                    qualified_name=f"{qname}.{member_name}",
                    file_path=file_path,
                    line_start=member_line,
                    line_end=member_line,
                    signature=stmt_stripped[:200],
                    docstring=None,
                    language="avdl",
                    source_snippet=stmt_stripped[:500],
                    is_method=True,
                    parent_class=qname,
                )
            )

    return classes + functions


# --- core walker / entry points ----------------------------------------------

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

# An import line in every language this parser handles: Python's `import x`
# and `from x import y`, JS/TS's `import ... from "..."` and re-exports,
# Kotlin/Java's `import x.y.Z`. Matched as text rather than through
# tree-sitter because it is the same shape everywhere and needs no grammar.
_IMPORT_LINE = re.compile(r"^\s*(?:import\b|from\s+\S+\s+import\b|export\s.*\bfrom\b)")
# Imports live at the top of a file; reading further is wasted work and
# risks picking up a dynamic import() buried in a function body.
_IMPORT_SCAN_LINES = 120
# Enough for a long import block, short enough that several files' worth
# still leaves room for the code itself in a prompt.
MAX_IMPORT_CHARS = 1200

# How much of a test file is kept. Measured against the tests that cover the
# ATS settings form: 1,605 / 2,235 / 3,920 characters, so a focused unit test
# fits whole. The one file this truncates is a 29,652-char suite covering a
# whole settings tab, and its first 6,000 characters are its imports and
# first fixture — the part worth copying. Raising it further buys more of
# the files that teach least.
MAX_TEST_SOURCE_CHARS = 6000


def extract_imports(source: str) -> str:
    """The import lines at the top of `source`, in order, as one block.

    Blank lines and comments between imports are dropped; the first run of
    real code ends the scan. Returns "" when a file imports nothing.
    """
    kept: list[str] = []
    for line in source.splitlines()[:_IMPORT_SCAN_LINES]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("//", "#", "/*", "*", "*/")):
            continue
        if _IMPORT_LINE.match(line):
            kept.append(line.rstrip())
            continue
        if kept:
            break  # past the import block
    block = "\n".join(kept)
    return block[:MAX_IMPORT_CHARS]


LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",  # plain javascript's own grammar already parses JSX
    ".kt": "kotlin",
    ".graphqls": "graphql",
    ".bpmn": "bpmn",  # custom parser — no tree-sitter grammar, see _CUSTOM_PARSERS
    ".avdl": "avdl",  # custom parser — no tree-sitter grammar, see _CUSTOM_PARSERS
}

# Custom (non-tree-sitter) parsers, dispatched before the generic tree-sitter
# path in parse_source() below — each returns ParsedSymbol objects directly
# since there's no tree-sitter Node to walk for these formats.
_CUSTOM_PARSERS = {
    "bpmn": parse_bpmn_source,
    "avdl": parse_avdl_source,
}

# Extensions with no code-shaped structure to extract (no classes/functions
# make sense for these) but still worth a File node — tracked separately
# from LANGUAGE_BY_EXTENSION, which implies "this produces real symbols".
# See RepoGraph.tracked_files.
TRACKED_ONLY_EXTENSIONS: dict[str, str] = {
    ".html": "html",
    ".css": "css",
    ".csv": "csv",
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


def parse_source(source: bytes, file_path: str, language: str) -> list[ParsedSymbol]:
    """Parse raw source bytes already in memory — the incremental-update
    path's entry point: a single changed file's content, fetched via the
    GitHub API, with no repo clone and no file on disk at all.

    Raises on a genuine parse failure (tree-sitter itself doesn't raise on
    malformed syntax — it produces error nodes — so this mainly covers
    decode errors); callers are expected to catch and count, not let one
    bad file abort a run.
    """
    custom_parser = _CUSTOM_PARSERS.get(language)
    if custom_parser is not None:
        return custom_parser(source, file_path)

    parser = get_parser(language)
    tree = parser.parse(source)
    adapter = ADAPTERS[language]

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


def parse_file(path: Path, repo_root: Path, language: str) -> list[ParsedSymbol]:
    """Parse one file already on disk, returning its classes + functions
    (kind-tagged, flat) — the full-repo-scan path's entry point (see
    parse_repo() below). Thin wrapper around parse_source(): reads the
    file's bytes and derives its repo-relative file_path, nothing else.
    """
    source = path.read_bytes()
    file_path = str(path.relative_to(repo_root).as_posix())
    return parse_source(source, file_path, language)


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
                tracked_language = TRACKED_ONLY_EXTENSIONS.get(path.suffix.lower())
                if tracked_language is not None:
                    rel_path = str(path.relative_to(repo_root).as_posix())
                    graph.tracked_files[rel_path] = tracked_language
                else:
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
            rel_path = str(path.relative_to(repo_root).as_posix())
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                text = ""
            imports = extract_imports(text) if text else ""
            if imports:
                graph.file_imports[rel_path] = imports
            # Read once, used twice — the import scan already has the text.
            if text and is_test_path(rel_path):
                graph.file_source[rel_path] = text[:MAX_TEST_SOURCE_CHARS]
            for symbol in symbols:
                if symbol.kind == "class":
                    graph.classes.append(symbol)
                else:
                    graph.functions.append(symbol)

    return graph
