"""tree-sitter based extraction of symbols, imports and chunks from a repo.

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

Version 4 of the pipeline (see ingest.PIPELINE_VERSION) changed three
things here, all measured against 13 real services:

- Every file is read and kept whole (see file_inventory.py), so nothing a
  parser skips is lost — 20% of code lines sat outside every symbol.
- More of the code is a symbol: type aliases, enums, records, Kotlin
  objects, top-level constants, test blocks, and functions handed to
  useCallback/useMemo. A nested symbol is named after its parent, so two
  `handleClick`s in one file no longer merge into one node.
- Chunks are cut by LINE OWNERSHIP: each line belongs to the innermost
  symbol that owns it, or to the file, and is embedded exactly once. A
  chunk's text is exactly its lines — no repeated signature prefix, no
  4,990-character `return (...)`, no chunk that is a signature plus `};`.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from tree_sitter import Node
from tree_sitter_language_pack import get_parser

from . import code_facts
from .file_inventory import FileRecord, build_record
from .test_paths import is_test_path

__all__ = [
    "CodeChunk",
    "ImportSpec",
    "LanguageAdapter",
    "ParsedFile",
    "ParsedSymbol",
    "RepoGraph",
    "DEFAULT_DENYLIST",
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "LANGUAGE_BY_EXTENSION",
    "TRACKED_ONLY_EXTENSIONS",
    "parse_file",
    "parse_repo",
    "parse_source",
    "parse_source_full",
]


# --- data shapes ----------------------------------------------------------


@dataclass
class CodeChunk:
    index: int
    line_start: int
    line_end: int
    text: str  # exactly the file's lines line_start..line_end
    # One line naming where the chunk sits, e.g.
    # "// view.tsx › InfiniteScrollGrid (lines 90–120)". Shown to a reader;
    # not embedded — embedding_text() already adds the path and name.
    header: str = ""


@dataclass
class ParsedSymbol:
    kind: str  # "class" | "function" | "variable" — which node label it gets
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
    chunks: list[CodeChunk] = field(default_factory=list)
    # What it is in its own language: class, interface, enum, record,
    # object, type, function, method, constructor, test, const, field, ...
    symbol_kind: str = ""
    # The innermost symbol this one sits inside (a class for a method, a
    # component for the handler declared in it), or None at the top level.
    parent_symbol: str | None = None
    # Names this symbol's own code uses (calls, JSX tags, types), excluding
    # code inside its nested symbols. Resolved into REFERENCES edges.
    identifiers: list[str] = field(default_factory=list)
    # Names it extends or implements, generics stripped.
    supertypes: list[str] = field(default_factory=list)
    # Domain facts read from its text: a JPA table, a Kafka message type, a
    # REST route... (see code_facts.py).
    facts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol_kind:
            self.symbol_kind = {
                "class": "class",
                "function": "method" if self.is_method else "function",
                "variable": "const",
            }.get(self.kind, self.kind)


@dataclass
class ImportSpec:
    """One import or re-export statement, structured.

    `names` holds (imported_name, local_name) pairs: ("default", "Grid") for a
    default import, ("*", "ns") for a namespace import, and the plain name
    twice for `import { A }`. A side-effect import has no names. `wildcard`
    is `export * from` in TypeScript and `import a.b.*` on the JVM."""

    module: str
    names: list[tuple[str, str]] = field(default_factory=list)
    reexport: bool = False
    wildcard: bool = False
    line: int = 0

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "names": [list(pair) for pair in self.names],
            "reexport": self.reexport,
            "wildcard": self.wildcard,
            "line": self.line,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImportSpec:
        return cls(
            module=data["module"],
            names=[(a, b) for a, b in data.get("names", [])],
            reexport=bool(data.get("reexport")),
            wildcard=bool(data.get("wildcard")),
            line=int(data.get("line") or 0),
        )


@dataclass
class ParsedFile:
    """Everything one parse of one file produced."""

    symbols: list[ParsedSymbol]
    imports: list[ImportSpec] = field(default_factory=list)
    package: str | None = None
    # Chunks of the lines no symbol owns: top-level statements, decorators,
    # `export default`, a constants table. Owned by the File itself.
    region_chunks: list[CodeChunk] = field(default_factory=list)
    import_lines: set[int] = field(default_factory=set)
    facts: dict = field(default_factory=dict)


@dataclass
class RepoGraph:
    classes: list[ParsedSymbol]
    functions: list[ParsedSymbol]
    variables: list[ParsedSymbol] = field(default_factory=list)
    # file_path -> language, for files with no symbols (an .html, a barrel,
    # a .sql) — kept for callers that only need "which files have no
    # symbols"; `files` below is the full picture.
    tracked_files: dict[str, str] = field(default_factory=dict)
    # file_path -> its import lines, verbatim (shown to the planner so code
    # it writes uses module paths that exist).
    file_imports: dict[str, str] = field(default_factory=dict)
    # file_path -> its whole text, every text file (capped, see
    # file_inventory.MAX_STORED_SOURCE_CHARS). Before version 4 only test
    # files kept their text; production files survived only as symbols,
    # and everything between symbols was lost.
    file_source: dict[str, str] = field(default_factory=dict)
    # file_path -> chunks of the lines no symbol owns (see ParsedFile).
    file_chunks: dict[str, list[CodeChunk]] = field(default_factory=dict)
    # file_path -> FileRecord: EVERY file under the root, parsed or not.
    files: dict[str, FileRecord] = field(default_factory=dict)
    file_import_specs: dict[str, list[ImportSpec]] = field(default_factory=dict)
    file_packages: dict[str, str] = field(default_factory=dict)
    file_facts: dict[str, dict] = field(default_factory=dict)
    # file_path -> (code lines, code lines inside a symbol or chunk), for
    # the coverage report.
    file_coverage: dict[str, tuple[int, int]] = field(default_factory=dict)
    files_parsed: int = 0
    files_skipped_unsupported: int = 0
    files_skipped_oversized: int = 0
    files_failed: int = 0

    @property
    def symbols(self) -> list[ParsedSymbol]:
        return [*self.classes, *self.functions, *self.variables]


@dataclass
class LanguageAdapter:
    """Everything specific to one language/grammar — the only place its
    quirks are allowed to live. Every hook receives tree-sitter nodes and
    returns plain data, never raises, and returns None/empty when nothing
    is found (never guess)."""

    language: str
    class_types: set[str]
    function_types: set[str]
    name_of: Callable[[Node], str | None]
    body_of: Callable[[Node], Node | None]
    docstring_of: Callable[[Node, Node | None, bytes], str | None]
    # node -> its symbol_kind ("interface", "enum", "test", ...); None means
    # the generic "class"/"function".
    kind_of: Callable[[Node], str | None] = lambda node: None
    # node -> the whole declaration it sits in (`export const X = ...` for
    # an arrow function), which is what its snippet and lines cover.
    root_of: Callable[[Node], Node] = lambda node: node
    supertypes_of: Callable[[Node], list[str]] = lambda node: []
    # Node types that may declare top-level variables, and the hook that
    # names them: node -> [(name, root_node, symbol_kind)].
    variable_types: set[str] = field(default_factory=set)
    variables_of: Callable[[Node], list[tuple[str, Node, str]]] = lambda node: []
    identifier_types: frozenset[str] = frozenset()
    # root node -> (imports, package, import line numbers)
    imports_of: Callable[[Node], tuple[list[ImportSpec], str | None, set[int]]] = (
        lambda root: ([], None, set())
    )


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
    """`file::name` at the top level, `file::Parent.name` inside a symbol —
    any symbol, not only a class, so nested names cannot collide."""
    if parent_class is None:
        return f"{file_path}::{name}"
    parent_simple = parent_class.split("::", 1)[1]
    return f"{file_path}::{parent_simple}.{name}"


def _node_key(node: Node) -> tuple[int, int, str]:
    return (node.start_byte, node.end_byte, node.type)


def _child_of_type(node: Node, *types: str) -> Node | None:
    for child in node.children:
        if child.type in types:
            return child
    return None


def _strip_generics(name: str) -> str:
    return re.sub(r"<.*", "", name).strip().split(".")[-1].strip("?")


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


def _types_in(node: Node | None, type_names: set[str]) -> list[str]:
    """Every descendant of `node` of one of `type_names`, as text, in order
    — used to read the supertypes out of a heritage clause."""
    if node is None:
        return []
    found: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in type_names:
            text = text_of(current)
            if text:
                found.append(_strip_generics(text))
            continue
        stack.extend(reversed(current.children))
    return found


# --- Python -----------------------------------------------------------------


def _python_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    if body is None or not body.children:
        return None
    first = body.children[0]
    if first.type == "expression_statement" and first.children:
        first = first.children[0]
    if first.type != "string":
        return None
    text = first.text.decode("utf-8", errors="replace").strip()
    return text.strip("\"'").strip() or None


def _python_root_of(node: Node) -> Node:
    if node.parent is not None and node.parent.type == "decorated_definition":
        return node.parent
    return node


def _python_supertypes_of(node: Node) -> list[str]:
    return _types_in(node.child_by_field_name("superclasses"), {"identifier", "attribute"})


def _python_variables_of(node: Node) -> list[tuple[str, Node, str]]:
    # Module level only: an assignment inside a function is a local. The
    # bundled grammar puts a module-level `X = 1` straight under `module`;
    # other versions wrap it in an expression_statement — accept both.
    if node.parent is None or node.parent.type != "module":
        return []
    assignment = node
    if node.type != "assignment":
        assignment = node.children[0] if node.children else None
    if assignment is None or assignment.type != "assignment":
        return []
    left = assignment.child_by_field_name("left")
    if left is None or left.type != "identifier":
        return []
    return [(text_of(left) or "", node, "const")]


def _python_imports_of(root: Node) -> tuple[list[ImportSpec], str | None, set[int]]:
    imports: list[ImportSpec] = []
    lines: set[int] = set()
    for child in root.children:
        if child.type == "import_statement":
            for name_node in child.children:
                if name_node.type == "dotted_name":
                    module = text_of(name_node) or ""
                    imports.append(ImportSpec(module, [(module, module.split(".")[0])]))
                elif name_node.type == "aliased_import":
                    module = text_of(name_node.child_by_field_name("name")) or ""
                    alias = text_of(name_node.child_by_field_name("alias")) or module
                    imports.append(ImportSpec(module, [(module, alias)]))
        elif child.type == "import_from_statement":
            module_node = child.child_by_field_name("module_name")
            module = text_of(module_node) or ""
            names: list[tuple[str, str]] = []
            wildcard = False
            for name_node in child.children_by_field_name("name"):
                if name_node.type == "aliased_import":
                    original = text_of(name_node.child_by_field_name("name")) or ""
                    alias = text_of(name_node.child_by_field_name("alias")) or original
                    names.append((original, alias))
                else:
                    plain = text_of(name_node) or ""
                    names.append((plain, plain))
            if _child_of_type(child, "wildcard_import") is not None:
                wildcard = True
            imports.append(ImportSpec(module, names, wildcard=wildcard))
        else:
            continue
        imports[-1].line = child.start_point[0] + 1
        lines.update(range(child.start_point[0] + 1, child.end_point[0] + 2))
    return imports, None, lines


PYTHON_ADAPTER = LanguageAdapter(
    language="python",
    class_types={"class_definition"},
    function_types={"function_definition"},
    name_of=field_name_of,
    body_of=field_body_of,
    docstring_of=_python_docstring_of,
    root_of=_python_root_of,
    supertypes_of=_python_supertypes_of,
    variable_types={"expression_statement", "assignment"},
    variables_of=_python_variables_of,
    identifier_types=frozenset({"identifier"}),
    imports_of=_python_imports_of,
)


# --- Java --------------------------------------------------------------------


def _java_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    return adjacent_comment_docstring(node, {"block_comment"})


_JAVA_KINDS = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "annotation",
    "constructor_declaration": "constructor",
    "compact_constructor_declaration": "constructor",
}


def _java_supertypes_of(node: Node) -> list[str]:
    types = {"type_identifier", "scoped_type_identifier", "generic_type"}
    found: list[str] = []
    for clause in ("superclass", "super_interfaces", "extends_interfaces"):
        found += _types_in(_child_of_type(node, clause), types)
    return found


def _java_variables_of(node: Node) -> list[tuple[str, Node, str]]:
    # Constants only — `static final` fields. Other fields are part of their
    # class's own lines and need no node of their own.
    modifiers = _child_of_type(node, "modifiers")
    words = set((text_of(modifiers) or "").split())
    if not {"static", "final"} <= words:
        return []
    names = []
    for declarator in node.children:
        if declarator.type == "variable_declarator":
            name = field_name_of(declarator)
            if name:
                names.append((name, node, "const"))
    return names


def _jvm_dotted(node: Node | None) -> str:
    return re.sub(r"\s+", "", text_of(node) or "")


def _java_imports_of(root: Node) -> tuple[list[ImportSpec], str | None, set[int]]:
    imports: list[ImportSpec] = []
    lines: set[int] = set()
    package: str | None = None
    for child in root.children:
        if child.type == "package_declaration":
            target = _child_of_type(child, "scoped_identifier", "identifier")
            package = _jvm_dotted(target) or None
        elif child.type == "import_declaration":
            target = _child_of_type(child, "scoped_identifier", "identifier")
            dotted = _jvm_dotted(target)
            wildcard = _child_of_type(child, "asterisk") is not None
            if wildcard:
                spec = ImportSpec(dotted, wildcard=True)
            else:
                last = dotted.rsplit(".", 1)[-1]
                spec = ImportSpec(dotted, [(last, last)])
            spec.line = child.start_point[0] + 1
            imports.append(spec)
            lines.update(range(child.start_point[0] + 1, child.end_point[0] + 2))
    return imports, package, lines


JAVA_ADAPTER = LanguageAdapter(
    language="java",
    class_types={
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        "record_declaration",
        "annotation_type_declaration",
    },
    function_types={
        "method_declaration",
        "constructor_declaration",
        "compact_constructor_declaration",
    },
    name_of=field_name_of,
    body_of=field_body_of,
    docstring_of=_java_docstring_of,
    kind_of=lambda node: _JAVA_KINDS.get(node.type),
    supertypes_of=_java_supertypes_of,
    variable_types={"field_declaration"},
    variables_of=_java_variables_of,
    identifier_types=frozenset({"identifier", "type_identifier"}),
    imports_of=_java_imports_of,
)


# --- TypeScript / TSX / JavaScript -------------------------------------------
#
# Same grammar family, JS is the base that TS/TSX extend, so all three share
# identical node shapes (confirmed against real grammars, not guessed — see
# the language-support design discussion).

# A function handed to one of these and assigned to a const takes the
# const's name: `const save = useCallback(() => ...)` is `save`. Measured on
# a real ticket: the function holding the bug (`updateArchiveButton`, a
# useCallback) was not a symbol at all, only text inside its component.
_JS_FUNCTION_WRAPPERS = {
    "useCallback", "useMemo", "memo", "forwardRef", "observer", "debounce", "throttle",
    "useEvent", "useEffectEvent", "React.memo", "React.forwardRef", "React.useCallback",
    "React.useMemo",
}
# Test-block calls whose function argument becomes a `test` symbol named
# after the call and its title: `it("renders with title prop")`.
_JS_TEST_BLOCKS = {
    "describe", "it", "test", "context", "suite", "beforeEach", "afterEach", "beforeAll",
    "afterAll",
}
_JS_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "generator_function"}

_TS_KINDS = {
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "class": "class",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
    "method_definition": "method",
}


def _call_of_argument(node: Node) -> Node | None:
    """The call a function is passed to, when it is passed straight in."""
    parent = node.parent
    if parent is None or parent.type != "arguments":
        return None
    call = parent.parent
    return call if call is not None and call.type == "call_expression" else None


def _test_block_name(call: Node) -> str | None:
    callee = text_of(call.child_by_field_name("function")) or ""
    base = callee.split(".")[0]
    if base not in _JS_TEST_BLOCKS or "(" in callee or "`" in callee:
        return None
    args = call.child_by_field_name("arguments")
    first = next((c for c in args.children if c.is_named), None) if args is not None else None
    if first is not None and first.type in ("string", "template_string"):
        title = (text_of(first) or "").strip("`'\"")
        title = re.sub(r"\s+", " ", title)[:100]
        return f'{base}("{title}")'
    return base if base.startswith(("before", "after")) else None


def _wrapper_declarator(call: Node) -> Node | None:
    callee = text_of(call.child_by_field_name("function")) or ""
    if callee not in _JS_FUNCTION_WRAPPERS:
        return None
    parent = call.parent
    return parent if parent is not None and parent.type == "variable_declarator" else None


def _typescript_name_of(node: Node) -> str | None:
    name = field_name_of(node)
    if name is not None:
        return name
    parent = node.parent
    if parent is None:
        return None
    # An arrow function has no name of its own (it's anonymous) — if it's
    # the direct value of `const x = (...) => {...}`, borrow x's name.
    if node.type in _JS_FUNCTION_VALUES and parent.type == "variable_declarator":
        return field_name_of(parent)
    # A class field holding a function: `handleClick = () => {...}`.
    if node.type in _JS_FUNCTION_VALUES and parent.type in (
        "public_field_definition",
        "field_definition",
    ):
        return text_of(parent.child_by_field_name("name"))
    call = _call_of_argument(node) if node.type in _JS_FUNCTION_VALUES else None
    if call is not None:
        declarator = _wrapper_declarator(call)
        if declarator is not None:
            return field_name_of(declarator)
        return _test_block_name(call)
    # Anything else (an inline callback like `.map(x => x+1)`) stays unnamed
    # and is walked through rather than guessed.
    return None


def _typescript_kind_of(node: Node) -> str | None:
    if node.type in _TS_KINDS:
        return _TS_KINDS[node.type]
    call = _call_of_argument(node) if node.type in _JS_FUNCTION_VALUES else None
    if call is not None and _wrapper_declarator(call) is None and _test_block_name(call):
        return "test"
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


def _typescript_root_of(node: Node) -> Node:
    if node.parent is not None and node.parent.type in (
        "public_field_definition",
        "field_definition",
    ):
        return node.parent
    call = _call_of_argument(node) if node.type in _JS_FUNCTION_VALUES else None
    if call is not None:
        if _wrapper_declarator(call) is not None:
            return _ts_statement_root(call)
        if call.parent is not None and call.parent.type == "expression_statement":
            return call.parent
        return call
    return _ts_statement_root(node)


def _typescript_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    doc = adjacent_comment_docstring(node, {"comment"})
    if doc is not None:
        return doc
    # A top-level exported class/function (and any decorator on it, e.g.
    # NestJS's @Controller()) is wrapped in export_statement — and a
    # `const`-assigned arrow function is wrapped twice more on top of that —
    # so its own prev_sibling is never the doc comment; check the outermost
    # wrapper instead.
    root = _typescript_root_of(node)
    if root is not node:
        return adjacent_comment_docstring(root, {"comment"})
    return None


def _typescript_supertypes_of(node: Node) -> list[str]:
    types = {"identifier", "type_identifier", "member_expression", "nested_type_identifier"}
    found: list[str] = []
    heritage = _child_of_type(node, "class_heritage")
    if heritage is not None:
        for clause in heritage.children:
            found += [t for t in _types_in(clause, types) if t]
    found += _types_in(_child_of_type(node, "extends_type_clause"), types)
    return found


def _typescript_variables_of(node: Node) -> list[tuple[str, Node, str]]:
    """Top-level `const X = <not a function>` — styled components, config
    objects, constants tables. A const holding a function is a function
    symbol instead (see _typescript_name_of)."""
    parent = node.parent
    if parent is None:
        return []
    top_level = parent.type == "program" or (
        parent.type == "export_statement" and parent.parent is not None
        and parent.parent.type == "program"
    )
    if not top_level:
        return []
    kind = "const" if (text_of(node) or "").lstrip().startswith("const") else "variable"
    found = []
    for declarator in node.children:
        if declarator.type != "variable_declarator":
            continue
        name_node = declarator.child_by_field_name("name")
        if name_node is None or name_node.type != "identifier":
            continue  # a destructuring pattern names nothing new worth a node
        value = declarator.child_by_field_name("value")
        if value is not None and value.type in _JS_FUNCTION_VALUES:
            continue
        if value is not None and value.type == "call_expression" and _wrapper_declarator(value):
            continue
        found.append((text_of(name_node) or "", _ts_statement_root(declarator), kind))
    return found


def _string_value(node: Node | None) -> str | None:
    if node is None:
        return None
    fragment = _child_of_type(node, "string_fragment")
    if fragment is not None:
        return text_of(fragment)
    text = text_of(node) or ""
    return text.strip("'\"`") or None


def _typescript_imports_of(root: Node) -> tuple[list[ImportSpec], str | None, set[int]]:
    imports: list[ImportSpec] = []
    lines: set[int] = set()
    for child in root.children:
        if child.type == "import_statement":
            module = _string_value(child.child_by_field_name("source"))
            if module is None:
                continue
            names: list[tuple[str, str]] = []
            clause = _child_of_type(child, "import_clause")
            if clause is not None:
                for part in clause.children:
                    if part.type == "identifier":
                        names.append(("default", text_of(part) or ""))
                    elif part.type == "namespace_import":
                        alias = _child_of_type(part, "identifier")
                        names.append(("*", text_of(alias) or ""))
                    elif part.type == "named_imports":
                        for spec in part.children:
                            if spec.type == "import_specifier":
                                original = text_of(spec.child_by_field_name("name")) or ""
                                alias = text_of(spec.child_by_field_name("alias")) or original
                                names.append((original, alias))
            imports.append(ImportSpec(module, names, line=child.start_point[0] + 1))
            lines.update(range(child.start_point[0] + 1, child.end_point[0] + 2))
        elif child.type == "export_statement" and child.child_by_field_name("source") is not None:
            module = _string_value(child.child_by_field_name("source")) or ""
            names = []
            wildcard = False
            clause = _child_of_type(child, "export_clause")
            namespace = _child_of_type(child, "namespace_export")
            if clause is not None:
                for spec in clause.children:
                    if spec.type == "export_specifier":
                        original = text_of(spec.child_by_field_name("name")) or ""
                        alias = text_of(spec.child_by_field_name("alias")) or original
                        names.append((original, alias))
            elif namespace is not None:
                names.append(("*", text_of(_child_of_type(namespace, "identifier")) or ""))
            else:
                wildcard = True
            imports.append(
                ImportSpec(module, names, reexport=True, wildcard=wildcard,
                           line=child.start_point[0] + 1)
            )
            lines.update(range(child.start_point[0] + 1, child.end_point[0] + 2))
    # require("x") and lazy import("x") anywhere in the file: a route table
    # importing pages lazily is still a real dependency.
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "call_expression":
            callee = node.child_by_field_name("function")
            is_require = callee is not None and callee.type == "identifier" and (
                text_of(callee) == "require"
            )
            if callee is not None and (callee.type == "import" or is_require):
                args = node.child_by_field_name("arguments")
                first = next((c for c in args.children if c.is_named), None) if args else None
                if first is not None and first.type == "string":
                    module = _string_value(first)
                    if module:
                        imports.append(ImportSpec(module, [], line=node.start_point[0] + 1))
        stack.extend(node.children)
    return imports, None, lines


def _make_js_family_adapter(language: str) -> LanguageAdapter:
    return LanguageAdapter(
        language=language,
        class_types={
            "class_declaration",
            "abstract_class_declaration",
            "interface_declaration",
            "type_alias_declaration",
            "enum_declaration",
        },
        function_types={
            "method_definition",
            "function_declaration",
            "generator_function_declaration",
            *_JS_FUNCTION_VALUES,
        },
        name_of=_typescript_name_of,
        body_of=field_body_of,
        docstring_of=_typescript_docstring_of,
        kind_of=_typescript_kind_of,
        root_of=_typescript_root_of,
        supertypes_of=_typescript_supertypes_of,
        variable_types={"lexical_declaration", "variable_declaration"},
        variables_of=_typescript_variables_of,
        identifier_types=frozenset(
            {"identifier", "type_identifier", "shorthand_property_identifier"}
        ),
        imports_of=_typescript_imports_of,
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
#
# The bundled grammar fails on `const val` inside an `object` and returns an
# ERROR node in its place. The walker descends through ERROR nodes, so the
# classes around one are still found; the constants themselves are read by
# code_facts.kotlin_constants() from the text instead.

_KOTLIN_NAME_CHILD_TYPE = {
    "class_declaration": "type_identifier",
    "object_declaration": "type_identifier",
    "companion_object": "type_identifier",
    "type_alias": "type_identifier",
    "function_declaration": "simple_identifier",
}


def _kotlin_name_of(node: Node) -> str | None:
    target_type = _KOTLIN_NAME_CHILD_TYPE.get(node.type)
    if target_type is None:
        return None
    for child in node.children:
        if child.type == target_type:
            return text_of(child)
    if node.type == "companion_object":
        return "Companion"
    return None


def _kotlin_body_of(node: Node) -> Node | None:
    for child in node.children:
        if child.type in ("class_body", "function_body", "enum_class_body"):
            return child
    return None


def _kotlin_docstring_of(node: Node, body: Node | None, source: bytes) -> str | None:
    return adjacent_comment_docstring(node, {"multiline_comment"})


def _kotlin_kind_of(node: Node) -> str | None:
    if node.type == "class_declaration":
        if _child_of_type(node, "enum_class_body") is not None:
            return "enum"
        if any(child.type == "interface" for child in node.children):
            return "interface"
        return "class"
    return {"object_declaration": "object", "companion_object": "object",
            "type_alias": "type"}.get(node.type)


def _kotlin_supertypes_of(node: Node) -> list[str]:
    found = []
    for child in node.children:
        if child.type == "delegation_specifier":
            user_type = _child_of_type(child, "user_type")
            if user_type is None:
                invocation = _child_of_type(child, "constructor_invocation")
                user_type = _child_of_type(invocation, "user_type") if invocation else None
            if user_type is not None:
                names = [text_of(c) for c in user_type.children if c.type == "type_identifier"]
                if names:
                    found.append(names[-1] or "")
    return found


def _kotlin_variables_of(node: Node) -> list[tuple[str, Node, str]]:
    if node.parent is None or node.parent.type != "source_file":
        return []
    declaration = _child_of_type(node, "variable_declaration")
    name = text_of(_child_of_type(declaration, "simple_identifier")) if declaration else None
    if not name:
        return []
    modifiers = text_of(_child_of_type(node, "modifiers")) or ""
    return [(name, node, "const" if "const" in modifiers.split() else "property")]


def _kotlin_imports_of(root: Node) -> tuple[list[ImportSpec], str | None, set[int]]:
    imports: list[ImportSpec] = []
    lines: set[int] = set()
    package: str | None = None
    headers: list[Node] = []
    for child in root.children:
        if child.type == "package_header":
            package = _jvm_dotted(_child_of_type(child, "identifier")) or None
        elif child.type == "import_list":
            headers.extend(c for c in child.children if c.type == "import_header")
        elif child.type == "import_header":
            headers.append(child)
    for header in headers:
        dotted = _jvm_dotted(_child_of_type(header, "identifier"))
        if not dotted:
            continue
        if _child_of_type(header, "wildcard_import") is not None:
            spec = ImportSpec(dotted, wildcard=True)
        else:
            alias_node = _child_of_type(header, "import_alias")
            last = dotted.rsplit(".", 1)[-1]
            alias = text_of(_child_of_type(alias_node, "type_identifier")) if alias_node else None
            spec = ImportSpec(dotted, [(last, alias or last)])
        spec.line = header.start_point[0] + 1
        imports.append(spec)
        lines.update(range(header.start_point[0] + 1, header.end_point[0] + 2))
    return imports, package, lines


KOTLIN_ADAPTER = LanguageAdapter(
    language="kotlin",
    class_types={"class_declaration", "object_declaration", "companion_object", "type_alias"},
    function_types={"function_declaration"},
    name_of=_kotlin_name_of,
    body_of=_kotlin_body_of,
    docstring_of=_kotlin_docstring_of,
    kind_of=_kotlin_kind_of,
    supertypes_of=_kotlin_supertypes_of,
    variable_types={"property_declaration"},
    variables_of=_kotlin_variables_of,
    identifier_types=frozenset({"simple_identifier", "type_identifier"}),
    imports_of=_kotlin_imports_of,
)


# --- GraphQL ----------------------------------------------------------------
#
# Types (object/interface/input/enum/scalar/union, and `extend type`) as
# classes, their fields as members, and — for a client `.graphql` file —
# named operations and fragments as functions. This grammar exposes NO named
# tree-sitter fields (`child_by_field_name` returns None for everything) —
# the tree is purely type-tagged, so name/body/docstring all do a type-based
# scan of direct children instead of the field-based lookup used above.
#
# Field ARGUMENTS (`user(id: ID!): User`) are also `input_value_definition`
# nodes, same as a real input-type field — but since a `field_definition`'s
# own "body" (per `_graphql_body_of` below) only recognizes
# fields_definition/input_fields_definition/enum_values_definition, the core
# walker never recurses into a field's `arguments_definition`, so arguments
# are never mistaken for top-level symbols.

_GRAPHQL_KINDS = {
    "object_type_definition": "type",
    "interface_type_definition": "interface",
    "input_object_type_definition": "input",
    "enum_type_definition": "enum",
    "scalar_type_definition": "scalar",
    "union_type_definition": "union",
    "object_type_extension": "extension",
    "interface_type_extension": "extension",
    "input_object_type_extension": "extension",
    "enum_type_extension": "extension",
    "field_definition": "field",
    "input_value_definition": "field",
    "operation_definition": "operation",
    "fragment_definition": "fragment",
}


def _graphql_child_of_type(node: Node, type_name: str) -> Node | None:
    return _child_of_type(node, type_name)


def _graphql_name_of(node: Node) -> str | None:
    if node.type == "fragment_definition":
        return text_of(_child_of_type(_child_of_type(node, "fragment_name") or node, "name"))
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


def _graphql_supertypes_of(node: Node) -> list[str]:
    return _types_in(_child_of_type(node, "implements_interfaces"), {"name"})


GRAPHQL_ADAPTER = LanguageAdapter(
    language="graphql",
    class_types={
        "object_type_definition",
        "interface_type_definition",
        "input_object_type_definition",
        "enum_type_definition",
        "scalar_type_definition",
        "union_type_definition",
        "object_type_extension",
        "interface_type_extension",
        "input_object_type_extension",
        "enum_type_extension",
    },
    function_types={
        "field_definition",
        "input_value_definition",
        "operation_definition",
        "fragment_definition",
    },
    name_of=_graphql_name_of,
    body_of=_graphql_body_of,
    docstring_of=_graphql_docstring_of,
    kind_of=lambda node: _GRAPHQL_KINDS.get(node.type),
    supertypes_of=_graphql_supertypes_of,
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
# Every line of a file has exactly one OWNER: the innermost symbol that owns
# lines and covers it, or the file. Each owner's lines are cut into chunks of
# about _CHUNK_TARGET_CHARS at line boundaries, preferring a line that ends
# a statement or block. A line is therefore embedded exactly once, and a
# component's chunks no longer repeat the code of the handlers declared in
# it — those are their own symbols with their own chunks.

# The ~500-800 char chunk-size target agreed in design: small enough to keep
# one embedding focused on one idea, large enough not to be pointlessly tiny.
_CHUNK_TARGET_CHARS = 700
# A chunk smaller than this is merged into its neighbour: before version 4,
# a "signature plus one line" chunk was common and matched nothing.
MIN_CHUNK_CHARS = 200
# A variable this small folds into its parent's (or the file's) lines rather
# than owning a chunk of its own: `const MAX = 5` is not a search target.
MIN_OWNED_VARIABLE_CHARS = 200
# Members that never own lines — a GraphQL field, an Avro field, a BPMN task
# is a line or two, and is searched as part of its type, not alone.
_NON_OWNING_KINDS = {"field", "flow_node"}
# How far back from the size limit a chunk may end early to land on a
# statement boundary instead of mid-block.
_BOUNDARY_LOOKBACK_LINES = 12
_BOUNDARY_ENDINGS = ("}", "};", ");", ")", "]", ";", "},", "),", ">", "/>")
# A run with less real content than this (`}` or `});` alone) is not
# embedded; its lines still count as covered.
_TRIVIAL_RUN_CHARS = 20
# The same, for what is left of a class or component once its methods and
# handlers have taken their own lines: `class OrderController:` alone is
# scaffolding, not something a search should land on. A function with no
# nested symbols always keeps its chunk, however small.
_TRIVIAL_SCAFFOLD_CHARS = 60


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


def _is_boundary(line: str) -> bool:
    stripped = line.strip()
    return not stripped or stripped.endswith(_BOUNDARY_ENDINGS)


def _is_trivial(lines: list[str], threshold: int = _TRIVIAL_RUN_CHARS) -> bool:
    content = re.sub(r"[\s{}()\[\];,]", "", "".join(lines))
    return len(content) < threshold


def _split_run(lines: list[str], run: list[int]) -> list[tuple[int, int]]:
    """Cut one run of consecutive line numbers into (start, end) chunks."""
    pieces: list[tuple[int, int]] = []
    start_idx = 0
    size = 0
    i = 0
    while i < len(run):
        length = len(lines[run[i] - 1]) + 1
        if size and size + length > _CHUNK_TARGET_CHARS:
            cut = i - 1
            for back in range(i - 1, max(start_idx, i - _BOUNDARY_LOOKBACK_LINES) - 1, -1):
                if _is_boundary(lines[run[back] - 1]):
                    cut = back
                    break
            pieces.append((run[start_idx], run[cut]))
            start_idx = cut + 1
            size = sum(len(lines[run[j] - 1]) + 1 for j in range(start_idx, i))
            continue
        size += length
        i += 1
    if start_idx < len(run):
        pieces.append((run[start_idx], run[-1]))

    def chars(piece: tuple[int, int]) -> int:
        return sum(len(lines[n - 1]) + 1 for n in range(piece[0], piece[1] + 1))

    merged: list[tuple[int, int]] = []
    for piece in pieces:
        if merged and (chars(piece) < MIN_CHUNK_CHARS or chars(merged[-1]) < MIN_CHUNK_CHARS):
            merged[-1] = (merged[-1][0], piece[1])
        else:
            merged.append(piece)
    return merged


def _chunks_for_lines(
    lines: list[str], owned: list[int], header_name: str, file_path: str,
    trivial_below: int = _TRIVIAL_RUN_CHARS,
) -> list[CodeChunk]:
    """Chunks covering exactly the line numbers in `owned` (sorted)."""
    runs: list[list[int]] = []
    for number in owned:
        if runs and number == runs[-1][-1] + 1:
            runs[-1].append(number)
        else:
            runs.append([number])
    chunks: list[CodeChunk] = []
    base = PurePosixPath(file_path).name
    for run in runs:
        # Leading/trailing blank lines carry nothing.
        while run and not lines[run[0] - 1].strip():
            run = run[1:]
        while run and not lines[run[-1] - 1].strip():
            run = run[:-1]
        if not run or _is_trivial([lines[n - 1] for n in run], trivial_below):
            continue
        for start, end in _split_run(lines, run):
            chunks.append(
                CodeChunk(
                    index=len(chunks),
                    line_start=start,
                    line_end=end,
                    text="\n".join(lines[start - 1 : end]),
                    header=f"// {base} › {header_name} (lines {start}–{end})",
                )
            )
    return chunks


def _owns_lines(symbol: ParsedSymbol) -> bool:
    if symbol.symbol_kind in _NON_OWNING_KINDS:
        return False
    if symbol.kind == "variable" and len(symbol.source_snippet) < MIN_OWNED_VARIABLE_CHARS:
        return False
    return True


def assign_chunks(
    text: str,
    file_path: str,
    symbols: list[ParsedSymbol],
    import_lines: set[int],
    file_chunks: bool = True,
) -> list[CodeChunk]:
    """Give every owning symbol its chunks (set on `symbol.chunks`) and
    return the file's own region chunks.

    Owners are applied outermost first, so a nested symbol takes its lines
    back from its parent. `file_chunks=False` leaves the file's own lines
    unchunked (a BPMN file's diagram coordinates are not worth a search)."""
    lines = text.split("\n")
    total = len(lines)
    owner: list[int] = [-1] * (total + 2)  # -1 = the file itself
    owning = [
        (i, s) for i, s in enumerate(symbols) if _owns_lines(s) and s.line_start <= total
    ]
    for i, symbol in sorted(owning, key=lambda pair: (pair[1].line_start, -pair[1].line_end)):
        for number in range(symbol.line_start, min(symbol.line_end, total) + 1):
            owner[number] = i
    owned_by: dict[int, list[int]] = {}
    for number in range(1, total + 1):
        if owner[number] == -1 and number in import_lines:
            continue
        owned_by.setdefault(owner[number], []).append(number)
    owners_with_children = {
        s.parent_symbol for _, s in owning if s.parent_symbol
    }
    for i, symbol in owning:
        display = symbol.qualified_name.split("::", 1)[-1]
        threshold = (
            _TRIVIAL_SCAFFOLD_CHARS if symbol.qualified_name in owners_with_children
            else 0 if symbol.kind == "function" else _TRIVIAL_RUN_CHARS
        )
        symbol.chunks = _chunks_for_lines(
            lines, owned_by.get(i, []), display, file_path, threshold
        )
    if not file_chunks or len(text) < MIN_CHUNKED_FILE_CHARS:
        return []
    return _chunks_for_lines(lines, owned_by.get(-1, []), "(file)", file_path)


def chunk_text(text: str, file_path: str = "") -> list[CodeChunk]:
    """Split a whole file's text into embeddable pieces, cut at line ends.

    For a file no parser reads — a .sql migration, a .yaml, a Jenkinsfile —
    and for one whose parse failed. Same target size and boundary rules as
    every other chunk."""
    lines = text.split("\n")
    return _chunks_for_lines(lines, list(range(1, len(lines) + 1)), "(file)", file_path)


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


_PROCESS_CLOSE = re.compile(r"</(?:\w+:)?process\s*>")


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
        close = _PROCESS_CLOSE.search(text, idx)
        end_line = text.count("\n", 0, close.start()) + 1 if close else line
        qname = f"{file_path}::{proc_id}"
        classes.append(
            ParsedSymbol(
                kind="class",
                name=proc_name,
                qualified_name=qname,
                file_path=file_path,
                line_start=line,
                line_end=end_line,
                signature=f"process {proc_id}",
                docstring=None,
                language="bpmn",
                source_snippet=_tag_snippet(text, idx),
                symbol_kind="process",
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
                    symbol_kind="flow_node",
                    parent_symbol=qname,
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
#
# Each record also carries its Avro namespace (facts["avro_namespace"]):
# the generated class's full name is namespace + "." + name, and that is
# how a Kafka handler typed `KafkaMessageHandler<UpdateHighRisk>` is matched
# to its schema exactly (see service_links.py).

_BLOCK_RE = re.compile(r"\b(protocol|record|error|enum)\s+(\w+)\s*\{")
_NESTED_BLOCK_RE = re.compile(r"\b(record|error|enum)\s+(\w+)\s*\{")
_NAMESPACE_RE = re.compile(r'@namespace\s*\(\s*"([^"]+)"\s*\)')


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


def _namespace_before(raw: str, position: int, default: str | None) -> str | None:
    """The @namespace annotation directly in front of a block, if any —
    only whitespace and other annotations may sit between them."""
    head = raw[:position]
    match = None
    for candidate in _NAMESPACE_RE.finditer(head):
        match = candidate
    if match is None:
        return default
    between = head[match.end() :]
    if re.fullmatch(r"(\s|@\w+\s*\([^)]*\))*", between):
        return match.group(1)
    return default


def parse_avdl_source(source: bytes, file_path: str) -> list[ParsedSymbol]:
    raw = source.decode("utf-8", errors="replace")
    text = _strip_comments(raw)
    classes: list[ParsedSymbol] = []
    functions: list[ParsedSymbol] = []
    protocol_namespace: str | None = None

    for m in _BLOCK_RE.finditer(text):
        keyword, name = m.group(1), m.group(2)
        open_idx = m.end() - 1
        close_idx = _find_matching_brace(text, open_idx)
        line_start = text.count("\n", 0, m.start()) + 1
        line_end = text.count("\n", 0, close_idx) + 1
        qname = f"{file_path}::{name}"
        namespace = _namespace_before(text, m.start(), protocol_namespace)
        if keyword == "protocol":
            protocol_namespace = namespace
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
                symbol_kind=keyword,
                facts={"avro_namespace": namespace} if namespace else {},
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
                    symbol_kind="field",
                    parent_symbol=qname,
                )
            )

    protocols = [c for c in classes if c.symbol_kind == "protocol"]
    for block in classes:
        for protocol in protocols:
            inside = protocol.line_start <= block.line_start <= block.line_end <= protocol.line_end
            if block is not protocol and inside:
                block.parent_symbol = protocol.qualified_name
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

# How much of a test file the planner is SHOWN (graph_reader caps at read
# time). Measured against the tests that cover the ATS settings form:
# 1,605 / 2,235 / 3,920 characters, so a focused unit test fits whole. The
# one file this truncates is a 29,652-char suite covering a whole settings
# tab, and its first 6,000 characters are its imports and first fixture —
# the part worth copying. The file itself is now stored whole.
MAX_TEST_SOURCE_CHARS = 6000

# A file shorter than this is given its node and text but no chunks. 1,609
# of ui's symbol-less files are under 200 bytes, and almost all are a single
# `export * from "./view"`. Embedded, they are noise competing with real
# code; as a File node alone they still tell a coding agent the file exists.
MIN_CHUNKED_FILE_CHARS = 200

# Identifiers kept per symbol for reference resolution. A long component
# mentions hundreds of names; the cap keeps the node small, and names are
# kept in order of first use, so the ones it is built around come first.
_MAX_IDENTIFIERS = 300


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
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".graphqls": "graphql",
    ".graphql": "graphql",
    ".gql": "graphql",
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

# Extensions no parser reads but whose `language` label is worth more than
# the bare extension. Their text is stored and chunked like any other file.
TRACKED_ONLY_EXTENSIONS: dict[str, str] = {
    ".html": "html",
    ".css": "css",
    ".csv": "csv",
    ".sql": "sql",
    ".yml": "yaml",
    ".yaml": "yaml",
    ".md": "markdown",
}


def language_of(file_path: str) -> str:
    """The parser language for a path, or a plain label for a file no parser
    reads (its extension, or the file name for `Dockerfile`/`Jenkinsfile`)."""
    path = PurePosixPath(file_path)
    if path.suffix in LANGUAGE_BY_EXTENSION:
        return LANGUAGE_BY_EXTENSION[path.suffix]
    suffix = path.suffix.lower()
    return TRACKED_ONLY_EXTENSIONS.get(suffix) or suffix.lstrip(".") or "text"


@dataclass
class _Scope:
    qname: str
    is_class: bool


def _walk(
    root: Node,
    *,
    file_path: str,
    source: bytes,
    adapter: LanguageAdapter,
) -> list[tuple[ParsedSymbol, Node, Node]]:
    """Every symbol in the tree, in document order, as (symbol, node, root).

    Iterative, not recursive: a 5,000-line string concatenation or a
    20KB union type nests thousands of levels deep, and the recursive
    version of this walk hit Python's recursion limit on exactly those files
    — 12 of them across 13 services — losing the whole file."""
    found: list[tuple[ParsedSymbol, Node, Node]] = []
    seen_qnames: dict[str, int] = {}

    def unique(qname: str) -> str:
        count = seen_qnames.get(qname, 0) + 1
        seen_qnames[qname] = count
        return qname if count == 1 else f"{qname}#{count}"

    stack: list[tuple[list[Node], int, _Scope | None]] = [(root.children, 0, None)]
    while stack:
        children, index, scope = stack[-1]
        if index >= len(children):
            stack.pop()
            continue
        stack[-1] = (children, index + 1, scope)
        child = children[index]

        is_class = child.type in adapter.class_types
        is_function = not is_class and child.type in adapter.function_types
        if is_class or is_function:
            name = adapter.name_of(child)
            if name is None:
                # Anonymous: walk through it, so what is declared inside an
                # unnamed callback (a useEffect body, a describe block) is
                # still found.
                stack.append((child.children, 0, scope))
                continue
            symbol_root = adapter.root_of(child)
            body = adapter.body_of(child)
            is_method = is_function and scope is not None and scope.is_class
            parent = scope.qname if scope is not None else None
            qname = unique(qualified_name(file_path, name, parent))
            signature = signature_of(child, body, source)
            symbol = ParsedSymbol(
                kind="class" if is_class else "function",
                name=name,
                qualified_name=qname,
                file_path=file_path,
                line_start=symbol_root.start_point[0] + 1,
                line_end=symbol_root.end_point[0] + 1,
                signature=signature,
                docstring=adapter.docstring_of(child, body, source),
                language=adapter.language,
                source_snippet=node_text(symbol_root, source),
                is_method=is_method,
                parent_class=parent if is_method else None,
                symbol_kind=adapter.kind_of(child) or "",
                parent_symbol=parent,
                supertypes=adapter.supertypes_of(child) if is_class else [],
            )
            if is_method and symbol.symbol_kind in ("function", ""):
                symbol.symbol_kind = "method"
            found.append((symbol, child, symbol_root))
            if body is not None:
                stack.append((body.children, 0, _Scope(qname, is_class)))
            continue

        if child.type in adapter.variable_types:
            variables = adapter.variables_of(child)
            new_scope = scope
            for name, var_root, var_kind in variables:
                if not name:
                    continue
                parent = scope.qname if scope is not None else None
                qname = unique(qualified_name(file_path, name, parent))
                symbol = ParsedSymbol(
                    kind="variable",
                    name=name,
                    qualified_name=qname,
                    file_path=file_path,
                    line_start=var_root.start_point[0] + 1,
                    line_end=var_root.end_point[0] + 1,
                    signature=node_text(var_root, source).split("\n", 1)[0][:200],
                    docstring=adapter.docstring_of(var_root, None, source),
                    language=adapter.language,
                    source_snippet=node_text(var_root, source),
                    symbol_kind=var_kind,
                    parent_symbol=parent,
                )
                found.append((symbol, child, var_root))
                if len(variables) == 1:
                    new_scope = _Scope(qname, False)
            stack.append((child.children, 0, new_scope))
            continue

        stack.append((child.children, 0, scope))
    return found


def _collect_identifiers(
    found: list[tuple[ParsedSymbol, Node, Node]], adapter: LanguageAdapter
) -> None:
    """Fill each symbol's `identifiers` from its own code — stopping at any
    nested symbol, whose code is that symbol's, not its parent's."""
    if not adapter.identifier_types:
        return
    boundaries = set()
    for _, node, symbol_root in found:
        boundaries.add(_node_key(node))
        boundaries.add(_node_key(symbol_root))
    for symbol, node, symbol_root in found:
        own = {_node_key(node), _node_key(symbol_root)}
        names: dict[str, None] = {}
        stack = [symbol_root]
        while stack and len(names) < _MAX_IDENTIFIERS:
            current = stack.pop()
            key = _node_key(current)
            if key in boundaries and key not in own:
                continue
            if current.type in adapter.identifier_types:
                text = text_of(current) or ""
                if len(text) > 1 and text != symbol.name:
                    names.setdefault(text, None)
                continue
            stack.extend(reversed(current.children))
        symbol.identifiers = list(names)


def parse_source_full(
    source: bytes, file_path: str, language: str, searchable: bool = True
) -> ParsedFile:
    """Parse raw source bytes already in memory — symbols, imports, package
    and chunks for one file. The incremental-update path's entry point too:
    a single changed file's content, fetched via the GitHub API, with no
    repo clone and no file on disk at all.

    Raises on a genuine parse failure (tree-sitter itself doesn't raise on
    malformed syntax — it produces error nodes — so this mainly covers
    decode errors); callers are expected to catch and count, not let one
    bad file abort a run.
    """
    text = source.decode("utf-8", errors="replace")
    custom_parser = _CUSTOM_PARSERS.get(language)
    if custom_parser is not None:
        symbols = custom_parser(source, file_path)
        parsed = ParsedFile(symbols=symbols)
    else:
        parser = get_parser(language)
        tree = parser.parse(source)
        adapter = ADAPTERS[language]
        found = _walk(tree.root_node, file_path=file_path, source=source, adapter=adapter)
        _collect_identifiers(found, adapter)
        imports, package, import_lines = adapter.imports_of(tree.root_node)
        parsed = ParsedFile(
            symbols=[symbol for symbol, _, _ in found],
            imports=imports,
            package=package,
            import_lines=import_lines,
        )
    code_facts.add_facts(parsed, text, file_path, language)
    if searchable and not is_test_path(file_path):
        parsed.region_chunks = assign_chunks(
            text, file_path, parsed.symbols, parsed.import_lines,
            file_chunks=language != "bpmn",
        )
    return parsed


def parse_source(source: bytes, file_path: str, language: str) -> list[ParsedSymbol]:
    """Symbols only — see parse_source_full for everything else."""
    return parse_source_full(source, file_path, language).symbols


def parse_file(path: Path, repo_root: Path, language: str) -> list[ParsedSymbol]:
    """Parse one file already on disk, returning its classes + functions
    (kind-tagged, flat) — the full-repo-scan path's entry point (see
    parse_repo() below). Thin wrapper around parse_source(): reads the
    file's bytes and derives its repo-relative file_path, nothing else.
    """
    source = path.read_bytes()
    file_path = str(path.relative_to(repo_root).as_posix())
    return parse_source(source, file_path, language)


def _coverage(text: str, symbols: list[ParsedSymbol], chunks: list[CodeChunk],
              import_lines: set[int]) -> tuple[int, int]:
    """(code lines, code lines inside a symbol, a chunk or the import block)."""
    lines = text.split("\n")
    covered: set[int] = set(import_lines)
    for symbol in symbols:
        covered.update(range(symbol.line_start, symbol.line_end + 1))
    for chunk in chunks:
        covered.update(range(chunk.line_start, chunk.line_end + 1))
    code = [n for n, line in enumerate(lines, 1) if line.strip()]
    return len(code), sum(1 for n in code if n in covered)


def _all_covered(text: str) -> tuple[int, int]:
    """A file stored whole but deliberately not searched (generated, a test,
    a lockfile) is accounted for by its stored text."""
    code = sum(1 for line in text.split("\n") if line.strip())
    return code, code


def _text_only(graph: RepoGraph, record: FileRecord, text: str, chunkable: bool) -> None:
    """Stored text plus line chunks, for a file with no usable parse."""
    chunks: list[CodeChunk] = []
    if text and chunkable and len(text) >= MIN_CHUNKED_FILE_CHARS:
        chunks = chunk_text(text, record.file_path)
        graph.file_chunks[record.file_path] = chunks
    if text:
        graph.file_coverage[record.file_path] = (
            _coverage(text, [], chunks, set()) if chunkable else _all_covered(text)
        )


def add_file(graph: RepoGraph, record: FileRecord, raw: bytes,
             max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> None:
    """Parse one file into `graph`, whatever it is. Used by parse_repo for
    every file and by the incremental path for one changed file."""
    rel_path = record.file_path
    graph.files[rel_path] = record
    text = record.source or ""
    if record.kind != "binary":
        graph.file_source[rel_path] = text
        imports_block = extract_imports(text) if text else ""
        if imports_block:
            graph.file_imports[rel_path] = imports_block
    language = LANGUAGE_BY_EXTENSION.get(PurePosixPath(rel_path).suffix)
    # Test files are stored and parsed but never embedded: the planner's
    # search excludes test code by design (graph_reader.search_service_
    # chunks), so their vectors were paid for and never read. Tests reach
    # the planner through File.source and TESTS edges instead.
    chunkable = record.searchable and not is_test_path(rel_path)
    if language is None or record.kind == "binary":
        graph.tracked_files[rel_path] = record.language
        graph.files_skipped_unsupported += 1
        _text_only(graph, record, text, chunkable)
        return
    if len(raw) > max_file_size_bytes:
        # Still a node, with its text — but too large to parse.
        record.parse_status = "oversized"
        graph.tracked_files[rel_path] = language
        graph.files_skipped_oversized += 1
        _text_only(graph, record, text, chunkable)
        return
    try:
        parsed = parse_source_full(raw, rel_path, language, searchable=chunkable)
    except Exception as e:  # noqa: BLE001 - one bad file must never abort the repo
        record.parse_status = "failed"
        record.parse_error = f"{type(e).__name__}: {e}"[:500]
        graph.tracked_files[rel_path] = language
        graph.files_failed += 1
        _text_only(graph, record, text, chunkable)
        return
    record.parse_status = "ok"
    graph.files_parsed += 1
    for symbol in parsed.symbols:
        if symbol.kind == "class":
            graph.classes.append(symbol)
        elif symbol.kind == "variable":
            graph.variables.append(symbol)
        else:
            graph.functions.append(symbol)
    if parsed.imports:
        graph.file_import_specs[rel_path] = parsed.imports
    if parsed.package:
        graph.file_packages[rel_path] = parsed.package
    if parsed.facts:
        graph.file_facts[rel_path] = parsed.facts
    if parsed.region_chunks:
        graph.file_chunks[rel_path] = parsed.region_chunks
    if not parsed.symbols:
        graph.tracked_files[rel_path] = language
    all_chunks = [c for s in parsed.symbols for c in s.chunks] + parsed.region_chunks
    graph.file_coverage[rel_path] = (
        _coverage(text, parsed.symbols, all_chunks, parsed.import_lines)
        if chunkable and language != "bpmn"
        # A BPMN file's processes are chunked; the rest is diagram layout
        # (<di:waypoint>, <dc:Bounds>), stored but deliberately not searched.
        else _all_covered(text)
    )


def parse_repo(
    repo_root: Path,
    service_name: str,
    denylist: set[str] | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> RepoGraph:
    """Walk `repo_root`: a FileRecord for EVERY file, and a parse of every
    file a parser reads.

    "Parse everything" by design (per the ingestion architecture decision) —
    the denylist only prunes build/dependency noise, and the size guardrail
    only protects against pathological single files (generated/minified/
    vendored) that slipped past it; neither is a node-count cap.
    """
    denylist = denylist if denylist is not None else DEFAULT_DENYLIST
    graph = RepoGraph(classes=[], functions=[])

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in denylist)
        for filename in sorted(filenames):
            path = Path(dirpath) / filename
            rel_path = str(path.relative_to(repo_root).as_posix())
            try:
                raw = path.read_bytes()
            except OSError:
                graph.files_failed += 1
                continue
            record = build_record(rel_path, raw, language_of(rel_path))
            add_file(graph, record, raw, max_file_size_bytes)

    return graph
