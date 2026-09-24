"""TypeScript / TSX / JavaScript's LanguageAdapter.

Shared: same grammar family, JS is the base that TS/TSX extend, so all
three share identical node shapes (confirmed against real grammars, not
guessed — see the language-support design discussion).
"""

from __future__ import annotations

from tree_sitter import Node

from ..models import LanguageAdapter
from .shared import adjacent_comment_docstring, field_body_of, field_name_of


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
