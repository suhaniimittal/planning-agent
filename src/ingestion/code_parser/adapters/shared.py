"""Building blocks reused by more than one LanguageAdapter: field-based
name/body lookup (for grammars that expose named fields — Python, Java,
TypeScript/TSX/JavaScript), and the "doc comment sitting immediately above
the declaration" convention (Javadoc/JSDoc/KDoc — Java, TypeScript-family,
Kotlin all follow this idiom, just with different comment node-type names).
"""

from __future__ import annotations

from tree_sitter import Node

from ..text_utils import text_of


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
