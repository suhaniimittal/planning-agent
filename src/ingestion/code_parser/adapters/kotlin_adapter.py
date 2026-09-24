"""Kotlin's LanguageAdapter.

Kotlin's grammar exposes NO named fields at all (unlike the others above),
so name/body have to be found positionally — by node TYPE, not field name.
It also reuses the same node type ("function_declaration") for both a
top-level function and a class method, same as Python — DECLARES vs.
CONTAINS is still decided purely by nesting context, not the node's type.
"""

from __future__ import annotations

from tree_sitter import Node

from ..models import LanguageAdapter
from ..text_utils import text_of
from .shared import adjacent_comment_docstring

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
