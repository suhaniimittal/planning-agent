"""Java's LanguageAdapter."""

from __future__ import annotations

from tree_sitter import Node

from ..models import LanguageAdapter
from .shared import adjacent_comment_docstring, field_body_of, field_name_of


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
