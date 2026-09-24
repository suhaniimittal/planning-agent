"""Python's LanguageAdapter."""

from __future__ import annotations

from tree_sitter import Node

from ..models import LanguageAdapter
from .shared import field_body_of, field_name_of


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
