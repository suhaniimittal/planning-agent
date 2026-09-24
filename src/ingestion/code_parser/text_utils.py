"""Pure byte-slicing/text helpers shared across chunking, the language
adapters, and the core walker. No dependency on any of them — kept here to
avoid circular imports between those modules.
"""

from __future__ import annotations

from tree_sitter import Node


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
