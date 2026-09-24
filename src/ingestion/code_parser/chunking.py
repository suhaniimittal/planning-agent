"""Splitting an oversized function's body into embeddable pieces."""

from __future__ import annotations

from tree_sitter import Node

from .models import CodeChunk
from .text_utils import node_text

# The ~500-800 char chunk-size target agreed in design: small enough to keep
# one embedding focused on one idea, large enough not to be pointlessly tiny.
_CHUNK_TARGET_CHARS = 700


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
