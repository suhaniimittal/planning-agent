"""Tool entrypoint for the query-time flow."""

from __future__ import annotations

from aetherion_sdk import tool

from src.query.query_flow import build_tdd
from src.query.render import render_markdown, render_text


@tool()
async def generate_technical_design(issue_text: str, top_k: int = 5, hops: int = 2) -> dict:
    """Search the service graph for services relevant to a requirement/bug
    report and produce a technical design doc for a coding agent: structured
    data, a compact Markdown summary, and the full section-numbered plain-text
    document (the same form try_query.py saves to technical_design.txt)."""
    tdd = await build_tdd(issue_text, top_k=top_k, hops=hops)
    return {
        "tdd": tdd.model_dump(),
        "markdown": render_markdown(tdd),
        "text": render_text(tdd),
    }
