"""Agent entrypoint for the query-time flow."""

from __future__ import annotations

from typing import Any

from aetherion_sdk import agent, toolExecutor


@agent()
async def query_agent(payload: dict[str, Any]) -> dict:
    """Given a new requirement or bug report, produce a structured
    technical design doc naming the services/functions likely to change."""

    issue_text = payload.get("issue_text")
    tdd = await toolExecutor.execute("generate_technical_design", issue_text)

    return {
        "issue_text": issue_text,
        "technical_design_doc": tdd,
        "status": "success",
    }
