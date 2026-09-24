"""Agent — planning_agent.

Thin Aetherion binding: given a plain-text requirement or bug report,
delegates to the existing ``generate_technical_design`` tool (the same one
``src/query_agent/agent.py`` calls) via the SDK's ``toolExecutor.execute``.
No new placeholder tools invented here — just a name and a wire-up to real
work, same convention as this project's other agent.

The one piece of logic here is the error boundary. A workflow that raises
shows up in the console as a failed run with *no output to summarise*, which
says nothing about what broke. Catching it, logging the traceback worker-side
and returning a structured error means the failure is readable in the run
itself. Nothing is swallowed: the full traceback still reaches the worker log.
"""

from __future__ import annotations

import logging
from typing import Any

from aetherion_sdk import agent, toolExecutor

logger = logging.getLogger(__name__)


def _workflow_id() -> str | None:
    """Best-effort workflow id for logging; ``None`` outside a workflow."""
    try:
        from temporalio import workflow

        return workflow.info().workflow_id
    except Exception:
        return None


@agent(name="planning_agent")
async def planning_agent(payload: dict[str, Any]) -> dict[str, Any]:
    workflow_id = _workflow_id()
    issue_text = payload.get("issue_text")
    try:
        result = await toolExecutor.execute("generate_technical_design", issue_text)
        return {"status": "success", "issue_text": issue_text, **result}
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        logger.error("planning_agent run failed: %s", error, exc_info=True)
        return {
            "status": "error",
            "error": type(error).__name__,
            "message": (
                f"The run failed before it could produce a technical design: "
                f"{type(error).__name__}: {error}"
            )[:500],
            "issue_text": issue_text,
            "details": {"workflow_id": workflow_id},
        }
