"""Agent — planning_agent.

Thin Aetherion binding covering the whole pipeline in one call: syncs the
service graph for exactly the repos named in ``repos`` (``run_ingestion`` —
a full clone the first time a service is ingested, a git-diff incremental
sync with no clone every run after), then produces a technical design doc
for ``issue_text`` (``generate_technical_design``). No pipeline logic lives
here — per the SDK's "no direct I/O in the agent" rule, both tools do all
the actual clone/parse/Neo4j/LLM work; this just orchestrates the two calls
via ``toolExecutor.execute`` and shapes the result for the platform UI.

``repos`` is the ONLY source of "which services exist" once this is
published — deliberately not services.yaml, which this agent never reads.
Each run supplies its own service list as input, so a user never needs
access to this project's source to point it at a different/new repo.

The one piece of logic here is the error boundary. A workflow that raises
shows up in the console as a failed run with *no output to summarise*, which
says nothing about what broke. Catching it, logging the traceback worker-side
and returning a structured error means the failure is readable in the run
itself. Nothing is swallowed: the full traceback still reaches the worker log.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from aetherion_sdk import agent, toolExecutor

# Neither tool call below specifies a timeout by default, which leaves it at
# whatever the platform's own default activity timeout is — observed in a
# real run to be far too short (~30s) for genuine work: a large repo's first
# ingestion embeds tens of thousands of chunks, and a TDD's one LLM
# call gets slower the more matched code it's given. Both are set generously
# here instead of trusting that default, since the real risk isn't a logic
# bug — it's a correct, slow run getting killed mid-flight by an undersized
# clock (confirmed via a real run: the PDF was uploaded successfully only
# seconds after Temporal had already given up and reported a timeout).
#
# Ingestion has no practical limit: Temporal requires some timeout on every
# activity (left unset, it falls back to a short default), so this is a
# ceiling only a truly stuck run would ever hit, not a cap on large repos.
_INGESTION_TIMEOUT = timedelta(hours=6)
_TDD_TIMEOUT = timedelta(minutes=10)

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
    repos_text = payload.get("repos") or ""
    top_k = payload.get("top_k", 5)
    hops = payload.get("hops", 2)
    try:
        await toolExecutor.execute(
            "run_ingestion", repos_text, start_to_close_timeout=_INGESTION_TIMEOUT
        )
        result = await toolExecutor.execute(
            "generate_technical_design",
            issue_text,
            top_k,
            hops,
            start_to_close_timeout=_TDD_TIMEOUT,
        )

        structured_results = [
            {
                "type": "s3_download_link",
                "title": "Technical Design Document",
                "file_key": result["s3_key"],
                "label": "Download TDD (PDF)",
                "extension": "pdf",
            }
        ]

        return {
            "status": "success",
            "results": structured_results,
        }
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
