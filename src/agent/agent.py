"""Agent — planning_agent.

Thin Aetherion binding covering the whole pipeline in one call: syncs the
service graph for exactly the repos named in ``repos`` (``run_ingestion`` —
a full clone the first time a service is ingested, a git-diff incremental
sync with no clone every run after), then produces a technical design doc
for ``issue_text`` (``generate_technical_design``). With ``repos`` and no
``issue_text`` it only syncs the graph — how the orchestrator refreshes it
after a merge. No pipeline logic lives
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
    # An empty form field arrives as null, not as a missing key, and both
    # mean "use the default". Deliberately passed on AS None rather than
    # resolved here: a second copy of the numbers in this workflow silently
    # overrode the real defaults, so generate_technical_design owns them.
    top_k = payload.get("top_k")
    hops = payload.get("hops")
    requested = str(payload.get("mode") or "").strip().lower()
    if requested == "discuss":
        return await _discuss(payload, workflow_id)
    if requested in _MEMORY_MODES:
        return await _MEMORY_MODES[requested](payload, workflow_id)
    # Optional override. Left unset (the normal case), the ticket is
    # classified from its own text during the issue split.
    mode = payload.get("mode") or None
    issue_key = str(payload.get("issue_key") or (payload.get("card") or {}).get("key") or "")
    try:
        if payload.get("card") and not str(issue_text or "").strip():
            # The agreed input (2026-10-29): the card, its ranked services and
            # the review's questions with their answers.
            built = await toolExecutor.execute(
                "handoff_text", dict(payload), start_to_close_timeout=timedelta(minutes=1)
            )
            issue_text = built["issue_text"]
            repos_text = repos_text or built["repos"]
        ingestion = await toolExecutor.execute(
            "run_ingestion", repos_text, start_to_close_timeout=_INGESTION_TIMEOUT
        )
        if not str(issue_text or "").strip():
            # Graph refresh only: the orchestrator sends this when a pull
            # request is merged, so the next design sees the latest code.
            # There is no requirement to design for, so no TDD is written.
            return {
                "status": "success",
                "mode": "graph_sync",
                "services": (ingestion or {}).get("services", []),
                "results": [],
            }
        # The orchestrator asks for the document itself, to publish it in
        # Confluence and give its text to Coding.
        return_document = bool(payload.get("return_document"))
        result = await toolExecutor.execute(
            "generate_technical_design",
            issue_text,
            top_k,
            hops,
            # The TDD considers only the services this run listed.
            ingestion["services"],
            return_document,
            # Their repositories: the build files give the design its versions.
            dict(ingestion.get("repos") or {}),
            mode,
            # Its decisions in force, and its project's, are kept (2026-10-26).
            issue_key or None,
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

        out = {"status": "success", "results": structured_results}
        if return_document:
            out["document"] = {k: result.get(k, "") for k in ("title", "confluence_html", "text")}
            # The decisions the design uses, for Orchestration to record (2026-10-26).
            out["document"]["decisions"] = list(result.get("decisions") or [])
        return out
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


async def _discuss(payload: dict[str, Any], workflow_id: str | None) -> dict[str, Any]:
    """A question about an existing TDD, from the comments on its page (2026-10-08)."""
    question = str(payload.get("question") or "").strip()
    tdd_text = str(payload.get("tdd_text") or "").strip()
    if not question or not tdd_text:
        return {
            "status": "error",
            "error": "MissingInput",
            "message": "A design question needs both the question and the TDD's text.",
        }
    try:
        answer = await toolExecutor.execute(
            "answer_design_question",
            question,
            tdd_text,
            list(payload.get("discussion") or []),
            str(payload.get("issue_key") or "") or None,
            # No kind = memory off: answered exactly as before 2026-10-26.
            str(payload.get("kind") or "") or None,
            # The TDD page the question was asked on: its Memsy session.
            str(payload.get("page_id") or ""),
            start_to_close_timeout=_TDD_TIMEOUT,
        )
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        logger.error("planning_agent discuss failed: %s", error, exc_info=True)
        return {
            "status": "error",
            "error": type(error).__name__,
            "message": f"The question could not be answered: {type(error).__name__}: {error}"[:500],
            "details": {"workflow_id": workflow_id},
        }
    return {"status": "success", "mode": "discuss", **answer}


# --- memory and bug investigation (2026-10-26) ------------------------------------------


def _failed(what: str, error: Exception, workflow_id: str | None) -> dict[str, Any]:
    logger.error("planning_agent %s failed: %s", what, error, exc_info=True)
    return {
        "status": "error",
        "error": type(error).__name__,
        "message": f"{what} failed: {type(error).__name__}: {error}"[:500],
        "details": {"workflow_id": workflow_id},
    }


async def _facts(payload: dict[str, Any], workflow_id: str | None) -> dict[str, Any]:
    """Is this story the requirement its TDD answered? (contract mode "facts")"""
    issue_key = str(payload.get("issue_key") or "").strip()
    story_text = str(payload.get("story_text") or "").strip()
    if not issue_key or not story_text:
        return {
            "status": "error",
            "error": "MissingInput",
            "message": "A facts check needs the issue_key and the story_text.",
        }
    try:
        found = await toolExecutor.execute(
            "story_facts", issue_key, story_text, start_to_close_timeout=_TDD_TIMEOUT
        )
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        return _failed("The facts check", error, workflow_id)
    return {"status": "success", "mode": "facts", **found}


async def _record(payload: dict[str, Any], workflow_id: str | None) -> dict[str, Any]:
    """A published TDD version, remembered (contract mode "record")."""
    if not str(payload.get("issue_key") or "").strip():
        return {
            "status": "error",
            "error": "MissingInput",
            "message": "record needs the issue_key.",
        }
    try:
        stored = await toolExecutor.execute(
            "record_tdd", dict(payload), start_to_close_timeout=timedelta(minutes=2)
        )
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        return _failed("Recording the TDD", error, workflow_id)
    return {"status": "success", "mode": "record", "stored": stored}


async def _investigate(payload: dict[str, Any], workflow_id: str | None) -> dict[str, Any]:
    """A bug's root cause in the code graph (contract mode "investigate")."""
    report_text = str(payload.get("report_text") or "").strip()
    if not report_text:
        return {
            "status": "error",
            "error": "MissingInput",
            "message": "An investigation needs the bug report's text.",
        }
    try:
        found = await toolExecutor.execute(
            "investigate_bug",
            report_text,
            str(payload.get("repos") or ""),
            start_to_close_timeout=_TDD_TIMEOUT,
        )
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        return _failed("The investigation", error, workflow_id)
    return {"status": "success", "mode": "investigate", "investigation": found}


_MEMORY_MODES = {"facts": _facts, "record": _record, "investigate": _investigate}
