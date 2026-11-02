"""Agent — planning_agent.

Thin Aetherion binding. A Jira card with its ranked services and questions
becomes a plain-English TDD and a work order (``plan_tdd``, src/tdd): the
code is seen only through GitHub's REST API, never cloned and never stored.
The other modes are Orchestration's: ``discuss`` (a question on a published
TDD, answered without code), ``facts`` and ``record`` (the ticket's memory).
No pipeline logic lives here — per the SDK's "no direct I/O in the agent"
rule, the tools do all the GitHub, storage and LLM work; this routes the
payload and shapes the result for the platform UI.

The one piece of logic here is the error boundary. A workflow that raises
shows up in the console as a failed run with *no output to summarise*, which
says nothing about what broke. Catching it, logging the traceback worker-side
and returning a structured error means the failure is readable in the run
itself. Nothing is swallowed: the full traceback still reaches the worker log.
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from aetherion_sdk import agent, toolExecutor
from temporalio.common import RetryPolicy

# Left unset, a tool call gets the platform's default activity timeout —
# observed in a real run to be far too short (~30s) for genuine work. A TDD
# makes up to three model calls (scout, writer, one repair) plus its GitHub
# calls, so it is given room; the real risk is a correct, slow run killed
# mid-flight by an undersized clock.
_TDD_TIMEOUT = timedelta(minutes=15)
# A TDD is never re-run on its own. A timed-out step was not stuck — it was
# slow — and running it again starts a second, identical, paid run beside
# the first one's still-unfinished work. A failure is reported instead.
_TDD_RETRY = RetryPolicy(maximum_attempts=1)

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
    requested = str(payload.get("mode") or "").strip().lower()
    if requested == "discuss":
        return await _discuss(payload, workflow_id)
    if requested in _MEMORY_MODES:
        return await _MEMORY_MODES[requested](payload, workflow_id)
    if payload.get("planning_input") or isinstance(payload.get("card"), dict):
        return await _plan(payload, workflow_id)
    return {
        "status": "error",
        "error": "MissingInput",
        "message": "Fill in Planning input: the card, its ranked services and its questions.",
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


# --- memory (2026-10-26) ----------------------------------------------------------------


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


_MEMORY_MODES = {"facts": _facts, "record": _record}


# --- the TDD with no stored code (2026-11-02) -------------------------------------------

_INPUT_KEYS = ("card", "services", "questions", "issue_key")


async def _plan(payload: dict[str, Any], workflow_id: str | None) -> dict[str, Any]:
    """A card, its ranked services and its questions -> a plain-English TDD and
    a work order, the code seen only through GitHub's API (src/tdd).

    The input comes as the form's `planning_input` (JSON text) or as the
    object's own keys (Orchestration). Uploaded files come as storage keys —
    the form folds every file field into `uploaded_files` — and/or an
    `attachments` list of keys or {key, name, description}; each is joined to
    the card file that describes it."""
    raw = payload.get("planning_input")
    if not raw:
        raw = json.dumps({k: payload[k] for k in _INPUT_KEYS if k in payload})
    elif not isinstance(raw, str):
        raw = json.dumps(raw)
    attachments = _uploads(payload)
    try:
        result = await toolExecutor.execute(
            "plan_tdd",
            raw,
            attachments,
            start_to_close_timeout=_TDD_TIMEOUT,
            retry_policy=_TDD_RETRY,
        )
    except Exception as error:  # noqa: BLE001 - surface the failure, never hide it
        logger.error("planning_agent plan failed: %s", error, exc_info=True)
        return {
            "status": "error",
            "error": type(_root(error)).__name__,
            # The innermost message: an invalid input lists every problem there.
            "message": f"The TDD could not be written: {_root(error)}"[:2000],
            "details": {"workflow_id": workflow_id},
        }
    # The run's result is the TDD itself (the PDF link) and whether it is final;
    # the PDF holds everything else, the work order and the warnings included.
    out = {
        "status": "success",
        "mode": "plan",
        "tdd_status": result["status"],
        "results": [
            {
                "type": "s3_download_link",
                "title": "Technical Design Document",
                "file_key": result["s3_key"],
                "label": "Download TDD (PDF)",
                "extension": "pdf",
            }
        ],
    }
    if result["status"] == "draft":
        out["draft_reason"] = result["draft_reason"]
    if result.get("confluence_url"):
        out["confluence_url"] = result["confluence_url"]
    if payload.get("return_document"):
        # Orchestration asks for the document itself: to publish it in
        # Confluence, give Coding the work order and record the decisions.
        out["document"] = {
            k: result.get(k) for k in ("title", "text", "confluence_html", "decisions")
        }
        out["work_order"] = result["work_order"]
        out["warnings"] = result["warnings"]
        out["evidence"] = result.get("evidence") or {}
    return out


def _uploads(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """[{key, name?, description?}]: every uploaded file, in order. A field
    holds one storage key, a list of them, or (Orchestration) dicts."""
    out = []
    for field in ("uploaded_files", "attachments"):
        value = payload.get(field) or []
        for item in [value] if isinstance(value, str | dict) else list(value):
            if isinstance(item, dict) and item.get("key"):
                out.append(dict(item))
            elif isinstance(item, str) and item.strip():
                out.append({"key": item.strip()})
    return out


def _root(error: BaseException) -> BaseException:
    """The error at the bottom of Temporal's wrapping (ActivityError ->
    ApplicationError -> ...), whose message says what actually went wrong."""
    while getattr(error, "cause", None) is not None:
        error = error.cause
    return error
