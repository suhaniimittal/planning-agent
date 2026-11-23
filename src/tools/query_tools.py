"""Tool entrypoint for questions on a published TDD (the agent's "discuss" mode).

The TDD itself is written by src/tools/tdd_tools.py (2026-11-02). A question
is answered from the TDD's text, the story's facts, the decisions in force
and the earlier discussion — no code, no graph, no clone.
"""

from __future__ import annotations

from aetherion_sdk import tool

from src.memory import conversation, record
from src.query.design_qa import answer_question, legacy_answer


@tool()
async def answer_design_question(
    question: str,
    tdd_text: str,
    discussion: list[str] | None = None,
    issue_key: str | None = None,
    kind: str | None = None,
    page_id: str = "",
) -> dict:
    """A comment on a TDD ("why this approach?", "why not X?", "update the TDD: X"),
    answered from the TDD, the story's facts, the decisions in force and the
    discussion — and judged: a change is applied only when it is valid
    (2026-10-26). Changes nothing itself; the caller revises. With no ``kind``
    (a caller with memory off) it answers in the shape from before then."""
    if not kind:
        return await legacy_answer(question, tdd_text, discussion)
    facts, decisions = [], []
    if issue_key:
        stored = record.load_requirement(issue_key)
        facts = (stored or {}).get("facts") or []
        decisions = record.decisions_in_force(issue_key)
    # What was said before about this TDD, from Memsy (2026-10-26): earlier
    # questions and answers, even from other days or another page's thread.
    earlier = await conversation.recall(issue_key or "", question) if issue_key else []
    context = list(discussion or []) + [f"(remembered) {line}" for line in earlier]
    answer = await answer_question(
        question, tdd_text, context, kind or "question", facts, decisions
    )
    out = answer.to_dict()
    if issue_key:
        await conversation.remember_exchange(
            issue_key, page_id, question, out.get("answer") or "", kind, out.get("verdict") or ""
        )
    return out
