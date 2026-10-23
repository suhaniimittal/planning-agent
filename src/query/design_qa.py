"""Answering questions about a TDD — "why this approach?", "why not X?" (2026-10-08).

A design discussion happens in comments on the TDD's Confluence page. The
orchestrator hands each question here with the TDD as it stands and the
discussion so far. The answer is grounded in the TDD and in the code the
question touches (the same graph a TDD is written from): it explains the
reasoning, weighs the alternative honestly, and says whether the design should
change. It never changes the TDD itself — a person asks for that explicitly.

One model call, same provider switch as planner.py: AI Gateway when
AGENTS_GATEWAY_KEY + AI_GATEWAY_URL are set, else OpenAI directly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from src.ingestion.embedder import embed

from . import graph_reader, planner

_MAX_TDD_CHARS = 30_000
_MAX_SNIPPETS = 8

SYSTEM_PROMPT = """You are the software architect who wrote a Technical Design Document (TDD).
A reviewer asks you a question about it in a comment on the design page.

Answer as the author, in plain language, grounded ONLY in the TDD, the discussion so far and
the code excerpts given. Never invent code, files or facts that are not there.

- "Why this approach?": give the concrete reasons from the TDD and the code (constraints,
  existing structure, risk, effort), and what was traded off.
- "Why not <alternative>?": evaluate the alternative fairly — where it would be better, where
  it would be worse for THIS codebase and requirement — and conclude.
- If the alternative, or the reviewer's point, is genuinely better, say so and recommend a change.

Respond with ONLY a JSON object:
{
  "answer": "the answer, 3-10 short paragraphs or bullet lines, no markdown headings",
  "change_recommended": true or false,
  "proposed_change": "if true: the change to make to the design, one or two sentences; else empty"
}"""


@dataclass(frozen=True)
class DesignAnswer:
    answer: str
    change_recommended: bool
    proposed_change: str

    def to_dict(self) -> dict:
        return asdict(self)


async def _related_code(question: str, tdd_text: str) -> list[str]:
    """The code the question is most likely about: chunks near the question.

    Every service is searched, best SERVICE.md match first (2026-10-23): a
    service without a SERVICE.md has no doc embedding, so the doc index alone
    never returned it and its questions were answered without seeing any code.
    The chunks' own similarity decides what is shown.
    """
    try:
        embedding = await embed(f"{question}\n\n{tdd_text[:2000]}")
        scores = graph_reader.summary_scores(embedding)
        services = sorted(scores, key=lambda name: -scores[name])
        hits = graph_reader.vector_search_chunks(embedding, services, k=_MAX_SNIPPETS * 3)
    except Exception as error:  # noqa: BLE001 - an answer from the TDD alone is still useful
        print(f"  [DESIGN Q&A] no code context: {type(error).__name__}: {error}")
        return []
    seen: set[tuple[str, str]] = set()
    out: list[str] = []
    for hit in hits:
        key = (hit["service"], hit["function_qualified_name"])
        if key in seen:
            continue
        seen.add(key)
        out.append(f"[{hit['service']}] {hit['function_qualified_name']}\n{hit['text']}")
        if len(out) >= _MAX_SNIPPETS:
            break
    return out


def build_user_prompt(question: str, tdd_text: str, discussion: list[str], code: list[str]) -> str:
    parts = [f"QUESTION:\n{question.strip()}", "", "TDD (as it stands):", tdd_text[:_MAX_TDD_CHARS]]
    if discussion:
        parts += ["", "DISCUSSION SO FAR (oldest first):", *discussion]
    if code:
        parts += ["", "RELATED CODE (from the code graph):", *code]
    return "\n".join(parts)


def parse_answer(raw: str) -> DesignAnswer:
    """The model's JSON as a DesignAnswer; plain text is taken as the answer."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return DesignAnswer(answer=raw.strip(), change_recommended=False, proposed_change="")
    return DesignAnswer(
        answer=str(data.get("answer") or "").strip(),
        change_recommended=bool(data.get("change_recommended")),
        proposed_change=str(data.get("proposed_change") or "").strip(),
    )


async def answer_question(
    question: str, tdd_text: str, discussion: list[str] | None = None
) -> DesignAnswer:
    code = await _related_code(question, tdd_text)
    raw = await planner.call_llm(
        SYSTEM_PROMPT, build_user_prompt(question, tdd_text, list(discussion or []), code)
    )
    return parse_answer(raw)
