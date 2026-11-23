"""Answering questions about a TDD — "why this approach?", "why not X?" (2026-10-08).

A design discussion happens in comments on the TDD's Confluence page. The
orchestrator hands each question here with the TDD as it stands and the
discussion so far. The answer is grounded in the TDD, the story's facts, the
decisions in force and the earlier discussion — never in code: since
2026-11-02 no code is read, stored or cloned to answer a question. Where an
answer depends on code the TDD does not describe, it says so and names the
file to check. It explains the reasoning, weighs the alternative honestly,
and says whether the design should change. It never changes the TDD itself —
a person asks for that explicitly.

One model call, same provider switch as planner.py: AI Gateway when
AGENTS_GATEWAY_KEY + AI_GATEWAY_URL are set, else OpenAI directly.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from . import planner

_MAX_TDD_CHARS = 30_000

SYSTEM_PROMPT = """You are the software architect who wrote a Technical Design Document (TDD).
A reviewer comments on the design page: a question, a proposed alternative, or a requested
change. You answer, and you are also the GATE for changes: the TDD changes only for a valid
modification.

Ground everything ONLY in the TDD, the story's facts, the decisions in force and the
discussion. You have no access to the code: never invent code, files or facts. Where the
answer depends on code the TDD does not describe, say so and name the file to check (only
files the TDD names). Answer in plain words, with no code.

The comment's KIND is given:
- question ("why this?"): explain the reasons and trade-offs. verdict "answer_only".
- proposal ("why not X?") or update ("update the TDD: X"): judge each requested change:
  1. Does it contradict the story's facts (the requirement)? Then refuse it, naming the fact.
  2. Is it technically worse or wrong for this design as the TDD describes it? Then refuse
     it, with the reason.
  3. Does it contradict a DECISION IN FORCE? Refuse it, citing the decision id — unless it is
     clearly better, then apply it and list that decision id in "supersedes".
  4. Is it already in the TDD? Then it is "already_present": say where.
  5. Otherwise, if it is correct and useful, apply it.
  verdict: "apply" (all applied), "partial" (some applied, some refused), "refuse" (none),
  "already_present" (nothing new to do).
For every change you apply that is a design choice (how money is computed, a check's place in
the validation order, how time is read, concurrency handling, ...), add it to
"new_decisions" with a short stable id slug, title, choice, reason, and applies_to (only files
the TDD names: {"service","file_path","symbol"}).

Respond with ONLY a JSON object:
{
  "answer": "the answer in plain, simple words: 2-5 short sentences or bullet lines, \
under 120 words, no markdown headings",
  "verdict": "answer_only" | "apply" | "partial" | "refuse" | "already_present",
  "apply": ["each change to make to the design, one or two sentences"],
  "refused": [{"change": "...", "reason": "..."}],
  "supersedes": ["decision-id"],
  "new_decisions": [{"id": "...", "title": "...", "choice": "...", "reason": "...",
                     "applies_to": [{"service": "...", "file_path": "...", "symbol": "..."}]}]
}"""

# Before 2026-10-26, and still when a caller sends no kind (ORCH_MEMORY=off):
# an answer and a suggestion, no validation.
LEGACY_SYSTEM_PROMPT = (
    """You are the software architect who wrote a Technical Design Document (TDD).
A reviewer asks you a question about it in a comment on the design page.

Answer as the author, in plain language, grounded ONLY in the TDD and the discussion so far.
You have no access to the code: never invent code, files or facts that are not there.

- "Why this approach?": give the concrete reasons from the TDD (constraints, existing
  structure, risk, effort), and what was traded off.
- "Why not <alternative>?": evaluate the alternative fairly — where it would be better, where
  it would be worse for THIS design and requirement — and conclude.
- If the alternative, or the reviewer's point, is genuinely better, say so and recommend a change.

Respond with ONLY a JSON object:
{
  "answer": "the answer in plain, simple words: 2-5 short sentences or bullet lines, \
under 120 words, no markdown headings",
  "change_recommended": true or false,
  "proposed_change": "if true: the change to make to the design, one or two sentences; else empty"
}"""
)

VERDICTS = ("answer_only", "apply", "partial", "refuse", "already_present")
KINDS = ("question", "proposal", "update", "update_override")


@dataclass(frozen=True)
class DesignAnswer:
    answer: str
    change_recommended: bool
    proposed_change: str
    verdict: str = "answer_only"
    apply: list = field(default_factory=list)
    refused: list = field(default_factory=list)
    supersedes: list = field(default_factory=list)
    new_decisions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def build_user_prompt(
    question: str,
    tdd_text: str,
    discussion: list[str],
    kind: str = "question",
    facts: list[dict] | None = None,
    decisions: list[dict] | None = None,
) -> str:
    parts = [f"KIND: {kind}", f"COMMENT:\n{question.strip()}"]
    if facts:
        parts += ["", "THE STORY'S FACTS (the requirement):"]
        parts += [f"- [{f['slot']}] {f['statement']}" for f in facts]
    if decisions:
        parts += ["", "DECISIONS IN FORCE:"]
        parts += [
            f"- [{d['id']}] {d['title']}: {d['choice']} (why: {d['reason']})" for d in decisions
        ]
    parts += ["", "TDD (as it stands):", tdd_text[:_MAX_TDD_CHARS]]
    if discussion:
        parts += ["", "DISCUSSION SO FAR (oldest first):", *discussion]
    return "\n".join(parts)


def _strings(items) -> list[str]:
    return [str(i).strip() for i in items or [] if str(i).strip()]


def parse_answer(raw: str) -> DesignAnswer:
    """The model's JSON as a DesignAnswer; plain text is taken as the answer.

    Older replies (answer / change_recommended / proposed_change only) still
    parse: a recommended change becomes verdict "apply"."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return DesignAnswer(answer=raw.strip(), change_recommended=False, proposed_change="")
    apply = _strings(data.get("apply"))
    if not apply and data.get("change_recommended") and data.get("proposed_change"):
        apply = [str(data["proposed_change"]).strip()]
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in VERDICTS:
        verdict = "apply" if apply else "answer_only"
    if verdict in ("answer_only", "refuse", "already_present"):
        apply = []
    refused = [
        {"change": str(r.get("change") or "").strip(), "reason": str(r.get("reason") or "").strip()}
        for r in data.get("refused") or []
        if isinstance(r, dict)
    ]
    decisions = [d for d in data.get("new_decisions") or [] if isinstance(d, dict) and d.get("id")]
    return DesignAnswer(
        answer=str(data.get("answer") or "").strip(),
        change_recommended=verdict in ("apply", "partial"),
        proposed_change="; ".join(apply),
        verdict=verdict,
        apply=apply,
        refused=refused,
        supersedes=_strings(data.get("supersedes")) if verdict in ("apply", "partial") else [],
        new_decisions=decisions if verdict in ("apply", "partial") else [],
    )


def override(change: str) -> DesignAnswer:
    """A person insists ("update the TDD anyway: X"): applied without checks, and
    recorded as a human override so later readers know it was not validated."""
    slug = "-".join(re.findall(r"[a-z0-9]+", change.lower()))[:48] or "override"
    return DesignAnswer(
        answer="Applied as requested, without the usual checks (a person's override).",
        change_recommended=True,
        proposed_change=change,
        verdict="apply",
        apply=[change],
        new_decisions=[
            {
                "id": f"override-{slug}",
                "title": change[:80],
                "choice": change,
                "reason": "Requested by a person as an override.",
                "override_by": "human",
                "applies_to": [],
            }
        ],
    )


def legacy_user_prompt(question: str, tdd_text: str, discussion: list[str]) -> str:
    parts = [f"QUESTION:\n{question.strip()}", "", "TDD (as it stands):", tdd_text[:_MAX_TDD_CHARS]]
    if discussion:
        parts += ["", "DISCUSSION SO FAR (oldest first):", *discussion]
    return "\n".join(parts)


async def legacy_answer(question: str, tdd_text: str, discussion: list[str] | None = None) -> dict:
    """The answer in its shape from before 2026-10-26: answer, change_recommended,
    proposed_change."""
    raw = await planner.call_llm(
        LEGACY_SYSTEM_PROMPT, legacy_user_prompt(question, tdd_text, list(discussion or []))
    )
    a = parse_answer(raw)
    return {
        "answer": a.answer,
        "change_recommended": a.change_recommended,
        "proposed_change": a.proposed_change,
    }


async def answer_question(
    question: str,
    tdd_text: str,
    discussion: list[str] | None = None,
    kind: str = "question",
    facts: list[dict] | None = None,
    decisions: list[dict] | None = None,
) -> DesignAnswer:
    kind = kind if kind in KINDS else "question"
    if kind == "update_override":
        return override(question.strip())
    raw = await planner.call_llm(
        SYSTEM_PROMPT,
        build_user_prompt(question, tdd_text, list(discussion or []), kind, facts, decisions),
    )
    return parse_answer(raw)
