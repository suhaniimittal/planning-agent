"""The TDD as the writer call returns it, and the work order (2026-11-02).

Plain English throughout: no code, no snippets. Files, components and routes
are named, never quoted. `decisions` is the memory's own DesignDecision
(src/query/models.py), so src/memory keeps and recalls them unchanged; the
card's answered questions are among them, with ids "q<n>-...".

The fields under "set by the flow" are never the model's to fill: status,
the pinned commits, what was and was not read, the warnings. The last three
are for the log and for Orchestration, not for the document.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from src.query.models import DesignDecision

# The id of a decision taken from the card's answers (questions.question_id).
CARD_DECISION = re.compile(r"^q\d+(-|$)")


class _Model(BaseModel):
    # A model's extra keys are ignored rather than failing the whole document.
    model_config = {"extra": "ignore"}


class FlowStep(_Model):
    step: str = Field(description="One step of the flow, in plain English.")
    service: str = Field(default="", description="The service this step happens in, if one.")


class Step(_Model):
    """One instruction of the implementation guide (2026-11-06). No file is
    named: with no code read, a file name is a guess, and a wrong one sends
    the coding agent to the wrong place. The step says what the component is,
    in words, and the folder it lives in; the coding agent finds the file."""

    title: str = ""
    what: str
    component: str = Field(default="", description="What changes, in words.")
    # The folders it lives in, from the paths given: one, or a few narrow ones
    # rather than one broad one (2026-11-12).
    areas: list[str] = Field(default_factory=list)
    new_area: bool = False  # the first area is a folder to create; its parent exists
    done_when: str = ""

    @model_validator(mode="before")
    @classmethod
    def _one_area(cls, value: object) -> object:
        # One folder written as "area", the shape before 2026-11-12.
        if isinstance(value, dict) and "areas" not in value and value.get("area"):
            return {**value, "areas": [value["area"]]}
        return value


class DataNeed(_Model):
    """One value a change needs (2026-11-12): where it may come from, and the
    contract to confirm it in. With no code read, whether the value exists
    there is not known (KS-XXXXX: the order's language was not on Order)."""

    service: str = ""
    what: str
    sources: list[str] = Field(default_factory=list)  # each source the ticket allows, best first
    confirm_in: str = ""  # the contract folder, from the paths given


class ServicePlan(_Model):
    service: str
    what_changes: str = Field(description="What changes in this service, in one sentence.")
    steps: list[Step] = Field(default_factory=list)
    reference_pr: str = Field(
        default="", description="'#142' when a listed pull request shows the pattern, else ''."
    )


class RisksAndTests(_Model):
    risks: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)


class Requirement(_Model):
    requirement: str
    check: str = ""  # how a tester confirms it, with the expected result


class OpenQuestion(_Model):
    question: str
    why_it_matters: str = ""
    source: Literal["card", "planner"] = "planner"
    # A card answer still "to confirm": planned on, but not settled.
    answer: str = ""
    # That answer as the one-line assumption the plan uses (the writer's).
    assumption: str = ""
    # The writer's own question that repeats a card question: that question.
    same_as: str = ""


class Assumption(_Model):
    question: str  # a card question with an answer so far, word for word
    assumption: str


class WorkStep(_Model):
    order: int = 0
    service: str
    summary: str = Field(default="", description="What to do, in one or two sentences.")
    reference_pr: str = ""
    # Set by the flow from the input and the pinned commit, never by the model.
    repo: str = ""
    path: str = ""
    base_branch: str = ""
    base_sha: str = ""


class WorkOrder(_Model):
    ticket: str = ""
    steps: list[WorkStep] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    ready: bool = False
    not_ready_reason: str = ""


class PinnedService(_Model):
    service: str
    repo: str
    branch: str
    sha: str
    path: str = ""


class TddDocument(_Model):
    title: str = ""
    problem: str = ""
    cause: str = ""
    fix: str = ""
    # What must be true after the change, each with how to check it.
    requirements: list[Requirement] = Field(default_factory=list)
    decisions: list[DesignDecision] = Field(default_factory=list)
    current_flow: list[FlowStep] = Field(default_factory=list)
    flow_after_change: list[FlowStep] = Field(default_factory=list)
    per_service_plan: list[ServicePlan] = Field(default_factory=list)
    order_of_work: list[str] = Field(default_factory=list)
    data_needed: list[DataNeed] = Field(default_factory=list)
    risks_and_tests: RisksAndTests = Field(default_factory=RisksAndTests)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)  # read into open_questions
    work_order: WorkOrder = Field(default_factory=WorkOrder)

    # --- set by the flow -----------------------------------------------------------------
    ticket: str = ""
    ticket_type: str = ""
    status: Literal["final", "draft"] = "final"
    draft_reason: str = ""
    high_level: bool = False  # code was not read: the plan names files to inspect
    pinned: list[PinnedService] = Field(default_factory=list)
    files_read: list[str] = Field(default_factory=list)  # "service/path"
    not_read: list[str] = Field(default_factory=list)
    # Paths whose names or contents match the ticket's words: unconfirmed,
    # for the coding agent to search from, never shown as where to change.
    hints: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = ""

    @field_validator("requirements", mode="before")
    @classmethod
    def _plain_requirements(cls, value: object) -> object:
        # A requirement written as a plain sentence has no check yet.
        if isinstance(value, list):
            return [{"requirement": v} if isinstance(v, str) else v for v in value]
        return value

    def card_decisions(self) -> list[DesignDecision]:
        return [d for d in self.decisions if CARD_DECISION.match(d.id)]

    def design_decisions(self) -> list[DesignDecision]:
        return [d for d in self.decisions if not CARD_DECISION.match(d.id)]
