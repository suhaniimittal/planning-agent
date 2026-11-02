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

from pydantic import BaseModel, Field

from src.query.models import DesignDecision

# The id of a decision taken from the card's answers (questions.question_id).
CARD_DECISION = re.compile(r"^q\d+(-|$)")


class _Model(BaseModel):
    # A model's extra keys are ignored rather than failing the whole document.
    model_config = {"extra": "ignore"}


class FlowStep(_Model):
    step: str = Field(description="One step of the flow, in plain English.")
    service: str = Field(default="", description="The service this step happens in, if one.")


class FileToLook(_Model):
    path: str
    why: str = ""


class Step(_Model):
    """One instruction of the implementation guide."""

    title: str = ""
    where: str = Field(default="", description="The file to change, as a path given.")
    new_file: bool = False  # `where` is a file to create, so it is not in the repository yet
    what: str
    why: str = ""
    done_when: str = ""


class ServicePlan(_Model):
    service: str
    what_changes: str = Field(description="What changes in this service, in one sentence.")
    steps: list[Step] = Field(default_factory=list)
    files_to_look_at: list[FileToLook] = Field(default_factory=list)
    reference_pr: str = Field(
        default="", description="'#142' when a listed pull request shows the pattern, else ''."
    )


class RisksAndTests(_Model):
    risks: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)


class OpenQuestion(_Model):
    question: str
    why_it_matters: str = ""
    source: Literal["card", "planner"] = "planner"


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
    # What the ticket and its answers require, one plain sentence each.
    requirements: list[str] = Field(default_factory=list)
    decisions: list[DesignDecision] = Field(default_factory=list)
    current_flow: list[FlowStep] = Field(default_factory=list)
    flow_after_change: list[FlowStep] = Field(default_factory=list)
    per_service_plan: list[ServicePlan] = Field(default_factory=list)
    order_of_work: list[str] = Field(default_factory=list)
    risks_and_tests: RisksAndTests = Field(default_factory=RisksAndTests)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
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
    warnings: list[str] = Field(default_factory=list)
    generated_at: str = ""

    def card_decisions(self) -> list[DesignDecision]:
        return [d for d in self.decisions if CARD_DECISION.match(d.id)]

    def design_decisions(self) -> list[DesignDecision]:
        return [d for d in self.decisions if not CARD_DECISION.match(d.id)]
