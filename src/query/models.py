"""The structured output of the planning LLM call: a technical design doc
(TDD) naming which services, files, and functions/symbols likely need to
change for a given requirement or bug report, with concrete change
descriptions and reasoning a downstream coding agent can act on.

The LLM only ever produces this structured data — never the final document
text itself. render.py's job is to turn this into the actual readable TDD
(prose, tables, diagrams). Splitting it this way is deliberate: it's what
lets query_flow.build_tdd() go back afterward and overwrite specific fields
(service_dependencies, has_real_source, generated_at) with real, verified
facts instead of whatever the LLM guessed — something that isn't possible
once a document is just one block of free text.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FileChange(BaseModel):
    file_path: str | None = None
    function_or_symbol: str
    # What this symbol currently does, distinct from what changes — None
    # when the symbol has no real source and no doc description to ground
    # this in (never invented).
    current_behavior: str | None = None
    change_description: str
    implementation_notes: str
    # Illustrative by default — but when has_real_source is True, the LLM
    # was actually shown this symbol's real source (via the code graph), so
    # the sketch may track the real signature/body rather than being purely
    # speculative. Still never a claim of verified, tested code either way.
    pseudocode_sketch: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    reasoning: str
    change_type: Literal["create", "modify", "delete"] = "modify"
    # Set programmatically in query_flow.build_tdd() by checking the real
    # code graph — never trusted from the LLM's own claim — same "verify
    # against the graph, don't guess" principle as service_dependencies
    # below.
    has_real_source: bool = False


class ServicePlan(BaseModel):
    service: str
    changes: list[FileChange] = Field(default_factory=list)
    complexity: str
    # What responsibility changes for this service vs. what stays the same
    # — separate from `reasoning` (why it's affected at all).
    responsibility_change: str | None = None
    affected_data_model: str | None = None
    api_changes: list[str] = Field(default_factory=list)
    reasoning: str


class ServiceDependency(BaseModel):
    source: str
    target: str


class Scenario(BaseModel):
    category: str  # "unit" | "integration" | "api" | "regression" | "edge_case"
    scenario: str
    expected_result: str


class ArchComponent(BaseModel):
    """One node in an architecture diagram, as a FACT the LLM states —
    never as hand-drawn ASCII art. render.py builds the actual diagram from
    a list of these; asking the model to draw boxes-and-arrows itself was
    unreliable (inconsistent alignment, frequently skipped entirely) and
    spent effort better used on the reasoning fields."""

    name: str
    depends_on: list[str] = Field(default_factory=list)


class FlowStep(BaseModel):
    """One step in a request/execution flow, as an ordered fact — replaces
    both a prose flow paragraph AND a hand-drawn flow diagram with one
    source of truth; render.py derives both a numbered list and an ASCII
    chain from the same steps, so nothing is stated twice."""

    step: str
    detail: str | None = None


class SequenceStep(BaseModel):
    """One call or response in a multi-participant interaction — render.py
    assembles these into an aligned ASCII sequence diagram."""

    from_participant: str
    to_participant: str
    action: str
    direction: Literal["call", "response"] = "call"


class TechnicalDesignDoc(BaseModel):
    title: str
    # Set programmatically after parsing, in query_flow.build_tdd() — a
    # generation timestamp is a fact about the run, not something an LLM
    # should be asked to guess.
    generated_at: str | None = None
    issue_summary: str

    functional_requirements: list[str] = Field(default_factory=list)
    non_functional_requirements: list[str] = Field(default_factory=list)

    # Existing-system section — `current_architecture` is the narrative WHY
    # (distinct from the components list, which is just the WHAT); omitted
    # (left None/empty) rather than guessed when the available docs/code
    # don't actually describe an architecture.
    current_architecture: str | None = None
    current_architecture_components: list[ArchComponent] = Field(default_factory=list)
    current_flow_steps: list[FlowStep] = Field(default_factory=list)

    root_cause_analysis: str | None = None

    # Proposed-system section — same shape as the current-system section.
    proposed_solution: str | None = None
    proposed_architecture: str | None = None
    proposed_architecture_components: list[ArchComponent] = Field(default_factory=list)
    proposed_flow_steps: list[FlowStep] = Field(default_factory=list)
    sequence_steps: list[SequenceStep] = Field(default_factory=list)

    services: list[ServicePlan]
    # Filled in programmatically from the real CALLS graph after parsing —
    # never by the LLM — so the dependency diagram is always accurate.
    service_dependencies: list[ServiceDependency] = Field(default_factory=list)
    # Free-text notes on HOW services communicate (REST/events/queues) and
    # why — the LLM's own account, layered on top of service_dependencies
    # (which only records THAT a real CALLS edge exists, not how/why).
    communication_notes: list[str] = Field(default_factory=list)

    error_handling_notes: str | None = None
    security_considerations: str | None = None
    backward_compatibility_notes: str | None = None

    testing_notes: str | None = None
    testing_plan: list[Scenario] = Field(default_factory=list)

    implementation_plan: list[str] = Field(default_factory=list)

    risks: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    overall_reasoning: str
