"""A design decision, as the memory keeps it (2026-10-26).

The TDD itself is src/tdd/models.py (2026-11-02); its `decisions` are these,
so src/memory stores and recalls them unchanged.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DecisionTarget(BaseModel):
    service: str
    file_path: str
    symbol: str | None = None


class DesignDecision(BaseModel):
    """One design choice the TDD makes, kept by later versions (2026-10-26)."""

    id: str = Field(description="A short stable slug for the choice, e.g. 'money-integer-paise'.")
    title: str = Field(
        description="The choice in a few words, e.g. 'Money summed as integer paise'."
    )
    choice: str = Field(description="What is decided, concretely enough to check code against.")
    reason: str = Field(description="Why, in one sentence.")
    applies_to: list[DecisionTarget] = Field(
        default_factory=list,
        description="The real files (and symbols) this choice governs, from the code shown.",
    )
