"""Shared test setup (2026-10-15)."""

import pytest


@pytest.fixture(autouse=True)
def _offline_stack(monkeypatch):
    """No test reads Spring Boot's versions from Maven Central."""
    monkeypatch.setenv("STACK_OFFLINE", "1")
