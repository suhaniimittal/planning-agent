"""Shared test setup (2026-10-15)."""

import base64

import pytest

# A fixed, test-only key — never a real CODE_GRAPH_KEY.
TEST_CODE_GRAPH_KEY = base64.b64encode(bytes(range(32))).decode()


@pytest.fixture(autouse=True)
def _offline_stack(monkeypatch):
    """No test reads Spring Boot's versions from Maven Central."""
    monkeypatch.setenv("STACK_OFFLINE", "1")


@pytest.fixture(autouse=True)
def _code_graph_key(monkeypatch):
    """The graph's code is encrypted (2026-10-22); tests use a fixed test key."""
    monkeypatch.setenv("CODE_GRAPH_KEY", TEST_CODE_GRAPH_KEY)
