"""The published code never builds an OpenAI client (2026-10-26)."""

import pathlib

import pytest

from src.ingestion import local_client


def test_without_an_installed_client_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setattr(local_client, "_factory", None)
    with pytest.raises(RuntimeError, match="scripts/local_worker.py"):
        local_client.new_client()


def test_an_installed_client_class_is_used(monkeypatch):
    class Fake:
        pass

    monkeypatch.setattr(local_client, "_factory", Fake)
    assert isinstance(local_client.new_client(), Fake)


def test_no_source_file_builds_an_openai_client():
    # The platform's publish policy refuses `AsyncOpenAI(...)` outside the gateway.
    src = pathlib.Path(__file__).parents[2] / "src"
    offenders = [p for p in src.rglob("*.py") if "AsyncOpenAI(" in p.read_text()]
    assert offenders == []
