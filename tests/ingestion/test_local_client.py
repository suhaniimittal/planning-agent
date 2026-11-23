"""The client used when there is no AI gateway (2026-10-26)."""

import sys
import types

from src.ingestion import local_client


def test_without_an_installed_client_openai_is_used(monkeypatch):
    class FakeOpenAI:
        pass

    monkeypatch.setattr(local_client, "_factory", None)
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))

    assert isinstance(local_client.new_client(), FakeOpenAI)


def test_an_installed_client_class_is_used(monkeypatch):
    class Fake:
        pass

    monkeypatch.setattr(local_client, "_factory", Fake)
    assert isinstance(local_client.new_client(), Fake)
