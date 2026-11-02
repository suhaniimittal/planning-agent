"""Planning asks the strongest Claude first and falls back (2026-10-23)."""

import sys
import types

import pytest

from src.query import planner


class FakeClient:
    def __init__(self, refuse=(), error="model claude-opus-5-5 not found", **_):
        self.refuse, self.error, self.asked = set(refuse), error, []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def chat(self, *, model_name, **kwargs):
        self.asked.append((model_name, kwargs["max_tokens"]))
        if model_name in self.refuse:
            raise RuntimeError(self.error.replace("claude-opus-5-5", model_name))
        return {"content": f"design by {model_name}"}


@pytest.fixture
def gateway(monkeypatch):
    def install(**kwargs):
        client = FakeClient(**kwargs)
        module = types.ModuleType("agent_lib.gateway.ai")
        module.AiGatewayClient = lambda **_: client
        monkeypatch.setitem(sys.modules, "agent_lib.gateway.ai", module)
        monkeypatch.setattr(planner.gateway_env, "gateway_key", lambda: "k")
        monkeypatch.setattr(planner.gateway_env, "gateway_url", lambda: "u")
        monkeypatch.setattr(planner, "_working_model", None)
        monkeypatch.delenv("PLANNING_LLM_MODEL", raising=False)
        return client

    return install


@pytest.mark.asyncio
async def test_opus_is_asked_first_with_room_to_think(gateway):
    client = gateway()
    assert await planner._plan_via_gateway("s", "u") == "design by claude-opus-5-5"
    assert client.asked == [("claude-opus-5-5", 32000)]


@pytest.mark.asyncio
async def test_a_model_the_gateway_lacks_falls_back_and_is_remembered(gateway):
    client = gateway(refuse={"claude-opus-5-5"})
    assert await planner._plan_via_gateway("s", "u") == "design by claude-opus-4-8"
    await planner._plan_via_gateway("s", "u")
    assert [m for m, _ in client.asked] == ["claude-opus-5-5", "claude-opus-4-8", "claude-opus-4-8"]


@pytest.mark.asyncio
async def test_sonnet_is_the_last_resort(gateway):
    client = gateway(refuse={"claude-opus-5-5", "claude-opus-4-8"})
    assert await planner._plan_via_gateway("s", "u") == "design by claude-sonnet-4-6"
    assert len(client.asked) == 3


@pytest.mark.asyncio
async def test_a_network_error_is_not_hidden_by_trying_other_models(gateway):
    client = gateway(refuse={"claude-opus-5-5"}, error="Connection reset by peer")
    with pytest.raises(RuntimeError, match="Connection reset"):
        await planner._plan_via_gateway("s", "u")
    assert len(client.asked) == 1


def test_a_chosen_model_goes_first(monkeypatch):
    monkeypatch.setattr(planner, "_working_model", None)
    monkeypatch.setenv("PLANNING_LLM_MODEL", "claude-opus-4-7")
    assert planner.gateway_models()[:2] == ["claude-opus-4-7", "claude-opus-5-5"]
