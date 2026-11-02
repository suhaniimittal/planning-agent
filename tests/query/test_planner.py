
import pytest


@pytest.mark.asyncio
async def test_both_llm_paths_use_a_low_temperature(monkeypatch):
    """The API default (1.0) made identical input produce very different
    TDDs from run to run."""
    import sys
    import types

    from src.query import planner

    sent = {}

    class Completions:
        async def create(self, **kwargs):
            sent["openai"] = kwargs
            message = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    # The laptop launcher installs the client class (src/ingestion/local_client.py).
    from src.ingestion import local_client

    monkeypatch.setattr(local_client, "_factory", FakeOpenAI)

    class FakeGateway:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def chat(self, **kwargs):
            sent["gateway"] = kwargs
            return {"content": "{}"}

    ai_module = types.SimpleNamespace(AiGatewayClient=FakeGateway)
    monkeypatch.setitem(sys.modules, "agent_lib.gateway.ai", ai_module)
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "k")
    monkeypatch.setenv("AI_GATEWAY_URL", "https://gateway.example")

    monkeypatch.setattr(planner, "_OPENAI_MODEL", "gpt-4o")
    await planner._plan_via_openai("system", "user")
    await planner._plan_via_gateway("system", "user")

    assert sent["openai"]["temperature"] == planner._TEMPERATURE
    assert sent["gateway"]["temperature"] == planner._TEMPERATURE
    assert planner._TEMPERATURE <= 0.3


def test_models_get_the_api_parameters_they_actually_accept():
    """Three families, verified against the live API: gpt-5 models think only
    when asked and then refuse `temperature`, gpt-5.5/5.6 also refuse `seed`,
    and pre-5 models take the plain parameters."""
    from src.query import planner

    refuses_seed = planner._openai_params("gpt-5.5")
    assert "temperature" not in refuses_seed and "seed" not in refuses_seed
    assert refuses_seed["reasoning_effort"] == "high"
    assert refuses_seed["max_completion_tokens"] >= 12000

    thinking = planner._openai_params("gpt-5.4")
    assert "temperature" not in thinking and thinking["seed"] == planner._SEED
    assert thinking["reasoning_effort"] == "high"
    assert thinking["max_completion_tokens"] >= 12000

    plain = planner._openai_params("gpt-4.1")
    assert plain == {"max_tokens": 12000, "temperature": 0, "seed": planner._SEED}


def test_tdd_model_thinks_before_writing():
    """With only temperature=0 the model spent zero reasoning tokens."""
    from src.query import planner

    assert planner._openai_params(planner._OPENAI_MODEL)["reasoning_effort"] == "high"


# ---------- bug / feature modes ----------


# ---------- test work as implementation work ----------


def _ui_symbol(file_path: str) -> dict:
    return {
        "file_path": file_path, "line_start": 1, "line_end": 9,
        "qualified_name": f"{file_path}::X", "name": "X", "signature": "X",
        "docstring": None, "language": "typescript", "matched_identifiers": [],
    }
