import asyncio
import sys
import types

import pytest

from src.ingestion import embedder


@pytest.fixture(autouse=True)
def _reset_singletons(monkeypatch):
    """The whole point of the fix under test is module-level caching, so
    every test must start from a clean slate — otherwise an earlier test's
    cached client would leak into a later one."""
    monkeypatch.setattr(embedder, "_openai_client", None)
    monkeypatch.setattr(embedder, "_gateway_client", None)
    monkeypatch.delenv("AGENTS_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_URL", raising=False)


class FakeEmbeddingData:
    def __init__(self, embedding):
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, embedding):
        self.data = [FakeEmbeddingData(embedding)]


class FakeEmbeddingsAPI:
    def __init__(self):
        self.calls = []

    async def create(self, model, input):
        self.calls.append((model, input))
        return FakeEmbeddingResponse([0.1, 0.2])


class FakeAsyncOpenAI:
    construction_count = 0

    def __init__(self):
        FakeAsyncOpenAI.construction_count += 1
        self.embeddings = FakeEmbeddingsAPI()


def _inject_fake_openai_module(monkeypatch):
    FakeAsyncOpenAI.construction_count = 0
    fake_module = types.ModuleType("openai")
    fake_module.AsyncOpenAI = FakeAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)


class FakeGatewayClient:
    construction_count = 0

    def __init__(self, gateway_key, gateway_url):
        FakeGatewayClient.construction_count += 1
        self.gateway_key = gateway_key
        self.gateway_url = gateway_url
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def embed(self, provider, model_name, text):
        self.calls.append((provider, model_name, text))
        return [0.3, 0.4]


def _inject_fake_gateway_module(monkeypatch):
    FakeGatewayClient.construction_count = 0
    ai_module = types.ModuleType("agent_lib.gateway.ai")
    ai_module.AiGatewayClient = FakeGatewayClient
    gateway_module = types.ModuleType("agent_lib.gateway")
    gateway_module.ai = ai_module
    agent_lib_module = types.ModuleType("agent_lib")
    agent_lib_module.gateway = gateway_module

    monkeypatch.setitem(sys.modules, "agent_lib", agent_lib_module)
    monkeypatch.setitem(sys.modules, "agent_lib.gateway", gateway_module)
    monkeypatch.setitem(sys.modules, "agent_lib.gateway.ai", ai_module)


# --- OpenAI path: client is constructed once, reused across calls ----------


@pytest.mark.asyncio
async def test_embed_via_openai_constructs_client_only_once_across_many_calls(monkeypatch):
    _inject_fake_openai_module(monkeypatch)

    for _ in range(5):
        vector = await embedder.embed("some chunk of code")
        assert vector == [0.1, 0.2]

    assert FakeAsyncOpenAI.construction_count == 1


@pytest.mark.asyncio
async def test_embed_via_openai_reuses_the_same_client_instance(monkeypatch):
    _inject_fake_openai_module(monkeypatch)

    await embedder.embed("first")
    await embedder.embed("second")

    client = embedder._openai_client
    assert client.embeddings.calls == [
        ("text-embedding-3-small", "first"),
        ("text-embedding-3-small", "second"),
    ]


@pytest.mark.asyncio
async def test_embed_via_openai_concurrent_calls_still_construct_client_once(monkeypatch):
    """No `await` sits between the None-check and the assignment in
    _get_openai_client, so this is safe without a lock — this test is the
    regression guard for that invariant."""
    _inject_fake_openai_module(monkeypatch)

    await asyncio.gather(*(embedder.embed(f"chunk {i}") for i in range(10)))

    assert FakeAsyncOpenAI.construction_count == 1


# --- Gateway path: client is constructed once, even under concurrency ------


@pytest.mark.asyncio
async def test_embed_via_gateway_constructs_client_only_once_across_many_calls(monkeypatch):
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "test-key")
    monkeypatch.setenv("AI_GATEWAY_URL", "https://gateway.test")
    _inject_fake_gateway_module(monkeypatch)

    for _ in range(5):
        vector = await embedder.embed("some chunk of code")
        assert vector == [0.3, 0.4]

    assert FakeGatewayClient.construction_count == 1


@pytest.mark.asyncio
async def test_embed_via_gateway_concurrent_calls_construct_client_exactly_once(monkeypatch):
    """_get_gateway_client's construction path awaits (unlike the OpenAI
    one), so without the lock + re-check, concurrent embed() calls could
    each see the cache empty and construct their own client. This is the
    regression guard for that race."""
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "test-key")
    monkeypatch.setenv("AI_GATEWAY_URL", "https://gateway.test")
    _inject_fake_gateway_module(monkeypatch)

    results = await asyncio.gather(*(embedder.embed(f"chunk {i}") for i in range(10)))

    assert results == [[0.3, 0.4]] * 10
    assert FakeGatewayClient.construction_count == 1


@pytest.mark.asyncio
async def test_use_gateway_requires_both_env_vars(monkeypatch):
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "test-key")
    # AI_GATEWAY_URL deliberately left unset.
    assert embedder._use_gateway() is False


# --- embed_many: one request for many texts ----------------------------------


class _Item:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


@pytest.mark.asyncio
async def test_embed_many_openai_sends_one_request_and_keeps_input_order(monkeypatch):
    calls = []

    class Embeddings:
        async def create(self, model, input):
            calls.append(input)
            # Returned out of order on purpose: the index, not the position, decides.
            return types.SimpleNamespace(data=[_Item(1, [2.0]), _Item(0, [1.0])])

    monkeypatch.setattr(
        embedder, "_get_openai_client", lambda: types.SimpleNamespace(embeddings=Embeddings())
    )

    vectors = await embedder.embed_many(["a", "b"])

    assert calls == [["a", "b"]]
    assert vectors == [[1.0], [2.0]]


@pytest.mark.asyncio
async def test_embed_many_empty_makes_no_request(monkeypatch):
    def must_not_be_called():
        raise AssertionError("no request expected for an empty list")

    monkeypatch.setattr(embedder, "_get_openai_client", must_not_be_called)

    assert await embedder.embed_many([]) == []


def _use_fake_gateway(monkeypatch, client):
    monkeypatch.setenv("AGENTS_GATEWAY_KEY", "key")
    monkeypatch.setenv("AI_GATEWAY_URL", "https://gateway.example")

    async def get_client():
        return client

    monkeypatch.setattr(embedder, "_get_gateway_client", get_client)


@pytest.mark.asyncio
async def test_embed_many_gateway_uses_batched_reply_when_one_vector_per_text(monkeypatch):
    class Client:
        calls = []

        async def embed(self, provider, model_name, text):
            self.calls.append(text)
            return [[1.0], [2.0]]

    client = Client()
    _use_fake_gateway(monkeypatch, client)

    assert await embedder.embed_many(["a", "b"]) == [[1.0], [2.0]]
    assert client.calls == [["a", "b"]]


@pytest.mark.asyncio
async def test_embed_many_gateway_falls_back_to_one_call_per_text_on_single_vector_reply(
    monkeypatch,
):
    class Client:
        calls = []

        async def embed(self, provider, model_name, text):
            self.calls.append(text)
            return [0.5, 0.5]

    client = Client()
    _use_fake_gateway(monkeypatch, client)

    assert await embedder.embed_many(["a", "b"]) == [[0.5, 0.5], [0.5, 0.5]]
    assert client.calls == [["a", "b"], "a", "b"]


def test_fit_leaves_short_text_alone_and_caps_long_text_by_bytes():
    assert embedder._fit("short") == "short"
    long_text = "é" * 10_000  # 2 bytes each: 20,000 bytes
    fitted = embedder._fit(long_text)
    assert len(fitted.encode("utf-8")) <= embedder._MAX_INPUT_BYTES
    assert long_text.startswith(fitted)


@pytest.mark.asyncio
async def test_embed_many_sends_capped_texts(monkeypatch):
    sent = []

    class Embeddings:
        async def create(self, model, input):
            sent.extend(input)
            return types.SimpleNamespace(data=[_Item(i, [0.0]) for i in range(len(input))])

    monkeypatch.setattr(
        embedder, "_get_openai_client", lambda: types.SimpleNamespace(embeddings=Embeddings())
    )

    await embedder.embed_many(["ok", "x" * 50_000])

    assert sent[0] == "ok"
    assert len(sent[1]) == embedder._MAX_INPUT_BYTES
