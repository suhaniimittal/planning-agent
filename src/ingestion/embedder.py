"""Wraps the embedding model behind one function, with a runtime switch:

- If AGENTS_GATEWAY_KEY + AI_GATEWAY_URL are set (the case once this is
  published on the Aetherion platform), route through the AI Gateway.
- Otherwise (local dev), call OpenAI directly using OPENAI_API_KEY.

Only this file needs to know that distinction — ingest.py just calls embed().

Both clients below are lazily-constructed, module-level singletons, reused
across every embed() call. A single ingestion run can call this thousands
of times (once per doc summary, once per code chunk) — constructing a fresh
client per call would pay a fresh TCP+TLS handshake on every single one
instead of reusing one already-open connection. Never explicitly closed;
process exit handles it, same as graph_writer.get_driver()'s singleton.
"""

from __future__ import annotations

import asyncio
import os

_MODEL = "text-embedding-3-small"
_PROVIDER = "openai"

# The model rejects any input over 8192 tokens — and in a batch, one such
# input fails the whole request. A token always covers at least one byte, so
# capping by UTF-8 bytes keeps every input under that limit without needing
# a tokenizer. Only what's embedded is shortened; callers keep the full text.
_MAX_INPUT_BYTES = 8000


def _fit(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_INPUT_BYTES:
        return text
    return encoded[:_MAX_INPUT_BYTES].decode("utf-8", errors="ignore")

_openai_client = None
_gateway_client = None
_gateway_client_lock = asyncio.Lock()


def _use_gateway() -> bool:
    return bool(os.environ.get("AGENTS_GATEWAY_KEY") and os.environ.get("AI_GATEWAY_URL"))


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        _openai_client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    return _openai_client


async def _get_gateway_client():
    global _gateway_client
    if _gateway_client is None:
        # Construction can await (it opens the underlying connection), so —
        # unlike the OpenAI client above — two concurrent embed() calls could
        # both see `_gateway_client is None` and race to create one each.
        # The lock plus re-check after acquiring it is what keeps this to
        # exactly one real client no matter how many chunks are embedding
        # concurrently (ingest.py runs up to 10 embed() calls at once).
        async with _gateway_client_lock:
            if _gateway_client is None:
                from agent_lib.gateway.ai import AiGatewayClient

                client = AiGatewayClient(
                    gateway_key=os.environ["AGENTS_GATEWAY_KEY"],
                    gateway_url=os.environ["AI_GATEWAY_URL"],
                )
                _gateway_client = await client.__aenter__()
    return _gateway_client


async def _embed_via_gateway(text: str) -> list[float]:
    gateway_client = await _get_gateway_client()
    return await gateway_client.embed(provider=_PROVIDER, model_name=_MODEL, text=text)


async def _embed_via_openai(text: str) -> list[float]:
    client = _get_openai_client()
    resp = await client.embeddings.create(model=_MODEL, input=text)
    return resp.data[0].embedding


async def embed(text: str) -> list[float]:
    text = _fit(text)
    if _use_gateway():
        return await _embed_via_gateway(text)
    return await _embed_via_openai(text)


async def _embed_many_via_gateway(texts: list[str]) -> list[list[float]]:
    # The gateway accepts a list, but its declared return type is a single
    # vector — so the batched reply is only trusted when it really is one
    # vector per text; otherwise fall back to one call per text.
    gateway_client = await _get_gateway_client()
    result = await gateway_client.embed(provider=_PROVIDER, model_name=_MODEL, text=texts)
    if (
        isinstance(result, list)
        and len(result) == len(texts)
        and all(isinstance(v, list) for v in result)
    ):
        return result
    return [await _embed_via_gateway(t) for t in texts]


async def _embed_many_via_openai(texts: list[str]) -> list[list[float]]:
    client = _get_openai_client()
    resp = await client.embeddings.create(model=_MODEL, input=texts)
    # OpenAI tags each vector with its input's index; sort rather than trust order.
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


async def embed_many(texts: list[str]) -> list[list[float]]:
    """One request for many texts — the vectors are identical to calling
    embed() per text, just far fewer round trips. Returns one vector per
    text, in the same order. Raises if the request fails as a whole."""
    if not texts:
        return []
    texts = [_fit(t) for t in texts]
    if _use_gateway():
        return await _embed_many_via_gateway(texts)
    return await _embed_many_via_openai(texts)
