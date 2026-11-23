"""The model call every part of Planning makes (2026-11-02: trimmed to it).

AI Gateway when a gateway key is set (the platform), else OpenAI directly (a
laptop). Only the OpenAI path can ask for native JSON and send images; the
gateway's chat() takes text only. Callers parse what comes back: the TDD
flow (src/tdd/flow.py), the requirement facts and the equivalence judge
(src/memory), and the design Q&A (design_qa.py).
"""

from __future__ import annotations

import base64
import os
import re

from src.ingestion import gateway_env

from .images import around_slot

_GATEWAY_PROVIDER = "anthropic"
# The strongest Claude first (2026-10-23), then the next, so a design is never
# lost to a model the gateway does not offer. PLANNING_LLM_MODEL puts one first.
_GATEWAY_MODELS = ("claude-opus-5-5", "claude-opus-4-8", "claude-sonnet-4-6")
# Room for the TDD itself plus the thinking an Opus model does before writing it.
_GATEWAY_MAX_TOKENS = 32000
# Words in a gateway error that mean "this model is not available here".
_MODEL_REJECTED = re.compile(
    r"model|not found|404|does not exist|unknown|invalid|unsupported|no such", re.IGNORECASE
)
_working_model: str | None = None
# gpt-5.4 accepts a seed; gpt-5.5 refuses it as well as `temperature`.
_OPENAI_MODEL = "gpt-5.4"
# Models that can think before writing, and whose API takes
# `max_completion_tokens` rather than `max_tokens`.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# Models in that family that additionally REFUSE `seed`. Checked before the
# prefix list above, which they also match. Verified against the API.
_OPENAI_NO_TEMPERATURE_PREFIXES = ("gpt-5.5", "gpt-5.6", "o1", "o3", "o4")
# Thinking is off unless asked for: gpt-5.4 with only temperature=0 spent zero
# reasoning tokens, and the TDDs showed it — no-op changes, contradictions,
# missed edge states. Asking for it means giving up temperature=0 ("only the
# default (1) is supported" once reasoning is on); a model that works the
# problem through first beats a model that is merely repeatable.
_OPENAI_REASONING_EFFORT = "high"
_OPENAI_REASONING_MAX_TOKENS = 40000

# 0, not the API default of 1.0: the same ticket and the same code must give
# the same TDD. At 1.0 one run named three real backend files and the next
# only a UI folder, from identical input.
_TEMPERATURE = 0
# Any fixed value; it only has to stay the same across runs so the model
# starts from the same point. Output is still not byte-identical every time
# (GPU arithmetic is not perfectly reproducible), but the large swings go.
_SEED = 7


def _use_gateway() -> bool:
    return gateway_env.use_gateway()


def gateway_models() -> list[str]:
    """The models to try, in order: the one that last worked, PLANNING_LLM_MODEL,
    then the strongest available Claude down to the one used before."""
    chosen = [_working_model, os.environ.get("PLANNING_LLM_MODEL", "").strip()]
    return list(dict.fromkeys(m for m in [*chosen, *_GATEWAY_MODELS] if m))


async def _plan_via_gateway(system_prompt: str, user_prompt: str) -> str:
    global _working_model
    from agent_lib.gateway.ai import AiGatewayClient

    models = gateway_models()
    async with AiGatewayClient(
        gateway_key=gateway_env.gateway_key(),
        gateway_url=gateway_env.gateway_url(),
    ) as client:
        for model in models:
            try:
                reply = await client.chat(
                    provider=_GATEWAY_PROVIDER,
                    model_name=model,
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    max_tokens=_GATEWAY_MAX_TOKENS,
                    temperature=_TEMPERATURE,  # dropped by the SDK for a thinking model
                )
            except Exception as error:
                if model == models[-1] or not _MODEL_REJECTED.search(str(error)):
                    raise
                print(f"  [PLANNER] {model} not available ({error}); trying the next model")
                continue
            if model != _working_model:
                print(f"  [PLANNER] answered by {model}")
            _working_model = model
            return reply["content"]
    raise RuntimeError("no model answered")  # unreachable: the last model raises


async def describe_image(image, prompt: str, system_prompt: str) -> str:
    """What an image shows, in words, from the gateway's vision call (2026-11-02):
    its chat() takes text only, so a ticket's screenshot reaches the planner
    on the platform as this description. ``image`` is an images.TicketImage."""
    from agent_lib.gateway.ai import AiGatewayClient

    models = gateway_models()
    async with AiGatewayClient(
        gateway_key=gateway_env.gateway_key(),
        gateway_url=gateway_env.gateway_url(),
    ) as client:
        for model in models:
            try:
                return await client.describe_image(
                    provider=_GATEWAY_PROVIDER,
                    model_name=model,
                    prompt=prompt,
                    image_b64=base64.b64encode(image.data).decode(),
                    media_type=image.media_type,
                    system_prompt=system_prompt,
                )
            except Exception as error:
                if model == models[-1] or not _MODEL_REJECTED.search(str(error)):
                    raise
    raise RuntimeError("no model answered")  # unreachable: the last model raises


def _openai_params(model: str) -> dict:
    """Output-length and sampling settings for `model`.

    Three shapes, because the families differ: a thinking model is asked to
    think, needs `max_completion_tokens` with room for that on top of the TDD
    itself, and cannot take `temperature`; some also refuse `seed`.
    """
    thinking = {
        "max_completion_tokens": _OPENAI_REASONING_MAX_TOKENS,
        "reasoning_effort": _OPENAI_REASONING_EFFORT,
    }
    if model.startswith(_OPENAI_NO_TEMPERATURE_PREFIXES):
        return thinking
    if model.startswith(_OPENAI_REASONING_PREFIXES):
        return {**thinking, "seed": _SEED}
    return {"max_tokens": 12000, "temperature": _TEMPERATURE, "seed": _SEED}


async def _plan_via_openai(system_prompt: str, user_prompt: str, images=None) -> str:
    from src.ingestion import local_client

    # OpenAI directly, off the gateway (a laptop); the platform always uses the gateway.
    client = local_client.new_client()
    # Screenshots from the ticket (images.TicketImage) go at images.SLOT.
    user_content = user_prompt
    if images:
        head, tail = around_slot(user_prompt)
        user_content = [{"type": "text", "text": head}] + [
            {"type": "image_url", "image_url": {"url": image.data_url(), "detail": "high"}}
            for image in images
        ] + ([{"type": "text", "text": tail}] if tail else [])
    resp = await client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        **_openai_params(_OPENAI_MODEL),
    )
    return resp.choices[0].message.content


async def call_llm(system_prompt: str, user_prompt: str, images=None) -> str:
    if _use_gateway():
        if images:
            # The gateway's chat() takes text only.
            print(f"  [TDD] images: the gateway path cannot send images; {len(images)} left out")
        return await _plan_via_gateway(system_prompt, user_prompt)
    return await _plan_via_openai(system_prompt, user_prompt, images)
