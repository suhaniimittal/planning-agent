"""Prompt construction, the LLM planning call, and parsing its output into a
TechnicalDesignDoc.

Same provider-fallback split as ingestion/embedder.py: AI Gateway if
AGENTS_GATEWAY_KEY + AI_GATEWAY_URL are set, else OpenAI directly. Only the
OpenAI path can request native JSON mode (the gateway's chat() has no
response_format parameter), so the prompt itself is the primary mechanism
for valid JSON, with parse_tdd() as the defensive safety net.

That safety net now also repairs, not just detects, one common real-world
failure: the schema is large (many optional narrative fields per service/
change), and a model occasionally omits one required field (e.g. a single
service's `reasoning`) while getting everything else right. Treating that
as a hard failure means the ENTIRE run produces nothing — the same
"one gap shouldn't abort the whole thing" principle already used throughout
ingestion (one bad file/repo/chunk never aborts that run either).
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import get_origin

from pydantic import BaseModel, ValidationError

from src.ingestion import gateway_env

from .images import around_slot
from .models import FileChange, ServicePlan, TechnicalDesignDoc

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


class TddParseError(Exception):
    """Raised when the LLM's raw output can't be parsed into a TechnicalDesignDoc."""


_SCHEMA_NODE_MARKERS = ("type", "$ref", "allOf", "anyOf", "enum")


def _strip_schema_titles(node: object) -> object:
    """Pydantic's model_json_schema() adds a "title" key to every schema
    NODE purely as cosmetic metadata (the class/field name, title-cased) —
    e.g. the whole schema's own top-level "title": "TechnicalDesignDoc"
    sits directly above the TDD's actual `title` FIELD's schema (itself
    carrying its own cosmetic "title": "Title"). Dumped raw into the
    prompt, that collision is a real, observed failure mode: the model
    echoed the schema's class name back as the TDD's title verbatim.

    Only removes "title" from a dict that looks like an actual schema node
    (one of the markers above present as a sibling key) — NOT from a
    "properties" mapping, whose keys are real field names (one of which is
    legitimately "title" itself, e.g. TechnicalDesignDoc.title) and must
    never be dropped just because a key happens to match this string."""
    if isinstance(node, dict):
        stripped = {k: _strip_schema_titles(v) for k, v in node.items()}
        if "title" in stripped and any(marker in stripped for marker in _SCHEMA_NODE_MARKERS):
            del stripped["title"]
        return stripped
    if isinstance(node, list):
        return [_strip_schema_titles(v) for v in node]
    return node


# ---------------------------------------------------------------------------
# Mode-specific prompt blocks.
#
# A bug report and a feature request need different thinking, and sending both
# sets of rules on every ticket is worse than sending the right one: the
# bulk-print feature ticket ran through the bug prompt and had to strain its
# gap analysis through "Not the cause:" lines, while every genuinely new symbol
# came back as a "modify" of the nearest existing one.
#
# Everything mode-independent stays in build_system_prompt's own text, so the
# shared rules cannot drift between two copies — that is the whole reason this
# is one function with two inserts rather than two prompts.
# ---------------------------------------------------------------------------

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


def _model_for_loc(loc: tuple) -> type[BaseModel] | None:
    """The known schema is only ever nested two levels deep — services[i]
    (ServicePlan), and services[i].changes[j] (FileChange) — so this is a
    fixed lookup by loc shape, not a generic walker."""
    if len(loc) == 1:
        return TechnicalDesignDoc
    if len(loc) == 3 and loc[0] == "services":
        return ServicePlan
    if len(loc) == 5 and loc[0] == "services" and loc[2] == "changes":
        return FileChange
    return None


def _placeholder_for(model_cls: type[BaseModel], field_name: str) -> object | None:
    field = model_cls.model_fields.get(field_name)
    if field is None:
        return None
    if get_origin(field.annotation) is list:
        return []
    if field.annotation is bool:
        return False
    return "Not provided by the model — verify manually."


def _repair_missing_fields(data: dict, errors: list[dict]) -> tuple[dict, list[str]] | None:
    """Best-effort: if EVERY validation error is a MISSING required field
    (never a wrong type/value for a field that IS present — that's a real
    problem, not a gap worth papering over), fill each with a clearly-
    labeled placeholder instead of failing the whole parse. Returns the
    repaired data plus a human-readable note per repair (surfaced in the
    final TDD's open_questions, never silently hidden), or None if any
    error can't be confidently repaired this way.
    """
    if not errors or not all(e["type"] == "missing" for e in errors):
        return None

    repaired = copy.deepcopy(data)
    notes = []
    for e in errors:
        loc = e["loc"]
        model_cls = _model_for_loc(loc)
        if model_cls is None:
            return None
        placeholder = _placeholder_for(model_cls, loc[-1])
        if placeholder is None:
            return None
        target = repaired
        try:
            for key in loc[:-1]:
                target = target[key]
            target[loc[-1]] = placeholder
        except (KeyError, IndexError, TypeError):
            return None
        notes.append(
            f"Model response was missing required field "
            f"'{'.'.join(str(part) for part in loc)}' — filled with a "
            "placeholder; verify this section manually."
        )
    return repaired, notes


_TRAILING_LINE_RANGE = re.compile(r":\d+-\d+$")


def _strip_line_ranges(tdd: TechnicalDesignDoc) -> TechnicalDesignDoc:
    """Defense-in-depth, not a substitute for the prompt fix: the model is
    shown each real code symbol's location as `path:START-END` (see
    _format_code_symbol) and occasionally echoes that whole string back
    into `file_path` instead of the bare path. A downstream coding-agent
    step resolves `file_path` as a literal path on disk — a stray
    `:START-END` suffix would make every lookup for that file silently
    fail (never found in the graph, never found in a clone), so it's
    stripped here unconditionally rather than trusted from the model's own
    output, same "verify/normalize, don't just hope" principle already
    used for service_dependencies/has_real_source elsewhere in this
    pipeline."""
    for service_plan in tdd.services:
        for change in service_plan.changes:
            if change.file_path:
                change.file_path = _TRAILING_LINE_RANGE.sub("", change.file_path)
    return tdd


def parse_tdd(raw_llm_output: str) -> TechnicalDesignDoc:
    text = raw_llm_output.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise TddParseError(f"not valid JSON: {e}\nRaw: {raw_llm_output[:500]}") from e

    try:
        return _strip_line_ranges(TechnicalDesignDoc.model_validate(data))
    except ValidationError as e:
        repair = _repair_missing_fields(data, e.errors())
        if repair is None:
            raise TddParseError(f"JSON didn't match schema: {e}") from e
        repaired_data, notes = repair
        try:
            tdd = TechnicalDesignDoc.model_validate(repaired_data)
        except ValidationError as e2:
            raise TddParseError(
                f"JSON didn't match schema even after filling missing fields: {e2}"
            ) from e2
        tdd.open_questions = [*notes, *tdd.open_questions]
        return _strip_line_ranges(tdd)
