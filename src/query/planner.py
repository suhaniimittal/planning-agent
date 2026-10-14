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
import re
from typing import get_origin

from pydantic import BaseModel, ValidationError

from src.ingestion import gateway_env

from .models import FileChange, ServicePlan, TechnicalDesignDoc

_GATEWAY_PROVIDER, _GATEWAY_MODEL = "anthropic", "claude-sonnet-4-6"
# gpt-5.4, not 5.5: the newer model refuses `temperature` outright ("does not
# support 0 with this model"), leaving the API default of 1.0, and the same
# ticket then produced visibly different TDDs run to run. 5.4 is one
# generation back but accepts both temperature and a seed.
_OPENAI_MODEL = "gpt-5.4"
# Models that spend part of the output budget thinking before writing, and
# whose API takes `max_completion_tokens` rather than `max_tokens`.
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
# Models in that family that additionally REFUSE `temperature` (only the
# default is allowed). Checked before the prefix list above, which they also
# match. Verified against the API, not assumed.
_OPENAI_NO_TEMPERATURE_PREFIXES = ("gpt-5.5", "gpt-5.6", "o1", "o3", "o4")
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


def build_system_prompt() -> str:
    schema = json.dumps(_strip_schema_titles(TechnicalDesignDoc.model_json_schema()), indent=2)
    return (
        "You are a senior software architect analyzing a requirement/bug "
        "report and producing the structured data behind an "
        "implementation-ready Technical Design Document (TDD) for a "
        "microservices repository. You determine WHAT needs to change, "
        "WHERE, WHY, and HOW — using only the requirement text and the "
        "documentation/code evidence given to you below. A separate "
        "rendering step (not you) turns your structured answer into the "
        "final document; your only job is to fill in accurate, "
        "well-evidenced fields.\n\n"
        "GENERAL RULES\n"
        "1. Base every factual statement about the existing system on the "
        "provided documentation or real source-code evidence — never on "
        "assumed common architecture patterns.\n"
        "2. Do not invent services, files, classes, functions, APIs, "
        "databases, events, queues, configurations, or dependencies.\n"
        "3. If information cannot be determined, leave the field null/empty "
        "(for a list) rather than guessing, and note the gap in "
        "`open_questions` or `risks` if it could affect implementation.\n"
        "4. ONLY FIX WHAT THE TICKET REPORTS. The ticket defines the scope. "
        "Every change you propose must be needed to fix the specific problem "
        "it describes — not an improvement you noticed along the way, not a "
        "related weakness in nearby code, not a second place with the same "
        "shape of bug that the ticket never mentions. Nobody asked for those, "
        "and a reviewer will reject them. If you genuinely spot a related "
        "problem outside the report, state it in ONE line under `risks` and "
        "propose no change for it. Do not include a service or file merely "
        "because it exists or could theoretically be involved.\n"
        "5. Distinguish clearly between existing behavior, proposed "
        "behavior, and the reason for the change — never conflate WHAT "
        "changes with WHY it changes.\n"
        "6. NEVER introduce generic domain or compliance knowledge that "
        "isn't actually stated in the evidence you were given — e.g. do "
        "not mention a compliance standard (PCI DSS, GDPR, SOC 2, ...), a "
        "common design pattern, or a 'typical' way payment/shipping/order "
        "systems work, unless the provided documentation or code literally "
        "says so. A topic being generically associated with the domain "
        "(payments are 'usually' PCI-relevant, orders are 'often' split by "
        "address) is not evidence — it's a plausible-sounding guess, and "
        "this document must never present a guess as an established fact. "
        "If something like this genuinely seems relevant but isn't "
        "evidenced, say so explicitly in `open_questions` (e.g. \"Whether "
        "this system has PCI DSS obligations is not addressed by the "
        "available documentation/code\") rather than asserting it.\n\n"
        "Some services below also include a '### Real code symbols' section, "
        "parsed directly from that service's actual source code (real file "
        "paths, line numbers, signatures, and — for a shortlisted subset — "
        "real source snippets). Each symbol there is listed as "
        "`path/to/File.ext:START-END` — that START-END line range is shown "
        "ONLY so you can see roughly how big/where the symbol is; it is "
        "NEVER part of the file_path itself. When you set `file_path` in "
        "your answer, copy ONLY the plain path before the colon (e.g. "
        "`path/to/File.ext`) — never append `:START-END` or any other line "
        "range to it, since a downstream tool resolves `file_path` as a "
        "literal path on disk. Do use the `function_or_symbol`/`file_path` "
        "values (path portion only) from this section when a service has "
        "it, since it comes from parsed code rather than prose. Services "
        "WITHOUT a 'Real code symbols' section have NO parsed source "
        "available and no other source of file paths either — for those, "
        "set `file_path` to null rather than guessing one. A single "
        "candidate set can be mixed — "
        "some services grounded in real code, others with only a short "
        "summary — so do not assume uniform confidence across all services "
        "in your answer.\n\n"
        "Some services are marked as included only because they're "
        "graph-connected to a directly relevant service (a real CALLS "
        "relationship), not because their own content matched the issue — "
        "only their short summary is shown (plus a 'Real code symbols' "
        "section if their code was still a real match). Treat these as "
        "background context for understanding dependencies, not primary "
        "candidates for changes, unless their 'Real code symbols' section "
        "itself shows something directly relevant.\n\n"
        "Separate WHAT should change (`change_description`) from WHY "
        "(`reasoning`) — do not conflate them.\n\n"
        "List anything that couldn't be determined from the documentation "
        "alone in `risks` or `open_questions` rather than stating it as fact.\n\n"
        "Every service that has a 'Real code symbols' section and is NOT "
        "marked as graph-connected background context was chosen as likely "
        "involved in this issue. For EACH of those services, either propose "
        "at least one concrete change in a file listed in its 'Real code "
        "symbols' section, or still include it in `services` with an empty "
        "`changes` list and explain in its `reasoning` exactly why it needs "
        "no change. Never silently leave one out. For a service the reported "
        "problem does not touch, \"no change needed\" IS the correct answer — "
        "do not invent work for it to look thorough.\n\n"
        "Leave `service_dependencies` as an empty list — it is filled in "
        "automatically afterward from the real service graph, not by you. "
        "Likewise, leave `has_real_source` on every change as false, and "
        "`verification`/`verification_note` as null — they are also filled "
        "in afterward, from the real code graph, not by you.\n\n"
        "For each change, also provide `implementation_notes` — a concrete, "
        "step-by-step description of the approach (not just what and why, "
        "but roughly how) — `current_behavior` describing what the symbol "
        "does today (null if genuinely unknown, e.g. a documentation-only "
        "symbol with no description) — and `acceptance_criteria`: specific, "
        "testable conditions that would confirm the change is correct. Set "
        "`change_type` to \"create\" only for a symbol that doesn't exist "
        "yet, \"delete\" only if it should be removed, and \"modify\" "
        "otherwise (the default). Estimate `complexity` for each service's "
        "overall change (\"low\", \"medium\", or \"high\"), describe "
        "`responsibility_change` (what that service's responsibilities "
        "become vs. what stays the same), list any `api_changes` (new/"
        "modified/removed endpoints, or an empty list if none), and "
        "identify `affected_data_model` from the service's own documented "
        "data stores/entities, or null if not described. Note any "
        "`backward_compatibility_notes` at the top level — e.g. API or data "
        "shape changes that could break callers, or null if none apply.\n\n"
        "For each change, also provide `pseudocode_sketch` — a short block "
        "showing the shape of your proposed fix (the control flow, key "
        "steps, or logic changes). For a symbol shown to you with real "
        "source (in a 'Real code symbols' section), you may ground this "
        "sketch in that actual signature/body. For any symbol you were only "
        "given documentation for, this must stay generic, illustrative "
        "pseudocode — you have not seen its actual source, so do not write "
        "language-specific syntax that could be mistaken for verified code, "
        "or claim exact current implementation details you were not shown.\n\n"
        "REQUIREMENTS: split `functional_requirements` (what the system must "
        "do after the change) from `non_functional_requirements` (performance, "
        "reliability, security, scalability, observability) — leave either "
        "empty if the issue/docs don't actually support one.\n\n"
        "CURRENT vs. PROPOSED SYSTEM — STATE FACTS, DO NOT DRAW PICTURES. "
        "You do NOT need to hand-draw any diagram. Instead, state the "
        "underlying facts as structured data, and a separate rendering "
        "step builds the actual diagram from them — this is deliberate: "
        "hand-drawn ASCII art is unreliable (inconsistent alignment, easy "
        "to skip) and spending effort on it takes attention away from the "
        "reasoning that actually matters. Filling these structured fields "
        "is NOT optional polish — whenever you can name at least one "
        "component and one relationship between components (true for "
        "almost every candidate set you'll see), fill them in; only leave "
        "them empty when there is truly a single, isolated component with "
        "nothing to relate it to.\n\n"
        "`current_architecture_components`: one entry per component you can "
        "name (a service, a gateway, a database, an external system), each "
        "with its `name` and a `depends_on` list of the OTHER component "
        "names it calls/uses. If a 'Known service dependencies' section "
        "was given to you above, that is CONFIRMED, already-verified real "
        "data — you MUST reflect every one of those pairs in `depends_on` "
        "for the relevant components; do not leave `depends_on` empty for a "
        "service that appears in that list. Beyond those confirmed edges, "
        "you may add further relationships ONLY when the documentation/code "
        "actually states them — never invented, and never omitted just "
        "because stating a relationship feels like an extra assertion — a "
        "component with real, known dependencies that gets listed with an "
        "empty `depends_on` is a wrong answer, not a cautious one. "
        "`current_flow_steps`: the ordered "
        "steps of what currently happens (e.g. \"Validate input\", "
        "\"Process request\", \"Persist data\"), each as a `step` label with "
        "an optional `detail` — use `detail` whenever you have something "
        "concrete and evidenced to add about that step, not just the label "
        "alone. Also describe `current_architecture` as prose — this is the "
        "narrative WHY behind the structure (not a restatement of the "
        "component list, which is just the WHAT).\n\n"
        "BEFORE writing `root_cause_analysis`, actually TRACE the real "
        "code you were shown, for yourself: what each relevant function "
        "receives, what it really does with that data, and what it returns "
        "— grounded in its actual lines, not in how a similar-sounding bug "
        "\"usually\" happens. Do this tracing in your head; WRITE only the "
        "conclusions it supports (see the format below), never the "
        "walk-through itself. If a symbol you need has NO real source, say "
        "so rather than describing its internals as if you'd seen them.\n\n"
        "Then write `root_cause_analysis` as a few short lines, each one "
        "starting with its certainty and naming its evidence (file and "
        "function), e.g.:\n"
        "Confirmed: <cause> — <file>: <what that code does>.\n"
        "Likely: <cause> — <why it's only likely>.\n"
        "Not the cause: <ruled-out part> — <why>.\n"
        "Never present a guess as confirmed. Mirror the same pattern for the proposed "
        "system: `proposed_solution` (what changes, what stays, which "
        "components participate, how it resolves the issue), "
        "`proposed_architecture` (prose) + `proposed_architecture_"
        "components`, `proposed_flow_steps`, and — whenever more than one "
        "component is involved anywhere in your answer, which is the "
        "common case — `sequence_steps`: one entry per call or response, "
        "each with `from_participant`, `to_participant`, `action` (what's "
        "being requested/returned), and `direction` (\"call\" or "
        "\"response\"). Only participants supported by the available "
        "information belong here — never invent a component just to fill "
        "it out.\n\n"
        "`proposed_architecture_components` MUST reflect what your own "
        "`proposed_solution` actually changes structurally. If the "
        "proposed solution introduces a new call, a new dependency, or a "
        "new component (e.g. writing to a database, calling another "
        "service, publishing to a queue), that new relationship MUST "
        "appear in `depends_on` here even if it wasn't present in "
        "`current_architecture_components` — copying the current list "
        "over unchanged is only correct when the fix genuinely introduces "
        "no new structural dependency at all. If your own "
        "`proposed_solution` text describes a new call or dependency that "
        "isn't reflected here, that is an inconsistency within your own "
        "answer — fix it before responding, don't leave the two "
        "disagreeing with each other.\n\n"
        "IMPACT ANALYSIS MUST NAME SPECIFIC, EVIDENCED THINGS, NOT GENERIC "
        "CONCERNS. When describing what the proposed change affects "
        "(`proposed_solution`, `responsibility_change`, `risks`, "
        "`backward_compatibility_notes`), trace consequences to the "
        "SPECIFIC other services/functions actually visible to you — the "
        "ones in `service_dependencies`, in another candidate service's "
        "'Real code symbols' section, or explicitly named in the "
        "documentation — never to a generically-named concern the domain "
        "merely reminds you of. \"This changes how `api_payment_auth` is "
        "called, which `OrdersController.createOrder` depends on\" is a "
        "real impact statement; \"this could impact payment processing "
        "and compliance\" with no specific symbol/service named is not — "
        "it's filler that sounds thorough without saying anything checkable. "
        "If you genuinely cannot trace a consequence to something specific "
        "you were shown, that itself belongs in `open_questions`, not as a "
        "vague statement dressed up as analysis.\n\n"
        "`communication_notes`: describe HOW services in `service_"
        "dependencies` actually communicate (REST, events, a queue, a shared "
        "store) and why — only for mechanisms the documentation/code "
        "actually shows, never inferred. `error_handling_notes`: relevant "
        "validation failures, missing/invalid input, service/database "
        "failures, timeouts, retries, duplicate requests, partial failures — "
        "separating documented behavior from what you're proposing. "
        "`security_considerations`: authentication, authorization, input "
        "validation, sensitive data, access control — only what's actually "
        "relevant and supported by the material given, never invented "
        "security mechanisms.\n\n"
        "`testing_plan`: a list of concrete test scenarios, each with a "
        "`category` (\"unit\", \"integration\", \"api\", \"regression\", or "
        "\"edge_case\"), a `scenario`, and its `expected_result` — alongside "
        "`testing_notes` for any overall testing narrative that doesn't fit "
        "a single scenario. `implementation_plan`: an ORDERED list of "
        "concrete steps across the whole change (e.g. \"Modify X\", \"Add "
        "validation to Y\", \"Add unit tests\", \"Run regression tests\") — "
        "only steps actually supported by your analysis above.\n\n"
        "Be as thorough as the documentation actually supports — but if "
        "something isn't described in the docs, say so in `open_questions` "
        "rather than inventing plausible-sounding detail. Thoroughness means "
        "using everything the documentation gives you, not adding facts it "
        "doesn't contain.\n\n"
        "WRITE FOR A BUSY DEVELOPER: short, specific, no repetition. "
        "The document is read top to bottom by someone who will make the "
        "change, so every sentence must add a fact they need.\n"
        "- `issue_summary`: 2-4 sentences restating the reported problem — "
        "what the user sees, where, and the trigger. No causes or fixes.\n"
        "- `proposed_solution`: 2-5 sentences — what changes, in which "
        "services, and how that resolves the issue.\n"
        "- Each change's `change_description`, `implementation_notes` and "
        "`reasoning`: 1-3 sentences each, naming the concrete thing (field, "
        "condition, call) — never restating the issue.\n"
        "- `function_or_symbol` is exactly ONE symbol name as listed in "
        "'Real code symbols' (e.g. `isPhoneValid` or `Form.validate`), never "
        "two names joined with '/' or 'and'. If two functions change, that "
        "is two separate changes.\n"
        "- `pseudocode_sketch`: only the lines that change, at most ~10.\n"
        "- Leave `sequence_steps` empty and `current_architecture` and "
        "`overall_reasoning` to one sentence each — they are not shown.\n"
        "- Don't repeat the same fact in several fields; say it once in the "
        "field it belongs to.\n\n"
        "Respond with ONLY one JSON object matching this schema, no markdown "
        "fences, no prose outside the JSON:\n" + schema
    )


_MAX_FILE_CONTEXT_NAMES = 20


def _format_code_symbol(
    symbol: dict, snippet: str | None, file_neighbors: list[str] | None = None
) -> str:
    location = f"{symbol['file_path']}:{symbol['line_start']}-{symbol['line_end']}"
    line = f"- `{location}` `{symbol['qualified_name']}` — `{symbol['signature']}`"
    if symbol.get("docstring"):
        line += f"\n  {symbol['docstring']}"
    if symbol.get("matched_identifiers"):
        named = ", ".join(f"`{w}`" for w in symbol["matched_identifiers"])
        line += f"\n  Contains {named}, named in the issue itself."
    others = [n for n in file_neighbors or [] if n != symbol["name"]]
    if others:
        shown = ", ".join(others[:_MAX_FILE_CONTEXT_NAMES])
        hidden = len(others) - _MAX_FILE_CONTEXT_NAMES
        more = f" (+{hidden} more)" if hidden > 0 else ""
        line += f"\n  Also defined in this file: {shown}{more}"
    if snippet:
        line += f"\n  ```{symbol.get('language') or ''}\n{snippet}\n  ```"
    return line


def build_user_prompt(
    issue_text: str,
    candidate_docs: list[dict],
    code_symbols_by_service: dict[str, list[dict]] | None = None,
    snippets_by_key: dict[tuple[str, str], str] | None = None,
    seed_names: set[str] | None = None,
    known_dependencies: list[dict] | None = None,
    file_context: dict[tuple[str, str], list[str]] | None = None,
) -> str:
    """`known_dependencies`: real CALLS edges among the candidate services,
    read from the graph BEFORE the LLM call — shown here as confirmed
    ground truth (distinct from `service_dependencies` in the output
    schema, which the model must leave empty and which gets filled in
    afterward the same way). Without this, the model has no way to
    *confirm* a cross-service relationship at generation time, and
    appropriately declines to state one — e.g. leaving architecture
    diagram `depends_on` lists empty rather than guessing from prose alone.
    Giving it the real edges as input directly fixes that.

    `seed_names`: services that were DIRECT semantic matches to the issue
    (vector_search_services hits), as opposed to ones only pulled in via a
    1-2 hop CALLS traversal from a seed. `None` (the default) treats every
    candidate as a seed, preserving old behavior for any caller that doesn't
    distinguish them.

    Every candidate service only ever gets its short, code-derived `summary`
    (see summarizer.py) — there is no longer a separate "full doc" tier to
    choose between; the real per-symbol detail comes entirely from
    `code_symbols_by_service`/`snippets_by_key` when a service's code graph
    matched. A seed with no matched code, or a graph-neighbor pulled in via
    CALLS, both get only the summary — "seed" just means "scored in the
    top-k service search," which can be a weak, coincidental match rather
    than a confirmed one, so it earns no special extra content beyond that
    summary and whatever real code symbols were separately found relevant.
    """
    code_symbols_by_service = code_symbols_by_service or {}
    snippets_by_key = snippets_by_key or {}
    file_context = file_context or {}

    sections = [f"## Issue\n{issue_text}\n"]

    if known_dependencies:
        edges = "\n".join(f"- {d['source']} -> {d['target']}" for d in known_dependencies)
        sections.append(
            "## Known service dependencies (verified from the graph — "
            "already confirmed real, not something to double-check or "
            "second-guess)\n" + edges + "\n"
        )

    for doc in candidate_docs:
        is_seed = seed_names is None or doc["name"] in seed_names
        code_symbols = code_symbols_by_service.get(doc["name"])
        summary_line = f"Summary: {doc.get('summary') or 'Not available.'}\n"

        if not is_seed:
            doc_body = summary_line + (
                "(Included only because it's graph-connected to a directly "
                "matched service, not a direct semantic match itself — "
                "treat as background context for dependencies, not a "
                "primary candidate for changes, unless its 'Real code "
                "symbols' section below shows something directly relevant.)\n"
            )
        elif code_symbols:
            doc_body = summary_line + (
                "(Real code was found for this service — see 'Real code "
                "symbols' below, which reflects the actual parsed source; "
                "this summary is kept only for business context the code "
                "itself doesn't carry.)\n"
            )
        else:
            doc_body = summary_line + (
                "(This service matched at the summary level only — no real "
                "code symbols were found relevant to this issue. A "
                "summary-only match is not automatically a confirmed one: "
                "treat this the same way you'd treat a graph-connected "
                "neighbor — background context, not a primary candidate for "
                "changes — unless this summary clearly and specifically "
                "supports it.)\n"
            )

        section = f"## Service: {doc['name']}\n\n{doc_body}"
        if code_symbols:
            lines = [
                _format_code_symbol(
                    symbol,
                    snippets_by_key.get((doc["name"], symbol["qualified_name"])),
                    file_context.get((doc["name"], symbol["file_path"])),
                )
                for symbol in code_symbols
            ]
            section += "\n### Real code symbols for " + doc["name"] + "\n" + "\n".join(lines) + "\n"
        sections.append(section)
    return "\n---\n".join(sections)


def _use_gateway() -> bool:
    return gateway_env.use_gateway()


async def _plan_via_gateway(system_prompt: str, user_prompt: str) -> str:
    from agent_lib.gateway.ai import AiGatewayClient

    async with AiGatewayClient(
        gateway_key=gateway_env.gateway_key(),
        gateway_url=gateway_env.gateway_url(),
    ) as client:
        reply = await client.chat(
            provider=_GATEWAY_PROVIDER,
            model_name=_GATEWAY_MODEL,
            prompt=user_prompt,
            system_prompt=system_prompt,
            # The schema grew substantially (architecture/flow/sequence
            # diagrams, requirements, testing plan, etc.), and diagrams are
            # now explicitly required rather than optional — 8000 was
            # already a big jump from the original 4000, but multiple
            # multi-line ASCII diagrams plus everything else can still eat
            # into it fast.
            max_tokens=12000,
            temperature=_TEMPERATURE,
        )
        return reply["content"]


def _openai_params(model: str) -> dict:
    """Output-length and sampling settings for `model`.

    Three shapes, because the families differ: a thinking model needs
    `max_completion_tokens` and room to think on top of the TDD itself, and
    some of those also refuse `temperature`/`seed` entirely.
    """
    if model.startswith(_OPENAI_NO_TEMPERATURE_PREFIXES):
        return {"max_completion_tokens": _OPENAI_REASONING_MAX_TOKENS}
    if model.startswith(_OPENAI_REASONING_PREFIXES):
        return {
            "max_completion_tokens": _OPENAI_REASONING_MAX_TOKENS,
            "temperature": _TEMPERATURE,
            "seed": _SEED,
        }
    return {"max_tokens": 12000, "temperature": _TEMPERATURE, "seed": _SEED}


async def _plan_via_openai(system_prompt: str, user_prompt: str) -> str:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
    resp = await client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        **_openai_params(_OPENAI_MODEL),
    )
    return resp.choices[0].message.content


async def call_llm(system_prompt: str, user_prompt: str) -> str:
    if _use_gateway():
        return await _plan_via_gateway(system_prompt, user_prompt)
    return await _plan_via_openai(system_prompt, user_prompt)


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
