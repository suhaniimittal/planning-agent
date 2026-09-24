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
from typing import get_origin

from pydantic import BaseModel, ValidationError

from .models import FileChange, ServicePlan, TechnicalDesignDoc

_GATEWAY_PROVIDER, _GATEWAY_MODEL = "anthropic", "claude-sonnet-4-6"
_OPENAI_MODEL = "gpt-4o"


class TddParseError(Exception):
    """Raised when the LLM's raw output can't be parsed into a TechnicalDesignDoc."""


def build_system_prompt() -> str:
    schema = json.dumps(TechnicalDesignDoc.model_json_schema(), indent=2)
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
        "4. Keep scope limited to what's actually relevant to the "
        "requirement — do not include a service or file merely because it "
        "exists or could theoretically be involved.\n"
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
        "real source snippets). When a service has this section, PREFER its "
        "`file_path`/`function_or_symbol` values over anything in the 'File "
        "Reference' table of that service's documentation, since it comes "
        "from parsed code rather than prose. Services WITHOUT a 'Real code "
        "symbols' section have no parsed source available — for those, fall "
        "back to the exact `file_path` as it appears in that service's own "
        "'File Reference' table, and never invent or guess a path that isn't "
        "written there. If no file path can be identified for a given symbol "
        "in either source, set `file_path` to null rather than guessing. A "
        "single candidate set can be mixed — some services grounded in real "
        "code, others in documentation only — so do not assume uniform "
        "confidence across all services in your answer.\n\n"
        "Some services are marked as included only because they're "
        "graph-connected to a directly relevant service (a real CALLS "
        "relationship), not because their own content matched the issue — "
        "their full documentation is omitted, showing only their domain and "
        "key symbols (plus a 'Real code symbols' section if their code was "
        "still a real match). Treat these as background context for "
        "understanding dependencies, not primary candidates for changes, "
        "unless their 'Real code symbols' section itself shows something "
        "directly relevant.\n\n"
        "Separate WHAT should change (`change_description`) from WHY "
        "(`reasoning`) — do not conflate them.\n\n"
        "List anything that couldn't be determined from the documentation "
        "alone in `risks` or `open_questions` rather than stating it as fact.\n\n"
        "Leave `service_dependencies` as an empty list — it is filled in "
        "automatically afterward from the real service graph, not by you. "
        "Likewise, leave `has_real_source` on every change as false — it is "
        "also filled in afterward, from the real code graph, not by you.\n\n"
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
        "BEFORE writing `root_cause_analysis`, you must actually TRACE the "
        "real code you were shown — do not summarize it generically. For "
        "every function in a 'Real code symbols' section that's relevant "
        "to this issue, walk through what it literally does, in order: "
        "what it receives as input, each real operation it performs on "
        "that data (quoting or closely paraphrasing the actual lines, not "
        "a generic description like \"processes the request\"), and what "
        "it returns or passes onward. Only after this literal trace should "
        "you state `root_cause_analysis` — and the confirmed/likely/"
        "unknown distinction below must be grounded in that trace, not in "
        "how a similar-sounding bug \"usually\" happens in systems like "
        "this. If a symbol you need to reason about has NO real source (a "
        "documentation-only service), say so explicitly rather than "
        "describing its internals as if you'd seen them.\n\n"
        "Then describe `root_cause_analysis`, explicitly distinguishing a "
        "confirmed cause (evidenced by real code/docs) from a likely cause "
        "(plausible but unconfirmed) from an unknown cause — never present a "
        "guess as confirmed. Mirror the same pattern for the proposed "
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
        "You no longer need to spend any effort hand-drawing diagrams — that "
        "is handled separately from structured facts you provide above. "
        "Put that freed-up effort into DEPTH on `root_cause_analysis`, "
        "`proposed_solution`, and every `reasoning`/`change_description`/"
        "`implementation_notes` field: write multiple sentences that "
        "actually explain the reasoning chain (this is true because X, "
        "which means Y, therefore Z), not a single terse label. A short "
        "phrase where a real explanation is possible is a worse answer, "
        "even if it's not technically wrong.\n\n"
        "Respond with ONLY one JSON object matching this schema, no markdown "
        "fences, no prose outside the JSON:\n" + schema
    )


def _format_code_symbol(symbol: dict, snippet: str | None) -> str:
    location = f"{symbol['file_path']}:{symbol['line_start']}-{symbol['line_end']}"
    line = f"- `{location}` `{symbol['qualified_name']}` — `{symbol['signature']}`"
    if symbol.get("docstring"):
        line += f"\n  {symbol['docstring']}"
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
    1-2 hop CALLS traversal from a seed. Neighbors get only a short summary
    — a neighbor is a speculative "might be related via the call graph"
    candidate, not a confirmed-relevant one, so giving it the entire doc
    (often several thousand words) was diluting the prompt with mostly-
    irrelevant text for every candidate that happened to be graph-adjacent.
    `None` (the default) treats every candidate as a seed, preserving old
    behavior for any caller that doesn't distinguish them.

    Among seeds, whether the FULL doc is worth including further depends on
    whether real code symbols exist for that service: once real, parsed
    code is available (see `code_symbols_by_service`), the doc's detailed
    file-by-file prose is largely superseded by it — only the short,
    business-context `summary` (extracted once at ingestion time) still
    adds something the code itself can't. A seed with NO matched code ALSO
    only gets the summary, never the full doc — "seed" only means "scored
    in the top-k service search," which can be a weak, coincidental match
    rather than a confirmed one (especially when several services' summary
    embeddings score close together, which happens often on a small graph).
    Rewarding that unconfirmed match with the single most expensive kind of
    content — its entire multi-thousand-word documentation — was exactly
    backwards: a seed that turned out to have nothing relevant in its code
    is the LEAST justified case for spending the most tokens, not the most.
    If a service really is the relevant one, its summary (plus real code
    symbols, if any were found) should already make that case on its own.
    """
    code_symbols_by_service = code_symbols_by_service or {}
    snippets_by_key = snippets_by_key or {}

    sections = [f"## Issue\n{issue_text}\n"]

    if known_dependencies:
        edges = "\n".join(f"- {d['source']} -> {d['target']}" for d in known_dependencies)
        sections.append(
            "## Known service dependencies (verified from the graph — "
            "already confirmed real, not something to double-check or "
            "second-guess)\n" + edges + "\n"
        )
    for doc in candidate_docs:
        symbols = ", ".join(doc.get("key_symbols") or [])
        is_seed = seed_names is None or doc["name"] in seed_names
        code_symbols = code_symbols_by_service.get(doc["name"])

        if not is_seed:
            doc_body = (
                "(Included only because it's graph-connected to a directly "
                "matched service, not a direct semantic match itself — "
                "summary only, full documentation omitted to keep this "
                "prompt focused.)\n"
            )
        elif code_symbols:
            doc_body = (
                f"Summary: {doc.get('summary') or 'Not available.'}\n"
                "(Real code was found for this service — see 'Real code "
                "symbols' below, which reflects the actual parsed source. "
                "The service's detailed file-by-file documentation is "
                "omitted here since it's superseded by that real code; this "
                "summary is kept only for the business context code alone "
                "doesn't carry.)\n"
            )
        else:
            doc_body = (
                f"Summary: {doc.get('summary') or 'Not available.'}\n"
                "(This service matched at the summary level only — no real "
                "code symbols were found relevant to this issue, so its "
                "full documentation is omitted here to keep this prompt "
                "focused. A summary-only match is not automatically a "
                "confirmed one: treat this the same way you'd treat a "
                "graph-connected neighbor — background context, not a "
                "primary candidate for changes — unless this summary or "
                "its key symbols clearly and specifically support it.)\n"
            )

        section = (
            f"## Service: {doc['name']}\nDomain: {doc.get('domain') or 'unknown'}\n"
            f"Key symbols: {symbols}\n\n{doc_body}"
        )
        if code_symbols:
            lines = [
                _format_code_symbol(
                    symbol, snippets_by_key.get((doc["name"], symbol["qualified_name"]))
                )
                for symbol in code_symbols
            ]
            section += "\n### Real code symbols for " + doc["name"] + "\n" + "\n".join(lines) + "\n"
        sections.append(section)
    return "\n---\n".join(sections)


def _use_gateway() -> bool:
    return bool(os.environ.get("AGENTS_GATEWAY_KEY") and os.environ.get("AI_GATEWAY_URL"))


async def _plan_via_gateway(system_prompt: str, user_prompt: str) -> str:
    from agent_lib.gateway.ai import AiGatewayClient

    async with AiGatewayClient(
        gateway_key=os.environ["AGENTS_GATEWAY_KEY"],
        gateway_url=os.environ["AI_GATEWAY_URL"],
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
        )
        return reply["content"]


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
        max_tokens=12000,
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
        return TechnicalDesignDoc.model_validate(data)
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
        return tdd
