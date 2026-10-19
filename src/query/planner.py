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

_BUG_SCOPE = (
    "4. ONLY FIX WHAT THE TICKET REPORTS — BUT FIX IT END TO END, AND "
    "EXAMINE THE WHOLE PATH IT DESCRIBES. Two different scopes, and "
    "conflating them is the most common way this document goes wrong.\n"
    "   ANALYSIS SCOPE: the description AND the reproduction steps "
    "together. The steps trace a path through the system — where data is "
    "submitted, where it travels, where it is checked, where the failure "
    "shows. EVERY service on that path must be EXAMINED and must APPEAR "
    "in `services` with an explicit verdict. A reproduction step naming a "
    "submission names the service that accepts it; that service gets "
    "examined and gets an entry.\n"
    "   CHANGE SCOPE: every change required to fix the reported problem "
    "END TO END, and nothing beyond that. Trace the issue across the "
    "relevant frontend, backend, persistence and downstream-consumer "
    "code, and include every change needed to resolve it — even when "
    "that means more than one service or layer. Do not assume fixing "
    "the first visible symptom is sufficient: check whether the "
    "underlying cause can recur, whether data is already persisted in "
    "an invalid or unnormalised state, and whether other entry points "
    "or consumers are affected. THE SMALLEST SET OF CHANGES MEANS THE "
    "SMALLEST SET THAT CORRECTLY ADDRESSES THE ISSUE END TO END — not "
    "the fewest files, and not a frontend-only or backend-only fix by "
    "default. A service whose stored value is what the consumer later "
    "rejects needs its own change, even when fixing the consumer alone "
    "would make the reported symptom disappear; say which failure each "
    "change prevents. A service enters the change scope only when "
    "inspected code shows it must change, or a requirement stated in the "
    "ticket requires it — and its `reasoning` must say which. Sitting on "
    "the data path is NOT by itself a reason to change a service, and an "
    "unrelated change does not become in scope merely because its file "
    "or service is part of the same data flow. Everything else is out: "
    "not an improvement you noticed along the way, not a related weakness "
    "in nearby code, not a second place with the same shape of bug the "
    "ticket never mentions. Nobody asked for those, and a reviewer will "
    "reject them. If you genuinely spot a related problem outside the "
    "report, state it in ONE line under `risks` and propose no change "
    "for it.\n"
    "5. WHERE A BAD VALUE IS STORED IS A ROOT CAUSE LOCATION, NOT A "
    "BYSTANDER. When the ticket describes a value that is supplied in one "
    "place and rejected in another, find where that exact value is first "
    "accepted and written down — the field assignment, the mapper, the "
    "builder, the persistence call. A service that saves a value a "
    "downstream consumer then rejects is part of the cause, and leaving "
    "it out is not restraint: it keeps the source producing bad data "
    "while the symptom is patched downstream. Propose the normalisation "
    "there as well, not only at the consumer. This is narrow — it covers "
    "the one value the ticket names, never general hardening of an "
    "upstream service. When an entry-side fix and a consumer-side fix "
    "would each resolve the reported failure, state both in "
    "`proposed_solution` and propose both as changes unless the evidence "
    "shows one of them is unnecessary, saying what failure each one "
    "prevents. Never present one as settled while the evidence supports "
    "both.\n"
)

_FEATURE_SCOPE = (
    "4. BUILD WHAT THE TICKET SPECIFIES — ALL OF IT, AND NOTHING BESIDE "
    "IT. Two different scopes, and conflating them is the most common way "
    "this document goes wrong.\n"
    "   ANALYSIS SCOPE: the requirements and acceptance criteria define "
    "what must exist. EVERY service the requirements reach must be "
    "EXAMINED and must APPEAR in `services` with an explicit verdict, "
    "including services that already provide part of what is asked for "
    "and therefore need no change.\n"
    "   CHANGE SCOPE: every change required to deliver the stated "
    "requirements, and nothing beyond them. Each acceptance criterion in "
    "the ticket maps to at least one proposed change, or to one line "
    "saying why it is out of scope — a criterion that appears nowhere in "
    "your answer is a gap in the answer, not in the ticket. THE SMALLEST "
    "SET OF CHANGES MEANS THE SMALLEST SET THAT DELIVERS THE STATED "
    "REQUIREMENTS — not the fewest files, and not a frontend-only or "
    "backend-only design by default. A service enters the change scope "
    "only when inspected code shows it must change, or a stated "
    "requirement puts it there — and its `reasoning` must say which. "
    "Everything else is out: no speculative abstraction, no capability "
    "for a future requirement nobody asked for, no unrelated refactoring, "
    "and nothing included merely because its file or service sits near "
    "the feature. If you spot a related problem outside the requirements, "
    "state it in ONE line under `risks` and propose no change for it.\n"
    "5. BUILD ON WHAT IS ALREADY THERE. Before proposing anything new, "
    "look through the evidence for a mechanism that already does this "
    "kind of work — an existing validation approach, notification path, "
    "merge or export routine, repository or client. Prefer extending it "
    "over writing a competing implementation, and say in `reasoning` "
    "which existing thing you are extending. A second mechanism standing "
    "beside one that already works is wrong even when it functions. Do "
    "not claim an API, field, schema or dependency already exists unless "
    "the evidence confirms it.\n"
)

_BUG_ANALYSIS = (
    "BEFORE writing `root_cause_analysis`, actually TRACE the real "
    "code you were shown, for yourself: what each relevant function "
    "receives, what it really does with that data, and what it returns "
    "— grounded in its actual lines, not in how a similar-sounding bug "
    '"usually" happens. Do this tracing in your head; WRITE only the '
    "conclusions it supports (see the format below), never the "
    "walk-through itself. If a symbol you need has NO real source, say "
    "so rather than describing its internals as if you'd seen them.\n\n"
    "TRACE THE REPORTED VALUE THROUGH ITS WHOLE LIFE, AND WRITE THAT "
    "TRACE DOWN. When the ticket names a specific value (a field, a "
    "parameter, an input the reporter typed), `root_cause_analysis` "
    "must carry one line for each stage of that value's life, in "
    "order. These lines are conclusions, not a walk-through:\n"
    "Accepted at: <service>/<file>:<symbol> — what is done to it on the "
    "way in.\n"
    "Validated at: <service>/<file>:<symbol> — what is checked, and "
    "what is NOT.\n"
    "Transformed at: <service>/<file>:<symbol> — any mapper, builder or "
    "converter it passes through, and whether it is normalised there.\n"
    "Stored at: <service>/<file>:<symbol> — the exact assignment, "
    "mapper or builder call that writes it, and whether it is "
    "normalised first.\n"
    "Consumed at: <service>/<file>:<symbol> — what reads it and what "
    "that reader expects.\n"
    'Write "not shown in the evidence" for any stage you cannot '
    "locate — never leave a stage out. A validation that does NOT "
    "reject the value is not exoneration: it is the reason the value "
    "survives unchanged to the next stage, so say that and keep tracing "
    "to where it is written. A service whose only involvement is "
    "storing the value as-is still gets its `Stored at:` line, whether "
    "or not you go on to propose a change there.\n\n"
    "Then write `root_cause_analysis` as a few short lines, each one "
    "starting with its certainty and naming its evidence (service, file "
    "and symbol), e.g.:\n"
    "Confirmed: <cause> — <service>/<file>:<symbol>: <what that code does>.\n"
    "Hypothesis: <cause> — <what you observed that suggests it> — <what "
    "would confirm it>.\n"
    "Unknown: <what you could not determine> — <the one file, symbol or "
    "log that would settle it>.\n"
    "Not the cause: <ruled-out part> — <why>.\n"
    "A Confirmed line requires VERIFIED evidence — source you were shown "
    "and traced. A REPORTED symptom plus an INFERRED connection is a "
    "Hypothesis however convincing it feels, and the ticket's own "
    "description of a symptom never establishes a cause on its own. "
    "Never present a guess as confirmed, and never drop the Unknown "
    "lines to look more certain than you are.\n\n"
    "FIX THE CAUSE, NOT ONLY THE SYMPTOM. Decide where the defect "
    "originates — input acceptance, validation, transformation, "
    "persistence, business logic, integration or consumption — and then "
    "answer one question explicitly: can the same invalid state be "
    "produced again after your proposed fix? If it can, the fix is "
    "incomplete. Include an UPSTREAM correction when the evidence shows "
    "an upstream component creates or stores the invalid value "
    "responsible for the defect. Include a DOWNSTREAM correction when "
    "its behaviour independently contributes to the reported failure. Do "
    "NOT modify every service on the data path merely because it "
    "participates in the flow.\n\n"
    "PERSISTED DATA. When the defect involves an incorrect or "
    "inconsistent stored value, say what already-stored data means for "
    "the fix: whether existing records still fail after the change, and "
    "which proposed change handles them. Do not recommend a data repair "
    "or migration unless the evidence shows existing data requires it "
    "and you can specify the operation safely. A local patch that hides "
    "the symptom while leaving the verified cause intact is "
    "insufficient; an extensive refactor that fixes the defect but "
    "introduces unrelated work is also insufficient.\n\n"
    "Mirror the same evidence discipline for the proposed system: "
    "`proposed_solution` (what changes, what stays, which components "
    "participate, how it resolves the issue), `proposed_architecture` "
    "(prose — the narrative WHY behind the structure, not a restatement "
    "of the component list) + `proposed_architecture_components`, and "
    "`proposed_flow_steps`.\n\n"
)

_FEATURE_ANALYSIS = (
    "BEFORE writing anything, actually READ the real code you were "
    "shown, for yourself: what each relevant symbol already does, what "
    "it receives and what it returns — grounded in its actual lines, "
    'not in how a feature like this is "usually" built. Do this in '
    "your head; WRITE only the conclusions it supports. If a symbol you "
    "need has NO real source, say so rather than describing its "
    "internals as if you'd seen them.\n\n"
    "REQUIREMENTS BEFORE DESIGN. Separate three things and never blur "
    "them: what the ticket explicitly states, what you are assuming, and "
    "what it leaves unresolved. State assumptions in `risks` and "
    "unresolved requirements in `open_questions`. Never invent a "
    "business rule, and never treat your own design decision as "
    "something the ticket asked for.\n\n"
    "`root_cause_analysis` IS A GAP ANALYSIS HERE, NOT A DEFECT HUNT. "
    "There is no failure to diagnose: the question is what already "
    "exists, what is missing, and what the missing part should be built "
    "on. Write short lines, each naming its evidence (service, file and "
    "symbol):\n"
    "Exists: <capability> — <service>/<file>:<symbol>: <what it already "
    "does>.\n"
    "Missing: <capability the ticket requires> — <why no shown code "
    "provides it>.\n"
    "Extend: <service>/<file>:<symbol> — <why this is the right thing to "
    "build on rather than writing a parallel implementation>.\n"
    "Unknown: <what you could not determine> — <the one file, symbol or "
    "schema that would settle it>.\n"
    "Out of scope: <exclusion the ticket names> — <why>.\n"
    "An `Exists:` line is how you show you are reusing rather than "
    "reinventing, so write one for every capability the ticket says the "
    "platform already has. A `Missing:` line must name a requirement, "
    "not a gap in your own reading: if the evidence simply does not "
    "cover something, that is an `Unknown:` line.\n\n"
    "EVALUATE ONLY THE LAYERS THE REQUIREMENTS REACH: entry points and "
    "request validation; business logic; API contracts and inter-service "
    "calls; persistence and data models; events and downstream "
    "consumers; error handling; authorization and sensitive data; "
    "backward compatibility. Include a layer only when the requirements "
    "or the evidence actually put it in play — a layer listed for "
    "completeness, with nothing specific to say about it, is filler.\n\n"
    "Mirror the same evidence discipline for the proposed system: "
    "`proposed_solution` (what gets built, what stays, which components "
    "participate, how it satisfies the requirements), "
    "`proposed_architecture` (prose — the narrative WHY behind the "
    "structure, not a restatement of the component list) + "
    "`proposed_architecture_components`, and `proposed_flow_steps`.\n\n"
)


def build_system_prompt(mode: str = "bug") -> str:
    """The planning prompt, with the one block that differs by ticket type.

    `mode` is "bug" or "feature"; anything else falls back to "bug",
    because the bug prompt degrades far better on a feature ticket than
    the reverse — a feature run through it merely strains its gap
    analysis through `Not the cause:` lines, whereas a bug run through
    the feature prompt loses the value trace that finds where a bad
    value is written.
    """
    mode = mode if mode in ("bug", "feature") else "bug"
    scope = _BUG_SCOPE if mode == "bug" else _FEATURE_SCOPE
    analysis = _BUG_ANALYSIS if mode == "bug" else _FEATURE_ANALYSIS
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
        "1. EVIDENCE FIRST. Base every factual statement about the existing "
        "system on the provided documentation or real source-code evidence — "
        "never on assumed common architecture patterns. Name the service, "
        "file and symbol each statement comes from.\n"
        "2. Do not invent services, files, classes, functions, APIs, "
        "databases, events, queues, configurations, or dependencies.\n"
        "3. If information cannot be determined, leave the field null/empty "
        "(for a list) rather than guessing, and note the gap in "
        "`open_questions` or `risks` if it could affect implementation. Each "
        "open question must be answerable by looking at one named thing.\n"
        + scope
        + "6. Distinguish clearly between existing behavior, proposed "
        "behavior, and the reason for the change — never conflate WHAT "
        "changes with WHY it changes.\n"
        "7. NEVER introduce generic domain or compliance knowledge that "
        "isn't actually stated in the evidence you were given — e.g. do "
        "not mention a compliance standard (PCI DSS, GDPR, SOC 2, ...), a "
        "common design pattern, or a 'typical' way payment/shipping/order "
        "systems work, unless the provided documentation or code literally "
        "says so. A topic being generically associated with the domain "
        "(payments are 'usually' PCI-relevant, orders are 'often' split by "
        "address) is not evidence — it's a plausible-sounding guess, and "
        "this document must never present a guess as an established fact. "
        "If something like this genuinely seems relevant but isn't "
        'evidenced, say so explicitly in `open_questions` (e.g. "Whether '
        "this system has PCI DSS obligations is not addressed by the "
        'available documentation/code") rather than asserting it.\n\n'
        "FOUR KINDS OF KNOWING, KEPT APART EVERYWHERE YOU WRITE:\n"
        "  VERIFIED — behavior you can read in source you were actually "
        "shown.\n"
        "  REPORTED — a symptom the ticket describes. Evidence that "
        "something is wrong; never evidence of where or why.\n"
        "  INFERRED — a conclusion connecting the two. Reasonable, not "
        "proven.\n"
        "  UNKNOWN — something the evidence does not settle.\n"
        "A file or symbol merely EXISTING is not evidence that it needs to "
        "change, and never makes a proposed change verified. Evidence that a "
        "symbol is relevant is evidence about what its code DOES.\n\n"
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
        "A service's own code is the only evidence about that service. Code "
        "belonging to a different service — even one with a similar name, or "
        "a front end for the same feature — says nothing about what this one "
        "does. Never clear a service of needing a change on the strength of "
        "another service's source.\n\n"
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
        "`changes` list and explain in its `reasoning` exactly what you "
        "examined and why it needs no change. Never silently leave one out. "
        "For a service the reported "
        'problem does not touch, "no change needed" IS the correct answer — '
        "do not invent work for it to look thorough.\n\n"
        "Leave `service_dependencies` as an empty list — it is filled in "
        "automatically afterward from the real service graph, not by you. "
        "Likewise, leave `has_real_source` on every change as false, and "
        "`verification`/`verification_note` as null — they are also filled "
        "in afterward, from the real code graph, not by you.\n\n"
        "THE CHAIN MUST BE VISIBLE. A reader follows every change as: "
        "reported requirement or failure -> what the code does today -> what "
        "it will do after -> the specific edit that causes that -> the test "
        "that proves it. If you cannot name the line of the ticket a change "
        "answers to, you have no basis for that change; drop it or move it "
        "to `open_questions`.\n\n"
        "For each change, also provide `implementation_notes` — a concrete, "
        "step-by-step description of the approach (not just what and why, "
        "but roughly how), naming the EXISTING function, component, "
        "validation pattern or notification mechanism you are reusing. This "
        "codebase's own conventions come first: a fix that adds a second "
        "mechanism beside one that already exists is wrong even when it "
        "works. If the right location or API for the change cannot be "
        "determined from the evidence, say exactly what must be inspected "
        "before coding rather than guessing. Also provide "
        "`current_behavior` describing what the named symbol does today, "
        "grounded in the lines you were shown and not in the feature "
        "generally (null if genuinely unknown, e.g. a documentation-only "
        "symbol with no description) — and `acceptance_criteria`: specific, "
        "testable conditions that would confirm the change is correct, each "
        "matching a scenario in `testing_plan`. Set "
        "`change_type` by what happens to the FILE, not to the symbol: "
        '"create" ONLY when the file itself does not exist yet, "modify" '
        "when the file already exists — including when the code you are "
        'adding to it is brand new — "delete" only if it should be '
        'removed, and "reference" for a symbol your design only reads, '
        "calls, or depends on without editing it. New code going into an "
        'existing file is a "modify": whoever implements a "create" does '
        "not read the file first, so labelling it that way replaces the "
        "file's existing contents with only your new code. Estimate "
        "`complexity` for each service's "
        'overall change ("low", "medium", or "high"), describe '
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
        "or claim exact current implementation details you were not shown. "
        "Every identifier a sketch references — a message key, a constant, a "
        "helper, a component — must be one you actually saw in the evidence. "
        "If the fix needs something that does not exist yet, write it in the "
        "sketch AND say in `implementation_notes` that it must be created. "
        "Presenting a hypothetical helper as an existing repository API is a "
        "failure, not a shortcut.\n\n"
        "WHEN A CHANGE ADDS SOMETHING THAT DOES NOT EXIST YET, NAME THE NEW "
        "THING. Put the name of the symbol to be CREATED in "
        "`function_or_symbol`, and name the file it belongs in as "
        "`file_path`. The nearest existing symbol goes in `reasoning` as "
        "the anchor you are building beside — never in `function_or_symbol`, "
        "which misdirects whoever implements it onto a symbol you did not "
        "intend to touch. Say in `change_description` that the symbol is "
        "new. The `change_type` still follows the file, so a new symbol "
        'going into a file that already exists is a "modify" — being new is '
        "a fact about the symbol, not about the file.\n\n"
        "A CHANGE THAT CHANGES NOTHING IS NOT A CHANGE. Never describe a "
        'symbol you are leaving alone as a "modify". Every "create", '
        '"modify" and "delete" entry becomes a line in the reader\'s list '
        "of files to edit, so one of them saying 'no change is proposed "
        "here' sends them to a file they must not touch. You have two "
        "correct places to put it instead. If the symbol matters to the "
        "design — your new code calls it, reads it, or depends on its "
        'current behaviour — give it `change_type` "reference", describe '
        "in `change_description` what the new code relies on it for, and "
        "state in `implementation_notes` that this file is NOT edited. If "
        "the symbol merely turned out to be irrelevant, leave it out of "
        "`changes` entirely and say so in that service's `reasoning`.\n\n"
        'A "reference" entry still has to be real and still has to earn '
        "its place. Name a symbol you were actually shown, and include it "
        "only when a reader implementing the change would otherwise go "
        'looking for it. Do not pad `changes` with "reference" entries '
        "for every symbol you happened to read.\n\n"
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
        "is NOT optional polish — an incomplete component list renders as a "
        "visibly broken, half-finished diagram, which is worse than no "
        "diagram at all.\n\n"
        "`proposed_architecture_components`: EVERY service you listed in "
        "`services` MUST appear here as a component — including ones you "
        "gave an empty `changes` list, since the reader needs to see where "
        "they sit. Appearing here is NOT a claim that a service changes; "
        "components show structure, `services` shows changes. Add "
        "non-service components (a gateway, a database, an external system) "
        "only when the evidence names them. Each component carries its "
        "`name` and a `depends_on` list of the OTHER component "
        "names it calls/uses. If a 'Known service dependencies' section "
        "was given to you above, that is CONFIRMED, already-verified real "
        "data — you MUST reflect every one of those pairs in `depends_on` "
        "for the relevant components; do not leave `depends_on` empty for a "
        "service that appears in that list. Beyond those confirmed edges, "
        "you may add further relationships ONLY when the documentation/code "
        "actually states them — never invented, and never omitted just "
        "because stating a relationship feels like an extra assertion — a "
        "component with real, known dependencies that gets listed with an "
        "empty `depends_on` is a wrong answer, not a cautious one. A "
        "component that genuinely calls nothing gets an empty list, but it "
        "must still be present.\n\n"
        "`current_flow_steps` and `proposed_flow_steps` are rendered side by "
        "side, so they must be comparable: the same journey at the same "
        "granularity in both, traversing the services named in the "
        "reproduction steps in order. If one has a step for submission, so "
        "does the other. Each step is a `step` label with an optional "
        "`detail` — use `detail` whenever you have something "
        "concrete and evidenced to add about that step, naming the service "
        "and symbol that performs it, not just the label alone. If the two "
        "lists describe different journeys, or one stops short of where the "
        "other ends, the rendered flow looks truncated — reread and fix that "
        "before responding.\n\n"
        + analysis
        + "`proposed_architecture_components` MUST reflect what your own "
        "`proposed_solution` actually changes structurally. If the "
        "proposed solution introduces a new call, a new dependency, or a "
        "new component (e.g. writing to a database, calling another "
        "service, publishing to a queue), that new relationship MUST "
        "appear in `depends_on` here even if it wasn't present before — "
        "copying the current list "
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
        'merely reminds you of. "This changes how `api_payment_auth` is '
        'called, which `OrdersController.createOrder` depends on" is a '
        'real impact statement; "this could impact payment processing '
        'and compliance" with no specific symbol/service named is not — '
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
        "TEST WORK IS PART OF THE CHANGE, NOT A TESTING RECOMMENDATION. A "
        "`testing_plan` scenario describes what must be proved; by itself it "
        "instructs nobody to write anything. Every scenario marked TO ADD "
        "must ALSO appear as its own step in `implementation_plan`, placed "
        "after the production change it covers, naming the production symbol "
        "being tested, the scenario(s) that step implements, and the test "
        "file it goes in. A test update required by a production change "
        "belongs in the plan the same way. A plan that edits production code "
        "with no step that writes the tests leaves whoever implements it free "
        "to skip them.\n\n"
        'A SYMBOL\'S "Tested by" LINE NAMES A TEST THAT ALREADY ASSERTS ITS '
        "CURRENT BEHAVIOUR. Each symbol in a 'Real code symbols' section says "
        "either which test file sits beside it or that none does. When your "
        "change alters what that symbol returns, accepts, rejects, logs, "
        "renders or otherwise does observably, a test in that file will very "
        "likely assert the behaviour you are removing. Say so, and propose "
        "updating it in the same implementation step as the production "
        "change. Leaving a test that asserts the old behaviour is a failure "
        "you are introducing, not a problem for someone else. You were given "
        "the test file's PATH, never its contents, so name the file and the "
        "behaviour that changed — never quote a test, name a test function, "
        "or state what it currently asserts.\n\n"
        "A TEST THAT MUST CHANGE IS A FILE CHANGE, NOT ONLY A PLAN STEP. An "
        "implementation step describes work; only an entry in `changes` "
        "names a file for someone to open. So when an existing test file "
        "sits beside a symbol you are changing and your change alters what "
        "that symbol does observably, add that TEST FILE to the same "
        "service's `changes` as its own entry: `file_path` is the test "
        "file's path exactly as you were given it, `change_type` is "
        '"modify", `function_or_symbol` is the PRODUCTION symbol whose '
        "behaviour changed, and `change_description` says which behaviour "
        "the test must assert instead. You were given that file's PATH and "
        "never its contents, so do not name a test function or state what "
        "it currently asserts.\n\n"
        "WHEN A SYMBOL NAMES WHERE A NEW TEST BELONGS, WRITE THE TEST "
        "THERE. A symbol with no test beside it may still say where one "
        'would go ("a new test for it belongs at ...") — that path was '
        "measured from this service's own layout, not guessed, so use it "
        "exactly as given: add a change with that `file_path` and "
        '`change_type` "create", with `function_or_symbol` naming the '
        "production symbol the test covers. Every scenario you marked TO "
        "ADD for that symbol must be covered by one of these changes.\n\n"
        "Only when no path is given anywhere do you leave the test out: "
        "do NOT invent a path, and do NOT add a change without one — a "
        "change with no `file_path` is skipped by whoever implements it, "
        "so the test silently never gets written. Put the question of "
        "where this project keeps tests for that code into "
        "`open_questions` instead.\n\n"
        "When a symbol says no test file sits beside it, do not conclude it "
        "is untested: tests may live elsewhere in the project. Treat it as "
        "unknown, and put the question of where this project keeps tests for "
        "that code into `open_questions`.\n\n"
        "For every change, `implementation_notes` states which of these is "
        "true: new tests are required; an existing test beside the symbol "
        "asserts the behaviour being changed and must be updated; or test "
        "coverage cannot be determined from the evidence, in which case name "
        "what must be inspected in `open_questions`. Understanding the "
        "production code is not evidence that it is tested, so never claim "
        "coverage is adequate.\n\n"
        "`testing_plan`: a list of concrete test scenarios, each with a "
        '`category` ("unit", "integration", "api", "regression", or '
        '"edge_case"), a `scenario`, and its `expected_result`. Every '
        "change you propose maps to at least one scenario, and every "
        "scenario maps back to a proposed change — a test for behavior you "
        "are not changing is out of scope. Include negative cases: invalid "
        "input, missing input, the boundary just outside what is accepted. "
        "At least one entry is a `regression` scenario written so it FAILS "
        "against the code as it exists today and PASSES after your change — "
        'state both halves explicitly ("today: <observed>; after: '
        '<expected>"). Write scenarios with concrete inputs and concrete '
        "expected outputs — an actual value, an actual status, an actual "
        'error message — never "test that validation works". Each '
        "scenario says whether it is a test TO ADD or an EXISTING test to "
        "run, naming an existing one only if you actually saw it in the "
        "evidence. There is no third kind: NOTHING here executes tests, you "
        "have run nothing, and no execution results exist — never write "
        "that a test passed, failed, ran or was verified. `testing_notes` "
        "carries any overall testing narrative that doesn't fit a single "
        "scenario.\n\n"
        "`implementation_plan`: an ORDERED list of concrete steps across "
        "the whole change. Each step names (a) the verified code location "
        "it touches, as `<service> <file_path> <symbol>` copied from a "
        "'Real code symbols' section, (b) the intended change in one line, "
        "(c) why that step is needed, and (d) which `testing_plan` scenario "
        "validates it. A step that cannot name a verified location is not "
        "an implementation step — it is an open question, and belongs "
        "there instead. The single exception is a step that writes or "
        "updates tests: test files are not part of the evidence you were "
        "given, so such a step names the production symbol it covers and "
        "the test file path it proposes, marked as proposed rather than "
        "existing — unless a test file was listed beside that symbol, in "
        "which case name that file. Every file appearing in any change must be "
        "justified by at least one step. Do not pad the plan with generic "
        'steps ("review the code", "deploy") your analysis does not '
        "support. Do NOT number the steps yourself — they are a list and the "
        'rendering step numbers them, so a step beginning with its own "1." '
        'is printed as "1. 1.".\n\n'
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
        "- Leave `sequence_steps` empty, leave `current_architecture_"
        "components` empty, and keep `current_architecture` and "
        "`overall_reasoning` to one sentence each — they are not shown.\n"
        "- Don't repeat the same fact in several fields; say it once in the "
        "field it belongs to.\n"
        '- No generic recommendation ("add logging", "consider caching", '
        '"follow best practices") the ticket did not ask for. A concise '
        "accurate document beats a long one padded with speculation, and an "
        "empty list is a valid answer.\n\n"
        "SELF-CHECK BEFORE YOU RESPOND. Your sections are rendered as one "
        "document and read top to bottom; a reader who finds two sections "
        "disagreeing stops trusting all of them. Verify:\n"
        "- every service on the reproduction-step path appears in `services`;\n"
        "- every service with a non-empty `changes` list has evidence in its "
        "`reasoning` that it must change, not merely that it sits on the "
        "data path;\n"
        '- no service you called "no change needed" also carries changes;\n'
        "- no root cause is marked Confirmed on reported-symptom evidence "
        "alone;\n"
        "- every `file_path` is a plain path with no line range appended;\n"
        "- no identifier appears in a `pseudocode_sketch` that you did not "
        "see in the evidence, unless `implementation_notes` says it must be "
        "created;\n"
        "- every service in `proposed_solution` appears in `services`, in "
        "`proposed_architecture_components` and in `proposed_flow_steps`;\n"
        "- every change has an `implementation_plan` step and a "
        "`testing_plan` scenario, and every acceptance criterion maps to a "
        "scenario;\n"
        "- `open_questions` and `risks` do not contradict certainty claimed "
        "elsewhere;\n"
        "- every change has test work in `implementation_plan`, or a reason "
        "grounded in the evidence that it needs none;\n"
        "- every symbol whose behaviour changes and that named a test file "
        "beside it has that file in a step;\n"
        "- nothing claims a test was run.\n\n"
        "Respond with ONLY one JSON object matching this schema, no markdown "
        "fences, no prose outside the JSON:\n" + schema
    )


_MAX_FILE_CONTEXT_NAMES = 20


def _format_code_symbol(
    symbol: dict,
    snippet: str | None,
    file_neighbors: list[str] | None = None,
    tests_beside: list[str] | None = None,
    new_test_path: str | None = None,
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
    if tests_beside is not None:
        # Paths only — enough to tell the model a test exists and must be
        # updated, without test source competing with the real code.
        if tests_beside:
            named = ", ".join(f"`{t}`" for t in tests_beside)
            line += f"\n  Tested by (existing test file beside it): {named}"
        elif new_test_path:
            # Where a NEW test belongs, measured from this service's own
            # habit (graph_reader.test_naming_conventions) — without it the
            # model is left with nowhere to put the test and can only ask
            # the question, which leaves the test unwritten.
            line += (
                "\n  No test file sits beside this one. Tests in this service sit "
                f"beside the code they cover, so a new test for it belongs at `{new_test_path}`."
            )
        else:
            line += "\n  No test file sits beside this one in its folder."
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
    sibling_tests: dict[tuple[str, str], list[str]] | None = None,
    new_test_paths: dict[tuple[str, str], str] | None = None,
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
    sibling_tests = sibling_tests or {}
    new_test_paths = new_test_paths or {}

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
                    sibling_tests.get((doc["name"], symbol["file_path"])),
                    new_test_paths.get((doc["name"], symbol["file_path"])),
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
