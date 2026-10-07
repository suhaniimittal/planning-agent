# coding_agent — build reference for the new, standalone project

Status: **design finalized, nothing built yet in the new project.** This file is meant to be
copied into (or referenced from) the new `coding_agent` project directory and handed to whoever
(human or Claude session) builds it there. It supersedes the older version of this doc, which
assumed `coding_agent` would live inside this repo and reuse this repo's code directly — that is
no longer the plan.

## The one rule that shapes everything below

**`coding_agent` must have ZERO dependency on this project (`plan-agent` / `planning_agent`) — no
Python import, no shared package, no shared Neo4j graph.** It is a fully separate
`aetherion init` project, published independently. Anything it needs from this codebase gets
**copied in**, not imported. This matters because Aetherion's publish boundary is one project
root = one published agent (`aetherion publish` bundles one `src/` tree into one artifact,
uploaded in one call) — the two agents can never be the same project.

## What it does, end to end

**Input:** a TDD as a PDF file (uploaded via the platform's `file` trigger type).
**Output:** one Pull Request per affected service, never an auto-merge.

```
PDF upload
  -> extract raw text from the PDF (pypdf)
  -> ONE LLM call: raw text -> structured TechnicalDesignDoc JSON
  -> validate structurally (real file check, see below — no LLM, no Neo4j)
  -> group changes by service, then by file
  -> clone each affected service's repo once
  -> ONE LLM call per service: every file in that service's changeset -> patched/new content
  -> commit to a new branch, never main
  -> run the repo's own real test command (no repair loop — report failure honestly)
  -> open a PR (never auto-merged)
```

Steps 4 onward (clone -> patch -> commit -> test -> PR) are **already built and working** in this
repo, at `src/coding/*`. They get copied over close to verbatim. Steps 1-3 (PDF -> structured TDD
-> validated) are new and don't exist anywhere yet — that's the actual new work.

## New piece 1: PDF text extraction

A plain library call, no LLM: `pypdf` (pure Python, permissive license — no licensing concern the
way a PDF-*rendering* library like PyMuPDF had, since this is only ever reading a PDF, never
producing one). Extract all page text and concatenate it — the PDF is `planning_agent`'s own
`render_text()` output turned into a PDF, so it's already a clean, section-numbered plain-text
document; no layout/table reconstruction needed.

## New piece 2: text -> structured TDD (one LLM call, same trusted pattern this whole system already uses)

The extracted text is *unstructured* even though it's well-organized — getting it back into a
real `TechnicalDesignDoc` (per-service file changes, `change_type`, symbols, acceptance criteria)
needs one LLM call with a JSON schema, exactly the same "never guess, validate the parse
afterward" pattern `src/query/planner.py`'s `parse_tdd()` already uses in the planning project. Do
**not** write a regex/section parser against the PDF text — it would silently break the moment the
wording varies at all, and this system's own design philosophy elsewhere is explicit about
preferring a schema-validated LLM call over a fragile text parser for exactly this reason.

Needs its own copy of the `TechnicalDesignDoc` / `ServicePlan` / `FileChange` pydantic models
(duplicated from `src/query/models.py` here — only the fields the coding pipeline actually reads:
`service`, `file_path`, `function_or_symbol`, `change_type`, `change_description`,
`implementation_notes`, `pseudocode_sketch`, `acceptance_criteria`, `current_behavior`,
`has_real_source`, plus `title`/`issue_summary` for the PR body). Nothing else from `models.py`
(architecture diagrams, sequence steps, etc.) is needed here — the coding pipeline never reads
those fields.

## New piece 3 (real design change from the old plan): structural validation without Neo4j

The old plan's `validation.py` checked a change's `file_path` against **this project's Neo4j
graph** (`src/ingestion/graph_writer.get_driver()`). That's a real, hidden dependency on
`planning_agent`'s own database — which breaks the "zero dependency" rule just as much as a Python
import would.

**Fix: check the file's existence directly in the freshly cloned repo instead of a graph
snapshot.** The pipeline already clones the repo in step 3 (`cloned_repo()`) before it needs this
check anyway, so this is strictly simpler than the old version, not just more independent — no
Neo4j driver, no `graph_writer` import, no risk of checking against a stale/out-of-sync graph
snapshot. Reuses the exact logic `coding_flow.py`'s own `_read_current_content()` already has
(`(repo_root / file_path).exists()`), just moved earlier, into validation, instead of being
duplicated at patch time.

Concretely: `verify_changes(tdd)` changes from "does this file exist in Neo4j for this service" to
"does this file exist in the repo clone for this service" — same verified/skipped split, same
never-required-symbol-preexistence rule (a `modify` adding a brand-new method to an existing file
must still pass), just checked against real disk state instead of a graph.

One consequence: this means cloning has to happen **before** validation now, not after (the old
order was validate-then-clone-per-service; the new order is clone-per-service-then-validate). Not
a problem — it was already cloning once per service either way.

`services.yaml`/`manifest.load_services()` is still needed (to resolve a service name to its repo
slug) — that's a **plain config file copied into the new project**, not a database or a code
dependency on `planning_agent`. Keep a copy of `services.yaml` here, kept in sync manually (or
consider it deliberately allowed to diverge — a coding_agent might reasonably support a different,
narrower service list than planning_agent does).

## Files to copy in verbatim (no changes needed)

| File | From | Notes |
|---|---|---|
| `src/coding/models.py` | this repo | trim its `from src.query.models import FileChange` to the new project's own duplicated model |
| `src/coding/patcher.py` | this repo | no changes |
| `src/coding/repo_ops.py` | this repo | no changes |
| `src/coding/test_runner.py` | this repo | no changes |
| `src/ingestion/repo_fetcher.py` | this repo | no changes — `cloned_repo()`, `current_branch_of()` |
| `src/ingestion/github_api.py` | this repo | no changes — only `create_pull_request()` is actually used here |
| `src/ingestion/manifest.py` | this repo | no changes — `load_services()` |
| `services.yaml` | this repo | copy as a starting point; may diverge over time |

## Files to rewrite

| File | Change from the old plan |
|---|---|
| `src/coding/validation.py` | Replace the Neo4j lookup with a real filesystem check against the repo clone (see above) — `verify_changes()` now takes the cloned `repo_root` per service as an argument, since the clone must happen first |
| `src/coding/coding_flow.py` | Reorder: clone happens before `verify_changes()` is called (per service), not before; everything else (`_build_patches`, `_pr_body`, `run_coding_agent`) is otherwise unchanged |
| `src/query/models.py` (renamed, e.g. `src/coding/tdd_models.py`) | Trimmed `TechnicalDesignDoc`/`ServicePlan`/`FileChange` — only the fields the coding pipeline reads (see New piece 2 above) |

## New files to build

| File | Purpose |
|---|---|
| `src/coding/pdf_extract.py` | `extract_text(pdf_bytes) -> str` via `pypdf` |
| `src/coding/tdd_extraction.py` | System/user prompt + one LLM call turning extracted text into the trimmed `TechnicalDesignDoc` schema, validated the same way `planner.parse_tdd()` is (schema-validated, repaired-if-needed, never trusted blind) |
| `src/tools/coding_tools.py` | `@tool()` wrapping: extract PDF text -> extract structured TDD -> `run_coding_agent(tdd)` -> return PR URLs. (`run_coding_agent_tool` already exists in this repo's `src/tools/coding_tools.py` as a reference for the *second half* of this — the TDD-to-PR part is copy-paste; only the PDF-handling front half is new) |
| `src/agent/agent.py` | `@agent(name="coding_agent")` — a `file`-type trigger (PDF), thin orchestration only: `toolExecutor.execute(...)` for the one tool above, same error-boundary pattern as this repo's `planning_agent` |
| `src/agent/metadata.json` | `"name": "coding_agent"`, one `file` trigger for the PDF upload |

## New project folder layout

```
coding-agent/                    <- new, separate `aetherion init` project root
  pyproject.toml                  <- own version, own dependencies (add `pypdf`)
  services.yaml                   <- copied starting point
  src/
    agent/
      __init__.py
      agent.py                    <- @agent(name="coding_agent")
      metadata.json
    tools/
      __init__.py
      coding_tools.py              <- @tool() — full flow, PDF in, PR URLs out
      metadata.json
    coding/
      __init__.py
      models.py                    <- copied verbatim
      patcher.py                   <- copied verbatim
      repo_ops.py                  <- copied verbatim
      test_runner.py                <- copied verbatim
      validation.py                 <- rewritten (no Neo4j)
      coding_flow.py                <- rewritten (clone-before-validate ordering)
      pdf_extract.py                <- new
      tdd_extraction.py              <- new
      tdd_models.py                  <- new (trimmed TechnicalDesignDoc/ServicePlan/FileChange)
    ingestion/
      __init__.py
      repo_fetcher.py                <- copied verbatim
      github_api.py                  <- copied verbatim
      manifest.py                    <- copied verbatim
```

(Package name `ingestion` is kept only because the copied files' own internal imports reference
`src.ingestion.*` — rename everything together if a different package name is preferred, just keep
it consistent.)

## Why no extra LLM calls for the actual patching/validation (carried over unchanged)

Considered and rejected in the original design: a second "critic" LLM pass reviewing each proposed
change before applying it. Rejected because it doesn't catch the failure mode that matters most —
a proposal that *sounds* correct from reading the code but is wrong about the real world. A second
LLM reading the same code has the identical blind spot and would approve the same broken plan.
That class of error can only be caught by **actually trying it** (a real test run), not by asking
for another opinion. So validation is either free (a real filesystem check) or real (actual test
execution) — never a second guess. The one new LLM call in this whole plan (PDF text -> structured
TDD) is not an exception to this: it's not validating a plan, it's recovering structure from a
document — a fundamentally different, unavoidable task, same as `parse_tdd()` already does on the
planning side.

## Guardrails (unchanged from the original plan)

- Never write to `main`/`master` — branch + PR only.
- Never auto-apply a change that failed the structural (file-existence) check.
- Never claim a fix works without real verification; say so honestly when it can't be confirmed.
- No repair/retry LLM calls — verification is one-shot; failures are reported, not auto-fixed.
- No "critic" LLM passes anywhere — validation is either free or real, never another opinion.

## Build order, when work starts in the new project

1. Copy the verbatim files first (`patcher.py`, `repo_ops.py`, `test_runner.py`,
   `repo_fetcher.py`, `github_api.py`, `manifest.py`, `services.yaml`) — zero new logic, just
   relocation.
2. `tdd_models.py` — the trimmed schema, independently reviewable against the old `models.py`.
3. `pdf_extract.py` — smallest new piece, trivially testable with any sample PDF.
4. `tdd_extraction.py` — the one new LLM call; test against a real PDF produced by
   `planning_agent`'s own `render_text()` output once that exists.
5. `validation.py` rewrite — testable against a real local clone, no network/LLM needed.
6. `coding_flow.py` reorder — wire the pieces together.
7. `coding_tools.py` + `agent.py` + `metadata.json` — the platform-facing wrapper, last.
