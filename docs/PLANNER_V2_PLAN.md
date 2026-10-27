# Planner v2: whole files, graph links, and a bounded tool loop

Status: **proposal, nothing built.** Depends on `INGESTION_V2_PLAN.md`:
`File.source` (Phase 1) and `IMPORTS`/`REFERENCES` (Phase 3) at minimum.

## Why

On the Client UI ticket "controls shift when the Archive button appears",
the TDD proposed changing only `ReportsListScreen/view.tsx`. A review with
the code open found that the button is rendered in
`InfiniteScrollGrid/view.tsx`, its type `ActionButton` lives in
`InfiniteScrollGrid/types.ts`, and keeping it mounted exposes an
empty-selection bug in `updateArchiveButton`. The planner could not have
known any of this:

- **It sees functions, not files.** `build_tdd` sends each matched symbol
  whole only if it is at most 1,500 chars, or fits the service's
  20,000-char budget (`_FULL_SOURCE_MAX_CHARS`,
  `_WHOLE_SOURCE_BUDGET_CHARS` in `query_flow.py`). `ReportsListScreen` is
  43,510 chars, so it arrived as one ~700-char chunk without the JSX.
- **It cannot follow code.** It saw the import line for `InfiniteScrollGrid`
  and nothing else. There is no link to follow and no way to ask.
- **It cannot check a guess.** The TDD said, in its own words: *"inspecting
  the JSX branch that renders action buttons in that file would confirm the
  exact element"*. Then it listed that as an open question.

The same pattern caused earlier misses: `toAvro` hidden beside
`isPhoneValid`, and a 12,135-char component shown as one chunk. Each was
patched with another size rule. This plan replaces the rules.

## Current flow (for reference)

```
issue_split.issue_queries  -> mode (bug/feature) + search queries
embed_many                 -> query vectors
graph_reader.summary_scores + search_service_chunks -> rank services and functions
load_symbols + size rules  -> snippet per symbol (whole, or one chunk)
_widen_to_small_classes    -> whole class when <= 8,000 chars
load_file_context / imports / tests / conventions
planner.build_system_prompt + build_user_prompt
planner.call_llm           -> ONE call
planner.parse_tdd + verify_tdd -> TechnicalDesignDoc -> render_pdf
```

## New workflow

```
1. Understand   issue_split (unchanged)              -> mode, queries
2. Find         search (unchanged), grouped by FILE   -> ranked files per service
3. Expand       follow graph links from top files     -> related files, tables, operations, messages
4. Assemble     whole files under one budget          -> seed context, each file WHOLE or PARTIAL
5. Investigate  LLM + read-only tools, bounded loop   -> more files as needed
6. Submit       submit_tdd(JSON) -> parse_tdd -> verify_tdd + read-check -> PDF
```

### Step 2: find

Chunk search stays as it is: it is good at finding *where* relevant code
is. Hits are then grouped by file. A file's score is its best chunk score
plus a small bonus for each further matching chunk, capped so one long
file cannot win on volume. Service selection (`_SUMMARY_WEIGHT`, named
services, `top_k`) is unchanged.

### Step 3: expand

From the top files (the focus files), follow one step of each link:

| Link | What it brings in | Why |
|---|---|---|
| `IMPORTS` out | Local files the focus file imports, followed through barrels | The grid case: screen → `InfiniteScrollGrid/view.tsx`, `types.ts` |
| `REFERENCES` out | The files of symbols the focus code uses | Callees and rendered components not found by search |
| `IMPORTS` in (importers) | Up to 5 files that import the focus file | Who breaks if its API changes |
| `TESTS` | Existing tests of the focus file and its neighbours | Test conventions and what is already asserted |
| `READS` / `WRITES` → `Table` | Table definitions the focus code touches | Schema changes without guessing columns |
| `RESOLVES` / `USES_OPERATION` | The GraphQL operation and the other side of it | Cross-service API changes |
| `PUBLISHES` / `HANDLES` → `Message` | The message, and who else publishes or handles it | Kafka contract changes |

The links to tables, operations and messages arrive with ingestion Phases
4–5. Until then, step 3 uses imports, references, importers and tests.

### Step 4: assemble

- **The unit is the file.** Text comes from `File.source`.
- **There is one budget for the whole prompt**, `CODE_CONTEXT_TOKENS`,
  proposed at 100k tokens and set from the model's context window. There is
  no per-service or per-symbol limit. It is filled in this order:
  1. focus files
  2. their imports and references
  3. matched files in the other selected services
  4. tests
  5. importers
  6. tables, operations and messages
- **A file that does not fit whole is sent as `PARTIAL`.** That means its
  outline (every symbol with its line range) plus each matched region with
  about 80 lines of context. It is never sent as a lone chunk.
- **Every file is labelled** in the prompt:
  `=== ui/src/.../view.tsx [WHOLE, 1,494 lines] ===` or
  `[PARTIAL, lines 1180–1260, 1400–1494 of 1,494]`.
- **Removed:** `_FULL_SOURCE_MAX_CHARS`, `_WHOLE_SOURCE_BUDGET_CHARS`,
  `_FULL_CLASS_MAX_CHARS`, `_widen_to_small_classes`, and the per-symbol
  `snippet_map`.
- **The system prompt gains two rules.** You may only say code is absent
  from a file marked `WHOLE`. To propose changing a file you have only
  partly seen, read it first with `read_file`.

For the Archive ticket, the seed context would be about 55k characters
(~14k tokens): the screen whole, the grid and its types whole, and its
tests. That is well inside the budget.

### Step 5: investigate (the tool loop)

```
messages = [system, user(seed context + issue)]
for round in 1..MAX_ROUNDS:
    reply = chat_with_tools(messages, TOOLS, tool_choice="auto")
    if reply calls submit_tdd: break
    for call in reply.tool_calls (up to the remaining call budget):
        result = run_tool(call)  # read-only, scoped, truncated, cached
        messages += [tool_call, tool_result]
if not submitted:
    reply = chat_with_tools(messages, [submit_tdd], tool_choice=submit_tdd)
```

Both providers support this. The AI gateway client has
`chat_with_tools(provider, model_name, messages, tools, tool_choice, ...)`,
and OpenAI chat completions support `tools` natively. `planner.call_llm`
gains a `call_llm_with_tools` beside it, with the same gateway-or-OpenAI
split.

### Step 6: submit and verify

- **The final answer is a `submit_tdd` call** whose arguments are the
  existing `TechnicalDesignDoc` JSON schema. That gives structured output on
  both providers, and the gateway's `chat()` has no `response_format`
  today. `parse_tdd`, with its repair step, still runs on the arguments.
- **`verify_tdd` runs unchanged.**
- **New read-check.** Every `FileChange` with `change_type` `modify` or
  `delete` must name a file the planner saw `WHOLE`, either in the seed or
  through `read_file`. If it did not, the change is flagged unverified with
  the note "proposed without reading the whole file". Nothing is dropped,
  same as `verify_tdd`.
- **Evidence log.** Each tool call (name, arguments, result size, round) is
  recorded. It is printed in the debug output, and the PDF gets a short
  "Code examined" appendix listing the files read and how.

## Tool definitions

All tools are read-only Neo4j lookups. None run code, call GitHub, or write
anything. Every tool is limited to the services in the run (the
`service_names` passed to `build_tdd`).

| Tool | Arguments | Returns | Limits |
|---|---|---|---|
| `search_code` | `query: str`, `service?: str`, `limit?: int` (≤10) | Ranked hits: service, file, symbol, line range, a 300-char preview | Same search as step 2 |
| `read_file` | `service: str`, `path: str`, `start_line?: int`, `end_line?: int` | The file text with line numbers, or the requested range. States `WHOLE` or `PARTIAL`. | `MAX_TOOL_RESULT_CHARS`. A larger file returns its outline and asks for a range. |
| `list_files` | `service: str`, `directory: str` | File names, kinds and line counts in that folder (not recursive) | 200 entries |
| `get_imports` | `service: str`, `path: str` | Resolved local imports (path and names), plus unresolved import strings | none |
| `find_importers` | `service: str`, `path: str` | Files that import this one, with the names they import | 30 entries |
| `find_usages` | `service: str`, `symbol: str` | Symbols that reference it, with file and line | 30 entries |
| `get_symbol` | `service: str`, `name: str` | Matching symbols: kind, file, line range, signature, parent | 10 entries |
| `get_table` | `service: str`, `table: str` | Columns, the changelogs that define it, mapped entities, readers and writers | after ingestion Phase 4 |
| `get_operation` | `name: str` | The exposing service and resolver, the schema text, and the client files that use it | after Phase 5 |
| `get_message` | `name: str` | Avro definition, topics, and publishing and handling symbols by service | after Phase 5 |
| `submit_tdd` | the `TechnicalDesignDoc` schema | — | ends the loop |

Example definition, in the format both providers accept:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read a file from a service in this run. Returns the whole file when it fits, otherwise its outline; pass start_line/end_line to read a range.",
    "parameters": {
      "type": "object",
      "properties": {
        "service": {"type": "string", "description": "Service name, e.g. \"ui\""},
        "path": {"type": "string", "description": "File path relative to the repo root"},
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1}
      },
      "required": ["service", "path"]
    }
  }
}
```

Each tool is a thin wrapper over a `graph_reader` function, and most of
those exist or are listed in the ingestion plan: `load_file_source`,
`get_imports`, `find_importers`, and so on. The wrappers live in a new
`src/query/planner_tools.py`, which holds the tool schemas, argument
checks, scoping, truncation and the call cache.

## Guardrails

| Guardrail | Value | Why |
|---|---|---|
| `MAX_ROUNDS` | 6 | Bounds latency. Most tickets should need 0–3 tool calls when the seed context is good. |
| `MAX_TOOL_CALLS` | 15 per run | Bounds cost if the model calls tools in parallel |
| `MAX_TOOL_RESULT_CHARS` | 40,000 per call | One file read cannot swamp the context |
| `MAX_TOOL_TOTAL_CHARS` | 150,000 per run | The seed plus tool results stay inside the model's window |
| Read-only and scoped | always | No writes, no code execution, no services outside the run |
| Call cache | per run | The same call twice returns the cached result with a note, so loops die out |
| Forced submit | after the caps | `tool_choice` set to `submit_tdd`. The run always produces a TDD. |
| Temperature | `_TEMPERATURE` (unchanged) | Keeps runs comparable |
| Tool errors | returned as tool results | A bad path is something the model can fix ("no such file; did you mean …"). It never aborts the run. |
| Fallback | single-call v1 path | If the tool call fails at the provider level, or `submit_tdd` never parses, the run falls back to today's flow and records why |
| No critic pass | always | Tools gather evidence. Nothing asks an LLM to judge the plan, in line with the rule in `CODING_AGENT_PLAN.md`. |

**Open item to test first:** the gateway's `chat_with_tools` has no
`max_tokens` argument, but `chat()` sets 12,000 for the TDD. If the
gateway's default is lower, a long `submit_tdd` call would be cut off.
Check this with one real call before building the loop. If it is cut off,
ask for `max_tokens` support, or end the loop with a normal `chat()` call
that sends the gathered context.

## Evaluation

### Benchmark set

1. **Known misses**, with ground truth from human or code review:
   - the Archive button ticket: expected `ReportsListScreen/view.tsx`,
     `InfiniteScrollGrid/view.tsx`, `InfiniteScrollGrid/types.ts`, and the
     empty-selection fix in `updateArchiveButton`
   - the phone whitespace ticket: expected `toAvro`, not only
     `isPhoneValid`
   - the 12,135-char component ticket: expected the four findings from its
     review
2. **Historical fixes.** Pick about 30 merged commits with a Jira key (for
   example reports-api `KS-17424`). The issue text is the Jira ticket, or
   the PR title and description. The ground truth is the non-test files the
   commit changed.
   - **Avoid leakage:** ingest each service at the fix commit's parent.
     Otherwise the fix is already in the code the planner reads.
   - If the "310-commit harness" mentioned earlier exists outside this
     repo, reuse it. It is not in `plan-agent`.

### Metrics, per ticket

| Metric | Meaning |
|---|---|
| File recall | Ground-truth files that appear as `modify`/`create` in the TDD |
| File precision | TDD changes that are in the ground truth, or judged needed on review |
| Service recall | Ground-truth services that the TDD includes |
| Unread modifies | `modify` changes the read-check flagged |
| Open questions | Count. These should fall as the planner can read code. |
| Tool calls, rounds | The loop's cost |
| Latency, input and output tokens | The run's cost |
| Stability | File recall across 3 runs of the same ticket |

### Variants compared

| Variant | Description |
|---|---|
| A | Today's planner (v1) |
| B | Steps 1–4 and 6: whole files and links, no loop |
| C | B plus the tool loop |

A `PLANNER_MODE` setting (`v1`, `v2-seed`, `v2-loop`) runs any variant on
the same graph, and a script runs the benchmark set and writes one CSV per
variant.

**Acceptance.** B beats A on file recall with no drop in precision. C beats
B on file recall, or on unread modifies, by enough to justify its added
latency and tokens. Agree the threshold before running, for example "+15
points recall, ≤ 2× latency".

### Tests (no LLM)

- Each tool against a fixture graph: scoping, truncation, outlines for
  large files, and error messages for bad paths.
- The loop with a scripted fake LLM: it stops at `MAX_ROUNDS` and
  `MAX_TOOL_CALLS`, forces submit, returns cached repeats, and falls back
  on a provider error.
- Context assembly: budget order, `WHOLE`/`PARTIAL` labels, and the
  Archive ticket's seed containing all three ground-truth files once
  ingestion Phase 3 is in.
- The read-check flags a `modify` on a file seen only partly.

## Rollout

| Step | Needs | Delivers |
|---|---|---|
| 1 | Ingestion Phase 1 (`File.source`) | Step 4 with whole files and the global budget. `read_file`/`list_files` tools built and tested, loop not yet on. Variant B without links. |
| 2 | Ingestion Phase 3 (imports, references) | Step 3 expansion, plus the `get_imports`, `find_importers`, `find_usages` and `get_symbol` tools. Run variant B. |
| 3 | The gateway `max_tokens` check | Steps 5–6: loop, `submit_tdd`, read-check, evidence log. Run variant C against A and B. |
| 4 | Ingestion Phases 4–5 | Table, operation and message expansion, and `get_table`, `get_operation` and `get_message` |

v1 stays the default until a variant passes acceptance on the benchmark.

## Costs and risks

- **Bigger prompts:** about 20–100k tokens of seed instead of 10–20k, plus
  loop rounds. With Anthropic, prompt caching on the seed makes later
  rounds much cheaper. Measure before relying on it.
- **More variation between runs.** Tool choices add it. Mitigated by
  temperature, the caps, and the stability metric.
- **Spending the budget on noise.** If expansion pulls in generic shared
  files (design-system atoms, constants), they take up the budget. Rank
  expanded files by link count from the focus files, and keep generated
  files out, as ingestion already marks them.
- **Gateway limits** (`max_tokens`, see the open item) could block the loop
  on the gateway path. The OpenAI path does not depend on them.
