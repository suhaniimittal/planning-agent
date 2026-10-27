# Ingestion v2: store every file, every line, every link

Status: **built (pipeline version 4), not yet run against Neo4j.** All five
phases are implemented; see "As built" at the end for where the build
differs from this plan and why. Supersedes the ingestion parts of
`INGESTION_PIPELINE.md`, which still describes the old doc-based pipeline.

## Why

A real ticket (Client UI: controls shift when the Archive button appears)
produced a TDD that changed the wrong file. The button is rendered in
`InfiniteScrollGrid/view.tsx`, and its type is `ActionButton` in
`InfiniteScrollGrid/types.ts`. The planner never saw either file:

- `ReportsListScreen` (43,510 chars) reached it as one ~700-char chunk.
- Nothing links `ReportsListScreen` to the grid it renders.
- `ActionButton` is a `type` alias, which the parser does not capture.

An audit of all 13 local repos against Neo4j (2026-10-26) shows this is
general, not a one-off:

| What | Today |
|---|---|
| Files on disk / `File` nodes | 21,377 / 21,365 (12 missing: files that failed to parse) |
| Files whose text is stored | 7,400 (test and symbol-less files only, cut at 6,000 chars) |
| Code lines inside any stored symbol or chunk | 80.4% (asurint-api 41.5%, ui 75.3%) |
| Text files with a node but no content | ~2,500, including 933 `.sql` Liquibase migrations |
| Edge types | `HAS_FILE`, `DEFINES`, `DECLARES`, `HAS_CHUNK`, and 29 `Service-CALLS->Service` |

There are no code-level links at all: no imports, calls, extends, tests,
operations, topics or tables.

## Principles

1. **Store everything, embed what is useful for search.** Storage is
   lossless: every text file's full content, every symbol, every link. The
   vector index is curated: generated code, lockfiles and diagram
   coordinates are stored but not embedded.
2. **The file is the source of truth.** `File.source` holds the exact text.
   Symbols and chunks are line ranges into it, not separate copies that can
   drift or get cut.
3. **Every line is accounted for.** Each non-blank line of every text file
   belongs to a symbol, a chunk, or a file marked as not searchable. A
   coverage report proves it on every run.
4. **Never guess a link.** A link is written only when it resolves to a real
   node. Anything unresolved is recorded and reported, never dropped
   silently.
5. **Re-ingesting is cheap.** Embeddings are cached by the hash of the text
   they were made from, so a version bump re-parses everything but only
   pays to embed text that actually changed.

## Graph schema v2

### Nodes

| Label | Key | New or changed properties |
|---|---|---|
| `Service` | `name` | unchanged |
| `File` | `service`, `file_path` | `source` (full text, every text file), `line_count`, `size`, `truncated`, `kind` (`source`, `test`, `migration`, `schema`, `config`, `build`, `doc`, `generated`, `binary`), `parse_status` (`ok`, `failed`, `oversized`, `binary`, `unsupported`), `parse_error`, `searchable` |
| `Class` / `Function` | `service`, `qualified_name` | Keep both labels. Add `kind` (`class`, `interface`, `enum`, `record`, `object`, `type`, `const`, `function`, `method`, `constructor`, `test`). Add the `Symbol` label to all of them. `qualified_name` includes the parent for nested symbols. `source_snippet` is derived from `File.source` by line range. |
| `Chunk` | `service`, `qualified_name` | `text` is the exact lines. `header` is `// view.tsx › InfiniteScrollGrid (lines 90–120)`. `kind` is `symbol` or `region`. `text_hash` is used for the embedding cache. |
| `Table` (new) | `service`, `name` | `columns` (`["id uuid not null", ...]`), `defined_in` (changelog paths) |
| `Operation` (new) | `service`, `type`, `name` | A GraphQL Query, Mutation or Subscription field the service exposes |
| `Endpoint` (new) | `service`, `method`, `path` | A REST route |
| `Message` (new) | `name` (fully qualified Avro name) | A Kafka message type, such as `com.asurint.models.orders.events.UpdateHighRisk`. `defined_in` is its `.avdl` path in `models`. |
| `Topic` (new) | `name` | A Kafka topic, such as `domain.orders`. `constant` is the `Topics` constant that names it, such as `Topics.domain.orders`. |

### Edges

| Edge | Meaning | Phase |
|---|---|---|
| `File -IMPORTS {names, via}-> File` | Resolved local import. `via` records the barrel file (`index.ts`) it was followed through. | 3 |
| `Symbol -REFERENCES-> Symbol` | A symbol's body uses another symbol: a call, a JSX render, a type use. Resolved through imports, the same file, or the same class. | 3 |
| `Symbol -CONTAINS-> Symbol` | Nesting (a function inside a component). Generalises `DECLARES`, which stays for methods. | 2 |
| `Class -EXTENDS / IMPLEMENTS-> Class` | Inheritance | 3 |
| `File -TESTS-> File` | A test file imports, or is named after, a source file | 3 |
| `Service -OWNS-> Table`, `File -DEFINES-> Table`, `Class -MAPS_TO-> Table` | Database schema from Liquibase, plus entities | 4 |
| `Symbol -READS / WRITES-> Table` | From SQL in repositories (`@Query`, JDBC strings) | 4 |
| `Service -EXPOSES-> Operation / Endpoint`, `Function -RESOLVES-> Operation` | API surface | 5 |
| `File -USES_OPERATION-> Operation` | A GraphQL client operation (`gql` template or `.graphql` file) matched to a schema field in another service | 5 |
| `Symbol -PUBLISHES-> Message` | A class given a `KafkaProducer<Key, M>` | 5 |
| `Symbol -HANDLES-> Message` | A class implementing `KafkaMessageHandler<M>` | 5 |
| `Message -ON_TOPIC-> Topic` | From a handler's `inputTopic`, or a producer's topic configuration | 5 |
| `Service -CALLS {via, evidence}-> Service` | Derived from the concrete links above. `via` is `graphql`, `kafka`, `rest` or `text-scan`. For Kafka, `evidence` names the message. The current text scan stays as the fallback. | 5 |

## Pipeline

```
Stage 0  Discover     walk repo -> classify every file -> read text -> File node (always)
Stage 1  Parse        per language -> symbols + imports + identifiers used + domain facts
Stage 2  Chunk/embed  symbol chunks + region chunks + text-file chunks -> embedding cache -> vectors
Stage 3  Link (in)    imports -> files, identifiers -> symbols, extends, tests
Stage 4  Link (out)   after ALL services: operations, topics, endpoints, tables -> Service CALLS
Stage 5  Verify       coverage report + unresolved-link report; fail loudly
```

### Stage 0: discover

- Walk with the existing denylist. **Every** file gets a node before any
  parsing, so a parse failure can no longer drop one (today: 11 generated
  GraphQL clients in reports-api and `ui/src/apps/candidate/types/TIcon.ts`).
- Binary files are detected with a null-byte sniff and stored as nodes
  without text (56 today).
- Text is stored up to `MAX_STORED_SOURCE_CHARS` (proposed 1,000,000). Above
  that the text is cut and `truncated` is set.
- `kind` and `searchable` are set by path rules. `generated`: `src/generated/`,
  `*.generated.*`, files whose header says `@generated` or `DO NOT EDIT`,
  and the Apollo test clients. `build`: lockfiles, `gradlew`. These are
  stored but not embedded.

### Stage 1: parse — what each language must capture

| Language | Captured today | Add |
|---|---|---|
| TS / TSX / JS | classes, interfaces, functions, `const x = () =>` | `type` aliases, `enum`, `abstract class`, top-level non-function `const` (styled components, config objects; skip tiny ones), arrow functions passed to `useCallback`, `useMemo`, `memo`, `forwardRef` and assigned to a `const`, test blocks (`describe`, `it`, `test`) named by their title |
| Java | classes, interfaces, methods, constructors | `enum`, `record`, annotation types, `static final` constants |
| Kotlin | classes, functions | `object` and `companion object`, `typealias`, top-level `val`/`var` |
| Python | classes, functions | module-level constants |
| GraphQL `.graphqls` | object, interface, input and enum types, fields | `scalar`, `union`, `extend type`, directives. Fields on `Query`, `Mutation` and `Subscription` also become `Operation` nodes. |
| GraphQL `.graphql` (client) | nothing | new: `query`, `mutation` and `fragment` definitions |
| SQL / Liquibase | nothing (933 `.sql` files) | new: parse with `sqlglot`. Replay `CREATE` and `ALTER` in changelog order into `Table` nodes. Statement-level chunks. Also read Liquibase XML/YAML changelogs. |
| Avro `.avdl`, BPMN | custom parsers | BPMN diagram coordinates (`di:waypoint`, `dc:Bounds`) are excluded from embedding |
| YAML, JSON, properties, XML, Gradle, Dockerfile, shell, `.feature` | name only | full text. Chunked by top-level key, block or `Scenario`. |

Parsing also records, for each symbol, the identifiers its body uses
(Stage 3 resolves them), and for each file its structured imports (module
plus names), taken from the **full** text, not the 1,200-char `imports`
field.

### Stage 2: chunk and embed

- **Exact text.** A chunk's text is exactly its lines. The repeated
  ~600-char signature prefix goes, replaced by a one-line header.
  `embedding_text()` already adds the path and name.
- **Snippet starts at the declaration.** It starts at `export const Name =`,
  using `_ts_statement_root`. Today it starts at `<A, R, …`.
- **Sizes.** When a statement is longer than the target (one `return (...)`
  becomes a 4,990-char chunk today), split inside it at its children (JSX
  elements, arguments), with lines as the last resort. Skip bare `{` and `}`.
  Merge chunks under ~200 chars into a neighbour.
- **Region chunks.** Lines inside no symbol (top-level code, decorators,
  `export default`) are chunked as `region` chunks, so coverage is complete.
  This also replaces today's rule that only symbol-less files get file
  chunks, which would otherwise break as soon as a file gains a `type` symbol.
- **Embedding cache.** The key is `sha256(embedding_text)`, looked up from
  existing `Chunk.text_hash` before calling the API. A version bump then
  costs only parsing plus embeddings for changed text.

### Stage 3: link inside a service

- **Imports.**
  - TS/JS: relative paths try `.ts`, `.tsx`, `.js`, `.jsx` and `/index.*`.
    Aliases come from `tsconfig.json` `paths`, parsed as JSONC; ui maps
    `@client/*` to `src/apps/client/*`. Barrels are followed: `index.ts`'s
    `export { X } from "./view"` and `export * from "./types"` are traced to
    the file that actually defines each imported name.
  - Java/Kotlin: package plus class name to file. Classes in the same
    package need no import.
  - Python: module path.
- **References.** A symbol's used identifiers are resolved in this order:
  same file, then imported names, then the same package. Unresolved
  identifiers are kept as a count, not written as edges.
- **Extends/implements** come from the class header. **Tests** come from
  test-file imports plus the existing naming conventions.

### Stage 4: link across services (after all services are done)

- **GraphQL.** Client operations (ui's ~181 files using `gql`/`useQuery`,
  asurint-api) match the schema fields exposed by the Kotlin services'
  `.graphqls`. This gives `USES_OPERATION`, then `CALLS {via: graphql}`.
- **Kafka.** See the next section.
- **REST.** Client base URL plus path match `Endpoint` nodes.
- The current text scan stays as `via: text-scan` for anything the above
  does not explain.

### Kafka design

The services do not use Spring's `@KafkaListener` or `KafkaTemplate`
directly. They use the in-house library in the `kafka` repo
(`kafka-spring`: `com.asurint.kafka.spring.KafkaProducer`,
`com.asurint.kafka.dispatching.KafkaMessageHandler`, and the topic constants
in `com.asurint.kafka.configuration.Topics`). Message types are Avro
classes from the `models` repo (`com.asurint.models...`), generated from
`.avdl` files that ingestion already parses. Read in `rhino-service`:

```kotlin
// consumer: kafka/handlers/events/UpdateHighRiskSSNHandler.kt
class UpdateHighRiskSSNHandler(...) : KafkaMessageHandler<UpdateHighRisk> {
    override val inputTopic = Topics.domain.orders
    override val messageType = UpdateHighRisk::class.java
    override fun handle(message: UpdateHighRisk) { ... }
}

// producer: kafka/publishers/CreditReportOrderCompletedPublisher.kt
class CreditReportOrderCompletedPublisher(
    producer: KafkaProducer<OrderKey, RhinoCreditReportResult>,
) : AbstractOrderCompletedPublisher<...>(producer, ...)
```

**The message type is the link key, not the topic.** One topic, such as
`domain.orders`, carries many message types, so matching by topic alone
would link every producer on it to every handler on it. Matching by type
says exactly who consumes the event being changed.

What ingestion extracts:

1. **Messages.** Every Avro `record` in `models`' `.avdl` files becomes a
   `Message` node with its full name, from the existing `avdl` parser's
   output.
2. **Handlers.** A Kotlin or Java class whose supertype is
   `KafkaMessageHandler<M>` gets a `HANDLES` edge to `M`. Its `inputTopic`
   value, when it is a `Topics.…` constant, gives `M -ON_TOPIC-> Topic`.
3. **Publishers.** A class with a constructor parameter or property typed
   `KafkaProducer<K, M>` gets a `PUBLISHES` edge to `M`. This includes
   subclasses of shared bases like `AbstractOrderCompletedPublisher`, which
   pass the producer up.
4. **Name resolution.** `M` is resolved through the file's imports to the
   fully qualified Avro name. If it does not resolve, it is reported, not
   guessed.
5. **Topics.** The `Topics` object in the `kafka` repo is parsed for its
   constant tree, mapping `Topics.domain.orders` to the topic string. Where
   a producer's topic comes from Spring configuration rather than code, it
   is read from `application*.yml`/`.properties`, which Stage 0 now stores.
6. **Service links.** For each message, every publishing service gets
   `CALLS {via: kafka, evidence: "<message name>"}` to every handling
   service.

Before building, confirm two things by reading the library: how a
`KafkaProducer` bean gets its topic, and whether any service handles
messages without `KafkaMessageHandler`, such as a raw consumer in
`kafka-backfill`. The rules above cover the pattern found in `rhino-service`
and should be checked against two more services before they are trusted.

### Stage 5: verify

`scripts/coverage_report.py` (the audit used for this doc, made permanent):

- per service: files on disk vs nodes, files with text, percent of code
  lines covered, symbol counts by `kind`, edge counts by type
- unresolved imports and references, top offenders listed
- exits non-zero if any file has no node or coverage is below the target

## Incremental sync (GitHub mode)

Per changed file, re-run Stages 0–3 for that file. Then re-run Stage 3 for
the files that import it (a renamed export changes their edges), and Stage 4
for the service. Two existing bugs get fixed:

- Incremental sync never writes `File.source`, so an edited test file keeps
  its old text (`SET f += row` leaves it untouched).
- A changed file with no symbols has its chunks pruned and never rebuilt.

## Phases

Each phase ends with a `PIPELINE_VERSION` bump and a re-ingest. The
embedding cache arrives in Phase 1, so every re-ingest after that is cheap.

| Phase | Scope | Done when |
|---|---|---|
| **1. Storage** | Stage 0 complete (every file a node, full text, `kind`, `parse_status`), embedding cache, incremental bug fixes, coverage report script | 21,377 nodes (0 missing). All ~21,300 text files have `source`. Second re-ingest makes ~0 embedding calls. |
| **2. Symbols and chunks** | All "Add" rows in the language table except SQL. Nested `CONTAINS` with unique names. Snippet from the declaration. Exact-text chunks, sizes, region chunks. Generated files excluded from search. | Line coverage ≥ 99% for every service (from 80.4%). No chunk over the size limit. `ActionButton` and `updateArchiveButton` exist as symbols. |
| **3. In-service links** | `IMPORTS`, `REFERENCES`, `EXTENDS`/`IMPLEMENTS`, `TESTS` | `ReportsListScreen/view.tsx -IMPORTS-> InfiniteScrollGrid/view.tsx` and `types.ts`. Under 2% of local imports unresolved. |
| **4. Database** | Liquibase/SQL to `Table`, entities `MAPS_TO`, then `READS`/`WRITES` | Every service with a `liquibase/` folder has `Table` nodes. Spot-check 3 tables against the live schema. |
| **5. Cross-service** | `Operation`, `Endpoint`, `Message`, `Topic`, `CALLS {via}` | Every existing text-scan `CALLS` edge is either explained by a concrete link or kept as `text-scan`. In `rhino-service`, every class under `kafka/handlers` has a `HANDLES` edge and every class under `kafka/publishers` has a `PUBLISHES` edge. New edges reviewed by hand. |

**Required follow-on:** the planner must use this graph. See
`PLANNER_V2_PLAN.md`. Until it does, Phases 1–3 improve the graph but not
the TDD.

## Tests

- **Fixture:** a copy of `InfiniteScrollGrid/` (`view.tsx`, `types.ts`,
  `styles.ts`, `index.ts`) plus a minimal `ReportsListScreen`. Assert the
  full text is stored, `ActionButton` exists with `kind: type`, the
  screen's import resolves through `index.ts` to `view.tsx`, and no chunk
  exceeds the limit.
- Per-language fixtures for every new symbol kind.
- A Liquibase fixture: a changelog with `CREATE TABLE` then `ALTER TABLE ADD
  COLUMN` produces one table with both columns.
- Each phase: run the coverage report before and after on all 13 services,
  and keep the outputs with the PR.
- Re-run the Archive-button ticket after Phase 3 plus the planner change.
  The TDD must name `InfiniteScrollGrid/view.tsx` and `types.ts`.

## Costs and risks

- **Neo4j size:** the full text of ~21k files, roughly 1.7M code lines, is
  tens of MB. Fine for Neo4j. `source_snippet` becomes redundant and can be
  dropped once the planner reads `File.source`.
- **First re-ingest** after Phase 1 pays for all embeddings once (reports-api
  is 40k chunks). After that the cache makes it cheap.
- **`REFERENCES` volume** could be large in ui. Write only resolved edges,
  batch them, and measure after the first run.
- **Kafka rules are based on one service.** They come from reading
  `rhino-service`. Check them against two more services and the
  `kafka-spring` library before building (see the Kafka design).

## As built

Where the code is, by stage:

| Stage | Module |
|---|---|
| 0 Discover | `src/ingestion/file_inventory.py` |
| 1 Parse | `src/ingestion/code_parser.py`, `src/ingestion/code_facts.py` |
| 2 Chunk/embed | `code_parser.assign_chunks`, `ingest._embed_chunks` (with the embedding cache) |
| 3 Link (in) | `src/ingestion/links.py`, `src/ingestion/import_resolver.py`, `src/ingestion/db_schema.py` |
| 4 Link (out) | `src/ingestion/service_links.py` |
| 5 Verify | `scripts/coverage_report.py` (`--local` is a dry run that writes and embeds nothing) |

Where it differs from the plan above, and why:

- **No `sqlglot`.** `db_schema.py` is a small reader for the DDL migrations
  actually use (CREATE/ALTER/DROP TABLE, column changes, renames, views)
  plus Liquibase's XML/YAML changelogs and structured changes. It avoids a
  new dependency in the published agent, and on reports-api's 489
  migrations it rebuilds 239 tables; what it skips is data changes, indexes,
  enum types and `DO` blocks, never a table's columns.
- **Test files are stored, parsed and linked, but not embedded.** The
  planner's search excludes test code by design, so those vectors were paid
  for and never read. Tests reach the planner through `File.source` (capped
  to 6,000 characters at read time) and `TESTS` edges.
- **Which symbols own chunks.** Functions always own their lines. Classes
  own what their methods leave (fields, interface members) when it is more
  than scaffolding. Constants own their lines from 200 characters up. A
  GraphQL field, an Avro field and a BPMN task fold into their type or
  process: a line each, searched as part of the whole.
- **Generic Kafka producers.** A producer typed only `SpecificRecord` names
  its message where it builds it (`X.newBuilder()`), but it builds its
  payload records the same way. Those are linked only to records that are
  messages by this codebase's Avro convention (record `X` in protocol
  `XProtocol`) or that some handler consumes.
- **Kotlin `const val` inside an `object`** is read from text: the bundled
  tree-sitter grammar returns an ERROR node for it.
- **An ambiguous GraphQL match** (one operation exposed by two services) is
  recorded on the client file as `USES_OPERATION {ambiguous: true}` but
  never becomes a `CALLS` edge.
- **Stage 4 runs over the full manifest on every run**, not just the
  services that changed, so a link from an unchanged service to a changed
  one moves too.
- **Planner compatibility.** Three read-only changes in
  `src/query/graph_reader.py` keep today's planner working on the new graph:
  a chunk reports its owner from its own `function_qualified_name`, symbol
  lookups include `Variable`, and the read-time 6,000-character cap on test
  text. The planner itself is unchanged; see `PLANNER_V2_PLAN.md`.
