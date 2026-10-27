"""Planner v2: which code the model is shown, as whole files under one budget.

v1 chose code per symbol: every matched function, up to 20 per service, each
whole up to 1,500 characters or a per-service 20,000 budget and otherwise as
one ~700-character chunk, each method widened to its whole class with no
overall cap. Measured on the Archive-button ticket it sent 1.27 million
characters (~276k tokens) across eight services, most of them unrelated —
and still not the two files the fix was in, because nothing linked the
screen to the grid component it renders.

v2 gives every file that might matter ONE score and packs the best first:

  1. Starting files. The ticket is split into parts, and each part's best
     matching files are starting points — so a ticket describing changes in
     four services finds a starting file for each. The whole ticket's best
     are the main ones; a part's best counts only if it connects to a main
     one, and is otherwise shown by its symbols with nothing followed.
  2. Candidates. Everything linked to a starting file: the other end of each
     Kafka message, GraphQL operation and REST route in another service, the
     tables it touches, the files it imports and the files of those imports'
     own folders, the files it references, and its tests.
  3. One score each:  how closely the file matches the ticket (its best chunk
     against any part of it)  x  how it was found (see HOW_FOUND)  x  how
     strong the file it was reached from is, next to the best starting file
     — so the imports of the strongest match outrank those of a weak one.
  4. Packing. Each service with a counted starting file is guaranteed a
     share of the budget, then files go in best-first. A file that does not fit whole is
     shown PARTIAL (its opening and matched symbols) if it was a starting or
     linked file, otherwise by its symbols only (OUTLINE); once even that
     does not fit, the rest are left out.

There are no per-type counts. An earlier version capped imports, links,
references and tests by count, and every one of those numbers was tuned on
a single ticket: a bug spanning four services could have its fourth
service's handler cut while a weak import of the first kept its place.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from src.ingestion.test_paths import is_test_path

from . import graph_reader

# ~60k tokens of code. The system prompt is ~4k and the service summaries a
# few thousand more, so a prompt stays well inside every model's window.
CONTEXT_BUDGET_CHARS = 240_000

# How a file was reached, as a multiplier on how well it matches the ticket.
# A file the ticket matches directly counts most; one reached through a
# concrete link to another service nearly as much, since that link is
# evidence the change crosses it; a file merely imported or used, less.
HOW_FOUND = {
    "focus": 1.0,  # a starting file: matched directly by a part of the ticket
    "linked": 0.9,  # the other end of a Kafka message, GraphQL operation or route
    "table": 0.85,  # an entity mapped to a table a starting file uses
    "import": 0.8,  # imported by a starting file
    "test": 0.7,  # a test of a starting file
    # A file of an imported component's own folder (its styles, types,
    # hooks) IS that component, so it gets the component's score, not its
    # own: a styles file is CSS and barely resembles a ticket written in
    # words, yet it held the wrapper whose wrapping was the bug.
    "folder": 1.0,
    "reference": 0.6,  # used by a starting file without an import
}
# Starting files are not a fixed number. For the whole ticket and for each
# part of it, the files whose scores stand clearly above the rest are taken:
# the ranked list is cut at its biggest drop. Most files score in a dense
# band of loosely related code (0.70-0.74 on the Archive-button ticket) and
# the relevant ones sit above it; the biggest drop at the top is where one
# ends and the other begins. A fixed "best two" took 0.79 for the screen
# with the bug and then 0.74 for an unrelated wizard step — no closer to
# the ticket than a dozen other files — whose imports took ~50k characters.
#
# The drop is looked for among each list's top SEED_WINDOW scores, and only
# counts when it is clearly bigger than the typical step between them
# (SEED_GAP_FACTOR x the median step); when nothing stands out, only the
# best file is taken.
SEED_WINDOW = 5
SEED_GAP_FACTOR = 2.0
# Half the budget is split evenly between the services that have a starting
# file, so one large service cannot use it all; the other half goes to
# whichever files score best.
SERVICE_SHARE = 0.5
# No single file may take more than this share of the budget whole.
MAX_FILE_SHARE = 0.25
# A test is shown up to its setup, which is what a new case copies.
TEST_SOURCE_CHARS = 6000
# Candidates loaded at most — a ceiling on reading from the graph, far above
# what the budget can hold, not a choice of what is shown.
_MAX_CANDIDATES = 400
# A partial file: its opening (imports), and each matched symbol's lines
# with this much around them.
_PARTIAL_HEAD_LINES = 40
_PARTIAL_CONTEXT_LINES = 10

_DISPLAY_ORDER = ("focus", "linked", "import", "reference", "test")

# Files whose own folder is followed as part of them: UI components. A file
# that only collects others (an index.ts of GraphQL queries, a constants.ts)
# is not a component, and following its folder once brought in 30k of
# unrelated query definitions at a component's score.
_COMPONENT_SUFFIXES = (".tsx", ".jsx")


def _is_component_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return name.endswith(_COMPONENT_SUFFIXES) and not name.startswith("index.")


@dataclass
class ShownFile:
    service: str
    path: str
    tier: str  # focus | linked | import | reference | test  (how it is grouped)
    reason: str
    status: str = "WHOLE"  # WHOLE | PARTIAL | OUTLINE
    text: str = ""
    line_count: int = 0
    kind: str = "source"
    symbols: list[dict] = field(default_factory=list)
    matched: list[str] = field(default_factory=list)
    tested_by: list[str] = field(default_factory=list)
    new_test_path: str | None = None
    shown_lines: list[tuple[int, int]] = field(default_factory=list)
    priority: float = 0.0
    # Who imports this file: {"count", "files"} — what a change to a shared
    # file reaches.
    used_by: dict = field(default_factory=dict)
    # Why an OUTLINE file's code is not shown.
    outline_note: str = "the budget was spent"

    @property
    def key(self) -> tuple[str, str]:
        return (self.service, self.path)


@dataclass
class Context:
    files: list[ShownFile] = field(default_factory=list)
    tables: list[dict] = field(default_factory=list)
    budget: int = CONTEXT_BUDGET_CHARS
    used: int = 0
    # Every file considered and left out, best first: (service, path, tier,
    # score) — for the debug log, so "why wasn't X shown?" has an answer.
    left_out: list[tuple[str, str, str, float]] = field(default_factory=list)

    def services(self) -> list[str]:
        return list(dict.fromkeys(f.service for f in self.files if f.status != "OUTLINE"))

    def shown_files_by_service(self) -> dict[str, set[str]]:
        shown: dict[str, set[str]] = {}
        for f in self.files:
            if f.status != "OUTLINE":
                shown.setdefault(f.service, set()).add(f.path)
        return shown

    def symbol_names_by_service(self) -> dict[str, set[str]]:
        """Names the model may cite, as it sees them in a file's outline."""
        names: dict[str, set[str]] = {}
        for f in self.files:
            if f.status == "OUTLINE":
                continue
            bucket = names.setdefault(f.service, set())
            for s in f.symbols:
                bucket.add(s["name"])
                bucket.add(s["qualified_name"].split("::", 1)[-1])
        return names

    @property
    def has_tests(self) -> bool:
        return any(f.tier == "test" and f.text for f in self.files)


def _file_of(qualified_name: str) -> str:
    return qualified_name.split("::", 1)[0]


def score_files(ranked_by_service: dict[str, list[dict]]) -> list[dict]:
    """Search hits rolled up to files: [{service, path, score, matched}],
    best first, test files excluded. `score` is the file's best chunk."""
    files: dict[tuple[str, str], dict] = {}
    for service, hits in ranked_by_service.items():
        for hit in hits:
            path = _file_of(hit["function_qualified_name"])
            if is_test_path(path):
                continue
            entry = files.setdefault(
                (service, path), {"service": service, "path": path, "score": 0.0, "matched": []}
            )
            entry["score"] = max(entry["score"], hit["score"])
            name = hit["function_qualified_name"].split("::", 1)[-1]
            if name != path and name not in entry["matched"]:
                entry["matched"].append(name)
    return sorted(files.values(), key=lambda e: (-e["score"], e["service"], e["path"]))


def standing_out(scores: list[float]) -> int:
    """How many of the best-first `scores` stand clearly above the rest:
    cut at the biggest drop among the top SEED_WINDOW, if that drop is at
    least SEED_GAP_FACTOR times the median step; otherwise 1."""
    top = scores[:SEED_WINDOW]
    if len(top) < 2:
        return len(top)
    steps = [a - b for a, b in zip(top, top[1:], strict=False)]
    biggest = max(range(len(steps)), key=lambda i: steps[i])
    typical = sorted(steps)[len(steps) // 2]
    if steps[biggest] <= 0 or steps[biggest] < SEED_GAP_FACTOR * typical:
        return 1
    return biggest + 1


def vague_parts(part_bests: list[float]) -> set[int]:
    """Parts of the ticket whose best match is far below every other part's
    best — a sentence too vague to point at any code ("…could cause
    mis-clicks"), whose "best" file is close to random. The same cut as
    standing_out, across the parts' best scores. Part 0, the whole ticket,
    is never vague; with fewer than three parts there is no typical step to
    compare against, so none is."""
    if len(part_bests) < 3:
        return set()
    order = sorted(range(len(part_bests)), key=lambda i: -part_bests[i])
    ranked = [part_bests[i] for i in order]
    steps = [a - b for a, b in zip(ranked, ranked[1:], strict=False)]
    biggest = max(range(len(steps)), key=lambda i: steps[i])
    typical = sorted(steps)[len(steps) // 2]
    if steps[biggest] <= 0 or steps[biggest] < SEED_GAP_FACTOR * typical:
        return set()
    return {i for i in order[biggest + 1 :] if i != 0}


def pick_seeds(
    scored: list[dict], per_part: dict[tuple[str, str], list[float]]
) -> list[dict]:
    """The starting files, as [{..., "priority", "part", "main"}]: for the
    whole ticket (part 0) and each part of it, the files that stand clearly
    above the rest (see standing_out). Those of the whole ticket are the
    MAIN starting files; a part's are extras, which build_context keeps only
    if they connect to a main one. Without per-part scores, the same rule
    over the overall scores, and every pick is main."""
    if not scored:
        return []
    by_key = {(e["service"], e["path"]): e for e in scored}
    parts = max((len(v) for v in per_part.values()), default=0)
    if parts == 0:
        take = standing_out([e["score"] for e in scored])
        return [{**e, "priority": e["score"], "part": 0, "main": True} for e in scored[:take]]
    def part_score(k: tuple[str, str], part: int) -> float:
        return per_part[k][part] if k in per_part else by_key[k]["score"]

    skip = vague_parts([max(part_score(k, part) for k in by_key) for part in range(parts)])
    seeds: dict[tuple[str, str], dict] = {}
    for part in range(parts):
        if part in skip:
            continue
        ranked = sorted(by_key, key=lambda k: -part_score(k, part))
        for key in ranked[: standing_out([part_score(k, part) for k in ranked])]:
            if key not in seeds:
                score = max(per_part.get(key, [by_key[key]["score"]]))
                seeds[key] = {**by_key[key], "priority": score, "part": part, "main": part == 0}
    return sorted(seeds.values(), key=lambda e: -e["priority"])


def _partial(
    text: str, symbols: list[dict], matched: list[str]
) -> tuple[str, list[tuple[int, int]]]:
    """The opening of the file and each matched symbol, with a little
    around it — what a model needs to change a file it cannot see whole."""
    lines = text.split("\n")
    wanted = {s["qualified_name"].split("::", 1)[-1] for s in symbols} & set(matched)
    ranges = [(1, min(_PARTIAL_HEAD_LINES, len(lines)))]
    for s in symbols:
        if s["qualified_name"].split("::", 1)[-1] in wanted and s.get("line_start"):
            ranges.append((
                max(1, s["line_start"] - _PARTIAL_CONTEXT_LINES),
                min(len(lines), (s.get("line_end") or s["line_start"]) + _PARTIAL_CONTEXT_LINES),
            ))
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    parts = [f"// … lines {a}–{b} …\n" + "\n".join(lines[a - 1 : b]) for a, b in merged]
    return "\n".join(parts), merged


def _outline_cost(symbols: list[dict]) -> int:
    """Roughly what a file's symbol list costs in the prompt."""
    return 120 + sum(len(s["qualified_name"].split("::", 1)[-1]) + 20 for s in symbols[:60])


def build_context(
    ranked_by_service: dict[str, list[dict]],
    services_in_run: list[str] | None,
    budget: int = CONTEXT_BUDGET_CHARS,
    reader=graph_reader,
    vectors: list[list[float]] | None = None,
) -> Context:
    """Choose and load the code for one ticket. `ranked_by_service` is the
    per-service search result query_flow already computed (rank_functions
    rows); `services_in_run` limits files to the services of this run (None:
    any service in the graph); `vectors` are the ticket's embeddings — the
    whole ticket first, then each of its parts."""
    context = Context(budget=budget)
    scored = score_files(ranked_by_service)
    if not scored:
        return context
    vectors = vectors or []
    per_part = (
        reader.file_relevance_by_vector([(e["service"], e["path"]) for e in scored], vectors)
        if vectors else {}
    )
    seeds = pick_seeds(scored, per_part)
    allowed = set(services_in_run) if services_in_run is not None else None

    candidates: dict[tuple[str, str], ShownFile] = {}
    for seed in seeds:
        part = "the whole issue" if seed["part"] == 0 else f"part {seed['part']} of the issue"
        candidates[(seed["service"], seed["path"])] = ShownFile(
            service=seed["service"], path=seed["path"], tier="focus",
            reason=f"best match for {part} (score {seed['priority']:.2f}"
            + (f"; matched {', '.join(seed['matched'][:4])}" if seed["matched"] else "") + ")",
            matched=seed["matched"], priority=seed["priority"],
        )
    seed_pairs = list(candidates)
    seed_priority = {k: c.priority for k, c in candidates.items()}
    main = {(s["service"], s["path"]) for s in seeds if s["main"]}
    best_seed = next(k for k in seed_pairs if k in main)

    # How each other file was reached: (key, how, reason, the file it was
    # reached from).
    reached: list[tuple[tuple[str, str], str, str, tuple[str, str]]] = []

    def reach(service: str, path: str, how: str, reason: str, parent: tuple[str, str]) -> None:
        if not path or (is_test_path(path) and how != "test"):
            return
        if allowed is not None and service not in allowed:
            return
        reached.append(((service, path), how, reason, parent))

    for link in reader.cross_service_links(seed_pairs):
        reach(link["service"], link["file_path"], "linked",
              f"{link['reason']} — linked from {link['from_file']}",
              (link["from_service"], link["from_file"]))

    tables = reader.tables_for(seed_pairs)

    imports = [
        row for row in reader.imports_of(seed_pairs)
        if row.get("kind") in (None, "source", "schema")
    ]
    for row in imports:
        names = ", ".join(row["names"][:6])
        reach(row["service"], row["target"], "import",
              f"imported by {row['source']}" + (f" ({names})" if names else ""),
              (row["service"], row["source"]))

    # An imported component's own folder: its styles, types and hooks.
    import_parent = {(r["service"], r["target"]): (r["service"], r["source"]) for r in imports}
    folder_parent: dict[tuple[str, str], tuple[str, str]] = {}
    components = sorted(k for k in import_parent if _is_component_file(k[1]))
    for row in reader.imports_of(components):
        same_folder = row["target"].rsplit("/", 1)[0] == row["source"].rsplit("/", 1)[0]
        if same_folder and row.get("kind") in (None, "source", "schema"):
            names = ", ".join(row["names"][:6])
            folder_parent.setdefault((row["service"], row["target"]), (row["service"], row["source"]))
            reach(row["service"], row["target"], "folder",
                  f"imported by {row['source']}" + (f" ({names})" if names else ""),
                  import_parent.get((row["service"], row["source"]), best_seed))

    for row in reader.references_from(seed_pairs):
        reach(row["service"], row["target"], "reference",
              f"used by {row['source']} ({', '.join(row['names'][:5])})",
              (row["service"], row["source"]))

    # An extra starting file (the best match for one part of the ticket
    # only) counts when it connects to a main one: either reaches the other
    # by an import, reference or cross-service link, or both use one table.
    # A part is a short sentence ("…when report selected"), and its best
    # match alone is often a screen that merely shares its words: five such
    # files once brought ~120k characters of wizard forms and modals, and
    # pushed out the grid the bug was in. Unconnected, it is shown by its
    # symbols only, and nothing is followed from it.
    def table_users(table: dict) -> set[str]:
        return {use.rsplit(" ", 1)[0] for use in table.get("used_by") or []}

    main_paths = {path for _, path in main}
    connected = {
        key for key in seed_pairs if key not in main and (
            any((k == key and parent in main) or (parent == key and k in main)
                for k, _, _, parent in reached)
            or any(key[1] in table_users(t) and table_users(t) & main_paths for t in tables)
        )
    }
    strong = main | connected
    weak = set(seed_pairs) - strong
    for key in weak:
        candidates[key].reason += "; nothing links it to the best match for the whole issue"
        candidates[key].outline_note = (
            "it matched only one part of the issue, and nothing links it to the best match"
        )
    reached = [r for r in reached if r[3] in strong]
    strong_paths = {path for _, path in strong}
    context.tables = [t for t in tables if table_users(t) & strong_paths]
    for table in context.tables:
        for entity in table.get("entity_files") or []:
            reach(table["service"], entity, "table", f"entity mapped to table {table['name']}",
                  best_seed)

    tests = reader.tests_of(sorted(strong))
    conventions = reader.test_naming_conventions(sorted({s for s, _ in strong}))
    for key in [k for k in seed_pairs if k in strong]:
        candidate = candidates[key]
        # The test named after the file first (view.tsx -> view.test.tsx).
        stem = candidate.path.rsplit(".", 1)[0] + "."
        candidate.tested_by = sorted(
            tests.get(key, []), key=lambda t: (not t.startswith(stem), len(t), t)
        )
        if not candidate.tested_by and conventions.get(candidate.service):
            candidate.new_test_path = reader.new_test_path(
                candidate.path, conventions[candidate.service]
            )
        # Every test of a file gets the same score, and packing keeps equal
        # scores in this order, so the one named after the file goes first.
        for test_path in candidate.tested_by:
            reach(candidate.service, test_path, "test", f"tests {candidate.path}", key)

    # One score per file: its match to the ticket x how it was found. A file
    # with no chunks of its own (a test, a barrel, generated code) borrows
    # the score of the file it was reached from.
    relevance: dict[tuple[str, str], float] = {}
    if vectors and reached:
        found = reader.file_relevance_by_vector(sorted({key for key, *_ in reached}), vectors)
        relevance = {k: max(v) for k, v in found.items() if v}
    best_score = max(seed_priority[k] for k in strong)

    def place(key, how, reason, priority) -> None:
        current = candidates.get(key)
        if current is not None and (current.tier == "focus" or current.priority >= priority):
            return
        tier = {"table": "linked", "folder": "import"}.get(how, how)
        candidates[key] = ShownFile(
            service=key[0], path=key[1], tier=tier, reason=reason, priority=priority
        )

    for key, how, reason, parent in reached:
        if how == "folder":
            continue
        parent_score = seed_priority.get(parent, 0.0)
        own = relevance.get(key) if how != "test" else None
        standing = parent_score / best_score if best_score else 1.0
        place(key, how, reason, (own if own is not None else parent_score) * HOW_FOUND[how] * standing)
    # Then each component's own files, from the score its import just got.
    for key, how, reason, _ in reached:
        if how == "folder":
            component = folder_parent.get(key)
            parent = candidates.get(component) if component else None
            if parent is not None:
                place(key, how, reason, parent.priority * HOW_FOUND["folder"])

    ordered = sorted(candidates.values(), key=lambda c: -c.priority)[:_MAX_CANDIDATES]
    sources = reader.load_file_sources([c.key for c in ordered])
    symbols = reader.load_file_symbols([c.key for c in ordered])

    seed_services = list(dict.fromkeys(key[0] for key in seed_pairs if key in strong))
    reserved = {s: budget * SERVICE_SHARE / len(seed_services) for s in seed_services}
    # A share is held only while its service has files still to place: one
    # held for a service whose only file was 1k once kept 59k of the budget
    # unused while 145 files were left out.
    pending = Counter(c.service for c in ordered)
    remaining = budget
    for candidate in ordered:
        pending[candidate.service] -= 1
        for service in reserved:
            if pending[service] <= 0 and service != candidate.service:
                reserved[service] = 0.0
        loaded = sources.get(candidate.key)
        if loaded is None:
            continue  # named by a link but not in the graph — never invented
        text = loaded["source"] or ""
        candidate.kind = loaded.get("kind") or "source"
        candidate.line_count = loaded.get("line_count") or text.count("\n") + 1
        candidate.symbols = symbols.get(candidate.key, [])
        if candidate.tier == "test" and len(text) > TEST_SOURCE_CHARS:
            text = text[:TEST_SOURCE_CHARS]
            candidate.status = "PARTIAL"
            candidate.shown_lines = [(1, text.count("\n") + 1)]
        # What this file may use: what is left, minus what is still held for
        # OTHER services' fair shares.
        held_for_others = sum(v for s, v in reserved.items() if s != candidate.service)
        available = remaining - held_for_others
        whole_ok = len(text) <= budget * MAX_FILE_SHARE and not loaded.get("truncated")
        if candidate.key in weak:
            candidate.status = "OUTLINE"
        elif whole_ok and len(text) <= available:
            candidate.text = text
        elif candidate.tier in ("focus", "linked") and available > 2000:
            partial, lines = _partial(text, candidate.symbols, candidate.matched)
            if len(partial) <= min(available, budget * MAX_FILE_SHARE):
                candidate.text, candidate.shown_lines = partial, lines
                candidate.status = "PARTIAL"
            else:
                candidate.status = "OUTLINE"
        else:
            candidate.status = "OUTLINE"
        cost = len(candidate.text) if candidate.text else _outline_cost(candidate.symbols)
        if candidate.status == "OUTLINE" and cost > available:
            # Not even its symbols fit: left out.
            context.left_out.append(
                (candidate.service, candidate.path, candidate.tier, candidate.priority)
            )
            continue
        remaining -= cost
        if candidate.service in reserved:
            reserved[candidate.service] = max(0.0, reserved[candidate.service] - cost)
        context.files.append(candidate)

    shown = [f.key for f in context.files if f.tier != "test" and f.status != "OUTLINE"]
    for key, users in reader.importers_of(shown).items():
        for f in context.files:
            if f.key == key:
                f.used_by = users

    order = {tier: i for i, tier in enumerate(_DISPLAY_ORDER)}
    context.files.sort(key=lambda f: (order.get(f.tier, 9), -f.priority))
    context.used = budget - remaining
    return context


_QUOTED_VALUE = re.compile(r"[\"'“‘`][^\"'”’`]{1,60}[\"'”’`]")


def names_a_value(issue_text: str, identifiers: list[str]) -> bool:
    """Whether the ticket names a specific value or field — the trigger for
    the value-trace section of the system prompt."""
    return bool(identifiers) or bool(_QUOTED_VALUE.search(issue_text))
