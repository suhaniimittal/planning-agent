from src.query import context_v2

SCREEN = "src/apps/client/modules/Reports/pages/ReportsListScreen/view.tsx"
GRID = "src/apps/client/components/InfiniteScrollGrid/view.tsx"
GRID_TYPES = "src/apps/client/components/InfiniteScrollGrid/types.ts"
GRID_STYLES = "src/apps/client/components/InfiniteScrollGrid/styles.ts"
SCREEN_TEST = "src/apps/client/modules/Reports/pages/ReportsListScreen/view.test.tsx"


def _hit(qname, score):
    return {"function_qualified_name": qname, "score": score, "identifiers": []}


class FakeReader:
    """The graph_reader functions build_context uses, over a small graph.
    `relevance` is {(service, path): [score per ticket part]}."""

    def __init__(self, sources, relevance=None, symbols=None, imports=(), references=(),
                 links=(), tables=(), tests=None, conventions=None):
        self.sources = sources
        self.relevance = relevance or {}
        self.symbols = symbols or {}
        self._imports = list(imports)
        self._references = list(references)
        self._links = list(links)
        self._tables = list(tables)
        self._tests = tests or {}
        self._conventions = conventions or {}

    def file_relevance_by_vector(self, pairs, vectors):
        return {p: self.relevance[p] for p in pairs if p in self.relevance}

    def cross_service_links(self, pairs):
        return [link for link in self._links if (link["from_service"], link["from_file"]) in pairs]

    def tables_for(self, pairs):
        return self._tables

    def imports_of(self, pairs):
        return [r for r in self._imports if (r["service"], r["source"]) in pairs]

    def references_from(self, pairs):
        return [r for r in self._references if (r["service"], r["source"]) in pairs]

    def tests_of(self, pairs):
        return {k: v for k, v in self._tests.items() if k in pairs}

    def test_naming_conventions(self, services):
        return self._conventions

    def new_test_path(self, path, infix):
        return path.rsplit(".", 1)[0] + infix + path.rsplit(".", 1)[1]

    def load_file_sources(self, pairs):
        return {
            p: {"source": self.sources[p], "line_count": self.sources[p].count("\n") + 1,
                "kind": "test" if "test" in p[1] else "source", "truncated": False}
            for p in pairs if p in self.sources
        }

    def load_file_symbols(self, pairs):
        return {p: self.symbols.get(p, []) for p in pairs}

    def importers_of(self, pairs):
        found = {}
        for r in self._imports:
            key = (r["service"], r["target"])
            if key in pairs:
                entry = found.setdefault(key, {"count": 0, "files": []})
                entry["count"] += 1
                entry["files"].append(r["source"])
        return found


def _import(service, source, target, names=()):
    return {"service": service, "source": source, "target": target, "names": list(names),
            "kind": "source"}


def test_hits_roll_up_to_files_and_tests_never_compete():
    scored = context_v2.score_files({
        "ui": [
            _hit(f"{SCREEN}::ReportsListScreen.updateArchiveButton", 0.60),
            _hit(f"{SCREEN}::ReportsListScreen", 0.55),
            _hit(f"{SCREEN_TEST}::describe", 0.90),
            _hit(f"{GRID}::InfiniteScrollGrid", 0.50),
        ]
    })

    assert [s["path"] for s in scored] == [SCREEN, GRID]
    assert scored[0]["score"] == 0.60
    assert scored[0]["matched"] == ["ReportsListScreen.updateArchiveButton", "ReportsListScreen"]


def test_each_part_of_the_ticket_gets_its_own_starting_files():
    """A ticket describing a UI change and a backend change: the whole
    ticket's best file is the UI one, but part 2 finds the backend file."""
    scored = [
        {"service": "ui", "path": "a.tsx", "score": 0.84, "matched": []},
        {"service": "ui", "path": "b.tsx", "score": 0.83, "matched": []},
        {"service": "ui", "path": "c.tsx", "score": 0.82, "matched": []},
        {"service": "reports-api", "path": "R.kt", "score": 0.70, "matched": []},
    ]
    per_part = {
        ("ui", "a.tsx"): [0.84, 0.84, 0.40],
        ("ui", "b.tsx"): [0.83, 0.83, 0.40],
        ("ui", "c.tsx"): [0.82, 0.82, 0.40],
        ("reports-api", "R.kt"): [0.70, 0.30, 0.79],
    }

    seeds = context_v2.pick_seeds(scored, per_part)

    # The whole ticket: three UI files stand clearly above R.kt (0.82 -> 0.70).
    # Part 2: R.kt stands alone above the rest (0.79 -> 0.40).
    assert [(s["path"], s["part"]) for s in seeds] == [
        ("a.tsx", 0), ("b.tsx", 0), ("c.tsx", 0), ("R.kt", 2)
    ]
    assert seeds[-1]["priority"] == 0.79


def test_a_four_service_bug_keeps_every_services_file_over_weak_imports():
    """The case per-type counts got wrong: the fourth service's handler must
    beat a weak import of the first when space is short."""
    services = ("rhino-service", "reports-api", "compliance-service")
    links = [
        {"from_service": "orders-service", "from_file": "Publisher.kt", "service": s,
         "file_path": f"{s}/Handler.kt", "reason": "handles CancelOrder"}
        for s in services
    ]
    reader = FakeReader(
        sources={
            ("orders-service", "Publisher.kt"): "p" * 2000,
            **{(s, f"{s}/Handler.kt"): "h" * 2000 for s in services},
            **{("orders-service", f"util{i}.kt"): "u" * 2000 for i in range(10)},
        },
        relevance={
            ("orders-service", "Publisher.kt"): [0.80],
            **{(s, f"{s}/Handler.kt"): [0.75] for s in services},
            **{("orders-service", f"util{i}.kt"): [0.40] for i in range(10)},
        },
        links=links,
        imports=[_import("orders-service", "Publisher.kt", f"util{i}.kt") for i in range(10)],
    )

    context = context_v2.build_context(
        {"orders-service": [_hit("Publisher.kt::Publisher", 0.8)]}, None,
        budget=12_000, reader=reader, vectors=[[0.1]],
    )

    whole = [f.path for f in context.files if f.status == "WHOLE"]
    for service in services:
        assert f"{service}/Handler.kt" in whole
    assert sum(1 for p in whole if p.startswith("util")) < 10


def test_each_starting_service_keeps_a_share_of_the_budget():
    big_ui = {("ui", f"ui{i}.tsx"): "x" * 3000 for i in range(10)}
    reader = FakeReader(
        sources={**big_ui, ("reports-api", "R.kt"): "r" * 3000},
        # Part 2 of the ticket is about reports-api, so R.kt starts there,
        # and it serves a query the screen uses.
        relevance={**{k: [0.9, 0.3] for k in big_ui}, ("reports-api", "R.kt"): [0.5, 0.95]},
        imports=[_import("ui", "ui0.tsx", f"ui{i}.tsx") for i in range(1, 10)],
        links=[{"from_service": "ui", "from_file": "ui0.tsx", "service": "reports-api",
                "file_path": "R.kt", "reason": "serves Query.reports used here"}],
    )

    context = context_v2.build_context(
        {"ui": [_hit("ui0.tsx::A", 0.9)], "reports-api": [_hit("R.kt::R", 0.5)]},
        None, budget=12_000, reader=reader, vectors=[[0.1], [0.2]],
    )

    statuses = {f.path: f.status for f in context.files}
    assert statuses["R.kt"] == "WHOLE"


def test_the_screen_brings_in_the_grid_and_the_grids_own_folder():
    reader = FakeReader(
        sources={("ui", SCREEN): "screen", ("ui", GRID): "grid", ("ui", GRID_TYPES): "types",
                 ("ui", GRID_STYLES): "styles", ("ui", "src/common/far.ts"): "far",
                 ("ui", SCREEN_TEST): "test"},
        relevance={("ui", SCREEN): [0.8], ("ui", GRID): [0.67], ("ui", GRID_TYPES): [0.6],
                   ("ui", GRID_STYLES): [0.55], ("ui", "src/common/far.ts"): [0.9]},
        imports=[
            _import("ui", SCREEN, GRID, ["InfiniteScrollGrid"]),
            _import("ui", GRID, GRID_TYPES, ["GridProps"]),
            _import("ui", GRID, GRID_STYLES, ["SearchAndButtonWrap"]),
            # Outside the grid's own folder: not followed on the second hop.
            _import("ui", GRID, "src/common/far.ts"),
        ],
        tests={("ui", SCREEN): ["a/_experiment.test.tsx", SCREEN_TEST]},
    )

    context = context_v2.build_context(
        {"ui": [_hit(f"{SCREEN}::ReportsListScreen", 0.8)]}, ["ui"],
        reader=reader, vectors=[[0.1]],
    )

    paths = [f.path for f in context.files]
    assert paths[:2] == [SCREEN, GRID]
    assert {GRID_TYPES, GRID_STYLES} <= set(paths)
    priority = {f.path: f.priority for f in context.files}
    # The styles file barely matches the ticket (0.55) but is part of the grid.
    assert priority[GRID_STYLES] > 0.55 * context_v2.HOW_FOUND["import"]
    assert "src/common/far.ts" not in paths
    assert context.files[0].tested_by[0] == SCREEN_TEST
    assert context.has_tests


def test_a_kafka_link_brings_in_the_handler_but_only_from_services_in_the_run():
    reader = FakeReader(
        sources={("orders-service", "P.kt"): "publisher", ("rhino-service", "H.kt"): "handler",
                 ("models", "u.avdl"): "schema"},
        links=[
            {"from_service": "orders-service", "from_file": "P.kt", "service": "rhino-service",
             "file_path": "H.kt", "reason": "handles UpdateHighRisk"},
            {"from_service": "orders-service", "from_file": "P.kt", "service": "models",
             "file_path": "u.avdl", "reason": "Avro schema of UpdateHighRisk"},
            {"from_service": "orders-service", "from_file": "P.kt", "service": "not-in-run",
             "file_path": "X.kt", "reason": "handles UpdateHighRisk"},
        ],
    )

    context = context_v2.build_context(
        {"orders-service": [_hit("P.kt::Publisher", 0.7)]},
        ["orders-service", "rhino-service", "models"], reader=reader,
    )

    assert {(f.service, f.tier) for f in context.files} == {
        ("orders-service", "focus"), ("rhino-service", "linked"), ("models", "linked"),
    }


def test_a_file_over_a_quarter_of_the_budget_is_partial_and_the_rest_fall_back():
    big = "\n".join(f"line {i} " + "x" * 60 for i in range(1, 400))
    reader = FakeReader(
        sources={("ui", SCREEN): big, ("ui", GRID): "g" * 9000},
        symbols={("ui", SCREEN): [
            {"name": "updateArchiveButton",
             "qualified_name": f"{SCREEN}::ReportsListScreen.updateArchiveButton",
             "label": "Function", "symbol_kind": "function", "line_start": 300, "line_end": 310},
        ]},
        imports=[_import("ui", SCREEN, GRID)],
    )

    context = context_v2.build_context(
        {"ui": [_hit(f"{SCREEN}::ReportsListScreen.updateArchiveButton", 0.6)]},
        ["ui"], budget=24_000, reader=reader,
    )

    screen, grid = context.files
    assert screen.status == "PARTIAL"
    assert screen.shown_lines == [(1, 40), (290, 320)]
    assert "line 305 " in screen.text and "line 200 " not in screen.text
    assert grid.status == "OUTLINE" and grid.text == ""
    assert context.shown_files_by_service() == {"ui": {SCREEN}}


def test_a_long_test_is_cut_to_its_setup():
    reader = FakeReader(
        sources={("ui", SCREEN): "s",
                 ("ui", SCREEN_TEST): "t" * (context_v2.TEST_SOURCE_CHARS + 500)},
        tests={("ui", SCREEN): [SCREEN_TEST]},
    )

    context = context_v2.build_context({"ui": [_hit(f"{SCREEN}::S", 0.6)]}, ["ui"], reader=reader)

    test = context.files[-1]
    assert test.status == "PARTIAL" and len(test.text) == context_v2.TEST_SOURCE_CHARS


def test_a_starting_file_without_tests_is_told_where_one_would_go():
    reader = FakeReader(sources={("ui", SCREEN): "s"}, conventions={"ui": ".test."})

    context = context_v2.build_context({"ui": [_hit(f"{SCREEN}::S", 0.6)]}, ["ui"], reader=reader)

    assert context.files[0].new_test_path == SCREEN.replace(".tsx", ".test.tsx")


def test_value_trace_trigger():
    assert context_v2.names_a_value('Phone "555 1234" keeps its spaces', [])
    assert context_v2.names_a_value("x", ["isPhoneValid"])
    assert not context_v2.names_a_value("Controls shift when a report is selected", [])


def _two_starts(**extra):
    """strong.tsx is the whole ticket's best match; other.tsx starts only
    because it is part 2's best."""
    return FakeReader(
        sources={("ui", "strong.tsx"): "s", ("ui", "other.tsx"): "o" * 500,
                 ("ui", "strong_dep.ts"): "a" * 3000, ("ui", "other_dep.ts"): "b" * 3000},
        relevance={("ui", "strong.tsx"): [0.80, 0.30], ("ui", "other.tsx"): [0.60, 0.70],
                   ("ui", "strong_dep.ts"): [0.60, 0.20], ("ui", "other_dep.ts"): [0.65, 0.20]},
        imports=[_import("ui", "strong.tsx", "strong_dep.ts"),
                 _import("ui", "other.tsx", "other_dep.ts")] + extra.pop("imports", []),
        **extra,
    )


def _build(reader):
    return context_v2.build_context(
        {"ui": [_hit("strong.tsx::S", 0.8), _hit("other.tsx::O", 0.6)]}, ["ui"],
        reader=reader, vectors=[[0.1], [0.2]],
    )


def test_a_parts_match_that_nothing_links_to_the_main_one_is_only_outlined():
    """A part's best match that merely shares its words (a wizard step for
    "…when report selected") brought its imports and pushed out the grid the
    bug was in. Unconnected, it is listed by its symbols, nothing followed."""
    context = _build(_two_starts())

    files = {f.path: f for f in context.files}
    assert files["strong.tsx"].status == "WHOLE" and "strong_dep.ts" in files
    assert files["other.tsx"].status == "OUTLINE"
    assert "nothing links it" in files["other.tsx"].reason
    assert "other_dep.ts" not in files


def test_a_parts_match_the_main_one_imports_counts_in_full():
    context = _build(_two_starts(imports=[_import("ui", "strong.tsx", "other.tsx")]))

    files = {f.path: f for f in context.files}
    assert files["other.tsx"].status == "WHOLE"
    assert "other_dep.ts" in files


def test_a_parts_match_sharing_a_table_with_the_main_one_counts_in_full():
    table = {"service": "ui", "name": "reports", "columns": [], "foreign_keys": [],
             "defined_in": [], "used_by": ["strong.tsx reads", "other.tsx writes"],
             "entity_files": []}
    context = _build(_two_starts(tables=[table]))

    files = {f.path: f for f in context.files}
    assert files["other.tsx"].status == "WHOLE" and "other_dep.ts" in files
    assert [t["name"] for t in context.tables] == ["reports"]


def test_a_share_held_for_a_service_with_nothing_left_to_place_is_released():
    """A service whose only file is small must not keep half the budget
    held while files of the other service are left out."""
    ui = {("ui", f"ui{i}.tsx"): "x" * 3000 for i in range(4)}
    reader = FakeReader(
        sources={**ui, ("reports-api", "R.kt"): "r" * 100},
        relevance={**{k: [0.9] for k in ui}, ("reports-api", "R.kt"): [0.89]},
        imports=[_import("ui", "ui0.tsx", f"ui{i}.tsx") for i in range(1, 4)],
    )

    context = context_v2.build_context(
        {"ui": [_hit("ui0.tsx::A", 0.9), _hit("y.tsx::Y", 0.5), _hit("z.tsx::Z", 0.49)],
         "reports-api": [_hit("R.kt::R", 0.89)]},
        None, budget=12_400, reader=reader, vectors=[[0.1]],
    )

    # Both start (they stand together above y and z); R.kt costs 100,
    # and the ~6k held for reports-api comes back for ui's imports.
    assert {f.path: f.status for f in context.files} == {
        "ui0.tsx": "WHOLE", "R.kt": "WHOLE",
        "ui1.tsx": "WHOLE", "ui2.tsx": "WHOLE", "ui3.tsx": "WHOLE",
    }


def test_only_a_components_own_folder_is_followed():
    """A barrel or constants file is not a component: following its folder
    once brought in 30k of unrelated GraphQL query files."""
    reader = FakeReader(
        sources={("ui", SCREEN): "s", ("ui", GRID): "g", ("ui", GRID_STYLES): "st",
                 ("ui", "gql/Reports/index.ts"): "i", ("ui", "gql/Reports/Report.ts"): "r"},
        relevance={("ui", SCREEN): [0.8]},
        imports=[
            _import("ui", SCREEN, GRID), _import("ui", GRID, GRID_STYLES),
            _import("ui", SCREEN, "gql/Reports/index.ts"),
            _import("ui", "gql/Reports/index.ts", "gql/Reports/Report.ts"),
        ],
    )

    context = context_v2.build_context(
        {"ui": [_hit(f"{SCREEN}::S", 0.8)]}, ["ui"], reader=reader, vectors=[[0.1]]
    )

    paths = {f.path for f in context.files}
    assert GRID_STYLES in paths
    assert "gql/Reports/Report.ts" not in paths


def test_shown_files_say_who_imports_them():
    reader = FakeReader(
        sources={("ui", SCREEN): "s", ("ui", GRID): "g", ("ui", "Other.tsx"): "o"},
        imports=[_import("ui", SCREEN, GRID), _import("ui", "Other.tsx", GRID)],
    )

    context = context_v2.build_context({"ui": [_hit(f"{SCREEN}::S", 0.8)]}, ["ui"], reader=reader)

    grid = next(f for f in context.files if f.path == GRID)
    assert grid.used_by == {"count": 2, "files": [SCREEN, "Other.tsx"]}


def test_one_clear_winner_is_one_starting_file():
    """The Archive-button ticket's real scores: the screen with the bug, then
    a band of loosely related files."""
    assert context_v2.standing_out([0.79, 0.74, 0.74, 0.73, 0.73]) == 1


def test_files_standing_together_above_the_rest_are_all_taken():
    assert context_v2.standing_out([0.81, 0.80, 0.72, 0.71, 0.71]) == 2


def test_when_nothing_stands_out_only_the_best_is_taken():
    assert context_v2.standing_out([0.75, 0.745, 0.74, 0.735, 0.73]) == 1
    assert context_v2.standing_out([0.6]) == 1


def test_a_vague_part_of_the_ticket_starts_nothing():
    """The Archive-button ticket's parts: three point at the reports list
    (0.75-0.79); the fourth, "could cause mis-clicks", matches nothing well
    (0.66) and once made an unrelated wizard step a starting file."""
    assert context_v2.vague_parts([0.753, 0.791, 0.779, 0.658]) == {3}


def test_two_parts_are_both_kept_and_the_whole_ticket_never_dropped():
    assert context_v2.vague_parts([0.80, 0.62]) == set()
    assert context_v2.vague_parts([0.60, 0.80, 0.79, 0.78]) == set()
