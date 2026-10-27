from src.ingestion.code_parser import ImportSpec
from src.ingestion.import_resolver import FileImports, ImportResolver


def _ts(path, imports=(), top_level=()):
    return FileImports(path, "typescript", list(imports), top_level=set(top_level))


def test_relative_index_and_extension_resolution():
    files = {
        "src/a.ts": _ts("src/a.ts", [ImportSpec("./b"), ImportSpec("./dir"), ImportSpec("../x")]),
        "src/b.tsx": _ts("src/b.tsx"),
        "src/dir/index.ts": _ts("src/dir/index.ts"),
    }
    resolver = ImportResolver(files)

    assert {(e.source, e.target) for e in resolver.resolve_all()} == {
        ("src/a.ts", "src/b.tsx"),
        ("src/a.ts", "src/dir/index.ts"),
    }
    # Relative, so it looked local — and it is missing.
    assert resolver.unresolved == {"src/a.ts": ["../x"]}


def test_package_imports_are_external_not_unresolved():
    files = {"a.ts": _ts("a.ts", [ImportSpec("react", [("default", "React")])])}
    resolver = ImportResolver(files)

    assert resolver.resolve_all() == []
    assert resolver.unresolved == {}


def test_tsconfig_paths_with_comments_and_barrels():
    files = {
        "app/screen.tsx": _ts(
            "app/screen.tsx",
            [
                ImportSpec("@lib/grid", [("Grid", "Grid"), ("Props", "Props")]),
                ImportSpec("@lib/missing", [("X", "X")]),
            ],
        ),
        "lib/grid/index.ts": _ts(
            "lib/grid/index.ts",
            [
                ImportSpec("./view", [("Grid", "Grid")], reexport=True),
                ImportSpec("./types", reexport=True, wildcard=True),
            ],
        ),
        "lib/grid/view.tsx": _ts("lib/grid/view.tsx", top_level={"Grid"}),
        "lib/grid/types.ts": _ts("lib/grid/types.ts", top_level={"Props"}),
    }
    sources = {
        "tsconfig.json": """{
  // comments and trailing commas are allowed here
  "compilerOptions": { "baseUrl": ".", "paths": { "@lib/*": ["lib/*"], }, },
}""",
    }
    resolver = ImportResolver(files, sources)
    edges = {(e.source, e.target): e for e in resolver.resolve_all()}

    screen_view = edges[("app/screen.tsx", "lib/grid/view.tsx")]
    assert screen_view.names == ["Grid"] and screen_view.via == ["lib/grid/index.ts"]
    assert edges[("app/screen.tsx", "lib/grid/types.ts")].names == ["Props"]
    # The barrel's own re-exports are edges too.
    assert ("lib/grid/index.ts", "lib/grid/view.tsx") in edges
    assert resolver.unresolved == {"app/screen.tsx": ["@lib/missing"]}
    assert resolver.local_bindings("app/screen.tsx") == {
        "Grid": ("lib/grid/view.tsx", "Grid"),
        "Props": ("lib/grid/types.ts", "Props"),
    }


def test_barrel_cycles_terminate():
    files = {
        "a/index.ts": _ts("a/index.ts", [ImportSpec("../b", reexport=True, wildcard=True)]),
        "b/index.ts": _ts("b/index.ts", [ImportSpec("../a", reexport=True, wildcard=True)]),
        "c.ts": _ts("c.ts", [ImportSpec("./a", [("Nothing", "Nothing")])]),
    }
    edges = ImportResolver(files).resolve_all()

    assert ("c.ts", "a/index.ts") in {(e.source, e.target) for e in edges}


def test_jvm_imports_resolve_by_package_and_outer_class():
    files = {
        "src/A.kt": FileImports(
            "src/A.kt",
            "kotlin",
            [
                ImportSpec("com.asurint.rhino.model.Order", [("Order", "Order")]),
                ImportSpec("com.asurint.rhino.model.Order.Status", [("Status", "Status")]),
                ImportSpec("com.asurint.rhino.model.Gone", [("Gone", "Gone")]),
                ImportSpec("com.asurint.models.Shared", [("Shared", "Shared")]),
            ],
            package="com.asurint.rhino.api",
        ),
        "src/model/Order.kt": FileImports(
            "src/model/Order.kt",
            "kotlin",
            [],
            package="com.asurint.rhino.model",
            top_level={"Order"},
        ),
    }
    resolver = ImportResolver(files)
    targets = {e.target for e in resolver.resolve_all()}

    assert targets == {"src/model/Order.kt"}
    # Under this service's own package root and missing — reported. The
    # shared `com.asurint.models` library is simply external.
    assert resolver.unresolved == {"src/A.kt": ["com.asurint.rhino.model.Gone"]}


def test_python_relative_and_src_layout_imports():
    files = {
        "src/pkg/a.py": FileImports(
            "src/pkg/a.py",
            "python",
            [
                ImportSpec(".b", [("x", "x")]),
                ImportSpec("pkg.c", [("pkg.c", "pkg")]),
            ],
        ),
        "src/pkg/b.py": FileImports("src/pkg/b.py", "python", []),
        "src/pkg/c/__init__.py": FileImports("src/pkg/c/__init__.py", "python", []),
    }
    targets = {e.target for e in ImportResolver(files).resolve_all()}

    assert targets == {"src/pkg/b.py", "src/pkg/c/__init__.py"}
