"""What pipeline version 4 added to the parser: more kinds of symbol, nested
names that cannot collide, structured imports, identifiers, supertypes,
line-ownership chunks, and files that no longer fail."""

from src.ingestion.code_parser import (
    assign_chunks,
    parse_repo,
    parse_source,
    parse_source_full,
)


def _by_name(symbols):
    return {s.name: s for s in symbols}


# --- TypeScript / TSX -----------------------------------------------------------


def test_typescript_type_alias_enum_and_abstract_class_are_symbols():
    source = b"""
export type ActionButton = { id: string };
export enum Color { Red, Blue }
export abstract class Base extends Root implements Shape {}
"""
    symbols = _by_name(parse_source(source, "t.ts", "typescript"))

    assert symbols["ActionButton"].symbol_kind == "type"
    assert symbols["Color"].symbol_kind == "enum"
    assert symbols["Base"].symbol_kind == "class"
    assert symbols["Base"].supertypes == ["Root", "Shape"]


def test_typescript_top_level_constants_are_variables_but_functions_are_not():
    source = b"""
export const Wrap = styled.div`display: flex;`;
const MAX = 5;
export const handler = () => 1;
export const Memo = memo(() => null);
"""
    symbols = parse_source(source, "s.tsx", "tsx")
    kinds = {s.name: (s.kind, s.symbol_kind) for s in symbols}

    assert kinds["Wrap"] == ("variable", "const")
    assert kinds["MAX"] == ("variable", "const")
    assert kinds["handler"][0] == "function"
    assert kinds["Memo"][0] == "function"


def test_a_snippet_starts_at_the_declaration_not_after_the_name():
    """22% of single-chunk functions did not contain their own name: an
    arrow function's text began at its parameter list."""
    source = b"export const Grid = <A,>({ title }: P<A>) => {\n  return title;\n};\n"
    grid = parse_source(source, "g.tsx", "tsx")[0]

    assert grid.source_snippet.startswith("export const Grid =")
    assert grid.line_start == 1


def test_functions_passed_to_wrappers_take_the_const_name_and_nest():
    source = b"""
export const Screen = () => {
  const save = useCallback(() => { persist(); }, []);
  const total = useMemo(() => sum(rows), [rows]);
  useEffect(() => {
    const onResize = () => layout();
    window.addEventListener("resize", onResize);
  }, []);
  return null;
};
"""
    symbols = _by_name(parse_source(source, "s.tsx", "tsx"))

    assert symbols["save"].qualified_name == "s.tsx::Screen.save"
    assert symbols["total"].parent_symbol == "s.tsx::Screen"
    # Declared inside an anonymous callback: still found, still nested.
    assert symbols["onResize"].qualified_name == "s.tsx::Screen.onResize"


def test_nested_names_include_the_parent_so_they_cannot_collide():
    source = b"""
export const A = () => { const handleClick = () => 1; return handleClick; };
export const B = () => { const handleClick = () => 2; return handleClick; };
"""
    names = [s.qualified_name for s in parse_source(source, "x.tsx", "tsx")]

    assert "x.tsx::A.handleClick" in names
    assert "x.tsx::B.handleClick" in names


def test_a_repeated_name_in_one_scope_is_numbered_not_merged():
    source = b"function f() { return 1; }\nfunction f() { return 2; }\n"
    names = [s.qualified_name for s in parse_source(source, "f.js", "javascript")]

    assert names == ["f.js::f", "f.js::f#2"]


def test_test_blocks_are_symbols_named_by_their_title():
    source = b"""
describe("ReportsList", () => {
  beforeEach(() => { setup(); });
  it("renders the archive button", () => { expect(1).toBe(1); });
});
"""
    symbols = parse_source(source, "v.test.tsx", "tsx")
    by_qname = {s.qualified_name: s for s in symbols}

    suite = by_qname['v.test.tsx::describe("ReportsList")']
    assert suite.symbol_kind == "test"
    case = by_qname['v.test.tsx::describe("ReportsList").it("renders the archive button")']
    assert case.parent_symbol == suite.qualified_name
    assert 'v.test.tsx::describe("ReportsList").beforeEach' in by_qname


def test_identifiers_are_the_symbols_own_and_exclude_nested_code():
    source = b"""
export const Screen = () => {
  const update = useCallback(() => { computeArchiveState(); }, []);
  return <Grid onSelect={update} />;
};
"""
    symbols = _by_name(parse_source(source, "s.tsx", "tsx"))

    assert "Grid" in symbols["Screen"].identifiers
    assert "update" in symbols["Screen"].identifiers
    assert "computeArchiveState" not in symbols["Screen"].identifiers
    assert "computeArchiveState" in symbols["update"].identifiers


def test_typescript_imports_are_structured():
    source = b"""
import React, { useState as useS } from "react";
import * as api from "./api";
import "./side-effect.css";
export { Grid } from "./view";
export * from "./types";
const Lazy = React.lazy(() => import("./Lazy"));
"""
    imports = parse_source_full(source, "i.tsx", "tsx").imports
    by_module = {i.module: i for i in imports}

    assert by_module["react"].names == [("default", "React"), ("useState", "useS")]
    assert by_module["./api"].names == [("*", "api")]
    assert by_module["./side-effect.css"].names == []
    assert by_module["./view"].reexport and by_module["./view"].names == [("Grid", "Grid")]
    assert by_module["./types"].wildcard
    assert "./Lazy" in by_module


# --- Kotlin / Java / Python / GraphQL -------------------------------------------


def test_kotlin_objects_typealiases_enums_and_top_level_properties():
    source = b"""package com.a.b

import com.x.Y
import com.x.Z as W
import com.x.*

typealias Ids = List<String>
const val LIMIT = 5
object Registry { fun get() = 1 }
enum class Color { RED }
interface Repo
class Handler : Base(), Repo {
    companion object { fun create() = Handler() }
}
"""
    parsed = parse_source_full(source, "a/H.kt", "kotlin")
    kinds = {s.name: s.symbol_kind for s in parsed.symbols}

    assert kinds["Ids"] == "type"
    assert kinds["LIMIT"] == "const"
    assert kinds["Registry"] == "object"
    assert kinds["Color"] == "enum"
    assert kinds["Repo"] == "interface"
    assert kinds["Companion"] == "object"
    assert _by_name(parsed.symbols)["Handler"].supertypes == ["Base", "Repo"]
    assert parsed.package == "com.a.b"
    assert [(i.module, i.names, i.wildcard) for i in parsed.imports] == [
        ("com.x.Y", [("Y", "Y")], False),
        ("com.x.Z", [("Z", "W")], False),
        ("com.x", [], True),
    ]


def test_java_enums_records_and_constants():
    source = b"""package a.b;
import a.c.D;
public enum Status { OPEN; String label() { return "o"; } }
public record Point(int x) {}
public class K extends B implements I {
    public static final String TOPIC = "orders";
    private int count;
}
"""
    parsed = parse_source_full(source, "K.java", "java")
    symbols = _by_name(parsed.symbols)

    assert symbols["Status"].symbol_kind == "enum"
    assert symbols["label"].is_method
    assert symbols["Point"].symbol_kind == "record"
    assert symbols["TOPIC"].kind == "variable"
    assert symbols["TOPIC"].qualified_name == "K.java::K.TOPIC"
    assert "count" not in symbols
    assert symbols["K"].supertypes == ["B", "I"]
    assert parsed.package == "a.b"


def test_python_module_constants_and_relative_imports():
    parsed = parse_source_full(
        b"from .models import Order as O\nimport a.b as ab\nTIMEOUT = 30\n", "p/svc.py", "python"
    )

    assert _by_name(parsed.symbols)["TIMEOUT"].kind == "variable"
    assert [(i.module, i.names) for i in parsed.imports] == [
        (".models", [("Order", "O")]),
        ("a.b", [("a.b", "ab")]),
    ]


def test_graphql_extensions_scalars_unions_and_client_operations():
    source = b"""scalar Date
union Result = A | B
extend type Query { reports: [String] }
query GetReports { reports }
fragment F on Report { id }
"""
    kinds = {s.name: s.symbol_kind for s in parse_source(source, "s.graphql", "graphql")}

    assert kinds["Date"] == "scalar"
    assert kinds["Result"] == "union"
    assert kinds["Query"] == "extension"
    assert kinds["GetReports"] == "operation"
    assert kinds["F"] == "fragment"


def test_avro_records_know_their_namespace_and_protocol():
    source = b"""@namespace("com.asurint.models.orders")
protocol UpdateHighRiskProtocol {
    record UpdateHighRiskPayload { string reason; }
    record UpdateHighRisk { UpdateHighRiskPayload payload; }
}
"""
    symbols = _by_name(parse_source(source, "u.avdl", "avdl"))

    assert symbols["UpdateHighRisk"].facts["avro_namespace"] == "com.asurint.models.orders"
    assert symbols["UpdateHighRisk"].parent_symbol == "u.avdl::UpdateHighRiskProtocol"


# --- chunks and coverage --------------------------------------------------------


def test_every_line_is_owned_once_and_nested_symbols_take_their_lines_back():
    text = "\n".join(
        ["export const Screen = () => {"]
        + [f"  const v{i} = compute{i}(props.value{i});" for i in range(20)]
        + ["  const save = useCallback(() => {"]
        + [f"    persistField{i}(state.field{i});" for i in range(20)]
        + ["  }, []);", "  return null;", "};"]
    )
    parsed = parse_source_full(text.encode(), "s.tsx", "tsx")
    owned: dict[int, int] = {}
    for symbol in parsed.symbols:
        for chunk in symbol.chunks:
            for n in range(chunk.line_start, chunk.line_end + 1):
                owned[n] = owned.get(n, 0) + 1

    assert all(count == 1 for count in owned.values())
    save = next(s for s in parsed.symbols if s.name == "save")
    assert "persistField3" in "".join(c.text for c in save.chunks)


def test_region_chunks_cover_lines_outside_every_symbol():
    text = "import x from 'y';\n" + "".join(
        f"registerRoute('/path{i}', handlerNumber{i});\n" for i in range(40)
    )
    parsed = parse_source_full(text.encode(), "routes.ts", "typescript")

    joined = "".join(c.text for c in parsed.region_chunks)
    assert "registerRoute('/path33'" in joined
    assert "import x" not in joined  # the import block is stored, not embedded


def test_a_deeply_nested_file_parses_instead_of_failing(tmp_path):
    """The recursive walk hit Python's recursion limit on a 5,000-line string
    concatenation and lost 12 whole files across 13 services."""
    body = " +\n".join(f'"part {i}\\n"' for i in range(6000))
    (tmp_path / "Big.java").write_text(f"class Big {{\n  static final String Q = {body};\n}}\n")

    graph = parse_repo(tmp_path, "svc")

    assert graph.files_failed == 0
    assert graph.files["Big.java"].parse_status == "ok"
    assert any(c.name == "Big" for c in graph.classes)


def test_every_file_gets_a_record_with_its_kind(tmp_path):
    (tmp_path / "liquibase").mkdir()
    (tmp_path / "liquibase" / "001.sql").write_text("create table t (id int);\n")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x01")
    (tmp_path / "Jenkinsfile").write_text("pipeline {}\n")
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen" / "Api.kt").write_text(
        "// Code generated by openapi. DO NOT EDIT.\nclass Api\n"
    )

    graph = parse_repo(tmp_path, "svc")

    assert graph.files["liquibase/001.sql"].kind == "migration"
    assert graph.files["logo.png"].kind == "binary"
    assert "logo.png" not in graph.file_source
    assert graph.files["Jenkinsfile"].kind == "build"
    assert graph.files["gen/Api.kt"].generated
    assert not graph.files["gen/Api.kt"].searchable
    # Stored and parsed, but not embedded.
    assert all(not s.chunks for s in graph.symbols if s.file_path == "gen/Api.kt")


def test_coverage_counts_code_lines_inside_symbols_or_chunks(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n\nX = 2\n")

    graph = parse_repo(tmp_path, "svc")

    assert graph.file_coverage["a.py"] == (3, 3)


def test_assign_chunks_is_empty_for_a_tiny_barrel():
    assert assign_chunks('export * from "./view";\n', "index.ts", [], {1}) == []
