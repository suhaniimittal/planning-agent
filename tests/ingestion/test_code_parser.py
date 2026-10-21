from pathlib import Path

from src.ingestion import code_parser

from src.ingestion.code_parser import (
    _CHUNK_TARGET_CHARS,
    LANGUAGE_BY_EXTENSION,
    TRACKED_ONLY_EXTENSIONS,
    parse_file,
    parse_repo,
    parse_source,
)


def test_parse_source_extracts_class_and_functions_with_no_disk_file():
    source = (
        b'class OrderController:\n'
        b'    def create_order(self):\n'
        b'        return {}\n\n'
        b'def format_total(amount):\n'
        b'    return f"${amount}"\n'
    )
    symbols = parse_source(source, "controller.py", "python")
    by_name = {s.name: s for s in symbols}

    assert by_name["OrderController"].kind == "class"
    assert by_name["OrderController"].file_path == "controller.py"
    assert by_name["create_order"].is_method is True
    assert by_name["create_order"].file_path == "controller.py"
    assert by_name["format_total"].is_method is False


def test_parse_file_delegates_to_parse_source(tmp_path):
    """parse_file is now a thin wrapper: same result as calling parse_source
    directly with the file's bytes and repo-relative path."""
    src = tmp_path / "app.py"
    src.write_text("def handler():\n    pass\n")

    from_file = parse_file(src, tmp_path, "python")
    from_source = parse_source(src.read_bytes(), "app.py", "python")

    assert [s.qualified_name for s in from_file] == [s.qualified_name for s in from_source]


def test_parse_file_python_extracts_class_and_methods(tmp_path):
    src = tmp_path / "controller.py"
    src.write_text(
        'class OrderController:\n'
        '    """Handles order creation and lookup."""\n\n'
        '    def create_order(self, user_id, items):\n'
        '        """Creates a new order for the given user."""\n'
        '        return {"user_id": user_id, "items": items}\n\n\n'
        'def format_total(amount):\n'
        '    return f"${amount}"\n'
    )
    symbols = parse_file(src, tmp_path, "python")
    by_name = {s.name: s for s in symbols}

    cls = by_name["OrderController"]
    assert cls.kind == "class"
    assert cls.qualified_name == "controller.py::OrderController"
    assert cls.docstring == "Handles order creation and lookup."
    assert cls.file_path == "controller.py"

    method = by_name["create_order"]
    assert method.kind == "function"
    assert method.is_method is True
    assert method.parent_class == "controller.py::OrderController"
    assert method.qualified_name == "controller.py::OrderController.create_order"
    assert method.docstring == "Creates a new order for the given user."
    assert "return" in method.source_snippet

    top_level = by_name["format_total"]
    assert top_level.is_method is False
    assert top_level.parent_class is None
    assert top_level.qualified_name == "controller.py::format_total"


def test_parse_file_java_extracts_class_and_method_with_javadoc(tmp_path):
    src = tmp_path / "Ctrl.java"
    src.write_text(
        "/** Handles order creation. */\n"
        "public class OrderController {\n"
        "    /** Creates a new order. */\n"
        "    public Order createOrder(String userId) {\n"
        "        return null;\n"
        "    }\n"
        "}\n"
    )
    symbols = parse_file(src, tmp_path, "java")
    by_name = {s.name: s for s in symbols}

    cls = by_name["OrderController"]
    assert cls.docstring == "Handles order creation."
    assert cls.language == "java"

    method = by_name["createOrder"]
    assert method.is_method is True
    assert method.docstring == "Creates a new order."
    assert "String userId" in method.signature


def test_parse_file_nested_function_is_not_a_method(tmp_path):
    src = tmp_path / "mod.py"
    src.write_text(
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner\n"
    )
    symbols = parse_file(src, tmp_path, "python")
    inner = next(s for s in symbols if s.name == "inner")
    assert inner.is_method is False
    assert inner.parent_class is None


def test_parse_repo_prunes_denylisted_directories(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def real_fn():\n    pass\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "vendored.py").write_text("def should_not_appear():\n    pass\n")

    graph = parse_repo(tmp_path, "orders")
    names = {f.name for f in graph.functions}
    assert "real_fn" in names
    assert "should_not_appear" not in names


def test_parse_repo_does_not_exclude_test_directories(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_something():\n    pass\n")

    graph = parse_repo(tmp_path, "orders")
    assert any(f.name == "test_something" for f in graph.functions)


def test_parse_repo_skips_oversized_files(tmp_path):
    big = tmp_path / "big.py"
    big.write_text("def fn():\n    pass\n" + ("# padding\n" * 10))

    graph = parse_repo(tmp_path, "orders", max_file_size_bytes=10)
    assert graph.files_skipped_oversized == 1
    assert graph.files_parsed == 0
    assert graph.functions == []


def test_parse_repo_counts_unsupported_extensions(tmp_path):
    (tmp_path / "readme.md").write_text("# hello\n")
    graph = parse_repo(tmp_path, "orders")
    assert graph.files_skipped_unsupported == 1


def test_parse_repo_continues_after_one_file_fails(tmp_path, monkeypatch):
    (tmp_path / "good.py").write_text("def good_fn():\n    pass\n")
    (tmp_path / "bad.py").write_text("def bad_fn():\n    pass\n")

    import src.ingestion.code_parser as code_parser_module

    real_parse_file = code_parser_module.parse_file

    def flaky_parse_file(path, repo_root, language):
        if path.name == "bad.py":
            raise ValueError("boom")
        return real_parse_file(path, repo_root, language)

    monkeypatch.setattr(code_parser_module, "parse_file", flaky_parse_file)
    graph = code_parser_module.parse_repo(tmp_path, "orders")

    assert graph.files_failed == 1
    assert graph.files_parsed == 1
    assert any(f.name == "good_fn" for f in graph.functions)


def test_language_by_extension_covers_python_and_java():
    assert LANGUAGE_BY_EXTENSION[".py"] == "python"
    assert LANGUAGE_BY_EXTENSION[".java"] == "java"


def test_language_by_extension_covers_js_family_and_kotlin():
    assert LANGUAGE_BY_EXTENSION[".ts"] == "typescript"
    assert LANGUAGE_BY_EXTENSION[".tsx"] == "tsx"
    assert LANGUAGE_BY_EXTENSION[".js"] == "javascript"
    assert LANGUAGE_BY_EXTENSION[".jsx"] == "javascript"
    assert LANGUAGE_BY_EXTENSION[".kt"] == "kotlin"


def test_parse_file_typescript_nestjs_style_controller_with_decorator(tmp_path):
    src = tmp_path / "orders.controller.ts"
    src.write_text(
        "/** Handles orders. */\n"
        "@Controller('orders')\n"
        "export class OrdersController {\n"
        "  /** Creates a new order. */\n"
        "  createOrder(userId: string, items: Item[]): Order {\n"
        "    return { userId, items };\n"
        "  }\n"
        "}\n"
    )
    symbols = parse_file(src, tmp_path, "typescript")
    by_name = {s.name: s for s in symbols}

    cls = by_name["OrdersController"]
    assert cls.kind == "class"
    assert cls.docstring == "Handles orders."

    method = by_name["createOrder"]
    assert method.is_method is True
    assert method.parent_class == "orders.controller.ts::OrdersController"
    assert method.docstring == "Creates a new order."


def test_parse_file_typescript_interface_is_class_kind(tmp_path):
    src = tmp_path / "dto.ts"
    src.write_text("interface OrderDto {\n  id: string;\n}\n")
    symbols = parse_file(src, tmp_path, "typescript")
    assert symbols[0].kind == "class"
    assert symbols[0].name == "OrderDto"


def test_parse_file_typescript_named_arrow_function_gets_name_from_const(tmp_path):
    src = tmp_path / "utils.ts"
    src.write_text(
        "/** Formats a total. */\n"
        "export const formatTotal = (amount: number): string => `$${amount}`;\n"
    )
    symbols = parse_file(src, tmp_path, "typescript")
    assert len(symbols) == 1
    fn = symbols[0]
    assert fn.name == "formatTotal"
    assert fn.is_method is False
    assert fn.docstring == "Formats a total."


def test_parse_file_javascript_anonymous_inline_callback_is_skipped(tmp_path):
    src = tmp_path / "app.js"
    src.write_text(
        "class Cart {\n"
        "  addItem(item) {\n"
        "    return item;\n"
        "  }\n"
        "}\n"
        "const total = (a, b) => a + b;\n"
        "[1, 2, 3].map(x => x + 1);\n"
    )
    symbols = parse_file(src, tmp_path, "javascript")
    names = {s.name for s in symbols}
    assert names == {"Cart", "addItem", "total"}


def test_parse_file_tsx_react_function_and_arrow_components(tmp_path):
    src = tmp_path / "OrderCard.tsx"
    src.write_text(
        "export function OrderCard(props: { total: number }) {\n"
        "  return <div>{props.total}</div>;\n"
        "}\n\n"
        "export const ShippingBadge = ({ status }: { status: string }) => {\n"
        "  return <span>{status}</span>;\n"
        "};\n"
    )
    symbols = parse_file(src, tmp_path, "tsx")
    names = {s.name for s in symbols}
    assert names == {"OrderCard", "ShippingBadge"}
    assert all(s.is_method is False for s in symbols)


def test_parse_file_kotlin_extracts_class_and_method_with_kdoc(tmp_path):
    src = tmp_path / "OrderController.kt"
    src.write_text(
        "/** Handles order creation. */\n"
        "class OrderController {\n"
        "    /** Creates a new order. */\n"
        "    fun createOrder(userId: String): Order {\n"
        "        return Order(userId)\n"
        "    }\n"
        "}\n\n"
        "fun formatTotal(amount: Double): String {\n"
        "    return \"$amount\"\n"
        "}\n"
    )
    symbols = parse_file(src, tmp_path, "kotlin")
    by_name = {s.name: s for s in symbols}

    cls = by_name["OrderController"]
    assert cls.kind == "class"
    assert cls.docstring == "Handles order creation."

    method = by_name["createOrder"]
    assert method.is_method is True
    assert method.docstring == "Creates a new order."
    assert method.parent_class == "OrderController.kt::OrderController"

    top_level = by_name["formatTotal"]
    assert top_level.is_method is False
    assert top_level.parent_class is None


# --- chunking ----------------------------------------------------------------


def test_parse_file_small_function_has_exactly_one_chunk_covering_whole_body(tmp_path):
    src = tmp_path / "controller.py"
    src.write_text(
        "def create_order(user_id, items):\n"
        "    order = {'user_id': user_id, 'items': items}\n"
        "    return order\n"
    )
    fn = parse_file(src, tmp_path, "python")[0]
    assert len(fn.chunks) == 1
    chunk = fn.chunks[0]
    assert chunk.index == 0
    assert chunk.line_start == 1
    assert chunk.line_end == 3
    assert chunk.text == fn.source_snippet


def test_parse_file_oversized_function_splits_into_multiple_anchored_chunks(tmp_path):
    src = tmp_path / "big.py"
    body_lines = "\n".join(f"    x{i} = {i} + total" for i in range(80))
    src.write_text(f"def process_huge_order(total):\n{body_lines}\n    return total\n")

    fn = parse_file(src, tmp_path, "python")[0]
    assert len(fn.source_snippet) > _CHUNK_TARGET_CHARS
    assert len(fn.chunks) > 1

    # chunks are in order, non-overlapping, and cover the whole function body
    # (the def line itself is part of every multi-chunk's anchor TEXT, but
    # not part of any chunk's own line range — only body statements are)
    assert fn.chunks[0].line_start == fn.line_start + 1
    assert fn.chunks[-1].line_end == fn.line_end
    for a, b in zip(fn.chunks, fn.chunks[1:]):
        assert a.index < b.index
        assert a.line_end < b.line_start

    # every chunk is anchored with the function's own signature, since a
    # body-interior slice alone has no indication of which function it's from
    for chunk in fn.chunks:
        assert chunk.text.startswith(fn.signature)

    # no chunk is wildly over target (a single oversized statement could
    # still exceed it, but these are small single-line statements)
    for chunk in fn.chunks:
        assert len(chunk.text) <= _CHUNK_TARGET_CHARS * 1.5


def test_parse_file_class_has_no_chunks(tmp_path):
    src = tmp_path / "controller.py"
    src.write_text(
        "class OrderController:\n"
        "    def create_order(self):\n"
        "        return None\n"
    )
    cls = next(s for s in parse_file(src, tmp_path, "python") if s.kind == "class")
    assert cls.chunks == []


# --- GraphQL SDL ---------------------------------------------------------


def test_graphql_extracts_object_type_and_fields_as_class_and_methods():
    src = (
        b'"""A user of the system."""\n'
        b"type User {\n"
        b'  """the id"""\n'
        b"  id: ID!\n"
        b"  name: String\n"
        b"}\n"
    )
    symbols = parse_source(src, "schema.graphqls", "graphql")
    by_name = {s.name: s for s in symbols}

    cls = by_name["User"]
    assert cls.kind == "class"
    assert cls.docstring == "A user of the system."
    assert cls.qualified_name == "schema.graphqls::User"

    field = by_name["id"]
    assert field.kind == "function"
    assert field.is_method is True
    assert field.parent_class == "schema.graphqls::User"
    assert field.docstring == "the id"


def test_graphql_extracts_input_interface_and_enum_as_classes():
    src = (
        b"input CreateUserInput {\n  name: String!\n}\n\n"
        b"interface Node {\n  id: ID!\n}\n\n"
        b"enum Role {\n  ADMIN\n  USER\n}\n"
    )
    symbols = parse_source(src, "schema.graphqls", "graphql")
    classes = {s.name for s in symbols if s.kind == "class"}
    assert classes == {"CreateUserInput", "Node", "Role"}


def test_graphql_query_type_fields_become_methods_not_top_level_functions():
    src = b"type Query {\n  user(id: ID!): User\n}\n"
    symbols = parse_source(src, "schema.graphqls", "graphql")
    query_field = next(s for s in symbols if s.name == "user")
    assert query_field.is_method is True
    assert query_field.parent_class == "schema.graphqls::Query"


def test_graphql_does_not_extract_field_arguments_as_separate_symbols():
    src = b"type Query {\n  user(id: ID!, includeArchived: Boolean): User\n}\n"
    symbols = parse_source(src, "schema.graphqls", "graphql")
    names = {s.name for s in symbols}
    assert "id" not in names
    assert "includeArchived" not in names
    assert names == {"Query", "user"}


def test_language_by_extension_covers_graphqls_bpmn_avdl():
    assert LANGUAGE_BY_EXTENSION[".graphqls"] == "graphql"
    assert LANGUAGE_BY_EXTENSION[".bpmn"] == "bpmn"
    assert LANGUAGE_BY_EXTENSION[".avdl"] == "avdl"


# --- BPMN ------------------------------------------------------------------


def test_bpmn_extracts_process_as_class_and_flow_nodes_as_methods():
    src = (
        b'<?xml version="1.0" encoding="UTF-8"?>\n'
        b'<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">\n'
        b'  <bpmn:process id="OrderProcess" name="Order Process">\n'
        b'    <bpmn:startEvent id="StartEvent_1" name="Order Received" />\n'
        b'    <bpmn:serviceTask id="Task_1" name="Validate Order" />\n'
        b'    <bpmn:endEvent id="EndEvent_1" name="Order Complete" />\n'
        b"  </bpmn:process>\n"
        b"</bpmn:definitions>\n"
    )
    symbols = parse_source(src, "order.bpmn", "bpmn")
    classes = [s for s in symbols if s.kind == "class"]
    functions = [s for s in symbols if s.kind == "function"]

    assert len(classes) == 1
    assert classes[0].name == "Order Process"
    assert classes[0].qualified_name == "order.bpmn::OrderProcess"

    names = {f.name for f in functions}
    assert names == {"Order Received", "Validate Order", "Order Complete"}
    assert all(f.is_method for f in functions)
    assert all(f.parent_class == "order.bpmn::OrderProcess" for f in functions)


def test_bpmn_falls_back_to_id_when_no_name_attribute():
    src = (
        b'<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">\n'
        b'  <bpmn:process id="Proc1">\n'
        b'    <bpmn:task id="Task_1" />\n'
        b"  </bpmn:process>\n"
        b"</bpmn:definitions>\n"
    )
    symbols = parse_source(src, "p.bpmn", "bpmn")
    task = next(s for s in symbols if s.kind == "function")
    assert task.name == "Task_1"


def test_bpmn_ignores_non_flow_node_children():
    src = (
        b'<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL">\n'
        b'  <bpmn:process id="Proc1">\n'
        b'    <bpmn:startEvent id="Start_1" />\n'
        b'    <bpmn:sequenceFlow id="Flow_1" sourceRef="Start_1" targetRef="Task_1" />\n'
        b"  </bpmn:process>\n"
        b"</bpmn:definitions>\n"
    )
    symbols = parse_source(src, "p.bpmn", "bpmn")
    names = {s.name for s in symbols if s.kind == "function"}
    assert names == {"Start_1"}  # sequenceFlow isn't a Task/Event/Gateway


def test_bpmn_malformed_xml_returns_empty_without_raising():
    assert parse_source(b"<not><valid", "broken.bpmn", "bpmn") == []


# --- Avro IDL (.avdl) --------------------------------------------------------


def test_avdl_extracts_record_fields_as_methods():
    src = b"record User {\n  string name;\n  int age;\n}\n"
    symbols = parse_source(src, "user.avdl", "avdl")
    cls = next(s for s in symbols if s.kind == "class")
    assert cls.name == "User"
    fields = {s.name for s in symbols if s.kind == "function"}
    assert fields == {"name", "age"}


def test_avdl_extracts_union_typed_field_with_default_value():
    """Regression test: a multi-word type before the field name (a union)
    must not cause the field to be dropped — the field name is the LAST
    identifier before the optional default, not the first word of the type."""
    src = b"record User {\n  union { null, string } email = null;\n}\n"
    symbols = parse_source(src, "user.avdl", "avdl")
    fields = {s.name for s in symbols if s.kind == "function"}
    assert fields == {"email"}


def test_avdl_extracts_error_and_enum_as_classes():
    src = (
        b"error UserNotFoundError {\n  string message;\n}\n\n"
        b"enum Status {\n  ACTIVE, INACTIVE\n}\n"
    )
    symbols = parse_source(src, "user.avdl", "avdl")
    classes = {s.name for s in symbols if s.kind == "class"}
    assert classes == {"UserNotFoundError", "Status"}


def test_avdl_protocol_messages_become_methods_not_fields():
    src = (
        b"protocol UserProtocol {\n"
        b"  record User {\n    string name;\n  }\n\n"
        b"  User getUser(string id) throws UserNotFoundError;\n"
        b"  void deleteUser(string id);\n"
        b"}\n"
    )
    symbols = parse_source(src, "user.avdl", "avdl")
    protocol = next(s for s in symbols if s.name == "UserProtocol")
    assert protocol.kind == "class"

    messages = {s.name for s in symbols if s.is_method and s.parent_class == protocol.qualified_name}
    assert messages == {"getUser", "deleteUser"}

    # the nested record's OWN field must not also show up as a protocol message
    user_record_fields = {
        s.name for s in symbols if s.is_method and s.parent_class == "user.avdl::User"
    }
    assert user_record_fields == {"name"}


def test_avdl_no_blocks_returns_empty():
    assert parse_source(b"// just a comment\n", "empty.avdl", "avdl") == []


# --- tracked-only extensions (.html/.css/.csv) — no symbols, but tracked ----


def test_tracked_only_extensions_have_no_language_by_extension_entry():
    """These are deliberately NOT in LANGUAGE_BY_EXTENSION — they have no
    code-shaped structure to extract, so they must never reach parse_source
    at all (only get tracked as a File node with zero symbols)."""
    for ext in TRACKED_ONLY_EXTENSIONS:
        assert ext not in LANGUAGE_BY_EXTENSION


def test_parse_repo_tracks_html_css_csv_without_fake_symbols(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "style.css").write_text(".foo { color: red; }")
    (tmp_path / "data.csv").write_text("a,b\n1,2\n")

    graph = parse_repo(tmp_path, "orders")

    assert graph.classes == []
    assert graph.functions == []
    assert graph.tracked_files == {
        "index.html": "html",
        "style.css": "css",
        "data.csv": "csv",
    }


def test_parse_repo_still_counts_truly_unsupported_extensions(tmp_path):
    (tmp_path / "index.html").write_text("<html></html>")
    (tmp_path / "photo.png").write_bytes(b"binary")

    graph = parse_repo(tmp_path, "orders")

    assert graph.tracked_files == {"index.html": "html"}
    assert graph.files_skipped_unsupported == 1


def test_parse_repo_mixes_real_symbols_and_tracked_only_files(tmp_path):
    (tmp_path / "app.py").write_text("def handler():\n    pass\n")
    (tmp_path / "index.html").write_text("<html></html>")

    graph = parse_repo(tmp_path, "orders")

    assert any(f.name == "handler" for f in graph.functions)
    assert graph.tracked_files == {"index.html": "html"}


def test_extract_imports_takes_the_block_at_the_top_and_stops_at_real_code():
    """Nothing else in the pipeline records imports — parse_file extracts
    only classes and functions — which is why generated code invents module
    paths."""
    source = (
        'import React from "react";\n'
        "// a comment between imports\n"
        'import { render } from "@testing-library/react";\n'
        "\n"
        'import stubs from "../../stubs";\n'
        "\n"
        "export const x = 1;\n"
        'import { late } from "nope";\n'
    )
    assert code_parser.extract_imports(source) == (
        'import React from "react";\n'
        'import { render } from "@testing-library/react";\n'
        'import stubs from "../../stubs";'
    )


def test_extract_imports_handles_every_language_this_parser_reads():
    assert "from pathlib import Path" in code_parser.extract_imports(
        "from pathlib import Path\nimport os\n\ndef f(): pass\n"
    )
    assert "import com.asurint.Thing" in code_parser.extract_imports(
        "package com.asurint\n\nimport com.asurint.Thing\n\nclass X\n"
    )
    assert 'export { a } from "./a"' in code_parser.extract_imports('export { a } from "./a";\n')


def test_extract_imports_is_empty_when_a_file_imports_nothing():
    assert code_parser.extract_imports("def f():\n    return 1\n") == ""


def test_extract_imports_is_capped():
    source = "\n".join(f'import a{i} from "m{i}";' for i in range(200))
    assert len(code_parser.extract_imports(source)) <= code_parser.MAX_IMPORT_CHARS


def test_embedding_text_carries_the_path_and_name_the_code_often_lacks():
    """An arrow function's chunk begins at its parameter list, so the name it
    was assigned to is lost. Measured on a real graph: 22% of single-chunk
    functions do not contain their own name anywhere in their text, and 93%
    of functions are a single chunk."""
    text = code_parser.embedding_text(
        "src/common/utils/utils/multipleEmailFieldValidation.tsx",
        "multipleEmailFieldValidation",
        None,
        "(value: string) => {",
    )
    assert text.startswith("src/common/utils/utils/multipleEmailFieldValidation.tsx\n")
    assert "multipleEmailFieldValidation" in text
    assert text.endswith("(value: string) => {")


def test_embedding_text_includes_the_parent_class_for_a_method():
    text = code_parser.embedding_text("A.kt", "toAvro", "EmploymentVerificationInput", "builder()")
    assert "EmploymentVerificationInput" in text
    assert "toAvro" in text


def test_embedding_text_leaves_the_code_last_and_unchanged():
    """Chunk.text is what the planner reads, and is stored unprefixed; only
    what is sent to the embedding model gains the header."""
    code = "return 1"
    assert code_parser.embedding_text("a.py", "f", None, code).endswith(code)


def test_parse_repo_stores_a_test_file_whole(tmp_path):
    """A test file reaches the graph as a fraction of itself: parse_file()
    extracts functions and classes, and a Jest file is top-level describe()
    calls and object literals, so a 3,920-char test was stored as 192
    characters — never the fixture, which is the one thing a planner writing
    a test cannot guess."""
    (tmp_path / "view.test.tsx").write_text(
        'const settings = { canEdit: true };\nit("works", () => {});\n'
    )

    graph = parse_repo(tmp_path, "ui")

    assert graph.file_source["view.test.tsx"] == (
        'const settings = { canEdit: true };\nit("works", () => {});\n'
    )


def test_parse_repo_stores_no_source_for_production_files(tmp_path):
    """Their symbols are already extracted; storing both would double the
    graph for nothing."""
    (tmp_path / "view.tsx").write_text("export const view = () => null;\n")

    graph = parse_repo(tmp_path, "ui")

    assert "view.tsx" not in graph.file_source


def test_parse_repo_caps_a_long_test_file(tmp_path):
    """One real suite covering a whole settings tab is 29,652 characters;
    its first 6,000 are its imports and first fixture, the part worth
    copying."""
    (tmp_path / "big.test.tsx").write_text("x" * (code_parser.MAX_TEST_SOURCE_CHARS + 500))

    graph = parse_repo(tmp_path, "ui")

    assert len(graph.file_source["big.test.tsx"]) == code_parser.MAX_TEST_SOURCE_CHARS
