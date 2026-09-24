from pathlib import Path

from src.ingestion.code_parser import _CHUNK_TARGET_CHARS, LANGUAGE_BY_EXTENSION, parse_file, parse_repo


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
