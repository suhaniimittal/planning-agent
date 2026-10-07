import pytest

from src.ingestion import summarizer


def _cls(name, file_path="app/service.py", docstring=None, qualified_name=None) -> dict:
    return {
        "qualified_name": qualified_name or f"{file_path}::{name}",
        "name": name,
        "file_path": file_path,
        "language": "python",
        "docstring": docstring,
    }


def _fn(name, parent_class=None, file_path="app/service.py") -> dict:
    suffix = f"{parent_class.split('::')[-1]}." if parent_class else ""
    return {
        "qualified_name": f"{file_path}::{suffix}{name}",
        "name": name,
        "file_path": file_path,
        "language": "python",
        "is_method": parent_class is not None,
        "parent_class": parent_class,
    }


def test_build_digest_picks_dominant_language_and_entry_point():
    classes = [_cls("OrderService", file_path="app/main.py")]
    functions = [_fn("create_order", parent_class="app/main.py::OrderService", file_path="app/main.py")]

    digest = summarizer.build_digest("orders", classes, functions)

    assert digest["service"] == "orders"
    assert digest["language"] == "python"
    assert digest["entry_points"] == ["app/main.py"]


def test_build_digest_ranks_classes_by_method_count():
    small = _cls("Small", file_path="a.py")
    big = _cls("Big", file_path="a.py")
    functions = [
        _fn("one", parent_class="a.py::Small", file_path="a.py"),
        _fn("two", parent_class="a.py::Big", file_path="a.py"),
        _fn("three", parent_class="a.py::Big", file_path="a.py"),
    ]

    digest = summarizer.build_digest("svc", [small, big], functions)

    assert [c["name"] for c in digest["classes"]] == ["Big", "Small"]


def test_build_digest_caps_classes_to_max():
    classes = [_cls(f"Cls{i}", file_path="a.py") for i in range(20)]

    digest = summarizer.build_digest("svc", classes, [])

    assert len(digest["classes"]) == summarizer._MAX_CLASSES_PER_SERVICE


def test_build_digest_lists_top_level_functions_only():
    classes = [_cls("Svc", file_path="a.py")]
    functions = [
        _fn("method_a", parent_class="a.py::Svc", file_path="a.py"),
        _fn("main", file_path="a.py"),
    ]

    digest = summarizer.build_digest("svc", classes, functions)

    assert digest["top_level_functions"] == ["main"]


def test_build_digest_handles_empty_input():
    digest = summarizer.build_digest("empty-svc", [], [])

    assert digest == {
        "service": "empty-svc",
        "language": None,
        "entry_points": [],
        "classes": [],
        "top_level_functions": [],
    }


def test_java_application_suffix_detected_as_entry_point():
    classes = [
        {
            "qualified_name": "src/OrdersApplication.java::OrdersApplication",
            "name": "OrdersApplication",
            "file_path": "src/OrdersApplication.java",
            "language": "java",
            "docstring": None,
        }
    ]

    digest = summarizer.build_digest("orders", classes, [])

    assert digest["entry_points"] == ["src/OrdersApplication.java"]


def test_parse_response_extracts_summaries():
    raw = (
        '[{"service": "orders", "summary": "Handles orders."}, '
        '{"service": "payment", "summary": "Handles payments."}]'
    )
    result = summarizer._parse_response(raw, {"orders", "payment"})
    assert result == {"orders": "Handles orders.", "payment": "Handles payments."}


def test_parse_response_strips_markdown_fences():
    raw = '```json\n[{"service": "orders", "summary": "Handles orders."}]\n```'
    result = summarizer._parse_response(raw, {"orders"})
    assert result == {"orders": "Handles orders."}


def test_parse_response_skips_malformed_entries_without_failing():
    raw = '[{"service": "orders", "summary": "ok"}, {"bad": "entry"}, "not even a dict"]'
    result = summarizer._parse_response(raw, {"orders"})
    assert result == {"orders": "ok"}


def test_parse_response_raises_on_invalid_json():
    with pytest.raises(summarizer.SummarizerError):
        summarizer._parse_response("not json at all", {"orders"})


def test_parse_response_raises_when_not_an_array():
    with pytest.raises(summarizer.SummarizerError):
        summarizer._parse_response('{"service": "orders", "summary": "ok"}', {"orders"})


@pytest.mark.asyncio
async def test_generate_summaries_returns_empty_for_no_digests():
    result = await summarizer.generate_summaries([])
    assert result == {}


@pytest.mark.asyncio
async def test_generate_summaries_calls_openai_path_and_parses(monkeypatch):
    monkeypatch.delenv("AGENTS_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_URL", raising=False)

    async def fake_call_via_openai(system_prompt, user_prompt):
        return '[{"service": "orders", "summary": "Handles orders."}]'

    monkeypatch.setattr(summarizer, "_call_via_openai", fake_call_via_openai)

    digests = [summarizer.build_digest("orders", [], [])]
    result = await summarizer.generate_summaries(digests)

    assert result == {"orders": "Handles orders."}


@pytest.mark.asyncio
async def test_generate_summaries_wraps_provider_errors(monkeypatch):
    monkeypatch.delenv("AGENTS_GATEWAY_KEY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_URL", raising=False)

    async def failing_call(system_prompt, user_prompt):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(summarizer, "_call_via_openai", failing_call)

    digests = [summarizer.build_digest("orders", [], [])]
    with pytest.raises(summarizer.SummarizerError):
        await summarizer.generate_summaries(digests)
