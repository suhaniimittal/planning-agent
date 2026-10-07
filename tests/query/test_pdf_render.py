from src.query.models import FileChange, ServicePlan, TechnicalDesignDoc
from src.query.pdf_render import render_pdf


def _tdd(**overrides) -> TechnicalDesignDoc:
    defaults = dict(
        title="Fix order totals",
        issue_summary="Orders show wrong total price",
        services=[
            ServicePlan(
                service="orders",
                changes=[
                    FileChange(
                        file_path="src/.../OrdersController.java",
                        function_or_symbol="calculateTotal",
                        change_description="Fix summation of item prices",
                        implementation_notes="Loop over items and sum unit price * quantity",
                        pseudocode_sketch="total = 0\nfor item in items:\n    total += item.price",
                        reasoning="This is where totals are computed",
                    )
                ],
                complexity="medium",
                reasoning="Orders owns total calculation",
            )
        ],
        overall_reasoning="The bug is in total calculation logic",
    )
    defaults.update(overrides)
    return TechnicalDesignDoc(**defaults)


def test_render_pdf_returns_real_pdf_bytes():
    pdf_bytes = render_pdf(_tdd())
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 100


def test_render_pdf_content_matches_render_text_when_extracted():
    """The whole point of reusing render_text() verbatim: whatever text goes
    into the PDF must come back out unchanged when read — this is exactly
    what coding_agent's own PDF-extraction step later depends on."""
    import io

    from pypdf import PdfReader

    from src.query.render import render_text

    tdd = _tdd()
    pdf_bytes = render_pdf(tdd)
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert "Fix order totals" in extracted
    assert "calculateTotal" in extracted
    assert "OrdersController.java" in extracted
    # Every non-blank line of the original text must survive extraction —
    # xhtml2pdf can reflow whitespace, so this checks content, not layout.
    original_lines = [line.strip() for line in render_text(tdd).splitlines() if line.strip()]
    for line in original_lines[:5]:  # a representative sample, not every line
        assert line in extracted


def test_wrap_lines_keeps_short_lines_and_wraps_long_ones_with_indent():
    from src.query.pdf_render import _WRAP_WIDTH, _wrap_lines

    long_line = "  " + " ".join(["word"] * 60)
    text = "short line\n" + long_line

    lines = _wrap_lines(text).splitlines()

    assert lines[0] == "short line"
    assert all(len(line) <= _WRAP_WIDTH for line in lines)
    assert lines[1].startswith("  word")
    assert all(line.startswith("    ") for line in lines[2:])
    assert " ".join(" ".join(lines[1:]).split()) == " ".join(long_line.split())


def test_wrap_lines_splits_an_unbreakable_path():
    from src.query.pdf_render import _WRAP_WIDTH, _wrap_lines

    path = "src/" + "a" * 150 + "/view.tsx"

    lines = _wrap_lines("File: " + path).splitlines()

    assert all(len(line) <= _WRAP_WIDTH for line in lines)
    assert "".join("".join(lines).split()) == "File:" + path


def test_render_pdf_never_draws_a_line_past_the_page_edge():
    """xhtml2pdf doesn't wrap <pre> text, so an unwrapped long line is drawn
    off the right edge — present in the file, but unreadable to a person."""
    import io

    from pypdf import PdfReader

    long_reasoning = "This sentence is deliberately very long " * 10
    pdf_bytes = render_pdf(_tdd(overall_reasoning=long_reasoning))
    reader = PdfReader(io.BytesIO(pdf_bytes))
    extracted = "\n".join(page.extract_text() or "" for page in reader.pages)

    assert max(len(line) for line in extracted.splitlines()) <= 105
    assert extracted.count("deliberately") == 10


def test_wrap_lines_breaks_a_long_path_only_after_a_slash():
    from src.query.pdf_render import _WRAP_WIDTH, _wrap_lines

    path = (
        "src/apps/client/modules/Reports/pages/Report/components/ReportDetails/"
        "components/modals/PreAdverseActionModal/view.tsx"
    )

    lines = _wrap_lines("File: " + path).splitlines()

    assert len(lines) > 1
    assert all(len(line) <= _WRAP_WIDTH for line in lines)
    path_lines = [line.strip() for line in lines if "/" in line]
    assert all(piece.endswith("/") for piece in path_lines[:-1])
    assert "".join(path_lines) == path
