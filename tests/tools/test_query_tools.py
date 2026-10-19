from src.tools.query_tools import _slug, _stamp


def test_stamp_comes_from_the_tdd_not_the_clock():
    """The filename stamp and the PDF's own "Generated:" line must name the
    same run — a stamp read off the clock could disagree with the document."""
    assert _stamp("2026-10-18T16:05:15.739113+00:00") == "20261018-160515"


def test_stamp_is_normalized_to_utc():
    assert _stamp("2026-10-18T18:05:15+02:00") == "20261018-160515"


def test_stamp_falls_back_to_now_for_an_unstamped_or_unparseable_tdd():
    for bad in (None, "", "not a timestamp"):
        assert len(_stamp(bad)) == len("20261018-160515")


def test_file_name_keeps_the_short_slug_beside_the_stamp():
    name = f"{_slug('Bulk print merged consumer report PDFs from the dashboard')}-{_stamp(None)}"
    assert name.startswith("bulk-print-merged-consumer-report-pdfs-from-")
    assert not name.startswith("bulk-print-merged-consumer-report-pdfs-from-the")
