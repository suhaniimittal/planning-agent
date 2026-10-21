from src.ingestion import service_doc


def test_reads_the_doc_from_a_checkout(tmp_path):
    (tmp_path / "SERVICE.md").write_text("# ui\n\nThe screens.", encoding="utf-8")
    assert service_doc.read_local_service_doc(tmp_path) == "# ui\n\nThe screens."


def test_a_missing_or_blank_doc_reads_as_none(tmp_path):
    """Blank counts as missing: an empty file would otherwise be embedded and
    the service would score against noise instead of scoring 0."""
    assert service_doc.read_local_service_doc(tmp_path) is None
    (tmp_path / "SERVICE.md").write_text("   \n\n", encoding="utf-8")
    assert service_doc.read_local_service_doc(tmp_path) is None


def test_the_hash_follows_the_text_not_the_file(tmp_path):
    """Keyed on content so a commit touching only code never moves a
    service's embedding, and an edit always does."""
    assert service_doc.doc_hash("same") == service_doc.doc_hash("same")
    assert service_doc.doc_hash("same") != service_doc.doc_hash("same ")


def test_oversize_is_reported_so_truncation_is_never_silent():
    """embedder._fit truncates past this limit without a word, so a doc that
    loses its tail would otherwise look like it had been embedded whole."""
    assert service_doc.oversize_note("ui", "x" * service_doc.MAX_EMBED_BYTES) is None
    note = service_doc.oversize_note("ui", "x" * (service_doc.MAX_EMBED_BYTES + 1))
    assert "[DOC TOO LONG] ui" in note
    assert str(service_doc.MAX_EMBED_BYTES + 1) in note


def test_oversize_counts_bytes_not_characters():
    """The embedder's cap is on UTF-8 bytes; a doc of multi-byte characters
    hits it at well under that many characters."""
    text = "é" * (service_doc.MAX_EMBED_BYTES // 2 + 1)
    assert len(text) < service_doc.MAX_EMBED_BYTES
    assert service_doc.oversize_note("ui", text) is not None
