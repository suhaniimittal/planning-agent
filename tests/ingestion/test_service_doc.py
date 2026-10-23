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


def test_embedding_text_drops_the_exclusions_section():
    """The section listing what a service does NOT do is written to push it
    down on tickets about those topics; in an embedding it pulls it up,
    because embeddings have no representation of negation."""
    doc = (
        "# notifications-service\n\n"
        "## What this service does\n\nSends SMS.\n\n"
        "## What it does NOT do\n\nIt does not send email or manage CC lists.\n"
    )

    embedded = service_doc.embedding_text(doc)

    assert "Sends SMS." in embedded
    assert "CC lists" not in embedded
    assert "What it does NOT do" not in embedded


def test_embedding_text_resumes_at_the_next_heading_of_the_same_level():
    doc = (
        "## What it owns\n\nThe screens.\n\n"
        "## Out of scope\n\nBilling.\n\n"
        "## Key flows\n\nLogin.\n"
    )

    embedded = service_doc.embedding_text(doc)

    assert "The screens." in embedded
    assert "Login." in embedded
    assert "Billing." not in embedded


def test_embedding_text_takes_subsections_of_the_exclusions_with_it():
    doc = "## Exclusions\n\nNot ours:\n\n### Email\n\nSent elsewhere.\n\n## Key flows\n\nLogin.\n"

    embedded = service_doc.embedding_text(doc)

    assert "Sent elsewhere." not in embedded
    assert "Login." in embedded


def test_embedding_text_keeps_a_doc_that_has_no_exclusions_section():
    doc = "# ui\n\n## What this service does\n\nThe screens.\n"

    assert service_doc.embedding_text(doc) == doc.strip()


def test_embedding_text_falls_back_to_the_whole_doc_if_nothing_survives():
    """A doc that is nothing but its exclusions would otherwise embed the
    empty string, which scores against every ticket alike."""
    doc = "## What it does NOT do\n\nAnything at all.\n"

    assert service_doc.embedding_text(doc) == doc.strip()


def test_doc_hash_changes_when_the_embedding_derivation_changes(monkeypatch):
    """Every service must re-embed once after the embedded text starts being
    derived differently, without anyone editing a SERVICE.md."""
    before = service_doc.doc_hash("# ui\n\nThe screens.")
    monkeypatch.setattr(service_doc, "DOC_EMBED_VERSION", service_doc.DOC_EMBED_VERSION + 1)

    assert service_doc.doc_hash("# ui\n\nThe screens.") != before


def test_doc_hash_covers_the_exclusions_section_too():
    """The whole file is stored on the node and read by the planner, so an
    edit confined to the exclusions must still refresh it — even though it
    leaves the embedding unchanged."""
    base = "# ui\n\n## What it does NOT do\n\nBilling.\n"
    edited = "# ui\n\n## What it does NOT do\n\nBilling and payroll.\n"

    assert service_doc.doc_hash(base) != service_doc.doc_hash(edited)
