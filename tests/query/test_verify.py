import pytest

from src.query import verify
from src.query.models import FileChange, ServicePlan, TechnicalDesignDoc


def _change(file_path, symbol, change_type="modify") -> FileChange:
    return FileChange(
        file_path=file_path,
        function_or_symbol=symbol,
        change_description="d",
        implementation_notes="i",
        pseudocode_sketch="p",
        reasoning="r",
        change_type=change_type,
    )


@pytest.fixture
def graph(monkeypatch):
    """A tiny fake code graph for service "ui": files, and where symbols live."""
    state = {
        "files": {
            "src/report/modals/PreAdverseActionModal/view.tsx",
            "src/report/modals/OtherModal/view.tsx",
            "src/report/ReportDetails.tsx",
        },
        "symbols": {
            "PreAdverseActionModal.view": ["src/report/modals/PreAdverseActionModal/view.tsx"],
            "view": [
                "src/report/modals/OtherModal/view.tsx",
                "src/report/modals/PreAdverseActionModal/view.tsx",
            ],
            "ReportDetails": ["src/report/ReportDetails.tsx"],
        },
    }
    monkeypatch.setattr(
        verify.graph_reader, "file_exists", lambda service, path: path in state["files"]
    )
    monkeypatch.setattr(
        verify.graph_reader,
        "folder_exists",
        lambda service, folder: any(f.startswith(folder.rstrip("/") + "/") for f in state["files"]),
    )
    monkeypatch.setattr(
        verify.graph_reader,
        "find_symbol_files",
        lambda service, symbol: sorted(state["symbols"].get(symbol, [])),
    )
    return state


def _check(change, shown=frozenset()):
    verify.verify_change("ui", change, set(shown))
    return change


def test_real_file_and_real_symbol_is_verified(graph):
    c = _check(_change("src/report/ReportDetails.tsx", "ReportDetails"))
    assert c.verification == "verified"
    assert c.file_path == "src/report/ReportDetails.tsx"


def test_wrong_file_is_corrected_to_where_the_symbol_really_is(graph):
    c = _check(_change("src/report/Wrong.tsx", "PreAdverseActionModal.view"))
    assert c.verification == "corrected"
    assert c.file_path == "src/report/modals/PreAdverseActionModal/view.tsx"
    assert "src/report/Wrong.tsx" in c.verification_note


def test_missing_file_is_filled_in_from_the_graph(graph):
    c = _check(_change(None, "ReportDetails"))
    assert c.verification == "corrected"
    assert c.file_path == "src/report/ReportDetails.tsx"


def test_folder_path_picks_the_one_matching_symbol_inside_it(graph):
    # "view" exists in two files; the folder narrows it to one.
    c = _check(_change("src/report/modals/PreAdverseActionModal", "view"))
    assert c.verification == "corrected"
    assert c.file_path == "src/report/modals/PreAdverseActionModal/view.tsx"


def test_ambiguous_symbol_is_unverified_and_lists_candidates(graph):
    c = _check(_change(None, "view"))
    assert c.verification == "unverified"
    assert c.file_path is None
    assert "2 files" in c.verification_note


def test_made_up_symbol_in_a_folder_is_flagged_with_related_files(graph):
    """The real failure seen: a folder given as the file, and a function that
    doesn't exist marked "modify"."""
    c = _check(
        _change("src/report", "addDownloadButton"),
        shown={"src/report/modals/PreAdverseActionModal/view.tsx", "elsewhere/x.ts"},
    )
    assert c.verification == "unverified"
    assert c.file_path == "src/report"  # never rewritten to a guess
    assert "is a folder" in c.verification_note
    assert "addDownloadButton" in c.verification_note
    assert "PreAdverseActionModal/view.tsx" in c.verification_note
    assert "elsewhere/x.ts" not in c.verification_note


def test_real_file_without_the_symbol_is_flagged(graph):
    c = _check(_change("src/report/ReportDetails.tsx", "addDownloadButton"))
    assert c.verification == "unverified"
    assert "may need to be created" in c.verification_note


def test_nothing_found_at_all_is_flagged(graph):
    c = _check(_change("src/nowhere/x.ts", "ghost"))
    assert c.verification == "unverified"


def test_create_in_existing_file_or_folder_is_new(graph):
    in_file = _check(_change("src/report/ReportDetails.tsx", "downloadNotice", "create"))
    in_folder = _check(_change("src/report/modals/DownloadButton.tsx", "DownloadButton", "create"))
    assert in_file.verification == "new"
    assert in_folder.verification == "new"
    assert "src/report/modals" in in_folder.verification_note


def test_create_with_a_folder_as_the_file_or_unknown_folder_is_flagged(graph):
    folder = _check(_change("src/report/modals", "DownloadButton", "create"))
    unknown = _check(_change("src/nowhere/New.tsx", "New", "create"))
    missing = _check(_change(None, "New", "create"))
    assert folder.verification == "unverified"
    assert "is a folder" in folder.verification_note
    assert unknown.verification == "unverified"
    assert missing.verification == "unverified"


def test_symbol_and_path_are_normalized_before_lookup(graph):
    c = _check(_change("./src/report/ReportDetails.tsx", "`ReportDetails()`"))
    assert c.verification == "verified"
    assert c.file_path == "src/report/ReportDetails.tsx"


def test_verify_tdd_checks_every_change_in_every_service(graph):
    tdd = TechnicalDesignDoc(
        title="t",
        issue_summary="s",
        overall_reasoning="r",
        services=[
            ServicePlan(
                service="ui",
                complexity="low",
                reasoning="r",
                changes=[
                    _change("src/report/ReportDetails.tsx", "ReportDetails"),
                    _change(None, "ghost"),
                ],
            )
        ],
    )

    verify.verify_tdd(tdd)

    assert [c.verification for c in tdd.services[0].changes] == ["verified", "unverified"]


def test_component_glued_to_its_file_name_is_corrected_to_the_real_symbol(graph):
    """Seen for real: the LLM wrote `PreAdverseActionModal.view` for the
    `PreAdverseActionModal` component defined in view.tsx."""
    graph["symbols"]["PreAdverseActionModal"] = ["src/report/modals/PreAdverseActionModal/view.tsx"]
    graph["symbols"].pop("PreAdverseActionModal.view")
    graph["symbols"]["view"] = ["src/report/modals/OtherModal/view.tsx"]

    path = "src/report/modals/PreAdverseActionModal/view.tsx"
    c = _check(_change(path, "PreAdverseActionModal.view"))

    assert c.verification == "corrected"
    assert c.function_or_symbol == "PreAdverseActionModal"
    assert c.file_path == "src/report/modals/PreAdverseActionModal/view.tsx"


def test_dotted_name_with_no_matching_part_stays_unverified(graph):
    c = _check(_change("src/report/ReportDetails.tsx", "Nope.alsoNope"))
    assert c.verification == "unverified"
    assert c.function_or_symbol == "Nope.alsoNope"
