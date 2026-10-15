"""A design fits the repository's real versions and keeps what already works (2026-10-15)."""

from __future__ import annotations

from pathlib import Path

from src.ingestion import github_api
from src.query import planner, stack
from src.tools import query_tools

POM = (Path(__file__).parent / "fixtures" / "bank-pom.xml").read_text()


def test_the_bank_service_is_read_as_java_17():
    found = stack.parse_pom(POM)
    assert found.language == "Java 17" and found.java_release == "17"
    assert "org.junit.jupiter:junit-jupiter 5.10.2" in found.libraries


def test_build_files_are_read_from_github(monkeypatch):
    calls = []

    def root(repo, ref=None, token=None):
        calls.append(("list", repo, ref))
        return ["README.md", "pom.xml", ".gitignore"]

    def fetch(repo, path, ref, token=None):
        calls.append(("read", repo, path, ref))
        return POM

    monkeypatch.setattr(github_api, "list_root_files", root)
    monkeypatch.setattr(github_api, "fetch_file_content", fetch)

    rules = query_tools.technology_constraints({"bank": "AetherionAgentDemo/bank-transfer-service"})

    assert rules[0].startswith("Language: Java 17 (from pom.xml (maven.compiler.source/target))")
    assert any(r.startswith("Build: Maven.") for r in rules)
    assert calls == [
        ("list", "AetherionAgentDemo/bank-transfer-service", None),
        ("read", "AetherionAgentDemo/bank-transfer-service", "pom.xml", "HEAD"),
    ], "only the build file that exists is read"


def test_several_services_are_named_and_an_unreadable_one_says_so(monkeypatch):
    def root(repo, ref=None, token=None):
        if repo == "o/broken":
            raise github_api.GitHubApiError("404")
        return ["go.mod"]

    monkeypatch.setattr(github_api, "list_root_files", root)
    monkeypatch.setattr(github_api, "fetch_file_content", lambda *a, **k: "module x\n\ngo 1.22\n")

    rules = query_tools.technology_constraints({"api": "o/api@dev", "broken": "o/broken"})

    assert rules[0].startswith("api: Language: Go 1.22")
    assert rules[-1] == "broken: build files could not be read (404); versions unknown."


def test_the_request_carries_the_constraints_and_the_rules_to_keep_what_works():
    user = planner.build_user_prompt(
        "Add countTransfersFor", [], technology_constraints=["Language: Java 17 (from pom.xml)."]
    )
    assert "## Technology constraints (read from the repositories' build files)" in user
    assert "- Language: Java 17 (from pom.xml)." in user
    system = planner.build_system_prompt()
    assert "extend it, never replace it" in system
    assert "`preserved_behavior`" in system and "Earlier designs" in system
    assert "no API or syntax from a later version" in system


def test_no_constraints_no_section():
    assert "Technology constraints" not in planner.build_user_prompt("x", [])
