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
        return ["README.md", "pom.xml", ".gitignore", "src/main/java/com/bank/A.java"]

    def fetch(repo, path, ref, token=None):
        calls.append(("read", repo, path, ref))
        return POM

    monkeypatch.setattr(github_api, "list_tree", root)
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

    monkeypatch.setattr(github_api, "list_tree", root)
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


# --- API contracts and the rollout order (2026-10-15) ---------------------------------

OPENAPI = "openapi: 3.0.0\npaths:\n  /payments/{id}/refund:\n    post: {}\n"


def test_contract_files_are_found_and_build_output_is_not():
    from src.query import contracts

    paths = [
        "src/main/resources/openapi.yaml",
        "api/payments.proto",
        "docs/asyncapi.yml",
        "target/classes/openapi.yaml",
        "node_modules/x/swagger.json",
        "src/main/java/A.java",
    ]
    assert contracts.contract_paths(paths) == [
        "api/payments.proto",
        "docs/asyncapi.yml",
        "src/main/resources/openapi.yaml",
    ]


def test_the_contracts_reach_the_design(monkeypatch):
    monkeypatch.setattr(
        github_api, "list_tree", lambda repo, ref=None, token=None: ["pom.xml", "openapi.yaml"]
    )
    monkeypatch.setattr(
        github_api,
        "fetch_file_content",
        lambda repo, path, ref, token=None: POM if path == "pom.xml" else OPENAPI,
    )
    rules, contract_text = query_tools.repository_facts({"payments": "org/payments"})
    assert rules[0].startswith("Language: Java 17")
    assert (
        "### payments: openapi.yaml" in contract_text and "/payments/{id}/refund" in contract_text
    )

    user = planner.build_user_prompt("Partial refunds", [], api_contracts=contract_text)
    assert "## API contracts (current, read from the repositories)" in user
    system = planner.build_system_prompt()
    assert "every API change must be backward-compatible" in system
    assert "`rollout_order`" in system and "a provider before its consumers" in system


def test_the_rollout_order_is_shown_and_read_by_coding():
    from src.query.models import TechnicalDesignDoc
    from src.query.render import render_confluence, render_text

    tdd = TechnicalDesignDoc(
        title="t",
        issue_summary="s",
        overall_reasoning="r",
        services=[],
        rollout_order=["payments: accepts an optional amount first", "orders: then sends it"],
    )
    page = render_confluence(tdd)
    assert "Rollout order (merge and deploy in this order)" in page
    assert "payments: accepts an optional amount first" in page
    text = render_text(tdd)
    assert "ROLLOUT ORDER" in text and "1. payments: accepts" in text
