import pytest

from src.tools import ingestion_tools


@pytest.mark.asyncio
async def test_run_ingestion_parses_repos_text_and_forwards_specs(monkeypatch):
    seen = {}

    async def fake_run_ingest(specs):
        seen["specs"] = specs

    monkeypatch.setattr(ingestion_tools, "run_ingest", fake_run_ingest)

    repos_text = "orders: myorg/orders-service\npayment: myorg/payment-service@develop"
    result = await ingestion_tools.run_ingestion(repos_text)

    assert [s.name for s in seen["specs"]] == ["orders", "payment"]
    assert seen["specs"][1].branch == "develop"
    assert result == {"status": "success", "services": ["orders", "payment"]}


@pytest.mark.asyncio
async def test_run_ingestion_malformed_text_raises_repo_list_error(monkeypatch):
    async def fail_run_ingest(specs):
        raise AssertionError("must not reach ingest.run with a malformed repos_text")

    monkeypatch.setattr(ingestion_tools, "run_ingest", fail_run_ingest)

    with pytest.raises(ingestion_tools.RepoListError):
        await ingestion_tools.run_ingestion("not a valid line at all")


@pytest.mark.asyncio
async def test_run_ingestion_empty_text_raises_repo_list_error(monkeypatch):
    async def fail_run_ingest(specs):
        raise AssertionError("must not reach ingest.run with no services")

    monkeypatch.setattr(ingestion_tools, "run_ingest", fail_run_ingest)

    with pytest.raises(ingestion_tools.RepoListError):
        await ingestion_tools.run_ingestion("\n\n  \n")
