import os

import pytest

pytestmark = pytest.mark.skipif(
    not (
        os.environ.get("NEO4J_URI")
        and os.environ.get("NEO4J_PASSWORD")
        and (
            os.environ.get("OPENAI_API_KEY")
            or (os.environ.get("AGENTS_GATEWAY_KEY") and os.environ.get("AI_GATEWAY_URL"))
        )
    ),
    reason="requires live Neo4j + LLM credentials",
)


@pytest.mark.asyncio
async def test_build_tdd_end_to_end_against_ingested_graph():
    from src.query.query_flow import build_tdd

    tdd = await build_tdd(
        "shipping estimates are wrong for orders with multiple items",
        top_k=5,
        hops=2,
    )

    assert tdd.services, "expected at least one candidate service in the TDD"
    service_names = {s.service for s in tdd.services}
    assert service_names & {"orders", "shipping"}
