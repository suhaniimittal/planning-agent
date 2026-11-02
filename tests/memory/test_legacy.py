"""The old Neo4j memory nodes are removed once per worker (2026-10-26)."""

import types

import pytest

from src.memory import legacy


class FakeSession:
    def __init__(self, graph, ran):
        self.graph, self.ran = graph, ran

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def run(self, query):
        self.ran.append(query)
        label = query.split(":")[1].split(")")[0] if query.startswith("MATCH") else ""
        if "DETACH DELETE" in query:
            self.graph[label] = 0
        return types.SimpleNamespace(single=lambda: {"n": self.graph.get(label, 0)})


@pytest.fixture
def neo4j(monkeypatch):
    graph = {"Requirement": 1, "Fact": 22, "TddVersion": 2, "Decision": 6, "Function": 50}
    ran = []
    driver = types.SimpleNamespace(session=lambda: FakeSession(graph, ran))
    monkeypatch.setattr(legacy, "get_driver", lambda: driver)
    monkeypatch.setattr(legacy, "_done", False)
    return graph, ran


def test_the_old_memory_goes_and_the_code_stays(neo4j):
    graph, ran = neo4j
    assert legacy.drop() == {"Requirement": 1, "Fact": 22, "TddVersion": 2, "Decision": 6}
    assert graph == {"Requirement": 0, "Fact": 0, "TddVersion": 0, "Decision": 0, "Function": 50}
    assert sum("DROP INDEX" in q for q in ran) == len(legacy.INDEXES)


def test_a_worker_cleans_only_once(neo4j):
    _, ran = neo4j
    legacy.drop_once()
    first = len(ran)
    legacy.drop_once()
    assert len(ran) == first


def test_an_unreachable_neo4j_never_stops_the_check_and_is_retried(monkeypatch):
    monkeypatch.setattr(legacy, "_done", False)
    monkeypatch.setattr(legacy, "get_driver", lambda: (_ for _ in ()).throw(OSError("down")))
    legacy.drop_once()
    assert legacy._done is False
