from src.ingestion import graph_writer


class FakeSession:
    def __init__(self, rows=None):
        self.queries = []
        self._rows = rows if rows is not None else []

    def run(self, query, **params):
        self.queries.append((query, params))
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeDriver:
    def __init__(self, rows=None):
        self._session = FakeSession(rows)

    def session(self):
        return self._session


def _patch_driver(monkeypatch, rows=None):
    fake_driver = FakeDriver(rows)
    monkeypatch.setattr(graph_writer, "get_driver", lambda: fake_driver)
    return fake_driver


def test_load_existing_service_records_returns_name_full_doc_key_symbols(monkeypatch):
    rows = [
        {"name": "orders", "full_doc": "# orders doc", "key_symbols": ["OrdersController"]},
        {"name": "shipping", "full_doc": "# shipping doc", "key_symbols": ["ShipmentController"]},
    ]
    driver = _patch_driver(monkeypatch, rows)

    records = graph_writer.load_existing_service_records()

    assert records == rows
    query, params = driver._session.queries[0]
    assert "MATCH (s:Service)" in query
    assert "s.name AS name" in query
    assert "s.full_doc AS full_doc" in query
    assert "s.key_symbols AS key_symbols" in query
    assert params == {}


def test_load_existing_service_records_defaults_null_doc_and_symbols(monkeypatch):
    rows = [{"name": "user", "full_doc": None, "key_symbols": None}]
    _patch_driver(monkeypatch, rows)

    records = graph_writer.load_existing_service_records()

    assert records == [{"name": "user", "full_doc": "", "key_symbols": []}]


def test_load_existing_service_records_empty_graph_returns_empty_list(monkeypatch):
    _patch_driver(monkeypatch, rows=[])

    assert graph_writer.load_existing_service_records() == []
