"""scripts/encrypt_code_graph.py encrypts old plain nodes in place (2026-10-22)."""

import importlib.util
from pathlib import Path

from src.ingestion import code_crypto

_SPEC = importlib.util.spec_from_file_location(
    "encrypt_code_graph", Path(__file__).parents[2] / "scripts" / "encrypt_code_graph.py"
)
script = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(script)

PLAIN = {
    "Function": [
        {
            "service": "bank",
            "qualified_name": "T::f",
            "source_snippet": "int f() {}",
            "signature": "f()",
            "docstring": None,
        }
    ],
    "Chunk": [{"service": "bank", "qualified_name": "T::f::chunk0", "text": "int f() {}"}],
    "Service": [{"service": "bank", "qualified_name": "bank", "summary": "Moves money."}],
}


class Session:
    def __init__(self):
        self.writes = []

    def run(self, query, **params):
        if query.startswith("UNWIND"):
            self.writes.append((query, params["rows"]))
            return []
        label = query.split("(n:")[1].split(")")[0]
        return PLAIN.get(label, [])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _run(monkeypatch, apply):
    session = Session()
    monkeypatch.setattr(
        script, "get_driver", lambda: type("D", (), {"session": lambda s: session})()
    )
    script.main(apply=apply)
    return session


def test_a_dry_run_changes_nothing(monkeypatch, capsys):
    assert _run(monkeypatch, apply=False).writes == []
    assert "Function: 1 node(s) holding plain text" in capsys.readouterr().out


def test_apply_encrypts_every_code_field_and_adds_search_hashes(monkeypatch):
    writes = {
        q.split("(n:")[1].split(" ")[0]: (q, rows) for q, rows in _run(monkeypatch, True).writes
    }

    query, (fn,) = writes["Function"]
    assert "{service: row.service, qualified_name: row.qualified_name}" in query
    assert (
        code_crypto.decrypt(
            fn["source_snippet"],
            label="Function",
            service="bank",
            qualified_name="T::f",
            field="source_snippet",
        )
        == "int f() {}"
    )
    assert fn["docstring"] is None

    _, (chunk,) = writes["Chunk"]
    assert code_crypto.is_encrypted(chunk["text"])
    assert chunk["text_terms"] == code_crypto.text_terms("bank", "int f() {}")

    query, (service,) = writes["Service"]
    assert "{name: row.qualified_name}" in query
    assert code_crypto.is_encrypted(service["summary"])
