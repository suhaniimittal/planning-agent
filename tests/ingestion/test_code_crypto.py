"""The code in Neo4j is encrypted (2026-10-22): unreadable without the key,
tamper-evident, bound to its node, and still searchable by exact name."""

import base64
import os

import pytest

from src.ingestion import code_crypto
from src.ingestion.code_crypto import CodeDecryptError, CodeKeyError
from src.query import graph_reader

SOURCE = "public void transfer(Account from, Account to, BigDecimal amount) { ... }"
WHERE = {"label": "Function", "service": "bank", "qualified_name": "T.java::T.transfer"}


def _seal(text=SOURCE, **where):
    return code_crypto.encrypt(text, **{**WHERE, "field": "source_snippet", **where})


def _open(value, **where):
    return code_crypto.decrypt(value, **{**WHERE, "field": "source_snippet", **where})


def test_round_trip_and_nothing_readable_in_the_database():
    sealed = _seal()
    assert sealed.startswith("enc:v1:")
    assert "transfer" not in sealed and "Account" not in sealed
    assert _open(sealed) == SOURCE


def test_every_write_uses_a_fresh_nonce():
    assert _seal() != _seal()


def test_a_changed_value_is_refused_not_returned():
    sealed = _seal()
    body = bytearray(base64.b64decode(sealed.rsplit(":", 1)[1]))
    body[-1] ^= 1
    tampered = sealed.rsplit(":", 1)[0] + ":" + base64.b64encode(bytes(body)).decode()
    with pytest.raises(CodeDecryptError, match="changed or moved"):
        _open(tampered)


@pytest.mark.parametrize(
    "elsewhere",
    [
        {"qualified_name": "T.java::T.other"},
        {"service": "payments"},
        {"field": "docstring"},
        {"label": "Class"},
    ],
)
def test_a_value_copied_to_another_node_or_field_is_refused(elsewhere):
    with pytest.raises(CodeDecryptError):
        _open(_seal(), **elsewhere)


def test_another_key_cannot_decrypt(monkeypatch):
    sealed = _seal()
    monkeypatch.setenv("CODE_GRAPH_KEY", base64.b64encode(os.urandom(32)).decode())
    with pytest.raises(CodeDecryptError, match="another CODE_GRAPH_KEY"):
        _open(sealed)


def test_no_key_means_nothing_is_written(monkeypatch):
    monkeypatch.delenv("CODE_GRAPH_KEY")
    with pytest.raises(CodeKeyError, match="not set"):
        _seal()


@pytest.mark.parametrize("bad", ["short", base64.b64encode(b"x" * 16).decode(), "%%%"])
def test_a_weak_or_malformed_key_is_refused(monkeypatch, bad):
    monkeypatch.setenv("CODE_GRAPH_KEY", bad)
    with pytest.raises(CodeKeyError):
        _seal()


def test_empty_values_and_old_plain_values_pass_through():
    assert _seal(None) is None and _seal("") == ""
    assert _open("written before encryption") == "written before encryption"
    sealed = _seal()
    assert _seal(sealed) == sealed  # never encrypted twice


def test_encrypt_row_leaves_names_paths_and_the_callers_row_alone():
    row = {
        "qualified_name": "T.java::T.transfer",
        "name": "transfer",
        "file_path": "T.java",
        "source_snippet": SOURCE,
        "signature": "transfer(a, b, c)",
        "docstring": None,
    }
    sealed = code_crypto.encrypt_row(row, label="Function", service="bank")
    assert sealed["name"] == "transfer" and sealed["file_path"] == "T.java"
    assert code_crypto.is_encrypted(sealed["signature"]) and sealed["docstring"] is None
    assert row["source_snippet"] == SOURCE
    opened = code_crypto.decrypt_row(
        sealed, label="Function", service="bank", qualified_name="T.java::T.transfer"
    )
    assert opened == row


def test_search_hashes_are_per_service_and_hide_the_words():
    terms = code_crypto.text_terms("bank", "x = applyDiscount(y)")
    assert not any("applyDiscount" in t for t in terms)
    assert code_crypto.query_terms("bank", "applyDiscount")[0] in terms
    assert code_crypto.query_terms("shop", "applyDiscount")[0] not in terms


def _session_returning(rows):
    class Session:
        queries = []

        def run(self, query, **params):
            self.queries.append((query, params))
            return rows if "groups" in params else []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    session = Session()
    return session, type("D", (), {"session": lambda self: session})()


def test_a_hash_match_is_checked_against_the_real_text(monkeypatch):
    def chunk(qname, text):
        sealed = code_crypto.encrypt(
            text, label="Chunk", service="bank", qualified_name=qname, field="text"
        )
        return {
            "function_qualified_name": qname.split("::chunk")[0],
            "chunk_qualified_name": qname,
            "text": sealed,
            "score": 0.4,
        }

    rows = [
        chunk("a::f::chunk0", 'throw new X("Insufficient funds for account");'),
        # every word of the phrase is there, but not as the phrase
        chunk("a::g::chunk0", "// account funds are insufficient for now"),
    ]
    session, driver = _session_returning(rows)
    monkeypatch.setattr(graph_reader, "get_driver", lambda: driver)

    found = graph_reader.search_service_chunks("bank", [0.1], ["Insufficient funds for account"])

    assert [r["function_qualified_name"] for r in found] == ["a::f"]
    query, params = session.queries[-1]
    assert "text_terms" in query and "CONTAINS" in query  # old plain nodes still found
    assert all("Insufficient" not in t for g in params["groups"] for t in g)
