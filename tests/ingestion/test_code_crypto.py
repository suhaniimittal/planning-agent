"""What Planning stores is encrypted (2026-10-22): unreadable without the key,
tamper-evident, and bound to where it belongs."""

import base64
import os

import pytest

from src.ingestion import code_crypto
from src.ingestion.code_crypto import CodeDecryptError, CodeKeyError

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
