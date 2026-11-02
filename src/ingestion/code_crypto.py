"""Encryption of what Planning stores outside itself (2026-10-22; trimmed 2026-11-02).

Written for the code graph, which no longer exists; it now seals the memory
snapshots kept in Memsy (src/memory/record.py): a story's requirement facts,
TDD versions and decisions are company data, and leave Planning encrypted.

- Cipher: AES-256-GCM (NIST SP 800-38D, FIPS-approved). It is authenticated:
  a changed value fails to decrypt instead of returning garbage. Every value
  gets a fresh random 96-bit nonce.
- Binding: each value is bound (as GCM associated data) to where it belongs —
  label, service (here: the Memsy actor), qualified_name, field — so a
  ciphertext copied somewhere else fails to decrypt.
- Keys: one master key, ``CODE_GRAPH_KEY`` (32 random bytes, base64), from a
  mounted secret file, else the environment. HKDF-SHA-512 derives a key per
  "service" from it; one's key says nothing about another's.

No key, a wrong key or a tampered value raises — nothing is ever written in
plain text, and nothing undecryptable is ever passed on.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
from functools import lru_cache

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.ingestion.secret_store import secret

KEY_NAME = "CODE_GRAPH_KEY"
PREFIX = "enc:v1:"
_KEY_BYTES = 32
_NONCE_BYTES = 12
# Unchanged since the code graph: every stored snapshot was sealed under it.
_INFO = b"aetherion/code-graph/v1/"

class CodeKeyError(RuntimeError):
    """CODE_GRAPH_KEY is missing or not a 256-bit key."""


class CodeDecryptError(RuntimeError):
    """A stored value could not be decrypted: another key, or it was changed."""


@lru_cache(maxsize=4)
def _master(raw: str) -> bytes:
    try:
        key = base64.b64decode(raw, validate=True)
    except (binascii.Error, ValueError):
        raise CodeKeyError(f"{KEY_NAME} is not valid base64") from None
    if len(key) != _KEY_BYTES:
        raise CodeKeyError(f"{KEY_NAME} must be {_KEY_BYTES} random bytes, base64-encoded")
    return key


def master_key() -> bytes:
    raw = secret(KEY_NAME)
    if not raw:
        raise CodeKeyError(
            f"{KEY_NAME} is not set — the memory is never stored or read without encryption"
        )
    return _master(raw)


@lru_cache(maxsize=256)
def _derive(master: bytes, purpose: str, service: str) -> bytes:
    return HKDF(
        algorithm=hashes.SHA512(),
        length=_KEY_BYTES,
        salt=None,
        info=_INFO + purpose.encode() + b"/" + service.encode("utf-8"),
    ).derive(master)


@lru_cache(maxsize=4)
def _key_id(master: bytes) -> str:
    """8 hex characters naming the key, so a value written under another key
    gives a clear error. Derived one-way; says nothing about the key."""
    return hmac.new(master, _INFO + b"key-id", hashlib.sha512).hexdigest()[:8]


def _aad(label: str, service: str, qualified_name: str, field: str) -> bytes:
    return "\x1f".join(("v1", label, service, qualified_name, field)).encode("utf-8")


def is_encrypted(value: object) -> bool:
    return isinstance(value, str) and value.startswith(PREFIX)


def encrypt(value: str | None, *, label: str, service: str, qualified_name: str, field: str):
    """``value`` encrypted for this node and field. None and "" stay as they
    are: there is nothing to hide, and the readers already treat them as empty."""
    if value is None or value == "" or is_encrypted(value):
        return value
    master = master_key()
    nonce = os.urandom(_NONCE_BYTES)
    sealed = AESGCM(_derive(master, "enc", service)).encrypt(
        nonce, str(value).encode("utf-8"), _aad(label, service, qualified_name, field)
    )
    body = base64.b64encode(nonce + sealed).decode("ascii")
    return f"{PREFIX}{_key_id(master)}:{body}"


def decrypt(value: object, *, label: str, service: str, qualified_name: str, field: str):
    """The plain text of a stored value. A value that was never encrypted
    comes back as it is; anything else that fails to decrypt raises."""
    if not is_encrypted(value):
        return value
    master = master_key()
    key_id, _, body = value[len(PREFIX) :].partition(":")
    where = f"{label} {service}/{qualified_name}.{field}"
    if key_id != _key_id(master):
        raise CodeDecryptError(f"{where} was encrypted with another {KEY_NAME}")
    try:
        blob = base64.b64decode(body, validate=True)
        plain = AESGCM(_derive(master, "enc", service)).decrypt(
            blob[:_NONCE_BYTES], blob[_NONCE_BYTES:], _aad(label, service, qualified_name, field)
        )
    except (InvalidTag, binascii.Error, ValueError):
        raise CodeDecryptError(f"{where} could not be decrypted: it was changed or moved") from None
    return plain.decode("utf-8")
