"""Encryption of the code stored in Neo4j (2026-10-22).

The graph holds a whole codebase: every function's source, its signature and
docstring, every chunk of code, each service's SERVICE.md, and each file's
import lines and (for tests) full text. Anyone who could
open Neo4j (a database login, a backup, the browser) could read all of it. Now
those fields are stored encrypted, and are decrypted only by the reads that
hand a specific function or chunk to the planner, at the moment it asks.

- Cipher: AES-256-GCM (NIST SP 800-38D, FIPS-approved). It is authenticated:
  a value changed in the database fails to decrypt instead of returning
  garbage. Every value gets a fresh random 96-bit nonce.
- Binding: each value is bound (as GCM associated data) to its node and field
  — label, service, qualified_name, field — so a ciphertext copied onto another
  node or field fails to decrypt.
- Keys: one master key, ``CODE_GRAPH_KEY`` (32 random bytes, base64), read like
  NEO4J_PASSWORD (a mounted secret file, else the environment). It never
  touches Neo4j. HKDF-SHA-512 derives two keys per service from it: one to
  encrypt, one for the search index below; one service's keys say nothing
  about another's.
- Search: "find chunks containing ``approveDeferredReport``" used to run as
  ``c.text CONTAINS ...`` inside Neo4j, which cannot see into ciphertext. Each
  chunk now also stores ``text_terms``: keyed HMAC-SHA-512 hashes (128 bits
  kept) of the words in it. A search hashes its words with the same key and
  matches hashes; the hits are then decrypted and checked against the real
  text, so a hash collision can never add a false match.

Not encrypted, on purpose: embeddings (Neo4j's vector search needs the raw
numbers), names and file paths (the graph's structure and keys), line numbers.

No key, a wrong key or a tampered value raises — nothing is ever written in
plain text, and nothing undecryptable is ever passed on as if it were code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import os
import re
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
_TERM_BYTES = 16
_INFO = b"aetherion/code-graph/v1/"

# The fields holding code or text derived from it, per node label.
ENCRYPTED_FIELDS = {
    "Class": ("source_snippet", "signature", "docstring"),
    "Function": ("source_snippet", "signature", "docstring"),
    "Chunk": ("text",),
    # `doc` is the committed SERVICE.md; `summary` the generated text it replaced.
    "Service": ("doc", "summary"),
    # A test file's stored text, and every file's import lines. A File is named
    # by its path, which a rename changes: see code_graph_writer.rename_file.
    "File": ("source", "imports"),
    # Planning memory (2026-10-26): a story's requirement facts and the design
    # decisions its TDDs keep. A Fact is named by its slot within its story,
    # a Decision by its uid within its project.
    "Fact": ("statement", "values"),
    "Decision": ("title", "choice", "reason"),
}

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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
            f"{KEY_NAME} is not set — code is never stored or read without encryption"
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
    """The plain text of a stored value. Values written before encryption
    existed come back as they are (run scripts/encrypt_code_graph.py to
    encrypt them); anything else that fails to decrypt raises."""
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


def encrypt_row(
    row: dict, *, label: str, service: str, qualified_name: str | None = None
) -> dict:
    """A copy of a node's row with its code fields encrypted (the input is
    left as it was). The node is named by ``qualified_name``, else by the
    row's own ``qualified_name``."""
    fields = [f for f in ENCRYPTED_FIELDS[label] if f in row]
    if not fields:
        return row
    name = qualified_name if qualified_name is not None else row["qualified_name"]
    out = dict(row)
    for field in fields:
        out[field] = encrypt(
            row[field], label=label, service=service, qualified_name=name, field=field
        )
    return out


def decrypt_row(row: dict, *, label: str, service: str, qualified_name: str) -> dict:
    """A copy of a read row with its code fields decrypted."""
    out = dict(row)
    for field in ENCRYPTED_FIELDS[label]:
        if field in out:
            out[field] = decrypt(
                out[field], label=label, service=service, qualified_name=qualified_name, field=field
            )
    return out


# --- search without seeing the text ------------------------------------------


def words(text: str) -> set[str]:
    """The identifier-like words in ``text``, case kept: code names are
    case-sensitive, and so was the CONTAINS search this replaces."""
    return set(_WORD.findall(text or ""))


def _term(index_key: bytes, word: str) -> str:
    digest = hmac.new(index_key, word.encode("utf-8"), hashlib.sha512).digest()
    return base64.b64encode(digest[:_TERM_BYTES]).decode("ascii")


def text_terms(service: str, text: str) -> list[str]:
    """The hashed words of a chunk, stored as ``text_terms`` beside its text."""
    index_key = _derive(master_key(), "index", service)
    return sorted(_term(index_key, w) for w in words(text))


def query_terms(service: str, phrase: str) -> list[str]:
    """The hashes a chunk must hold ALL of to possibly contain ``phrase`` — a
    camelCase name is one word, a quoted message several."""
    index_key = _derive(master_key(), "index", service)
    return sorted({_term(index_key, w) for w in words(phrase)})
