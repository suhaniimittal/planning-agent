"""Pure text-extraction functions for a single service's documentation.md.

No I/O here — every function takes the raw markdown string and returns
plain Python data. Keeping these side-effect-free makes them easy to test
against the real docs (orders/user/shipping) independently of Neo4j,
GitHub, or the embedding API.
"""

from __future__ import annotations

import re

# Last-resort escape hatch for a mention that's genuinely not derivable from
# any service's own doc text (e.g. a narrative typo/rebrand mismatch) — NOT
# the primary mechanism. Cases like "customer/address/card -> user" are
# instead derived automatically at ingestion time by extract_owned_entities()
# + build_entity_index(), and "cart -> carts" by the pluralization check in
# resolve_service_name(). An entry here is a signal that RepoDocAI's doc
# generation or extract_owned_entities needs improving, not a fixture to keep.
MANUAL_OVERRIDES: dict[str, str] = {}

# Symbols starting with one of these (camel-case first subword) name an
# action/handler, not a business entity — e.g. AddCustomer, GetUserUri.
_ACTION_PREFIXES = {
    "add", "get", "set", "make", "new", "init", "test", "create", "update",
    "delete", "remove", "list", "find", "fetch", "load", "save", "build",
    "register", "login", "logout", "validate", "check", "handle", "process",
    "send", "receive", "parse", "convert", "map", "mask", "is", "has", "to",
    "from",
}

# Symbols ending with one of these (camel-case last subword) name an
# infrastructure/role concept, not an owned business entity — e.g.
# UserController, PaymentException, LoginEndpoint.
_ROLE_SUFFIXES = {
    "endpoint", "handler", "controller", "service", "repository", "test",
    "tests", "config", "configuration", "exception", "error", "request",
    "response", "dto", "impl", "factory", "provider", "listener", "filter",
    "interceptor", "resolver", "middleware", "client", "util", "utils",
    "helper", "manager",
}


def extract_service_name(md_text: str) -> str | None:
    """Canonical lowercase service name, from the 'Generated from `org/repo`' line."""
    m = re.search(r"Generated from `[\w-]+/([\w-]+)`", md_text)
    return m.group(1).lower() if m else None


def extract_repo_slug(md_text: str) -> str | None:
    """Full 'org/repo' slug from the same 'Generated from `org/repo`' line.

    Needed for cloning (repo_fetcher.py) — extract_service_name() deliberately
    keeps only the repo part for use as the Service node key, so this is a
    separate function rather than a change to that one's return shape.
    """
    m = re.search(r"Generated from `([\w-]+/[\w-]+)`", md_text)
    return m.group(1) if m else None


def extract_project_overview(md_text: str) -> str | None:
    """The one-paragraph summary under '## Project Overview' — used for embedding."""
    m = re.search(
        r"^##\s+Project Overview\s*\n+(.*?)(?=\n\s*\n|\n>|\n\|)",
        md_text,
        re.MULTILINE | re.DOTALL,
    )
    return m.group(1).strip() if m else None


def extract_technology_stack(md_text: str) -> list[tuple[str, str]]:
    """[(technology, purpose), ...] from the '## Technology Stack' table.

    Used as a fallback summary source for docs with no Project Overview
    section (e.g. library/non-service repos like Flask, where RepoDocAI's
    leaner template skips the business-overview sections entirely).
    """
    section = re.search(r"##\s+Technology Stack.*?(?=\n---|\Z)", md_text, re.DOTALL)
    if not section:
        return []
    rows = re.findall(r"^\|\s*\*\*([^*]+)\*\*\s*\|\s*([^|]+?)\s*\|", section.group(0), re.MULTILINE)
    return [(tech.strip(), purpose.strip()) for tech, purpose in rows]


def build_fallback_summary(md_text: str, name: str) -> str | None:
    """Synthetic one-line summary from the tech stack, when there's no
    Project Overview paragraph to embed directly."""
    stack = extract_technology_stack(md_text)
    if not stack:
        return None
    tech_desc = ", ".join(f"{tech} ({purpose})" for tech, purpose in stack[:8])
    return f"{name}: repository using {tech_desc}."


def extract_overview_table(md_text: str) -> dict[str, str]:
    """Key-value rows from the Project Overview table (Domain, Entry points, ...).

    Returns {} gracefully if the section or a given row is absent — not every
    doc has every row (e.g. `user`'s doc has no 'External systems' row).
    """
    section = re.search(r"##\s+Project Overview.*?(?=\n---)", md_text, re.DOTALL)
    if not section:
        return {}
    rows = re.findall(r"^\|\s*([^|\n]+?)\s*\|\s*([^|\n]+?)\s*\|$", section.group(0), re.MULTILINE)
    return {k.strip(): v.strip() for k, v in rows if k.strip() not in ("", "-", "---")}


def extract_key_symbols(md_text: str) -> list[str]:
    """Flat list of every symbol in the 'Key symbols' column of the File Reference table.

    Finds the column by its HEADER NAME, not a fixed position — different
    docs can have different column counts/orders, so position-based parsing
    breaks silently across repos.

    Scoped to the '## File Reference' section specifically — richer docs also
    have a '## Components' section where each component's own property table
    has an unrelated row literally containing the text "Key symbols" (e.g.
    "| Key symbols | `newOrder`, `calculateTotal` |"). Searching the whole
    doc for the first line containing that text grabs the wrong table.
    """
    file_ref_section = re.search(r"##\s+File Reference.*", md_text, re.DOTALL)
    if not file_ref_section:
        return []
    lines = file_ref_section.group(0).splitlines()

    header_idx, col_idx = None, None
    for i, line in enumerate(lines):
        if line.strip().startswith("|") and "Key symbols" in line:
            headers = [h.strip() for h in line.strip("|").split("|")]
            col_idx = headers.index("Key symbols")
            header_idx = i
            break

    if col_idx is None:
        return []

    symbols: list[str] = []
    for line in lines[header_idx + 2 :]:  # +2 skips header row + '|---|---|' separator
        if not line.strip().startswith("|"):
            break  # table ended
        cells = [c.strip() for c in line.strip("|").split("|")]
        if col_idx >= len(cells):
            continue
        symbols += [s.strip(" `") for s in cells[col_idx].split(",") if s.strip(" `")]
    return symbols


def _camel_subwords(symbol: str) -> list[str]:
    return re.findall(r"[A-Z][a-z0-9]*", symbol)


def extract_owned_entities(md_text: str, symbols: list[str]) -> set[str]:
    """Lowercased 'entity' nouns this service owns (e.g. {'address', 'card',
    'customer'} for the `user` service), derived from its own key_symbols.

    Conservative by design: only symbols that look like nouns naming a thing
    (not a verb-prefixed action or an infrastructure/role name) are kept, and
    when a Project Overview exists, candidates must also appear in the
    service's own Domain/overview text. When in doubt, drop the symbol rather
    than include it — a missing entity signal just means a mention stays
    unresolved (logged for review); a wrong one creates a wrong edge silently.
    """
    name = extract_service_name(md_text)

    candidates: set[str] = set()
    for symbol in symbols:
        if len(symbol) < 3 or not re.match(r"^[A-Z][a-zA-Z]*$", symbol):
            continue
        subwords = _camel_subwords(symbol)
        if not subwords:
            continue
        if subwords[0].lower() in _ACTION_PREFIXES:
            continue
        if subwords[-1].lower() in _ROLE_SUFFIXES:
            continue
        candidates.add(symbol.lower())

    domain = extract_overview_table(md_text).get("Domain", "")
    overview = extract_project_overview(md_text) or ""
    context_text = f"{domain} {overview}".strip().lower()

    if context_text:
        def _in_context(word: str) -> bool:
            variants = {word, f"{word}s", f"{word}es"}
            if word.endswith("s"):
                variants.add(word[:-1])
            if word.endswith("es"):
                variants.add(word[:-2])
            return any(re.search(rf"\b{re.escape(v)}\b", context_text) for v in variants)

        candidates = {c for c in candidates if _in_context(c)}

    if name:
        candidates.discard(name.lower())

    return candidates


def build_entity_index(records: list[dict]) -> dict[str, str]:
    """Global {entity_name: owning_service} index across all services.

    An entity claimed by more than one service is ambiguous and is dropped
    from the index entirely — same "never guess" philosophy as
    resolve_service_name itself.
    """
    claims: dict[str, set[str]] = {}
    for record in records:
        owned = extract_owned_entities(record["full_doc"], record["key_symbols"])
        for entity in owned:
            claims.setdefault(entity, set()).add(record["name"])

    index: dict[str, str] = {}
    for entity, owners in claims.items():
        if len(owners) == 1:
            index[entity] = next(iter(owners))
        else:
            print(f"  [AMBIGUOUS ENTITY] {entity!r} claimed by {sorted(owners)} — dropped")
    return index


def extract_config_uri_targets(symbols: list[str]) -> list[str]:
    """Signal A: config getters like getPaymentUri / getShippingUrl -> ['Payment', 'Shipping'].

    Most reliable cross-service signal we have: a property literally named
    after the service it points to.
    """
    targets = []
    for s in symbols:
        m = re.match(r"get(\w+?)(?:Uri|Url)$", s)
        if m:
            targets.append(m.group(1))
    return targets


def extract_narrative_targets(md_text: str) -> list[str]:
    """Signal B (singular): 'to/using/calls the payment service' style mentions."""
    return [
        m.strip()
        for m in re.findall(r"(?:to|using|calls?)\s+the\s+([\w\s]+?)\s+service", md_text, re.IGNORECASE)
    ]


def extract_narrative_list_targets(md_text: str) -> list[str]:
    """Signal B (list): 'calls to customer, address, card, ... services' style mentions.

    Separate from the singular pattern above — this grammar has no 'the' and
    ends in the plural 'services', so it needs its own regex or it's silently
    missed (found this gap by testing against the real Orders doc).
    """
    m = re.search(r"calls?\s+to\s+([\w,\s]+?)\s+services?\b", md_text, re.IGNORECASE)
    if not m:
        return []
    return [x.strip() for x in re.split(r",|\band\b", m.group(1)) if x.strip()]


def _singular(s: str) -> str:
    return s[:-1] if s.endswith("s") and len(s) > 3 else s


def resolve_service_name(
    mention: str,
    known_services: set[str],
    entity_index: dict[str, str] | None = None,
) -> str | None:
    """Resolve a raw text mention to a canonical service name, or None if unresolvable.

    Deliberately returns None instead of guessing — an unresolved mention
    should be logged and reviewed, not silently dropped or fuzzy-matched
    into a wrong edge. Tries, in order: exact match, singular/plural
    normalization (e.g. "cart" -> "carts"), the derived entity index (e.g.
    "customer" -> "user"), then a manual override, before giving up.
    """
    entity_index = entity_index or {}
    key = mention.strip().lower()

    for known in known_services:
        if key == known.lower():
            return known

    for known in known_services:
        if _singular(key) == _singular(known.lower()):
            return known

    if key in entity_index:
        return entity_index[key]

    if key in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[key]

    return None
