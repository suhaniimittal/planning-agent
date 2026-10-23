"""Splitting an issue report into self-contained chunks with an LLM, for the
per-part code search in query_flow.

Why not the regex splitter alone (code_search.issue_queries): English
sentences lean on the ones before them, and embedding one sentence in
isolation severs that. Measured on the deferred-submit ticket, 5 of its 11
regex-split parts named nothing from the problem domain at all — including
the one carrying the root cause, "This regex is anchored at the start of the
string...", which never says WHICH regex. That part's vector lands near
generic parsing code instead of near isPhoneValid. Two further parts were cut
mid-sentence at "e.g.", which a period-based splitter cannot see coming.

An LLM resolves "this regex" back to `internationalPhoneNumber` using the
ticket's own words, so every chunk stands on its own. The risks that buys are
handled here rather than hoped away:

- Drift between runs: the split is cached on a hash of the issue text, so the
  same ticket re-uses the same chunks. The TDD for one ticket must not change
  run to run — the same reason planner.py pins temperature and a seed.
- Invention: a model asked to resolve a reference can just as easily invent
  one ("the regex in candidate-service PhoneValidator.ts"), which would aim
  retrieval at a service the ticket never mentioned. Every code-shaped token
  in the output must appear verbatim in the input, or the split is rejected
  whole.
- Unavailability: any failure — rate limit, timeout, unparseable JSON, a
  failed check — falls back to the regex splitter. This module is additive.
  It never becomes the only path to a query list.

The caller's contract is unchanged from code_search.issue_queries: element 0
is always the WHOLE ticket, because query_flow scores services on vectors[0]
alone and must keep doing so. Only elements 1..N differ.
"""

from __future__ import annotations

import hashlib
import json
import re

from . import code_search, planner

# Chunks shorter than this are fragments with no meaning of their own — the
# same floor code_search applies to regex-split sentences, for the same
# reason: their vectors match bland prose everywhere.
MIN_CHUNK_CHARS = 40
# A chunk longer than this means the model did not really split. Generous
# against the 300 the prompt asks for: an occasional long chunk is fine,
# whereas the whole ticket returned as one chunk (1319 chars on the
# deferred-submit ticket) is the failure this catches.
MAX_CHUNK_CHARS = 600
# Beyond this the split is rejected rather than truncated. Truncating would
# drop the LAST chunks, which are the reproduction steps — exactly the
# content-blind loss this module exists to avoid. The budget is the same one
# code_search.MAX_QUERY_PARTS sets, less the slot element 0 already spends.
MAX_CHUNKS = code_search.MAX_QUERY_PARTS - 1
# A ticket this long that comes back as one or two chunks was not split.
UNSPLIT_TEXT_CHARS = 600
MIN_CHUNKS_FOR_LONG_TEXT = 3

SPLIT_SYSTEM_PROMPT = (
    "You split software issue reports into chunks for semantic code search.\n\n"
    "Each chunk becomes one embedding, compared against embeddings of real "
    "source code to find the functions a developer must change. A chunk that "
    "cannot be understood on its own retrieves nothing useful.\n\n"
    "RULES\n\n"
    "1. MEANING — each chunk expresses one complete technical thought, or a "
    "closely related group of thoughts. Never cut mid-sentence.\n\n"
    "2. SIZE — aim for 40-300 characters. These are heuristics: never split a "
    "thought to hit the minimum, never pad to reach it, never truncate a "
    "thought to stay under the maximum. Produce between 5 and 15 chunks.\n\n"
    "3. CONTEXT — a chunk must stand alone. If a sentence depends on an "
    'earlier one ("this regex", "the error", "it fails", "the '
    'underlying mutation"), resolve that reference inside the chunk. Overlap '
    "between neighbouring chunks is expected and fine — repeat whatever a "
    "chunk needs to make sense by itself.\n\n"
    "4. PRESERVE NAMES EXACTLY — when you resolve a reference, use the "
    "ticket's own identifier verbatim (internationalPhoneNumber, "
    "approveDeferredReport, DEFERRED), never a description of it. Writing "
    '"the phone validation regex" where the ticket wrote '
    "internationalPhoneNumber destroys the strongest signal in the text. Copy "
    "quoted strings, error messages, regex literals and example values "
    "exactly as written.\n\n"
    "5. DROP FURNITURE — headers and labels carrying no technical content "
    '("Key details", "Description", "Steps to reproduce:-", page '
    "numbers, generation stamps, ticket-id banners) do not become chunks. "
    "Fold one into the chunk it introduces only if it adds meaning.\n\n"
    "6. PRESERVE, DO NOT INFER — rewrite only to resolve references and "
    "complete thoughts, using wording already present in the ticket. Do not "
    "add a cause, mechanism, file, service or component the ticket does not "
    "state. Do not conclude anything the reporter did not. Every identifier, "
    "quoted string and literal value in your output must appear verbatim in "
    'the input. A causal link the ticket itself asserts ("X is never sent, '
    'so Y stays stuck") may be preserved; one it does not assert may not be '
    "added.\n\n"
    "Keep chunks in the order they appear in the ticket.\n\n"
    'Also classify the report. "bug" — it describes something broken: a '
    "failure, a wrong result, reproduction steps, behaviour that used to "
    'work. "feature" — it asks for something that does not exist yet: a '
    "new capability, acceptance criteria, a user story. Judge the whole "
    "report, not a single phrase.\n\n"
    'Respond with ONLY a JSON object of the form {"mode": "bug" or '
    '"feature", "chunks": ["...", "..."]}, no markdown fences and no '
    "prose outside the JSON."
)

# Token shapes a model could plausibly invent while rewriting, and that would
# misdirect retrieval if it did: camelCase and PascalCase symbols, hyphenated
# service names (candidate-service), and source filenames. Ordinary prose
# words are not checked — they carry no retrieval weight and would make this
# reject almost every legitimate rewrite.
# Filenames first: alternation is ordered, and the PascalCase branch would
# otherwise claim "PhoneValidator" and leave the ".ts" behind, reporting half
# an invented name.
_CODE_SHAPED = re.compile(
    r"\b\w+\.(?:ts|tsx|js|jsx|java|py|go|rb|kt|cs|php|rs)\b"
    r"|\b[a-z]+[A-Z][A-Za-z0-9]*\b"
    r"|\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"
    r"|\b[a-z0-9]+(?:-[a-z0-9]+)+\b"
)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
_FIRST_OBJECT = re.compile(r"\{.*\}", re.S)

# Vocabulary that separates the two kinds of ticket. A bug report narrates a
# failure; a feature request states what shall exist. Measured on two real
# tickets: the deferred-submit bug scored 9 bug signals to 0 feature, and the
# bulk-print enhancement scored 0 to 11 — no overlap in either direction.
# Used as the fallback when the model does not classify, and as the
# cross-check that gets logged when the two disagree.
_BUG_SIGNALS = (
    r"steps?\s+to\s+reproduce",
    r"\breproduc",
    r"\bobserve[d:]?\b",
    r"expected\s+(behaviou?r|result)",
    r"actual\s+(behaviou?r|result)",
    r"\bbug\b",
    r"\bdefect\b",
    r"silently",
    r"does nothing",
    r"instead of",
    r"stuck",
    r"\bfails?\b",
    r"\berror\b",
    r"regression",
    r"no longer",
    r"broken",
)
_FEATURE_SIGNALS = (
    r"\bas an?\b[^.\n]{0,40}\b(user|client|admin)\b",
    r"\bi want\b",
    r"\bso that\b",
    r"enhancement",
    r"\bin scope\b",
    r"\bout of scope\b",
    r"acceptance criteria",
    r"\bshall\b",
    r"\bnew (platform|feature|capability)\b",
    r"\bparity\b",
    r"given .{0,80}when .{0,80}then",
    r"\bmust be able\b",
    r"\brequirement\b",
)
MODES = ("bug", "feature")


def keyword_mode(issue_text: str) -> tuple[str, int, int]:
    """(mode, bug_hits, feature_hits) from vocabulary alone — no model call.

    Ties and silence resolve to "bug": the bug prompt degrades gracefully on
    a feature ticket, while the feature prompt run on a real bug drops the
    value trace that locates where a bad value is written. Fail toward the
    mode that loses less.
    """
    low = issue_text.lower()
    bug = sum(1 for pattern in _BUG_SIGNALS if re.search(pattern, low))
    feature = sum(1 for pattern in _FEATURE_SIGNALS if re.search(pattern, low))
    return ("feature" if feature > bug else "bug"), bug, feature


# Process-local: one agent run plans one ticket, sometimes more than once
# (a retry, a test). Nothing persists between runs, which is fine — the point
# is that a single ticket cannot split two different ways within a run.
_cache: dict[str, tuple[str, list[str]]] = {}


def _cache_key(issue_text: str) -> str:
    return hashlib.sha256(issue_text.strip().encode()).hexdigest()


def _parse_object(raw: str) -> dict | list | None:
    """The model's reply -> the JSON value it carries, or None.

    Tolerant on purpose: only the OpenAI path can ask for native JSON mode,
    so the gateway path may wrap the object in markdown fences or prose
    (the same asymmetry planner.parse_tdd defends against). Never raises —
    a failure here means a failed split, not a failed run.
    """
    text = _FENCE.sub("", raw or "").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        match = _FIRST_OBJECT.search(text)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def parse_mode(raw: str) -> str | None:
    """The model's `mode` field, or None when absent or not a known mode.

    Unknown values are discarded rather than passed through: an
    unrecognised mode would silently fall back inside
    build_system_prompt, hiding the fact that the model answered
    something nonsensical.
    """
    data = _parse_object(raw)
    mode = data.get("mode") if isinstance(data, dict) else None
    return mode if mode in MODES else None


def parse_chunks(raw: str) -> list[str]:
    """The model's reply -> a list of chunk strings.

    Tolerant on purpose: only the OpenAI path can ask for native JSON mode,
    so the gateway path may wrap the object in markdown fences or prose
    (the same asymmetry planner.parse_tdd defends against).

    Returns [] for anything unparseable — the caller treats that as a failed
    split and falls back, rather than raising into the middle of a run.
    """
    data = _parse_object(raw)
    if data is None:
        return []
    chunks = data.get("chunks") if isinstance(data, dict) else data
    if not isinstance(chunks, list):
        return []
    return [c.strip() for c in chunks if isinstance(c, str) and c.strip()]


def invented_tokens(chunks: list[str], issue_text: str) -> list[str]:
    """Code-shaped tokens in `chunks` that do not appear in `issue_text`.

    A non-empty result means the model wrote a symbol, service or filename
    the ticket never contained. Retrieval would then search for something
    that does not exist, so the whole split is discarded — one invented name
    is enough, because there is no way to tell which other chunks the same
    reasoning touched.
    """
    found: set[str] = set()
    for chunk in chunks:
        found.update(_CODE_SHAPED.findall(chunk))
    return sorted(t for t in found if t not in issue_text)


def usable(chunks: list[str], issue_text: str) -> tuple[list[str], str | None]:
    """`chunks` filtered to what's worth embedding, or ([], reason).

    Short chunks are dropped rather than failing the split — one stray
    fragment says nothing about the rest. The other checks reject the split
    whole, because each one means the model misread the task.
    """
    kept = [c for c in chunks if len(c) >= MIN_CHUNK_CHARS]
    if not kept:
        return [], "no chunk reached the minimum length"
    if len(kept) > MAX_CHUNKS:
        return [], f"{len(kept)} chunks exceeds the {MAX_CHUNKS}-part budget"
    longest = max(len(c) for c in kept)
    if longest > MAX_CHUNK_CHARS:
        return [], f"a chunk of {longest} chars was not really split"
    if len(issue_text) > UNSPLIT_TEXT_CHARS and len(kept) < MIN_CHUNKS_FOR_LONG_TEXT:
        return [], f"only {len(kept)} chunks for a {len(issue_text)}-char issue"
    invented = invented_tokens(kept, issue_text)
    if invented:
        return [], f"invented {', '.join(invented)}"
    return kept, None


def _keyword_only(issue_text: str) -> str:
    """The mode when the model gave us nothing usable."""
    mode, bug, feature = keyword_mode(issue_text)
    print(f"  [TDD] mode: {mode} (keywords; {bug} bug / {feature} feature signals)")
    return mode


def _decide_mode(raw: str, issue_text: str) -> str:
    """The model's classification, with keywords as fallback and cross-check.

    A disagreement is logged rather than resolved quietly: the model reads
    the whole report and wins, but a run where the two differ is worth
    noticing, because it is the shape of ticket most likely to be
    classified wrong.
    """
    stated = parse_mode(raw)
    guess, bug, feature = keyword_mode(issue_text)
    if stated is None:
        print(f"  [TDD] mode: {guess} (keywords; {bug} bug / {feature} feature signals)")
        return guess
    agreement = "agree" if stated == guess else f"DISAGREE, keywords said {guess}"
    print(f"  [TDD] mode: {stated} (llm; keywords {agreement} — {bug} bug / {feature} feature)")
    return stated


async def split_issue(issue_text: str) -> tuple[str, list[str]]:
    """`issue_text` -> (mode, chunks); chunks is [] if the split can't be trusted.

    The mode is always usable: the model classifies in the same call that
    splits, and keyword_mode decides whenever that call fails, omits the
    field, or answers with something unrecognised. Both are cached, so one
    ticket cannot be classified two ways within a run.

    Never raises: every failure path returns [] so the caller falls back to
    the regex splitter. A broken split must cost the run its improvement,
    not the run itself.
    """
    key = _cache_key(issue_text)
    if key in _cache:
        return _cache[key]
    try:
        raw = await planner.call_llm(
            SPLIT_SYSTEM_PROMPT, f"<issue>\n{issue_text.strip()}\n</issue>"
        )
    except Exception as e:  # noqa: BLE001 - any provider failure falls back
        print(f"  [TDD] issue split unavailable ({type(e).__name__}), using sentence split")
        return _keyword_only(issue_text), []
    chunks = parse_chunks(raw)
    if not chunks:
        print("  [TDD] issue split returned no usable JSON, using sentence split")
        return _keyword_only(issue_text), []
    kept, reason = usable(chunks, issue_text)
    if reason:
        print(f"  [TDD] issue split rejected ({reason}), using sentence split")
        return _keyword_only(issue_text), []
    mode = _decide_mode(raw, issue_text)
    _cache[key] = (mode, kept)
    return mode, kept


async def issue_queries(issue_text: str) -> tuple[str, list[str]]:
    """The whole ticket first, then its self-contained chunks.

    Drop-in async replacement for code_search.issue_queries, with the same
    contract: element 0 is the entire ticket and is what query_flow scores
    services with. Falls back to that function's own output when the LLM
    split is unavailable or untrustworthy.
    """
    whole = issue_text.strip()
    mode, chunks = await split_issue(issue_text)
    if not chunks:
        return mode, code_search.issue_queries(issue_text)
    print(f"  [TDD] issue split into {len(chunks)} self-contained chunks")
    return mode, [whole] + chunks
