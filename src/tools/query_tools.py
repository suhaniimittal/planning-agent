"""Tool entrypoint for the query-time flow."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime

from aetherion_sdk import tool
from common_lib.storage.storage_client import storage

from src.ingestion import github_api
from src.ingestion.confluence_api import ConfluenceApiError, create_page
from src.memory import record
from src.query import contracts, stack
from src.query.design_qa import answer_question, legacy_answer
from src.query.pdf_render import render_pdf
from src.query.query_flow import build_tdd
from src.query.render import render_confluence, render_text

_NON_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def _publish_to_confluence(tdd) -> str | None:
    """Best-effort: publishes the TDD as a new Confluence page and returns
    its URL, or None if Confluence isn't configured or the call fails for
    any reason. Never raises — a Confluence outage or a missing credential
    must never stop TDD generation from succeeding, same "one failure never
    aborts the run" principle used throughout this pipeline (SummarizerError,
    GitHubApiError, RepoFetchError all follow it too)."""
    space_key = os.environ.get("CONFLUENCE_SPACE_KEY")
    if not space_key:
        return None
    try:
        body_html = render_confluence(tdd)
        return create_page(space_key, tdd.title, body_html)
    except ConfluenceApiError as e:
        print(f"  [CONFLUENCE SKIP] could not publish {tdd.title!r}: {e}")
        return None


def _team_id() -> str:
    """Same bucket-resolution convention the ingestion pipeline used for its
    (now-removed) S3 doc source: TENANT_ID, falling back to
    STORAGE_AGENTS_BUCKET — the bucket name must equal the team_id per the
    storage client's own contract."""
    return os.environ.get("TENANT_ID") or os.environ["STORAGE_AGENTS_BUCKET"]


def _slug(text: str, max_len: int = 45) -> str:
    """A filename-safe slug, cut at a word boundary.

    Cutting at a fixed character count left names ending mid-word
    ("...silently-fails-wit"), so the truncation backs up to the last
    hyphen instead when one is close enough to the limit."""
    slug = _NON_SLUG_CHARS.sub("-", text.lower()).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    if len(slug) <= max_len:
        return slug or "technical-design"
    cut = slug.rfind("-", max_len // 2, max_len)
    return slug[: cut if cut > 0 else max_len].rstrip("-") or "technical-design"


def _stamp(generated_at: str | None) -> str:
    """The run's own generation time as a compact, sortable filename part.

    Taken from the TDD's `generated_at` rather than the clock, so the
    filename and the "Generated:" line inside the PDF always name the same
    run — the whole point of the stamp is telling two runs of one ticket
    apart, which a filename that disagreed with the document would defeat.
    Falls back to now() only for a TDD that was never stamped."""
    try:
        when = datetime.fromisoformat(generated_at) if generated_at else datetime.now(UTC)
    except ValueError:
        when = datetime.now(UTC)
    return when.astimezone(UTC).strftime("%Y%m%d-%H%M%S")


@tool()
async def answer_design_question(
    question: str,
    tdd_text: str,
    discussion: list[str] | None = None,
    issue_key: str | None = None,
    kind: str | None = None,
) -> dict:
    """A comment on a TDD ("why this approach?", "why not X?", "update the TDD: X"),
    answered from the TDD, the story's facts, the decisions in force, the
    discussion and the related code — and judged: a change is applied only when
    it is valid (2026-10-26). Changes nothing itself; the caller revises.
    With no ``kind`` (a caller with memory off) it answers exactly as before."""
    if not kind:
        return await legacy_answer(question, tdd_text, discussion)
    facts, decisions = [], []
    if issue_key:
        stored = record.load_requirement(issue_key)
        facts = (stored or {}).get("facts") or []
        decisions = record.decisions_in_force(issue_key)
    answer = await answer_question(
        question, tdd_text, discussion, kind or "question", facts, decisions
    )
    return answer.to_dict()


def repository_facts(service_repos: dict[str, str]) -> tuple[list[str], str]:
    """(technology constraints, API contracts) of each service's repository (2026-10-15).

    One call lists the repository's files; then only its build files (the root
    one, every module's, the Gradle catalog) and its API contracts (OpenAPI,
    AsyncAPI, protobuf) are read. A repository that cannot be read adds a line
    saying so — the design then knows its versions are unknown."""
    constraints: list[str] = []
    contract_files: dict[str, list[tuple[str, str]]] = {}
    for name, slug in sorted(service_repos.items()):
        repo, _, branch = slug.partition("@")
        ref = branch or "HEAD"
        try:
            paths = github_api.list_tree(repo, branch or None)
            def read(path: str, repo: str = repo, ref: str = ref) -> str:
                return github_api.fetch_file_content(repo, path, ref)

            found = stack.read_stack(paths, read)
            files = [(p, read(p)) for p in contracts.contract_paths(paths)]
        except github_api.GitHubApiError as exc:
            constraints.append(f"{name}: build files could not be read ({exc}); versions unknown.")
            continue
        prefix = f"{name}: " if len(service_repos) > 1 else ""
        constraints.extend(prefix + rule for rule in found.constraints())
        if files:
            contract_files[name] = files
    return constraints, contracts.contracts_section(contract_files)


def technology_constraints(service_repos: dict[str, str]) -> list[str]:
    """Each service's real stack, from its build files (see repository_facts)."""
    return repository_facts(service_repos)[0]


@tool()
async def generate_technical_design(
    issue_text: str,
    top_k: int | None = None,
    hops: int | None = None,
    service_names: list[str] | None = None,
    return_document: bool = False,
    service_repos: dict[str, str] | None = None,
    mode: str | None = None,
    issue_key: str | None = None,
) -> dict:
    """Search the service graph for services relevant to a requirement/bug
    report and produce a technical design doc: structured data plus a real
    downloadable PDF — the file a user downloads and hands, unmodified, to
    coding_agent's own PDF input. Uploaded via the platform's storage
    client, with its S3 key returned so the calling agent can surface it as
    a structured_results download link (this is the platform's only file-
    download mechanism, but nothing about it is ever exposed to the user —
    they just see a "Download TDD" button). Also published as a new
    Confluence page when CONFLUENCE_SPACE_KEY is configured — best-effort,
    never blocks the PDF output if it fails or isn't configured.

    `service_names`: the services this run covers; None considers every
    service in the graph. `service_repos` ("name" -> "org/repo[@branch]"): the
    repositories whose build files give the design its technology
    constraints. `return_document`: also return the title, the
    Confluence HTML and the plain text, for the orchestrator. `mode`:
    "bug" or "feature" to override how the ticket is classified. `issue_key`:
    the story, whose decisions in force — and its project's, on the same
    services — the design must keep (2026-10-26)."""
    constraints, api_contracts = repository_facts(service_repos or {})
    decisions_text = ""
    if issue_key:
        decisions_text = record.decisions_block(
            record.decisions_in_force(issue_key, service_names), issue_key
        )
    # Only what the caller actually chose is forwarded, so build_tdd's own
    # signature stays the single definition of what top_k and hops default
    # to — nothing here restates the numbers.
    chosen = {
        name: value for name, value in (("top_k", top_k), ("hops", hops)) if value is not None
    }
    tdd = await build_tdd(
        issue_text,
        service_names=service_names,
        technology_constraints=constraints,
        api_contracts=api_contracts,
        mode=mode,
        decisions_text=decisions_text,
        **chosen,
    )
    pdf_bytes = render_pdf(tdd)

    storage.init_client()
    # Named from the TDD's own title, not the raw issue text: the title is
    # written to describe the change, while the ticket text opens with
    # whatever the reporter typed first and truncated to something like
    # "deferred-order-submit-silently-fails-wit".
    #
    # The timestamp suffix gives every run its own name and its own storage
    # key. Without it, re-running a ticket overwrote its previous PDF at the
    # same key, and a browser downloading the result still appended "(2)",
    # "(3)" to avoid a local filename collision — so a stale download and a
    # fresh one looked identical and only the "Generated:" line inside the
    # PDF could tell them apart. The cost is accepted: one ticket run twice
    # now leaves two files.
    file_name = f"technical_designs/{_slug(tdd.title)}-{_stamp(tdd.generated_at)}.pdf"
    _, s3_key = storage.store_object(_team_id(), file_name, pdf_bytes, "application/pdf")

    _publish_to_confluence(tdd)

    result = {"s3_key": s3_key}
    if return_document:
        # For the orchestrator, which publishes the TDD in the project's
        # Confluence space and hands its text to Coding (2026-10-08).
        result |= {
            "title": tdd.title,
            "confluence_html": render_confluence(tdd),
            "text": render_text(tdd),
            "decisions": [d.model_dump() for d in tdd.decisions],
        }
    return result
