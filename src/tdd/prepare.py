"""Steps 1 and 2 of the TDD flow (2026-11-02): which services can be read at
all, and the one commit each is read at.

Best rank first, each service's repo must look like owner/name, the token
must see it (one call per repo), and its branch — the repo's default branch
when none was given — is pinned to the commit it is at now. A pin is made
once per (repo, branch), so two services in one shared repo read the same
commit. Every later call uses that sha, never the branch name.

A service that fails any of this is skipped with the reason and the others go
on. A rate limit, or the run's own call cap, stops the checks for every
service still waiting: the next call would fail the same way. No usable
service at all is not an error here; the flow then writes a ticket-only
draft.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .github import GitHubError, GitHubReader, GitHubStopped
from .inputs import PlanInput, PullRef, repo_problem


@dataclass
class ReadyService:
    name: str
    repo: str
    branch: str  # as pinned: the input's branch, else the repo's default branch
    sha: str
    default_branch: str
    path: str  # folder inside a shared repo; "" for the whole repo
    rank: int
    reason: str
    prs: list[PullRef]


@dataclass
class Prepared:
    services: list[ReadyService] = field(default_factory=list)  # best rank first
    skipped: list[str] = field(default_factory=list)  # "<service>: why"
    warnings: list[str] = field(default_factory=list)


def prepare(plan: PlanInput, reader: GitHubReader) -> Prepared:
    out = Prepared()
    if not reader.has_token:
        out.warnings.append("GITHUB_TOKEN is not set: only public repositories can be read.")
    stopped = ""
    for service in plan.ranked_services():
        name = service.service
        if stopped:
            out.skipped.append(f"{name}: not checked, because {stopped}")
            continue
        problem = repo_problem(service.repo)
        if problem:
            out.skipped.append(f"{name}: {problem}")
            continue
        try:
            info = reader.repo(service.repo)
            branch = service.branch or info.default_branch
            sha = reader.head_sha(service.repo, branch)
        except GitHubStopped as error:
            stopped = str(error)
            out.skipped.append(f"{name}: {error}")
            continue
        except GitHubError as error:
            out.skipped.append(f"{name}: {error}")
            continue
        out.services.append(
            ReadyService(
                name=name,
                repo=service.repo,
                branch=branch,
                sha=sha,
                default_branch=info.default_branch,
                path=service.path,
                rank=service.rank,
                reason=service.reason,
                prs=list(service.prs),
            )
        )
        where = f" (folder {service.path})" if service.path else ""
        print(f"  [TDD] {name}: {service.repo}@{branch}{where} pinned to {sha[:12]}")
    for line in out.skipped:
        print(f"  [TDD] skipped {line}")
    for line in out.warnings:
        print(f"  [TDD] warning: {line}")
    return out
