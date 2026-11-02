"""Step 3 of the TDD flow (2026-11-02): the service map, for this run's
services only.

With code reads on, for each service the files that show how it connects —
contracts, config, build files, controllers, clients, handlers (classify.py)
— are read at the pinned commit, facts are pulled out of them (extract.py)
and checked (verify.py), then calls are matched with offers across the
services (link.py). A service's file text is dropped as soon as its facts
are out, and once the links are made every quote and every config value is
blanked: the map keeps names, routes, topics and paths, never code or
configuration values (a config file can hold passwords).

With code reads off it runs paths-only: classify still picks the files by
their paths, nothing is read, so there are no facts and no links — the view
lists each service's contract, config, build and connection files by path.

``view()`` is what the LLM sees. ``pointers()`` is what evidence.py matches
the ticket's terms against, to find files the map points to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings
from ..github import GitHubError, GitHubReader, GitHubStopped
from ..prepare import ReadyService
from .classify import classify_with_cut
from .extract import extract_into
from .link import LinkResult, link_all
from .schema import ServiceFacts
from .verify import verify

# How much of each kind the LLM is shown per service, and links in total.
_VIEW_ITEMS = 40
_VIEW_FILES = 15
_VIEW_LINKS = 60


@dataclass
class MapSlice:
    paths_only: bool
    services: list[ServiceFacts] = field(default_factory=list)
    # service -> kind ("contract" | "config" | "build" | "code") -> paths
    files: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # service -> the folder actually used ("" = the whole repo)
    folders: dict[str, str] = field(default_factory=dict)
    links: LinkResult | None = None
    notes: list[str] = field(default_factory=list)

    def view(self) -> dict:
        """Names, routes, topics, links and paths: no quotes, no config values."""
        services = []
        for facts in self.services:
            services.append(
                {
                    "service": facts.name,
                    "folder": self.folders.get(facts.name, ""),
                    "stack": facts.stack,
                    "offers": [
                        {"route": e.route, "file": e.evidence.file}
                        for e in facts.exposes[:_VIEW_ITEMS]
                    ],
                    "calls": [
                        {
                            "route": c.route,
                            "target": c.target_hint or c.url_hint or "?",
                            "file": c.evidence.file,
                        }
                        for c in facts.calls[:_VIEW_ITEMS]
                    ],
                    "kafka": [
                        {
                            "role": t.role,
                            "what": t.topic or f"message {t.message_type}",
                            "file": t.evidence.file,
                        }
                        for t in facts.topics[:_VIEW_ITEMS]
                    ],
                    "files": {
                        kind: paths[:_VIEW_FILES]
                        for kind, paths in self.files.get(facts.name, {}).items()
                        if paths
                    },
                }
            )
        links = []
        if self.links:
            links = [
                {
                    "from": link.from_service,
                    "to": link.to_service,
                    "via": link.via,
                    "detail": link.detail,
                    "confidence": link.confidence,
                }
                for link in self.links.links[:_VIEW_LINKS]
            ]
        return {
            "paths_only": self.paths_only,
            "services": services,
            "links": links,
            "calls_with_no_known_receiver": len(self.links.unmatched_calls) if self.links else 0,
        }

    def pointers(self, service: str) -> list[tuple[str, str]]:
        """[(what, file)]: each route, call and topic the map found in
        ``service``, with the file it is in."""
        for facts in self.services:
            if facts.name == service:
                return (
                    [(e.route, e.evidence.file) for e in facts.exposes]
                    + [(f"{c.route} {c.target_hint}", c.evidence.file) for c in facts.calls]
                    + [(f"{t.topic} {t.message_type}", t.evidence.file) for t in facts.topics]
                )
        return []


def build_slice(reader: GitHubReader, services: list[ReadyService], settings: Settings) -> MapSlice:
    out = MapSlice(paths_only=not settings.allow_code_reads)
    stopped = ""
    for svc in services:
        if stopped:
            out.notes.append(f"{svc.name}: not mapped, because {stopped}")
            continue
        try:
            tree = reader.tree(svc.repo, svc.sha)
        except GitHubStopped as error:
            stopped = str(error)
            out.notes.append(f"{svc.name}: not mapped: {error}")
            continue
        except GitHubError as error:
            out.notes.append(f"{svc.name}: not mapped: {error}")
            continue
        folder = svc.path
        if folder and not any(p.startswith(folder + "/") for p in tree.paths):
            out.notes.append(
                f"{svc.name}: the folder '{folder}' is not in {svc.repo} at {svc.sha[:12]}; "
                "the whole repository is used"
            )
            folder = ""
        out.folders[svc.name] = folder
        picked, cut = classify_with_cut(list(tree.paths), folder, settings.map_files_per_service)
        kinds: dict[str, list[str]] = {}
        for kind, path in picked:
            kinds.setdefault(kind, []).append(path)
        out.files[svc.name] = kinds
        if cut:
            out.notes.append(
                f"{svc.name}: the service map used {len(picked)} of {len(picked) + cut} "
                "connection files (TDD_MAP_FILES_PER_SERVICE)"
            )
        facts = ServiceFacts(
            name=svc.name, repo=svc.repo, path=folder, branch=svc.branch, commit=svc.sha
        )
        out.services.append(facts)
        if out.paths_only:
            continue
        texts: dict[str, str] = {}
        try:
            for _, path in picked:
                try:
                    texts[path] = reader.read(svc.repo, path, svc.sha).text
                except GitHubStopped:
                    raise
                except GitHubError as error:
                    facts.files_skipped.append(f"{path}: {error}")
        except GitHubStopped as error:
            stopped = str(error)
            out.notes.append(f"{svc.name}: mapped from {len(texts)} files only: {error}")
        finally:
            extract_into(facts, texts)
            verify(facts, texts)
            texts.clear()
        print(
            f"  [TDD] map {svc.name}: {len(facts.files_read)} files, {len(facts.exposes)} offers, "
            f"{len(facts.calls)} calls, {len(facts.topics)} kafka uses"
        )
    if not out.paths_only and out.services:
        out.links = link_all(out.services)
        print(
            f"  [TDD] map: {len(out.links.links)} links, "
            f"{len(out.links.unmatched_calls)} calls with no known receiver"
        )
    _forget_text(out)
    return out


def _forget_text(out: MapSlice) -> None:
    """Blanks every quote (a line of code) and every config value, now that
    verification and linking no longer need them."""
    for facts in out.services:
        facts.config.clear()
        for item in [*facts.exposes, *facts.calls, *facts.topics, *facts.libs]:
            item.evidence.quote = ""
    if out.links:
        for link in out.links.links:
            for evidence in (link.evidence_from, link.evidence_to):
                if evidence is not None:
                    evidence.quote = ""
        for group in (out.links.unmatched_calls, out.links.unmatched_topics):
            for item in group:
                if item.get("evidence") is not None:
                    item["evidence"].quote = ""
