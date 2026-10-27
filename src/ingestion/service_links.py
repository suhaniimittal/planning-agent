"""Stage 4 of ingestion: links ACROSS services, once every service is in.

Three concrete ways one service talks to another, each matched exactly:

- Kafka. A handler (`KafkaMessageHandler<UpdateHighRisk>`) and a publisher
  (`KafkaProducer<OrderKey, UpdateHighRisk>`) meet at the MESSAGE TYPE —
  its fully qualified name, resolved through each file's imports — not at
  the topic: one topic (`domain.orders`) carries many message types, so
  matching by topic would link every producer on it to every consumer.
  Topics are still recorded (`Message -ON_TOPIC-> Topic`), resolved from
  the `Topics.domain.orders` constants in the `kafka` repo.
- GraphQL. A client's root field (`query { reports }`) meets the service
  whose schema exposes `Query.reports`.
- REST. A client's literal request path meets a route a controller serves.

Each becomes `Service -CALLS {via, evidence}-> Service`. The older text
scan (a service name near "url"/"graphql" in some file) stays as the
fallback, recorded as `via: ["text-scan"]`, for any pair the concrete
links do not explain. A match that is ambiguous — the same operation
exposed by two services — is recorded on the file but never becomes a
service link: two candidates is not evidence for either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import code_graph_writer, graph_writer

# Evidence kept per service link: enough to show why it exists.
_MAX_EVIDENCE = 20


@dataclass
class CrossServiceLinks:
    messages: list[dict] = field(default_factory=list)
    handles: list[dict] = field(default_factory=list)
    publishes: list[dict] = field(default_factory=list)
    on_topic: list[dict] = field(default_factory=list)
    uses_operation: list[dict] = field(default_factory=list)
    calls_endpoint: list[dict] = field(default_factory=list)
    # (source service, target service) -> {via: set, evidence: list}
    service_links: dict[tuple[str, str], dict] = field(default_factory=dict)
    unresolved_topics: list[str] = field(default_factory=list)
    ambiguous_operations: int = 0

    def link(self, source: str, target: str, via: str, evidence: str) -> None:
        if source == target:
            return
        entry = self.service_links.setdefault((source, target), {"via": set(), "evidence": []})
        entry["via"].add(via)
        if evidence not in entry["evidence"] and len(entry["evidence"]) < _MAX_EVIDENCE:
            entry["evidence"].append(evidence)


def _topic_resolver(constant_rows: list[dict]):
    """Resolves `Topics.domain.orders` -> "orders" through the constant tree
    declared in Kotlin objects (`val domain = DomainTopics`, `const val
    orders = "orders"`). Constants from a service named `kafka` win a
    conflict: that repo is where the topic list is defined."""
    constants: dict[str, dict] = {}
    # The `kafka` service's rows go last, so on a conflict they overwrite.
    for row in sorted(constant_rows, key=lambda r: r["service"] == "kafka"):
        constants.update(row["constants"])

    def resolve(chain: str) -> str | None:
        parts = chain.split(".")
        current = parts[0]
        for i, member in enumerate(parts[1:], 1):
            entry = constants.get(f"{current}.{member}")
            if entry is None:
                return None
            if "value" in entry:
                return entry["value"] if i == len(parts) - 1 else None
            current = entry["alias"]
        return None

    return resolve


def _normalize_path(path: str) -> str:
    path = re.sub(r"\{[^}]*\}|:\w+", "{}", path.split("?")[0])
    return path.rstrip("/") or "/"


def compute(data: dict) -> CrossServiceLinks:
    result = CrossServiceLinks()

    # --- Kafka -------------------------------------------------------------
    messages = {row["fqn"]: row for row in data["avro"]}
    for fqn, row in messages.items():
        result.messages.append(
            {
                "name": fqn,
                "service": row["service"],
                "file_path": row["file_path"],
                "is_message": bool(row["is_message"]),
                "short_name": fqn.rsplit(".", 1)[-1],
            }
        )
    handled = {fqn for row in data["kafka"] for fqn in row["handles"]}
    resolve_topic = _topic_resolver(data["constants"])
    publishers: dict[str, set[str]] = {}
    handlers: dict[str, set[str]] = {}
    on_topic: set[tuple[str, str, str]] = set()
    for row in data["kafka"]:
        service, qname = row["service"], row["qualified_name"]
        built = [
            f
            for f in row["publishes_built"]
            if (f in messages and messages[f]["is_message"]) or f in handled
        ]
        sent = sorted(set(row["publishes"]) | set(built))
        for fqn in row["handles"]:
            result.handles.append({"service": service, "qualified_name": qname, "message": fqn})
            handlers.setdefault(fqn, set()).add(service)
        for fqn in sent:
            result.publishes.append({"service": service, "qualified_name": qname, "message": fqn})
            publishers.setdefault(fqn, set()).add(service)
        for constants_key, fqns in (("input_topics", row["handles"]), ("output_topics", sent)):
            for constant in row[constants_key]:
                topic = resolve_topic(constant)
                if topic is None:
                    result.unresolved_topics.append(constant)
                    continue
                for fqn in fqns:
                    on_topic.add((fqn, topic, constant))
    result.on_topic = [{"message": m, "topic": t, "constant": c} for m, t, c in sorted(on_topic)]
    for fqn, sources in publishers.items():
        for source in sources:
            for target in handlers.get(fqn, ()):
                result.link(source, target, "kafka", fqn.rsplit(".", 1)[-1])

    # --- GraphQL -----------------------------------------------------------
    exposed: dict[tuple[str, str], set[str]] = {}
    for row in data["operations"]:
        exposed.setdefault((row["type"], row["name"]), set()).add(row["service"])
    for row in data["client_operations"]:
        for operation in row["operations"]:
            kind, _, name = operation.partition(":")
            root = kind.capitalize()
            targets = exposed.get((root, name), set()) - {row["service"]}
            ambiguous = len(targets) > 1
            if ambiguous:
                result.ambiguous_operations += 1
            for target in sorted(targets):
                result.uses_operation.append(
                    {
                        "service": row["service"],
                        "file_path": row["file_path"],
                        "target_service": target,
                        "type": root,
                        "name": name,
                        "ambiguous": ambiguous,
                    }
                )
                if not ambiguous:
                    result.link(row["service"], target, "graphql", f"{root}.{name}")

    # --- REST --------------------------------------------------------------
    routes: dict[str, list[dict]] = {}
    for row in data["endpoints"]:
        routes.setdefault(_normalize_path(row["path"]), []).append(row)
    for row in data["http_calls"]:
        for call in row["paths"]:
            method, _, path = call.partition(" ")
            candidates = [
                e
                for e in routes.get(_normalize_path(path), [])
                if e["service"] != row["service"]
                and (method == "ANY" or e["method"] in ("ANY", method))
            ]
            if len({e["service"] for e in candidates}) != 1:
                continue
            for endpoint in candidates:
                result.calls_endpoint.append(
                    {
                        "service": row["service"],
                        "file_path": row["file_path"],
                        "target_service": endpoint["service"],
                        "method": endpoint["method"],
                        "path": endpoint["path"],
                    }
                )
                result.link(
                    row["service"],
                    endpoint["service"],
                    "rest",
                    f"{endpoint['method']} {endpoint['path']}",
                )

    # --- the text scan, for whatever the above does not explain -----------
    for row in data["text_scan"]:
        key = (row["service"], row["target"])
        if key not in result.service_links:
            result.link(row["service"], row["target"], "text-scan", f"{row['files']} file(s)")
    return result


def link_services(services: list[str]) -> CrossServiceLinks:
    """Compute and write every cross-service link among `services`, and
    rewrite each one's CALLS edges with how it talks to the others."""
    data = code_graph_writer.load_cross_service_facts(services)
    result = compute(data)
    code_graph_writer.write_cross_service_links(services, result)
    by_source: dict[str, list[dict]] = {name: [] for name in services}
    for (source, target), entry in result.service_links.items():
        if source in by_source:
            by_source[source].append(
                {
                    "target": target,
                    "via": sorted(entry["via"]),
                    "evidence": entry["evidence"],
                }
            )
    for source, links in by_source.items():
        graph_writer.set_service_links(source, links)
    return result
