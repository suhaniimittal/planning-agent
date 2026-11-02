"""Matches what one service calls with what another offers.

  REST     call "POST /payments/validate"  <->  offer "POST /payments/validate"
  GraphQL  client root field "reports"     <->  schema "Query.reports"
  Kafka    producer topic / message type   <->  consumer topic / message type
  Shared   a build-file dependency         <->  a declared shared library

How sure a link is:
  exact     the target service is known (by name, host or config) AND the route matches
  probable  the route (or topic / message type) matches, the target was not named
  guess     more than one service could be the receiver; all are listed

Paths ignore parameter names: /payments/{id} equals /payments/:id.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schema import Evidence, Link, ServiceFacts

RANK = {"exact": 0, "probable": 1, "guess": 2}


@dataclass
class LinkResult:
    links: list[Link] = field(default_factory=list)
    unmatched_calls: list[dict] = field(default_factory=list)  # calls nobody offers
    unused_offers: list[dict] = field(default_factory=list)  # offers nobody calls
    unmatched_topics: list[dict] = field(default_factory=list)
    shared: list[dict] = field(default_factory=list)


# --- names, routes, config ---------------------------------------------------------------


def norm_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def norm_route(route: str) -> tuple[str, list[str]]:
    """("POST", ["payments", "{}", "validate"]) from "POST /payments/{id}/validate"."""
    method, sep, path = route.partition(" ")
    if not sep:
        method, path = "ANY", route
    segments = []
    for seg in path.split("?")[0].lower().strip("/").split("/"):
        if not seg:
            continue
        param = re.fullmatch(r"\{[^}]*\}|:\w+|<[^>]*>|\$\{[^}]*\}|\$\w+", seg)
        segments.append("{}" if param else seg)
    return method.upper(), segments


def _methods_ok(a: str, b: str) -> bool:
    return a == b or "ANY" in (a, b)


def _same(a: list[str], b: list[str]) -> bool:
    return len(a) == len(b) and all(x == y or "{}" in (x, y) for x, y in zip(a, b, strict=True))


def _suffix(a: list[str], b: list[str]) -> bool:
    short, long = sorted((a, b), key=len)
    return len(short) >= 2 and _same(short, long[-len(short) :])


def lookup_config(config: dict[str, str], key: str) -> str:
    """The value of a config key; matches "PAYMENTS_URL" with "payments.url"."""
    if key in config:
        return config[key]
    wanted = norm_name(key)
    for name, value in config.items():
        if norm_name(name) == wanted:
            return value
    for name, value in config.items():
        if wanted and norm_name(name).endswith(wanted):
            return value
    return ""


def find_service(
    hint: str, services: list[ServiceFacts], config: dict[str, str], key: str
) -> str | None:
    """The service a name, host or config key points to, or None."""
    host = hint
    if not host and key:
        value = lookup_config(config, key)
        match = re.match(r"https?://([^/:]+)", value)
        host = match.group(1) if match else value.split("/")[0].split(":")[0]
    wanted = norm_name(host)
    if len(wanted) < 3:
        return None
    for s in services:
        if norm_name(s.name) == wanted:
            return s.name
    # A loose match ("payments" ~ "payments-service") needs at least 4 characters on the
    # shorter side, so a service called "ui" never claims every host that starts with "ui".
    loose = [
        s.name
        for s in services
        if (len(norm_name(s.name)) >= 4 and wanted.startswith(norm_name(s.name)))
        or (len(wanted) >= 4 and norm_name(s.name).startswith(wanted))
    ]
    return loose[0] if len(loose) == 1 else None


def resolve_topic(topic: str, config: dict[str, str]) -> str:
    """ "${app.topic}" -> its value from config, or its default, else unchanged."""
    m = re.fullmatch(r"\$\{([^}:]+)(?::([^}]*))?\}", topic)
    if not m:
        return topic
    return lookup_config(config, m.group(1)) or m.group(2) or topic


# --- the three kinds of link -----------------------------------------------------------------


def _confidence(owners: int, target_known: bool, strong: bool) -> str:
    if owners > 1:
        return "guess"
    return "exact" if (target_known and strong) else "probable"


def link_rest(services: list[ServiceFacts], result: LinkResult) -> None:
    for src in services:
        for call in (c for c in src.calls if c.kind == "rest"):
            method, segs = norm_route(call.route)
            target = find_service(call.target_hint, services, src.config, call.url_hint)
            if target == src.name:
                continue  # a call to itself is not a link
            if call.target_hint and target is None:
                result.unmatched_calls.append(
                    {
                        "from": src.name,
                        "call": call.route,
                        "evidence": call.evidence,
                        "reason": (
                            f"calls '{call.target_hint}', which is not one of the mapped services"
                        ),
                    }
                )
                continue
            hits = []
            for other in services:
                if other is src or (target and other.name != target):
                    continue
                for offer in (e for e in other.exposes if e.kind == "rest"):
                    o_method, o_segs = norm_route(offer.route)
                    if not _methods_ok(method, o_method):
                        continue
                    if _same(segs, o_segs):
                        hits.append((other, offer, True))
                    elif _suffix(segs, o_segs):
                        hits.append((other, offer, False))
            if any(strong for _, _, strong in hits):
                hits = [h for h in hits if h[2]]
            if not hits:
                reason = (
                    f"{target} is the target, but it offers no matching route"
                    if target
                    else "no mapped service offers this route"
                )
                result.unmatched_calls.append(
                    {
                        "from": src.name,
                        "call": call.route,
                        "evidence": call.evidence,
                        "reason": reason,
                    }
                )
                continue
            owners = len({other.name for other, _, _ in hits})
            for other, offer, strong in hits:
                result.links.append(
                    Link(
                        src.name,
                        other.name,
                        "rest",
                        call.route,
                        _confidence(owners, target is not None, strong),
                        call.evidence,
                        offer.evidence,
                    )
                )


def link_graphql(services: list[ServiceFacts], result: LinkResult) -> None:
    for src in services:
        for call in (c for c in src.calls if c.kind == "graphql"):
            target = find_service(call.target_hint, services, src.config, call.url_hint)
            hits = [
                (other, offer)
                for other in services
                if other is not src and (not target or other.name == target)
                for offer in other.exposes
                if offer.kind == "graphql" and offer.route.split(".", 1)[-1] == call.route
            ]
            if not hits:
                result.unmatched_calls.append(
                    {
                        "from": src.name,
                        "call": f"GraphQL {call.route}",
                        "evidence": call.evidence,
                        "reason": "no mapped service has this field in its schema",
                    }
                )
                continue
            owners = len({other.name for other, _ in hits})
            for other, offer in hits:
                result.links.append(
                    Link(
                        src.name,
                        other.name,
                        "graphql",
                        offer.route,
                        _confidence(owners, target is not None, True),
                        call.evidence,
                        offer.evidence,
                    )
                )


def link_kafka(services: list[ServiceFacts], result: LinkResult) -> None:
    pairs: dict[tuple[str, str], list[Link]] = {}
    matched_p: set[int] = set()
    matched_c: set[int] = set()
    for prod in services:
        for p in (t for t in prod.topics if t.role == "produces"):
            p_topic = resolve_topic(p.topic, prod.config) if p.topic else ""
            for cons in services:
                if cons is prod:
                    continue
                for c in (t for t in cons.topics if t.role == "consumes"):
                    c_topic = resolve_topic(c.topic, cons.config) if c.topic else ""
                    if p_topic and p_topic == c_topic:
                        link = Link(
                            prod.name, cons.name, "kafka", p_topic, "exact", p.evidence, c.evidence
                        )
                    elif p.message_type and p.message_type == c.message_type:
                        link = Link(
                            prod.name,
                            cons.name,
                            "kafka",
                            f"message {p.message_type}",
                            "probable",
                            p.evidence,
                            c.evidence,
                        )
                    else:
                        continue
                    pairs.setdefault((prod.name, cons.name), []).append(link)
                    matched_p.add(id(p))
                    matched_c.add(id(c))
    for links in pairs.values():
        best = min(RANK[link.confidence] for link in links)
        result.links.extend(link for link in links if RANK[link.confidence] == best)
    # A class that names the message type and a send() that names the topic describe ONE
    # event; when one of them matched, the other in the same file is not "unmatched".
    matched_files = {
        (svc.name, t.role, t.evidence.file)
        for svc in services
        for t in svc.topics
        if id(t) in (matched_p if t.role == "produces" else matched_c)
    }
    for svc in services:
        for t in svc.topics:
            seen = matched_p if t.role == "produces" else matched_c
            if id(t) not in seen and (svc.name, t.role, t.evidence.file) not in matched_files:
                what = (
                    resolve_topic(t.topic, svc.config) if t.topic else f"message {t.message_type}"
                )
                result.unmatched_topics.append(
                    {"service": svc.name, "role": t.role, "what": what, "evidence": t.evidence}
                )


def shared_users(services: list[ServiceFacts], declared: list[dict]) -> list[dict]:
    def uses(lib_name: str, artifact: str) -> bool:
        return lib_name == artifact or lib_name.endswith((":" + artifact, "/" + artifact))

    out = []
    for decl in declared:
        artifact = str(decl.get("artifact") or decl.get("name") or "")
        users = sorted(
            s.name
            for s in services
            if s.name != decl.get("name") and any(uses(lib.name, artifact) for lib in s.libs)
        )
        out.append({"name": decl.get("name") or artifact, "artifact": artifact, "used_by": users})
    return out


def link_all(services: list[ServiceFacts], declared_shared: list[dict] | None = None) -> LinkResult:
    result = LinkResult()
    link_rest(services, result)
    link_graphql(services, result)
    link_kafka(services, result)
    result.shared = shared_users(services, declared_shared or [])
    called = {
        (link.to_service, link.evidence_to.quote) for link in result.links if link.evidence_to
    }
    for s in services:
        for offer in s.exposes:
            if (s.name, offer.evidence.quote) not in called:
                result.unused_offers.append({"service": s.name, "offer": offer.route})
    return result


def evidence_dict(ev: Evidence | None) -> dict | None:
    if ev is None:
        return None
    out = {"file": ev.file, "quote": ev.quote}
    if not ev.verified:
        out["unverified"] = True
    return out
