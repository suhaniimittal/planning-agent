"""Data shapes for the service-map POC.

Every fact carries Evidence: the file it came from and a quote copied from that
file. verify.py checks the quote really is in the file, so a fact nobody can
point to is dropped instead of trusted.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    file: str  # full path inside the repo
    quote: str  # one line copied from that file
    verified: bool = False  # set by verify.py


@dataclass
class Endpoint:
    """Something a service offers."""

    kind: str  # "rest" | "graphql"
    route: str  # "POST /imps/transfer" or "Query.reports"
    evidence: Evidence
    source: str = "rule"  # "rule" (found by a pattern) | "llm"


@dataclass
class Outbound:
    """Something a service calls."""

    kind: str  # "rest" | "graphql"
    route: str  # "POST /payments/validate", or the GraphQL root field "reports"
    evidence: Evidence
    target_hint: str = ""  # a service name or host, e.g. "payments-service"
    url_hint: str = ""  # a config key behind the base url, e.g. "payments.url"
    source: str = "rule"


@dataclass
class TopicUse:
    """A Kafka topic or message type a service sends or listens to."""

    role: str  # "produces" | "consumes"
    evidence: Evidence
    topic: str = ""  # a name, or "${config.key}"
    message_type: str = ""  # e.g. "ReportArchived"
    source: str = "rule"


@dataclass
class Lib:
    """A library a service's build file depends on."""

    name: str  # "bank-models", "com.calfus:bank-models" or "@bank/models"
    evidence: Evidence


@dataclass
class ServiceFacts:
    name: str
    repo: str
    path: str = ""
    branch: str = ""
    commit: str = ""
    stack: str = ""
    summary: str = ""
    files_read: list[str] = field(default_factory=list)
    files_skipped: list[str] = field(default_factory=list)
    exposes: list[Endpoint] = field(default_factory=list)
    calls: list[Outbound] = field(default_factory=list)
    topics: list[TopicUse] = field(default_factory=list)
    libs: list[Lib] = field(default_factory=list)
    config: dict[str, str] = field(default_factory=dict)


@dataclass
class Link:
    """One connection between two services."""

    from_service: str
    to_service: str
    via: str  # "rest" | "graphql" | "kafka"
    detail: str  # the route, field, topic or message type
    confidence: str  # "exact" | "probable" | "guess"
    evidence_from: Evidence | None = None
    evidence_to: Evidence | None = None
