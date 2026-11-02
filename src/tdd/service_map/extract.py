"""Pulls connection facts out of a service's files. No LLM: plain patterns.

Each finder reads one file's text and returns facts that carry a quote, the
source line the fact came from. verify.py later checks every quote is really in
its file. What it finds:

  offers   REST routes (Spring, NestJS), GraphQL root fields (schema files)
  calls    Feign clients, RestTemplate / WebClient / axios / fetch / http calls,
           GraphQL operations in client code
  kafka    @KafkaListener topics, KafkaMessageHandler<M>, KafkaProducer<K, M>,
           KafkaTemplate<K, M>, Nest event patterns, send("topic")
  libs     dependencies in build.gradle(.kts), pom.xml, package.json
  config   key/value pairs of application*.yml|properties, docker-compose, ...
           (used to resolve ${payments.url} and ${topic.name} placeholders)

Limits, on purpose: topic names held in constants, a NestJS global prefix, and
routes built from variables are not seen. They show up as unmatched in the report.
"""

from __future__ import annotations

import json
import re

from .classify import CODE_EXT, _kind
from .schema import Endpoint, Evidence, Lib, Outbound, ServiceFacts, TopicUse

# --- small helpers -------------------------------------------------------------------


def _line_at(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    end = len(text) if end == -1 else end
    return text[start:end].strip()[:160]


def _ev(path: str, text: str, pos: int) -> Evidence:
    return Evidence(path, _line_at(text, pos))


def _join(prefix: str, sub: str) -> str:
    parts = [p for p in (prefix.strip("/"), sub.strip("/")) if p]
    return "/" + "/".join(parts)


_STR = re.compile(r'"([^"]*)"')

# --- REST offers: Spring -----------------------------------------------------------

_SPRING_VERB = re.compile(r"@(Get|Post|Put|Delete|Patch)Mapping\b\s*(?:\(([^)]*)\))?")
_SPRING_REQ = re.compile(r"@RequestMapping\b\s*(?:\(([^)]*)\))?")
_DECL = re.compile(r"\b(?:class|interface|object)\s+\w+")


def _mapping_paths(args: str | None) -> list[str]:
    """The path strings of a mapping annotation's arguments ("" when it has none)."""
    if not args or not args.strip():
        return [""]
    keyed = re.search(r'(?:value|path)\s*=\s*(\[[^\]]*\]|\{[^}]*\}|"[^"]*")', args)
    found = _STR.findall(keyed.group(1)) if keyed else _STR.findall(args.split("=")[0])
    return found or [""]


def spring_routes(text: str) -> list[tuple[str, str, int]]:
    """[(METHOD, route, position)] of every Spring mapping in a file; a
    @RequestMapping before the class is the prefix of all the others."""
    decl = _DECL.search(text)
    decl_pos = decl.start() if decl else 0
    prefix = ""
    for m in _SPRING_REQ.finditer(text):
        if m.start() < decl_pos:
            prefix = _mapping_paths(m.group(1))[0]
            break
    routes: list[tuple[str, str, int]] = []
    for m in _SPRING_VERB.finditer(text):
        for sub in _mapping_paths(m.group(2)):
            routes.append((m.group(1).upper(), _join(prefix, sub), m.start()))
    for m in _SPRING_REQ.finditer(text):
        if m.start() < decl_pos:
            continue
        verb = re.search(r"RequestMethod\.(\w+)", m.group(1) or "")
        for sub in _mapping_paths(m.group(1)):
            routes.append((verb.group(1).upper() if verb else "ANY", _join(prefix, sub), m.start()))
    return routes


def find_spring_endpoints(path: str, text: str) -> list[Endpoint]:
    if not path.endswith((".kt", ".java")) or "@FeignClient" in text:
        return []
    if not re.search(r"@(RestController|Controller|RequestMapping)\b", text):
        return []
    return [
        Endpoint("rest", f"{method} {route}", _ev(path, text, pos))
        for method, route, pos in spring_routes(text)
    ]


# --- REST offers: NestJS ---------------------------------------------------------------

_NEST_CTRL = re.compile(
    r"""@Controller\s*\(\s*(?:['"]([^'"]*)['"]|\{[^}]*?path\s*:\s*['"]([^'"]*)['"][^}]*\})?"""
)
_NEST_VERB = re.compile(r"""@(Get|Post|Put|Delete|Patch)\s*\(\s*(?:['"]([^'"]*)['"])?""")


def find_nest_endpoints(path: str, text: str) -> list[Endpoint]:
    if not path.endswith((".ts", ".js")) or "@Controller" not in text:
        return []
    ctrl = _NEST_CTRL.search(text)
    prefix = (ctrl.group(1) or ctrl.group(2) or "") if ctrl else ""
    return [
        Endpoint(
            "rest",
            f"{m.group(1).upper()} {_join(prefix, m.group(2) or '')}",
            _ev(path, text, m.start()),
        )
        for m in _NEST_VERB.finditer(text)
    ]


# --- GraphQL ----------------------------------------------------------------------------

_GQL_ROOT = re.compile(
    r"\b(?:extend\s+)?type\s+(Query|Mutation|Subscription)\b[^{]*\{([^}]*)\}", re.S
)
_GQL_OP = re.compile(r"\b(query|mutation|subscription)\s+\w*\s*(?:\([^)]*\))?\s*\{\s*(\w+)")


def find_graphql_endpoints(path: str, text: str) -> list[Endpoint]:
    """The root fields a schema file offers: type Query { reports(...): [Report] }."""
    if not path.endswith((".graphqls", ".graphql", ".gql")):
        return []
    found = []
    for m in _GQL_ROOT.finditer(text):
        body = re.sub(r'"""[\s\S]*?"""', "", m.group(2))
        body = re.sub(r"\([^)]*\)", "", body, flags=re.S)
        for line in body.splitlines():
            field = re.match(r"\s*(\w+)\s*:", line.split("#", 1)[0])
            if field:
                pos = max(text.find(field.group(1), m.start(2)), 0)
                found.append(
                    Endpoint("graphql", f"{m.group(1)}.{field.group(1)}", _ev(path, text, pos))
                )
    return found


def find_graphql_calls(path: str, text: str) -> list[Outbound]:
    """The root field of each query/mutation a client sends: query Get { reports { id } }."""
    if path.endswith(".graphqls") or _GQL_ROOT.search(text):
        return []
    if not (path.endswith(CODE_EXT + (".graphql", ".gql"))):
        return []
    if (
        not path.endswith((".graphql", ".gql"))
        and "gql" not in text.lower()
        and "graphql" not in text.lower()
    ):
        return []
    return [
        Outbound("graphql", m.group(2), _ev(path, text, m.start())) for m in _GQL_OP.finditer(text)
    ]


# --- Kafka ---------------------------------------------------------------------------------

_KAFKA_LISTENER = re.compile(r"@KafkaListener\s*\(([^)]*)\)", re.S)
_TOPIC_ARG = re.compile(r'topics\s*=\s*(\[[^\]]*\]|\{[^}]*\}|"[^"]*")')
_KAFKA_HANDLER = re.compile(r"KafkaMessageHandler\s*<\s*([\w.]+)")
_KAFKA_PRODUCER = re.compile(r"Kafka(?:Producer|Template)\s*<\s*[\w.?]+\s*,\s*([\w.]+)")
_KAFKA_SEND = re.compile(r"""\.\s*(?:send|emit)\s*\(\s*["']([^"']+)["']""")
# Base types of the message libraries: they say nothing about WHICH message is sent.
_GENERIC_TYPES = {"SpecificRecord", "SpecificRecordBase", "GenericRecord", "Any", "Object"}

_NEST_EVENT = re.compile(r"""@(?:EventPattern|MessagePattern)\s*\(\s*['"]([^'"]+)['"]""")


def find_kafka(path: str, text: str) -> list[TopicUse]:
    if not path.endswith((".kt", ".java", ".ts", ".js")):
        return []
    found: list[TopicUse] = []
    seen: set[tuple[str, str, str]] = set()

    def add(role: str, topic: str, message_type: str, pos: int) -> None:
        message_type = message_type.rsplit(".", 1)[-1]
        if (role, topic, message_type) in seen or not (topic or message_type):
            return
        if message_type and (
            message_type in _GENERIC_TYPES or (len(message_type) == 1 and message_type.isupper())
        ):
            return  # a generic parameter or a base type, not a real message type
        seen.add((role, topic, message_type))
        found.append(TopicUse(role, _ev(path, text, pos), topic, message_type))

    for m in _KAFKA_LISTENER.finditer(text):
        arg = _TOPIC_ARG.search(m.group(1))
        for topic in _STR.findall(arg.group(1)) if arg else []:
            add("consumes", topic.replace("\\", ""), "", m.start())
    for m in _NEST_EVENT.finditer(text):
        add("consumes", m.group(1), "", m.start())
    for m in _KAFKA_HANDLER.finditer(text):
        add("consumes", "", m.group(1), m.start())
    for m in _KAFKA_PRODUCER.finditer(text):
        add("produces", "", m.group(1), m.start())
    if "kafka" in text.lower():
        for m in _KAFKA_SEND.finditer(text):
            add("produces", m.group(1), "", m.start())
    return found


# --- calls to other services ---------------------------------------------------------------

_PLACEHOLDER = re.compile(r"^(?:\$\{([^}:]+)(?::[^}]*)?\}|\$\{?([A-Za-z_][\w.]*)\}?|\{([^}]+)\})")


def split_url(raw: str) -> tuple[str, str, str] | None:
    """(path, host, config key) of a url literal, or None when it has no path.

    "http://payments-service:8080/payments/validate"
        -> ("/payments/validate", "payments-service", "")
    "${payments.url}/payments/validate" -> ("/payments/validate", "", "payments.url")
    "/payments/validate"                -> ("/payments/validate", "", "")"""
    s = raw.strip().strip("`'\"").replace("\\", "")
    host = key = ""
    if re.match(r"https?://", s):
        rest = s.split("://", 1)[1]
        authority, _, tail = rest.partition("/")
        path = "/" + tail
        placeholder = _PLACEHOLDER.match(authority)
        if placeholder:
            key = next(g for g in placeholder.groups() if g)
        else:
            host = authority.split(":")[0]
    else:
        placeholder = _PLACEHOLDER.match(s)
        if placeholder:
            key = next(g for g in placeholder.groups() if g)
            path = s[placeholder.end() :]
        else:
            path = s
    path = path.split("?")[0]
    return (path, host, key) if path.startswith("/") else None


_Q = r"""(?P<q>["'`])(?P<url>.+?)(?P=q)"""
_HTTP = [
    re.compile(
        r"\b\w*(?:resttemplate|restclient)\w*\s*\.\s*(?P<verb>get|post|put|delete|patch)\w*\s*(?:<[^>]*>)?\s*\(\s*"
        + _Q,
        re.I,
    ),
    re.compile(r"\.exchange\s*\(\s*" + _Q + r"""\s*,\s*HttpMethod\.(?P<verb>\w+)"""),
    re.compile(r"\.(?P<verb>get|post|put|delete|patch)\s*\(\s*\)\s*\.uri\s*\(\s*" + _Q, re.I),
    re.compile(
        r"\b(?:axios|http|httpclient|ky|api|client)\w*\s*\.\s*(?P<verb>get|post|put|delete|patch)\s*(?:<[^>]*>)?\s*\(\s*"
        + _Q,
        re.I,
    ),
    re.compile(
        r"\bfetch\s*\(\s*" + _Q + r"""(?:\s*,\s*\{[^}]*?method\s*:\s*["'](?P<verb>\w+)["'])?"""
    ),
]
_BASE_URL = re.compile(r"base[uU][rR][lL]\s*[:(=]\s*(?:\(\s*)?" + _Q)
_BASE_ENV = re.compile(r"base[uU][rR][lL]\s*:\s*process\.env\.(\w+)")
_FEIGN = re.compile(r"@FeignClient\s*\(([^)]*)\)", re.S)


def find_http_calls(path: str, text: str) -> list[Outbound]:
    if not path.endswith(CODE_EXT):
        return []
    # A client built once with a base url: calls with only a path go there.
    base_host = base_key = ""
    base = _BASE_URL.search(text)
    if base:
        parts = split_url(base.group("url") + "/")
        if parts:
            _, base_host, base_key = parts
    env = _BASE_ENV.search(text)
    if env and not base_key:
        base_key = env.group(1)
    found: list[Outbound] = []
    seen: set[tuple[str, str, str, str]] = set()
    for pattern in _HTTP:
        for m in pattern.finditer(text):
            parts = split_url(m.group("url"))
            if not parts:
                continue
            route_path, host, key = parts
            host, key = host or base_host, key or base_key
            verb = (m.groupdict().get("verb") or "GET").upper()
            sig = (verb, route_path, host, key)
            if sig in seen:
                continue
            seen.add(sig)
            found.append(
                Outbound("rest", f"{verb} {route_path}", _ev(path, text, m.start()), host, key)
            )
    return found


def find_feign(path: str, text: str) -> list[Outbound]:
    m = _FEIGN.search(text)
    if not m or not path.endswith((".kt", ".java")):
        return []
    args = m.group(1).replace("\\", "")

    def named(*names: str) -> str:
        for n in names:
            a = re.search(rf'\b{n}\s*=\s*"([^"]*)"', args)
            if a:
                return a.group(1)
        return ""

    name = named("name", "value")
    if not name:
        bare = re.match(r'\s*"([^"]*)"', args)
        name = bare.group(1) if bare else ""
    url = named("url")
    key = ""
    host = name
    if url:
        placeholder = re.search(r"\$\{([^}:]+)", url)
        if placeholder:
            key = placeholder.group(1)
        elif not name and re.match(r"https?://", url):
            host = url.split("://", 1)[1].split("/")[0].split(":")[0]
    return [
        Outbound("rest", f"{method} {route}", _ev(path, text, pos), host, key)
        for method, route, pos in spring_routes(text)
    ]


# --- libraries and config ------------------------------------------------------------------

_GRADLE_DEP = re.compile(
    r"""(?:implementation|api|compileOnly|runtimeOnly)\s*\(?\s*["']([^"':]+):([^"':]+)(?::[^"']*)?["']"""
)
_MAVEN_ART = re.compile(r"<artifactId>\s*([^<\s]+)\s*</artifactId>")


def find_libs(path: str, text: str) -> list[Lib]:
    name = path.rsplit("/", 1)[-1]
    libs: list[Lib] = []
    if name.startswith("build.gradle"):
        for m in _GRADLE_DEP.finditer(text):
            libs.append(Lib(f"{m.group(1)}:{m.group(2)}", _ev(path, text, m.start())))
    elif name == "pom.xml":
        for m in _MAVEN_ART.finditer(text):
            libs.append(Lib(m.group(1), _ev(path, text, m.start())))
    elif name == "package.json":
        try:
            data = json.loads(text)
        except ValueError:
            return []
        for section in ("dependencies", "devDependencies", "peerDependencies"):
            for dep in data.get(section) or {}:
                libs.append(Lib(dep, _ev(path, text, max(text.find(f'"{dep}"'), 0))))
    return libs


def flatten_config(path: str, text: str) -> dict[str, str]:
    """{"payments.url": "http://payments-service:8080", ...} from a config file."""
    name = path.rsplit("/", 1)[-1].lower()
    out: dict[str, str] = {}
    if name.endswith((".yml", ".yaml")):
        try:
            import yaml

            docs = list(yaml.safe_load_all(text))
        except Exception:  # noqa: BLE001 - a config we cannot read adds nothing
            return {}

        def walk(prefix: str, node: object) -> None:
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(f"{prefix}.{k}" if prefix else str(k), v)
            elif isinstance(node, list):
                for i, v in enumerate(node):
                    walk(f"{prefix}[{i}]", v)
            elif node is not None:
                out[prefix] = str(node)

        for doc in docs:
            walk("", doc)
    elif name.endswith(".properties"):
        for line in text.splitlines():
            line = line.strip()
            m = (
                None
                if not line or line.startswith(("#", "!"))
                else re.match(r"([^=:\s]+)\s*[=:]\s*(.*)", line)
            )
            if m:
                out[m.group(1)] = m.group(2).strip()
    return out


def guess_stack(paths: list[str], libs: list[Lib]) -> str:
    names = " ".join(lib.name for lib in libs)
    exts = {p.rsplit(".", 1)[-1] for p in paths if "." in p}
    parts = []
    if "kt" in exts:
        parts.append("Kotlin")
    elif "java" in exts:
        parts.append("Java")
    if "@nestjs/core" in names:
        parts.append("NestJS")
    elif "react" in names:
        parts.append("React")
    elif exts & {"ts", "tsx", "js", "jsx"}:
        parts.append("TypeScript/JS")
    return " + ".join(parts)


# --- one file, one service -------------------------------------------------------------------


def extract_file(path: str, text: str) -> dict:
    """Every fact one file holds: {"exposes", "calls", "topics", "libs", "config"}."""
    hit = _kind(path)
    return {
        "exposes": find_spring_endpoints(path, text)
        + find_nest_endpoints(path, text)
        + find_graphql_endpoints(path, text),
        "calls": find_feign(path, text)
        + find_http_calls(path, text)
        + find_graphql_calls(path, text),
        "topics": find_kafka(path, text),
        "libs": find_libs(path, text),
        "config": flatten_config(path, text) if hit and hit[0] == "config" else {},
    }


def extract_into(facts: ServiceFacts, files: dict[str, str]) -> ServiceFacts:
    """Fills ``facts`` from the files read ({path: text}), without repeating a fact."""
    have_exposes = {(e.kind, e.route) for e in facts.exposes}
    have_calls = {(c.kind, c.route, c.target_hint, c.url_hint) for c in facts.calls}
    have_topics = {(t.role, t.topic, t.message_type) for t in facts.topics}
    have_libs = {lib.name for lib in facts.libs}
    for path, text in files.items():
        found = extract_file(path, text)
        for e in found["exposes"]:
            if (e.kind, e.route) not in have_exposes:
                have_exposes.add((e.kind, e.route))
                facts.exposes.append(e)
        for c in found["calls"]:
            sig = (c.kind, c.route, c.target_hint, c.url_hint)
            if sig not in have_calls:
                have_calls.add(sig)
                facts.calls.append(c)
        for t in found["topics"]:
            if (t.role, t.topic, t.message_type) not in have_topics:
                have_topics.add((t.role, t.topic, t.message_type))
                facts.topics.append(t)
        for lib in found["libs"]:
            if lib.name not in have_libs:
                have_libs.add(lib.name)
                facts.libs.append(lib)
        facts.config.update(found["config"])
    facts.files_read = list(files)
    facts.stack = guess_stack(list(files), facts.libs)
    return facts
