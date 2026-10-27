"""Domain facts read off parsed symbols and files: what a class maps to in
the database, what it sends or receives over Kafka, which API it serves or
calls.

Every fact is RAW — a name exactly as the code writes it (`UpdateHighRisk`,
`Topics.domain.orders`, `reports.reports`). Nothing here decides what a
name refers to: that needs the rest of the service (imports, the table
list) or other services (the Avro schemas in `models`, the topic constants
in `kafka`), and is done by links.py and service_links.py, which write an
edge only when the name resolves to a real node.

Patterns are the ones found in this codebase, read from real files:
JPA entities with `@Table(name, schema)` and `JpaRepository<Entity, Id>`
repositories; graphql-java-tools resolvers (`GraphQLQueryResolver`,
`GraphQLMutationResolver`, 386 classes in reports-api) plus the Spring and
DGS annotations; the in-house Kafka library's `KafkaMessageHandler<M>` with
`inputTopic = Topics.x.y`, and `KafkaProducer<K, V>` sending with
`topicName = Topics.x.y`; Spring MVC `@GetMapping`-style routes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .code_parser import ParsedFile, ParsedSymbol

_JVM = {"kotlin", "java"}
_JS = {"typescript", "tsx", "javascript"}

# --- JPA ---------------------------------------------------------------------

_TABLE_ANNOTATION = re.compile(
    r"@(?:javax\.persistence\.|jakarta\.persistence\.)?Table\s*\(([^)]*)\)"
)
_ANNOTATION_ARG = re.compile(r'(\w+)\s*=\s*"([^"]*)"')
_REPOSITORY_SUPER = re.compile(
    r"\b(?:JpaRepository|CrudRepository|PagingAndSortingRepository|ReactiveCrudRepository|"
    r"CoroutineCrudRepository|JpaSpecificationExecutor|ListCrudRepository)\s*<\s*([\w.]+)"
)

# --- SQL inside string literals ---------------------------------------------

_STRING_LITERAL = re.compile(r'"""(.*?)"""|"((?:[^"\\\n]|\\.)*)"|`((?:[^`\\]|\\.)*)`', re.DOTALL)
_TABLE_NAME = r'"?([A-Za-z_][\w]*(?:"?\."?[A-Za-z_][\w]*)?)"?'
_SQL_READS = re.compile(rf"\b(?:from|join)\s+{_TABLE_NAME}", re.IGNORECASE)
_SQL_WRITES = re.compile(
    rf"\b(?:insert\s+into|update|delete\s+from|merge\s+into|truncate(?:\s+table)?)\s+{_TABLE_NAME}",
    re.IGNORECASE,
)
_SQL_HINT = re.compile(r"\b(select|insert|update|delete|merge)\b", re.IGNORECASE)

# --- Kafka (the in-house library in the `kafka` repo) -----------------------

_KAFKA_HANDLER = re.compile(r"\bKafkaMessageHandler\s*<\s*([\w.]+)\s*>")
_KAFKA_PRODUCER = re.compile(r"\bKafkaProducer\s*<\s*([\w.]+)\s*,\s*([\w.]+)\s*>")
_TOPIC_CONSTANT = re.compile(r"\bTopics(?:\.\w+)+")
_INPUT_TOPIC_LINE = re.compile(r"\binputTopics?\b[^\n]*")
_OUTPUT_TOPIC = re.compile(
    r"\b(?:topicName\s*=\s*|synchronousSend\s*\(\s*|send\s*\(\s*)(Topics(?:\.\w+)+)"
)
_NEW_BUILDER = re.compile(r"\b([A-Z]\w*)\s*\.\s*newBuilder\s*\(")
_GENERIC_RECORDS = {"SpecificRecord", "GenericRecord", "Any", "Object", "IndexedRecord"}

# --- GraphQL resolvers ------------------------------------------------------

_RESOLVER_SUPERTYPES = {
    "GraphQLQueryResolver": "Query",
    "GraphQLMutationResolver": "Mutation",
    "GraphQLSubscriptionResolver": "Subscription",
}
_RESOLVER_FOR_TYPE = re.compile(r"\bGraphQLResolver\s*<\s*([\w.]+)\s*>")
_MAPPING_ANNOTATION = re.compile(
    r"@(QueryMapping|MutationMapping|SubscriptionMapping|DgsQuery|DgsMutation|DgsSubscription)"
    r"(?:\s*\(([^)]*)\))?"
)
_SCHEMA_MAPPING = re.compile(r"@(?:SchemaMapping|DgsData)\s*\(([^)]*)\)")

# --- REST -------------------------------------------------------------------

_ROUTE_ANNOTATION = re.compile(r"@(Get|Post|Put|Delete|Patch|Request)Mapping\s*(?:\(([^)]*)\))?")
_FIRST_STRING = re.compile(r'"([^"]*)"')
_REQUEST_METHOD = re.compile(r"RequestMethod\.(\w+)")
_JS_HTTP_CALL = re.compile(
    r"\b(?:fetch|axios|request|http|api|client|apiClient|httpClient)"
    r"(?:\s*\.\s*(get|post|put|delete|patch|request))?\s*(?:<[^>]*>)?\s*\(\s*"
    r"([`'\"])(/[^`'\"\s]*)\2",
    re.IGNORECASE,
)
_JVM_HTTP_CALL = re.compile(r'\.uri\s*\(\s*"(/[^"\s]*)"')

# --- GraphQL client operations ---------------------------------------------

_GQL_TEMPLATE = re.compile(
    r"(?:\bgql|\bgraphql|/\*\s*GraphQL\s*\*/)\s*(?:\(\s*)?`([^`]*)`", re.IGNORECASE
)
_TEMPLATE_INTERPOLATION = re.compile(r"\$\{[^}]*\}")

# --- Kotlin constant objects ------------------------------------------------

_KT_OBJECT = re.compile(r"\bobject\s+(\w+)\s*(?::[^{]*)?\{")
_KT_CONST = re.compile(
    r'^\s*(?:const\s+)?val\s+(\w+)\s*(?::\s*String)?\s*=\s*"([^"]*)"\s*$', re.MULTILINE
)
_KT_ALIAS = re.compile(r"^\s*val\s+(\w+)\s*=\s*([A-Z]\w*)\s*$", re.MULTILINE)


def string_literals(text: str) -> list[str]:
    return [next(g for g in m.groups() if g is not None) for m in _STRING_LITERAL.finditer(text)]


def _header(symbol: ParsedSymbol) -> str:
    """A declaration up to its body — annotations, name, supertypes,
    constructor, parameters — so a fact about a class is not read from one
    of its methods. The parser's signature is exactly that; cutting at the
    first `{` instead would stop inside `@GetMapping("/health/{id}")`."""
    return symbol.signature or symbol.source_snippet


def sql_tables(text: str) -> tuple[list[str], list[str]]:
    """(tables read, tables written) named by SQL inside string literals."""
    reads: dict[str, None] = {}
    writes: dict[str, None] = {}
    for literal in string_literals(text):
        if not _SQL_HINT.search(literal):
            continue
        for m in _SQL_READS.finditer(literal):
            reads.setdefault(m.group(1).replace('"', "").lower(), None)
        for m in _SQL_WRITES.finditer(literal):
            writes.setdefault(m.group(1).replace('"', "").lower(), None)
    return list(reads), list(writes)


def _route_path(args: str | None) -> str:
    if not args:
        return ""
    match = _FIRST_STRING.search(args)
    return match.group(1) if match else ""


def _join_path(base: str, path: str) -> str:
    joined = "/" + "/".join(p.strip("/") for p in (base, path) if p.strip("/"))
    return joined if joined != "/" or not (base or path) else "/"


def _class_facts(symbol: ParsedSymbol) -> dict:
    facts: dict = {}
    header = _header(symbol)
    table = _TABLE_ANNOTATION.search(header)
    if table:
        args = dict(_ANNOTATION_ARG.findall(table.group(1)))
        name = args.get("name") or args.get("value")
        if name:
            facts["db_table"] = (f"{args['schema']}.{name}" if args.get("schema") else name).lower()
    if re.search(r"@(?:javax\.persistence\.|jakarta\.persistence\.)?Entity\b", header):
        facts["jpa_entity"] = True
    repository = _REPOSITORY_SUPER.search(header)
    if repository:
        facts["repository_entity"] = repository.group(1).split(".")[-1]

    handles = _KAFKA_HANDLER.findall(header)
    if handles:
        facts["kafka_handles"] = sorted(set(handles))
    input_topics = sorted(
        {
            topic
            for line in _INPUT_TOPIC_LINE.findall(symbol.source_snippet)
            for topic in _TOPIC_CONSTANT.findall(line)
        }
    )
    if input_topics:
        facts["kafka_input_topics"] = input_topics
    producers = _KAFKA_PRODUCER.findall(symbol.source_snippet)
    if producers:
        published = {
            value for _, value in producers if value.split(".")[-1] not in _GENERIC_RECORDS
        }
        if published:
            facts["kafka_publishes"] = sorted(published)
        if any(value.split(".")[-1] in _GENERIC_RECORDS for _, value in producers):
            # A producer typed only `SpecificRecord` names its message where
            # it builds it: `ReportAttachmentAdded.newBuilder()`. It builds
            # its payload and context records the same way, so these are
            # kept apart and linked only to records that are messages (see
            # service_links.py), never to every record it touches.
            built = sorted(set(_NEW_BUILDER.findall(symbol.source_snippet)))
            if built:
                facts["kafka_publishes_built"] = built
    output_topics = sorted(set(_OUTPUT_TOPIC.findall(symbol.source_snippet)))
    if output_topics:
        facts["kafka_output_topics"] = output_topics

    for supertype in symbol.supertypes:
        if supertype in _RESOLVER_SUPERTYPES:
            facts["graphql_root_resolver"] = _RESOLVER_SUPERTYPES[supertype]
    resolver_for = _RESOLVER_FOR_TYPE.search(header)
    if resolver_for:
        facts["graphql_type_resolver"] = resolver_for.group(1).split(".")[-1]

    mapping = re.search(r"@RequestMapping\s*(?:\(([^)]*)\))?", header)
    if mapping:
        facts["route_base"] = _route_path(mapping.group(1))
    return facts


def _method_facts(symbol: ParsedSymbol, class_facts: dict) -> dict:
    facts: dict = {}
    # Annotations sit before the method's body, which is where they are read.
    head = _header(symbol)

    root_type = class_facts.get("graphql_root_resolver") or class_facts.get("graphql_type_resolver")
    resolves: set[str] = set()
    if root_type and symbol.is_method and symbol.symbol_kind == "method":
        resolves.add(f"{root_type}.{symbol.name}")
        # graphql-java-tools also binds field `reports` to getReports/isReports.
        for prefix in ("get", "is"):
            if (
                symbol.name.startswith(prefix)
                and symbol.name[len(prefix) : len(prefix) + 1].isupper()
            ):
                field_name = symbol.name[len(prefix)].lower() + symbol.name[len(prefix) + 1 :]
                resolves.add(f"{root_type}.{field_name}")
    for annotation, args in _MAPPING_ANNOTATION.findall(head):
        root = (
            "Query"
            if "Query" in annotation
            else "Mutation" if "Mutation" in annotation else "Subscription"
        )
        named = dict(_ANNOTATION_ARG.findall(args or ""))
        field = named.get("field") or named.get("name") or named.get("value") or _route_path(args)
        resolves.add(f"{root}.{field or symbol.name}")
    for args in _SCHEMA_MAPPING.findall(head):
        named = dict(_ANNOTATION_ARG.findall(args))
        parent = named.get("typeName") or named.get("parentType")
        if parent:
            resolves.add(f"{parent}.{named.get('field') or symbol.name}")
    if resolves:
        facts["graphql_resolves"] = sorted(resolves)

    routes = []
    for verb, args in _ROUTE_ANNOTATION.findall(head):
        method = verb.upper()
        if verb == "Request":
            found = _REQUEST_METHOD.search(args or "")
            method = found.group(1).upper() if found else "ANY"
        routes.append(
            f"{method} {_join_path(class_facts.get('route_base', ''), _route_path(args))}"
        )
    if routes:
        facts["routes"] = sorted(set(routes))
    return facts


_ROOT_TYPES = {"Query", "Mutation", "Subscription"}


def graphql_documents(text: str, language: str) -> tuple[list[str], list[str]]:
    """(root fields a client asks for, as "query:reports"; root fields a
    server defines, as "Query.reports") — from gql templates in JS/TS, or a
    whole `.graphql` document. Parsed with graphql-core, so nothing is
    counted from a document that does not parse.

    The second list matters for a Node GraphQL server, whose schema lives in
    gql`type Query {...}` templates rather than in a .graphqls file."""
    from graphql import GraphQLError, parse
    from graphql.language import (
        FieldNode,
        ObjectTypeDefinitionNode,
        ObjectTypeExtensionNode,
        OperationDefinitionNode,
    )

    if language == "graphql":
        documents = [text]
    else:
        documents = [
            _TEMPLATE_INTERPOLATION.sub("", m.group(1)) for m in _GQL_TEMPLATE.finditer(text)
        ]
    uses: dict[str, None] = {}
    exposes: dict[str, None] = {}
    for document in documents:
        try:
            parsed = parse(document, no_location=True)
        except (GraphQLError, RecursionError):
            continue
        for definition in parsed.definitions:
            if isinstance(definition, OperationDefinitionNode):
                operation = definition.operation.value
                for selection in definition.selection_set.selections:
                    if isinstance(selection, FieldNode) and not selection.name.value.startswith(
                        "__"
                    ):
                        uses.setdefault(f"{operation}:{selection.name.value}", None)
            elif isinstance(definition, ObjectTypeDefinitionNode | ObjectTypeExtensionNode):
                if definition.name.value in _ROOT_TYPES and language != "graphql":
                    for field in definition.fields or ():
                        exposes.setdefault(f"{definition.name.value}.{field.name.value}", None)
    return list(uses), list(exposes)


def http_paths(text: str, language: str) -> list[str]:
    """Literal request paths a client calls, as "GET /api/x/{}" (method ANY
    when the call does not say). Interpolations become `{}`."""
    found: dict[str, None] = {}
    if language in _JS:
        for verb, _, path in _JS_HTTP_CALL.findall(text):
            path = _TEMPLATE_INTERPOLATION.sub("{}", path)
            found.setdefault(f"{(verb or 'ANY').upper()} {path}", None)
    elif language in _JVM:
        for path in _JVM_HTTP_CALL.findall(text):
            found.setdefault("ANY " + _KOTLIN_TEMPLATE.sub("{}", path), None)
    return list(found)


_KOTLIN_TEMPLATE = re.compile(r"\$\{?\w+\}?")


def kotlin_constants(text: str) -> dict[str, dict]:
    """String constants declared in Kotlin objects, keyed "Object.member":
    {"value": "orders"} for `const val orders = "orders"`, {"alias":
    "DomainTopics"} for `val domain = DomainTopics`. Read from text because
    the bundled grammar returns an ERROR node for `const val` inside an
    object (see code_parser.py's Kotlin section)."""
    constants: dict[str, dict] = {}

    def scan(body: str, prefix: str) -> None:
        position = 0
        depth_zero = []
        while True:
            m = _KT_OBJECT.search(body, position)
            if m is None:
                depth_zero.append(body[position:])
                break
            depth_zero.append(body[position : m.start()])
            close = _matching_brace(body, m.end() - 1)
            scan(body[m.end() : close], f"{prefix}{m.group(1)}.")
            position = close + 1
        own = "".join(depth_zero)
        if prefix:
            for name, value in _KT_CONST.findall(own):
                constants[f"{prefix}{name}"] = {"value": value}
            for name, target in _KT_ALIAS.findall(own):
                constants[f"{prefix}{name}"] = {"alias": target}

    scan(text, "")
    return constants


def _matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(text)


def add_facts(parsed: ParsedFile, text: str, file_path: str, language: str) -> None:
    """Attach every fact this file and its symbols carry, in place."""
    if language in _JVM:
        class_facts: dict[str, dict] = {}
        for symbol in parsed.symbols:
            if symbol.kind == "class":
                facts = _class_facts(symbol)
                class_facts[symbol.qualified_name] = facts
                symbol.facts.update(facts)
        for symbol in parsed.symbols:
            if symbol.kind == "function":
                symbol.facts.update(
                    _method_facts(symbol, class_facts.get(symbol.parent_class or "", {}))
                )
        if language == "kotlin" and "object " in text and "val " in text:
            constants = kotlin_constants(text)
            if constants:
                parsed.facts["kotlin_constants"] = constants
    if language in _JVM or language in _JS or language == "python":
        for symbol in parsed.symbols:
            if symbol.kind in ("function", "class") and _SQL_HINT.search(symbol.source_snippet):
                reads, writes = sql_tables(_own_text(symbol, parsed.symbols))
                if reads:
                    symbol.facts["sql_reads"] = reads
                if writes:
                    symbol.facts["sql_writes"] = writes
    if language in _JS or (language == "graphql" and not file_path.endswith(".graphqls")):
        if "gql" in text or "graphql" in text.lower() or language == "graphql":
            uses, exposes = graphql_documents(text, language)
            if uses:
                parsed.facts["graphql_operations"] = uses
            if exposes:
                parsed.facts["graphql_exposes"] = exposes
    if language in _JS or language in _JVM:
        paths = http_paths(text, language)
        if paths:
            parsed.facts["http_paths"] = paths


def _own_text(symbol: ParsedSymbol, symbols: list[ParsedSymbol]) -> str:
    """A class's text without its methods' text, a method's text whole —
    so a query string inside a method is attributed to the method only."""
    if symbol.kind != "class":
        return symbol.source_snippet
    lines = symbol.source_snippet.split("\n")
    for other in symbols:
        if other.parent_symbol == symbol.qualified_name and other.kind == "function":
            for n in range(other.line_start, other.line_end + 1):
                index = n - symbol.line_start
                if 0 <= index < len(lines):
                    lines[index] = ""
    return "\n".join(lines)
