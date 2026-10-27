"""Stage 3 of ingestion: every link INSIDE one service.

Imports to the files they load, identifiers to the symbols they name,
classes to what they extend, tests to what they test, and the service's own
domain: its tables (rebuilt from its migrations), the GraphQL operations
and REST routes it serves, and the Kafka messages its classes handle or
publish, as fully qualified names.

Pure computation over a ServiceIndex — built from a fresh parse after a
full ingest, or read back from Neo4j after an incremental one, so both
paths produce identical links. Writing them is code_graph_writer's job;
links that cross services are service_links.py's.

The rule throughout: a link is made only when the name resolves to
something that exists. A name that does not resolve is counted, never
matched to its nearest lookalike.
"""

from __future__ import annotations

import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from . import db_schema
from .code_parser import ImportSpec, RepoGraph
from .import_resolver import FileImports, ImportResolver, ResolvedImport
from .test_paths import is_test_path

_JS_LANGUAGES = {"typescript", "tsx", "javascript"}
_JVM_LANGUAGES = {"kotlin", "java"}
_ROOT_TYPES = {"Query", "Mutation", "Subscription"}
_TEST_CLASS_SUFFIXES = ("IntegrationTests", "IntegrationTest", "Tests", "Test", "Spec", "IT")
_TEST_FILE_INFIX = re.compile(r"\.(test|spec|stories)(?=\.[^.]+$)")


@dataclass
class IndexedSymbol:
    qualified_name: str
    name: str
    label: str  # Class | Function | Variable
    symbol_kind: str
    file_path: str
    parent_symbol: str | None = None
    identifiers: list[str] = field(default_factory=list)
    supertypes: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    language: str = ""

    @property
    def is_test(self) -> bool:
        return is_test_path(self.file_path)


@dataclass
class IndexedFile:
    path: str
    language: str
    kind: str
    imports: list[ImportSpec] = field(default_factory=list)
    package: str | None = None
    facts: dict = field(default_factory=dict)


@dataclass
class ServiceIndex:
    service: str
    files: dict[str, IndexedFile]
    symbols: list[IndexedSymbol]
    # Text of the files Stage 3 must read: tsconfig/jsconfig files and every
    # migration or changelog file.
    sources: dict[str, str] = field(default_factory=dict)


@dataclass
class Operation:
    type: str
    name: str
    file_path: str | None = None
    field_qname: str | None = None


@dataclass
class ServiceLinks:
    imports: list[ResolvedImport] = field(default_factory=list)
    unresolved_imports: dict[str, list[str]] = field(default_factory=dict)
    references: list[tuple[str, str]] = field(default_factory=list)
    extends: list[tuple[str, str]] = field(default_factory=list)
    implements: list[tuple[str, str]] = field(default_factory=list)
    tests: list[tuple[str, str]] = field(default_factory=list)  # (test file, tested file)
    tables: list[db_schema.Table] = field(default_factory=list)
    maps_to: list[tuple[str, str]] = field(default_factory=list)  # (class qname, table)
    reads: list[tuple[str, str]] = field(default_factory=list)  # (symbol qname, table)
    writes: list[tuple[str, str]] = field(default_factory=list)
    operations: list[Operation] = field(default_factory=list)
    resolves: list[tuple[str, str, str]] = field(default_factory=list)  # (qname, type, name)
    endpoints: list[tuple[str, str, str]] = field(default_factory=list)  # (qname, METHOD, path)
    kafka: dict[str, dict] = field(default_factory=dict)  # class qname -> resolved facts
    avro_messages: list[dict] = field(default_factory=list)
    unresolved_references: int = 0


# --- building an index ------------------------------------------------------


def _needs_source(path: str, kind: str) -> bool:
    name = PurePosixPath(path).name
    return kind == "migration" or bool(re.match(r"^(tsconfig(\.[\w-]+)?|jsconfig)\.json$", name))


def index_from_repo_graph(service: str, graph: RepoGraph) -> ServiceIndex:
    files: dict[str, IndexedFile] = {}
    sources: dict[str, str] = {}
    for path, record in graph.files.items():
        files[path] = IndexedFile(
            path=path,
            language=record.language,
            kind=record.kind,
            imports=graph.file_import_specs.get(path, []),
            package=graph.file_packages.get(path),
            facts=graph.file_facts.get(path, {}),
        )
        if _needs_source(path, record.kind) and record.source:
            sources[path] = record.source
    labels = {"class": "Class", "function": "Function", "variable": "Variable"}
    symbols = [
        IndexedSymbol(
            qualified_name=s.qualified_name,
            name=s.name,
            label=labels[s.kind],
            symbol_kind=s.symbol_kind,
            file_path=s.file_path,
            parent_symbol=s.parent_symbol,
            identifiers=list(s.identifiers),
            supertypes=list(s.supertypes),
            facts=dict(s.facts),
            language=s.language,
        )
        for s in graph.symbols
    ]
    return ServiceIndex(service, files, symbols, sources)


def _json(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def index_from_rows(
    service: str, file_rows: list[dict], symbol_rows: list[dict], load_sources
) -> ServiceIndex:
    """A ServiceIndex read back from Neo4j (code_graph_writer.
    load_service_index_rows) — what an incremental sync recomputes links
    from, since it never parses the whole service. `load_sources(paths)`
    fetches the text of just the files Stage 3 reads."""
    files = {
        row["file_path"]: IndexedFile(
            path=row["file_path"],
            language=row.get("language") or "",
            kind=row.get("kind") or "source",
            imports=[ImportSpec.from_dict(d) for d in _json(row.get("import_specs"), [])],
            package=row.get("package"),
            facts=_json(row.get("facts"), {}),
        )
        for row in file_rows
    }
    symbols = [
        IndexedSymbol(
            qualified_name=row["qualified_name"],
            name=row.get("name") or "",
            label=row.get("label") or "Function",
            symbol_kind=row.get("symbol_kind") or "",
            file_path=row.get("file_path") or "",
            parent_symbol=row.get("parent_symbol"),
            identifiers=list(row.get("identifiers") or []),
            supertypes=list(row.get("supertypes") or []),
            facts=_json(row.get("facts"), {}),
            language=row.get("language") or "",
        )
        for row in symbol_rows
    ]
    needed = [p for p, f in files.items() if _needs_source(p, f.kind)]
    return ServiceIndex(service, files, symbols, load_sources(needed))


# --- the links --------------------------------------------------------------


class _Scopes:
    """Name lookup from inside a symbol: its own nested symbols, then each
    enclosing symbol's, then the file's top level, then imports, then (JVM)
    the rest of its package."""

    def __init__(self, index: ServiceIndex, resolver: ImportResolver):
        self.index = index
        self.resolver = resolver
        self.by_qname = {s.qualified_name: s for s in index.symbols}
        self.children: dict[str, dict[str, str]] = {}
        self.top: dict[str, dict[str, str]] = {}
        self.package_top: dict[str, dict[str, str]] = {}
        for s in index.symbols:
            if s.parent_symbol:
                self.children.setdefault(s.parent_symbol, {}).setdefault(s.name, s.qualified_name)
            else:
                self.top.setdefault(s.file_path, {}).setdefault(s.name, s.qualified_name)
                file = index.files.get(s.file_path)
                if file and file.package and file.language in _JVM_LANGUAGES:
                    self.package_top.setdefault(file.package, {}).setdefault(
                        s.name, s.qualified_name
                    )
        self._bindings: dict[str, dict[str, tuple[str, str]]] = {}

    def bindings(self, path: str) -> dict[str, tuple[str, str]]:
        if path not in self._bindings:
            self._bindings[path] = self.resolver.local_bindings(path)
        return self._bindings[path]

    def ancestors(self, symbol: IndexedSymbol) -> list[str]:
        chain = []
        current = symbol.parent_symbol
        while current and len(chain) < 50:
            chain.append(current)
            parent = self.by_qname.get(current)
            current = parent.parent_symbol if parent else None
        return chain

    def lookup(self, symbol: IndexedSymbol, name: str) -> str | None:
        for scope in (symbol.qualified_name, *self.ancestors(symbol)):
            found = self.children.get(scope, {}).get(name)
            if found:
                return found
        found = self.top.get(symbol.file_path, {}).get(name)
        if found:
            return found
        binding = self.bindings(symbol.file_path).get(name)
        if binding:
            found = self.top.get(binding[0], {}).get(binding[1])
            if found:
                return found
        file = self.index.files.get(symbol.file_path)
        if file and file.language in _JVM_LANGUAGES:
            if file.package:
                found = self.package_top.get(file.package, {}).get(name)
                if found:
                    return found
            for spec in file.imports:
                if spec.wildcard:
                    found = self.package_top.get(spec.module, {}).get(name)
                    if found:
                        return found
        return None


def _jvm_fqn(name: str, file: IndexedFile | None, scopes: _Scopes) -> str | None:
    """The fully qualified name a type name refers to in a JVM file: through
    its imports, else a class of that name in its own package. None when
    neither — a type parameter like `MessageType` resolves to nothing."""
    if "." in name and name[0].islower():
        return name
    if file is None:
        return None
    for spec in file.imports:
        if spec.wildcard:
            continue
        for original, alias in spec.names:
            if alias == name:
                return spec.module
    if file.package and name in scopes.package_top.get(file.package, {}):
        return f"{file.package}.{name}"
    return None


def compute_links(index: ServiceIndex) -> ServiceLinks:
    links = ServiceLinks()
    file_imports = {
        path: FileImports(path=path, language=f.language, imports=f.imports, package=f.package)
        for path, f in index.files.items()
    }
    for s in index.symbols:
        if s.parent_symbol is None and s.file_path in file_imports:
            file_imports[s.file_path].top_level.add(s.name)
    resolver = ImportResolver(file_imports, index.sources)
    links.imports = resolver.resolve_all()
    links.unresolved_imports = resolver.unresolved
    scopes = _Scopes(index, resolver)

    # References.
    seen_refs: set[tuple[str, str]] = set()
    for symbol in index.symbols:
        own_chain = {symbol.qualified_name, *scopes.ancestors(symbol)}
        for name in symbol.identifiers:
            target = scopes.lookup(symbol, name)
            if target is None:
                links.unresolved_references += 1
                continue
            if target in own_chain:
                continue
            pair = (symbol.qualified_name, target)
            if pair not in seen_refs:
                seen_refs.add(pair)
                links.references.append(pair)

    # Extends / implements.
    for symbol in index.symbols:
        if symbol.label != "Class":
            continue
        for supertype in symbol.supertypes:
            target_qname = scopes.lookup(symbol, supertype)
            target = scopes.by_qname.get(target_qname or "")
            if target is None or target.label != "Class" or target_qname == symbol.qualified_name:
                continue
            if target.symbol_kind == "interface" and symbol.symbol_kind != "interface":
                links.implements.append((symbol.qualified_name, target_qname))
            else:
                links.extends.append((symbol.qualified_name, target_qname))

    links.tests = _test_links(index, links.imports, scopes)
    _database_links(index, links)
    _api_links(index, links)
    _kafka_links(index, links, scopes)
    return links


def _test_links(
    index: ServiceIndex, imports: list[ResolvedImport], scopes: _Scopes
) -> list[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for edge in imports:
        if is_test_path(edge.source) and not is_test_path(edge.target):
            if index.files.get(edge.target) and index.files[edge.target].kind == "source":
                pairs.add((edge.source, edge.target))
    # Sibling naming: view.test.tsx tests view.tsx.
    paths = set(index.files)
    for path in index.files:
        if not is_test_path(path):
            continue
        stripped = _TEST_FILE_INFIX.sub("", path)
        if stripped != path:
            stem = posixpath.splitext(stripped)[0]
            for ext in (".tsx", ".ts", ".jsx", ".js"):
                if stem + ext in paths and not is_test_path(stem + ext):
                    pairs.add((path, stem + ext))
                    break
    # JVM naming: OrderServiceTests tests the one class named OrderService.
    classes_by_name: dict[str, list[IndexedSymbol]] = {}
    for s in index.symbols:
        if s.label == "Class" and s.parent_symbol is None and not s.is_test:
            classes_by_name.setdefault(s.name, []).append(s)
    for s in index.symbols:
        if s.label != "Class" or s.parent_symbol is not None or not s.is_test:
            continue
        if s.language not in _JVM_LANGUAGES:
            continue
        for suffix in _TEST_CLASS_SUFFIXES:
            if s.name.endswith(suffix) and len(s.name) > len(suffix):
                candidates = classes_by_name.get(s.name[: -len(suffix)], [])
                if len(candidates) == 1:
                    pairs.add((s.file_path, candidates[0].file_path))
                break
    return sorted(pairs)


def _database_links(index: ServiceIndex, links: ServiceLinks) -> None:
    migration_sources = {
        p: t
        for p, t in index.sources.items()
        if index.files.get(p) and index.files[p].kind == "migration"
    }
    schema = db_schema.build_schema(migration_sources)
    links.tables = list(schema.tables.values())
    tables = schema.tables
    if not tables:
        return
    entity_tables: dict[str, str] = {}
    for symbol in index.symbols:
        if symbol.label == "Class" and symbol.facts.get("db_table"):
            table = db_schema.match_table(symbol.facts["db_table"], tables)
            if table:
                links.maps_to.append((symbol.qualified_name, table))
                entity_tables.setdefault(symbol.name, table)
    reads: set[tuple[str, str]] = set()
    writes: set[tuple[str, str]] = set()
    for symbol in index.symbols:
        if symbol.is_test:
            continue
        entity = symbol.facts.get("repository_entity")
        if entity and entity in entity_tables:
            reads.add((symbol.qualified_name, entity_tables[entity]))
            writes.add((symbol.qualified_name, entity_tables[entity]))
        for name in symbol.facts.get("sql_reads", []):
            table = db_schema.match_table(name, tables)
            if table:
                reads.add((symbol.qualified_name, table))
        for name in symbol.facts.get("sql_writes", []):
            table = db_schema.match_table(name, tables)
            if table:
                writes.add((symbol.qualified_name, table))
    links.reads = sorted(reads)
    links.writes = sorted(writes)


def _api_links(index: ServiceIndex, links: ServiceLinks) -> None:
    by_qname = {s.qualified_name: s for s in index.symbols}
    operations: dict[tuple[str, str], Operation] = {}
    for symbol in index.symbols:
        if symbol.is_test or symbol.language != "graphql" or symbol.symbol_kind != "field":
            continue
        parent = by_qname.get(symbol.parent_symbol or "")
        if (
            parent is None
            or parent.name not in _ROOT_TYPES
            or parent.symbol_kind not in ("type", "extension")
        ):
            continue
        operations.setdefault(
            (parent.name, symbol.name),
            Operation(parent.name, symbol.name, symbol.file_path, symbol.qualified_name),
        )
    for path, file in index.files.items():
        if is_test_path(path):
            continue
        for exposed in file.facts.get("graphql_exposes", []):
            root, _, name = exposed.partition(".")
            operations.setdefault((root, name), Operation(root, name, path, None))
    links.operations = list(operations.values())
    for symbol in index.symbols:
        if symbol.is_test:
            continue
        for resolved in symbol.facts.get("graphql_resolves", []):
            root, _, name = resolved.partition(".")
            if (root, name) in operations:
                links.resolves.append((symbol.qualified_name, root, name))
        for route in symbol.facts.get("routes", []):
            method, _, path = route.partition(" ")
            links.endpoints.append((symbol.qualified_name, method, path))


def _kafka_links(index: ServiceIndex, links: ServiceLinks, scopes: _Scopes) -> None:
    by_qname = scopes.by_qname
    parents: dict[str, list[str]] = {}
    for child, parent in links.extends:
        parents.setdefault(child, []).append(parent)

    def inherited(qname: str, key: str, depth: int = 0) -> list[str]:
        own = list(by_qname[qname].facts.get(key, [])) if qname in by_qname else []
        if depth < 8:
            for parent in parents.get(qname, []):
                own += inherited(parent, key, depth + 1)
        return own

    for symbol in index.symbols:
        if symbol.label != "Class" or symbol.is_test:
            continue
        facts = symbol.facts
        if not any(
            k in facts for k in ("kafka_handles", "kafka_publishes", "kafka_publishes_built")
        ):
            continue
        file = index.files.get(symbol.file_path)
        resolved: dict = {}
        for key, out in (
            ("kafka_handles", "handles"),
            ("kafka_publishes", "publishes"),
            ("kafka_publishes_built", "publishes_built"),
        ):
            fqns = sorted({f for f in (_jvm_fqn(n, file, scopes) for n in facts.get(key, [])) if f})
            if fqns:
                resolved[out] = fqns
        if not resolved:
            continue
        resolved["input_topics"] = sorted(set(facts.get("kafka_input_topics", [])))
        resolved["output_topics"] = sorted(
            set(inherited(symbol.qualified_name, "kafka_output_topics"))
        )
        links.kafka[symbol.qualified_name] = resolved

    for symbol in index.symbols:
        if symbol.language != "avdl" or symbol.symbol_kind != "record":
            continue
        namespace = symbol.facts.get("avro_namespace")
        if not namespace:
            continue
        protocol = by_qname.get(symbol.parent_symbol or "")
        siblings = [
            s
            for s in index.symbols
            if s.file_path == symbol.file_path and s.symbol_kind == "record"
        ]
        # The message in `protocol UpdateHighRiskProtocol { record
        # UpdateHighRiskPayload {...} record UpdateHighRisk {...} }` is the
        # record the protocol is named after; the others are its parts.
        is_message = (protocol is not None and protocol.name == f"{symbol.name}Protocol") or len(
            siblings
        ) == 1
        links.avro_messages.append(
            {
                "qualified_name": symbol.qualified_name,
                "fqn": f"{namespace}.{symbol.name}",
                "file_path": symbol.file_path,
                "is_message": is_message,
            }
        )
