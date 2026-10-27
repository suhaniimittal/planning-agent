"""A service's database tables, rebuilt from its own migrations.

Every service in this codebase owns its database and defines it with
Liquibase: a master changelog (`changelog-master.xml`) lists changesets in
order, and most changesets point at a plain PostgreSQL file
(`<sqlFile path="liquibase/20190123_01_packages.sql"/>`). Replaying those
files in order — CREATE TABLE, then every ALTER after it — gives the
schema the service runs against today, without connecting to any
database: no credentials, and the same answer every run.

The SQL side is a small reader for the DDL that migrations actually use,
not a full SQL grammar. A statement it does not understand is skipped and
counted, never guessed at: a wrong column would be worse than a missing
one, because the planner treats what it is shown as fact.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import PurePosixPath

import yaml


@dataclass
class Column:
    name: str
    type: str
    nullable: bool = True

    def render(self) -> str:
        return f"{self.name} {self.type}{'' if self.nullable else ' not null'}".strip()


@dataclass
class Table:
    name: str  # lower-case, schema-qualified when the migration qualifies it
    columns: dict[str, Column] = field(default_factory=dict)
    defined_in: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    kind: str = "table"  # or "view"

    def touch(self, path: str) -> None:
        if path not in self.defined_in:
            self.defined_in.append(path)


@dataclass
class Schema:
    tables: dict[str, Table] = field(default_factory=dict)
    statements_applied: int = 0
    statements_skipped: int = 0
    files_read: list[str] = field(default_factory=list)


# --- changelog order --------------------------------------------------------

_MASTER_NAME = re.compile(
    r"(changelog[-_.]?master|master[-_.]?changelog|db\.changelog-master)", re.I
)
_CHANGELOG_SUFFIXES = {".xml", ".yaml", ".yml", ".json"}


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _resolve(path: str, relative_to: str, files: set[str]) -> str | None:
    """A path written in a changelog, matched to a real file: relative to the
    changelog, then to the repo root, then by unique suffix (Liquibase
    resolves against the classpath, which this cannot see)."""
    path = path.lstrip("/")
    candidates = [
        str(PurePosixPath(relative_to).parent / path) if relative_to else path,
        path,
    ]
    for candidate in candidates:
        normalized = str(PurePosixPath(*[p for p in PurePosixPath(candidate).parts if p != "."]))
        if normalized in files:
            return normalized
    suffix_matches = [f for f in files if f.endswith("/" + path) or f == path]
    return suffix_matches[0] if len(suffix_matches) == 1 else None


def _changelog_steps(path: str, text: str, files: set[str], seen: set[str]) -> list[tuple]:
    """Ordered steps from one changelog: ("sql", file_path) for a SQL file,
    ("inline", sql_text, path) for inline SQL, ("change", dict, path) for a
    structured Liquibase change, and nested changelogs expanded in place."""
    if path in seen:
        return []
    seen.add(path)
    steps: list[tuple] = []
    suffix = PurePosixPath(path).suffix.lower()
    if suffix == ".xml":
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return steps
        for element in root.iter():
            tag = _local(element.tag)
            if tag == "include" and element.get("file"):
                target = _resolve(element.get("file"), path, files)
                if target:
                    steps.append(("include", target))
            elif tag == "includeAll" and element.get("path"):
                folder = element.get("path").strip("/")
                base = str(PurePosixPath(path).parent / folder)
                for candidate in sorted(files):
                    if candidate.startswith((base + "/", folder + "/")):
                        steps.append(("include", candidate))
            elif tag == "sqlFile" and element.get("path"):
                target = _resolve(element.get("path"), path, files)
                if target:
                    steps.append(("sql", target))
            elif tag == "sql" and element.text:
                steps.append(("inline", element.text, path))
            elif tag in _STRUCTURED_CHANGES:
                steps.append(("change", _xml_change(element), path))
    elif suffix in (".yaml", ".yml", ".json"):
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError:
            return steps
        for entry in (data or {}).get("databaseChangeLog", []) or []:
            if not isinstance(entry, dict):
                continue
            if "include" in entry and isinstance(entry["include"], dict):
                target = _resolve(str(entry["include"].get("file", "")), path, files)
                if target:
                    steps.append(("include", target))
            change_set = entry.get("changeSet")
            if not isinstance(change_set, dict):
                continue
            for change in change_set.get("changes", []) or []:
                if not isinstance(change, dict):
                    continue
                for name, body in change.items():
                    if name == "sqlFile" and isinstance(body, dict):
                        target = _resolve(str(body.get("path", "")), path, files)
                        if target:
                            steps.append(("sql", target))
                    elif name == "sql":
                        sql = body.get("sql") if isinstance(body, dict) else body
                        if isinstance(sql, str):
                            steps.append(("inline", sql, path))
                    elif name in _STRUCTURED_CHANGES and isinstance(body, dict):
                        steps.append(("change", {"type": name, **_yaml_change(body)}, path))
    return steps


def migration_order(sources: dict[str, str]) -> list[tuple]:
    """Every step in order, starting from each master changelog. With no
    master changelog, every .sql file in path order (Flyway's V1__, V2__ and
    date-prefixed names both sort correctly)."""
    files = set(sources)
    masters = sorted(
        (
            p
            for p in files
            if _MASTER_NAME.search(PurePosixPath(p).name)
            and PurePosixPath(p).suffix.lower() in _CHANGELOG_SUFFIXES
        ),
        key=lambda p: (p.count("/"), p),
    )
    if not masters:
        return [("sql", p) for p in sorted(files) if p.lower().endswith(".sql")]
    seen: set[str] = set()
    ordered: list[tuple] = []

    def expand(path: str) -> None:
        for step in _changelog_steps(path, sources.get(path, ""), files, seen):
            if step[0] == "include":
                if PurePosixPath(step[1]).suffix.lower() == ".sql":
                    ordered.append(("sql", step[1]))
                else:
                    expand(step[1])
            else:
                ordered.append(step)

    for master in masters:
        expand(master)
    return ordered


# --- structured Liquibase changes ------------------------------------------

_STRUCTURED_CHANGES = {
    "createTable",
    "addColumn",
    "dropColumn",
    "renameColumn",
    "dropTable",
    "renameTable",
    "modifyDataType",
    "addNotNullConstraint",
    "dropNotNullConstraint",
    "createView",
}


def _xml_change(element: ET.Element) -> dict:
    change = {"type": _local(element.tag), **dict(element.attrib), "columns": []}
    for column in element:
        if _local(column.tag) != "column":
            continue
        nullable = True
        for constraint in column:
            if _local(constraint.tag) == "constraints":
                if constraint.get("nullable") == "false" or constraint.get("primaryKey") == "true":
                    nullable = False
        change["columns"].append(
            {"name": column.get("name"), "type": column.get("type") or "", "nullable": nullable}
        )
    return change


def _yaml_change(body: dict) -> dict:
    change = {k: v for k, v in body.items() if k != "columns"}
    columns = []
    for entry in body.get("columns", []) or []:
        column = entry.get("column", {}) if isinstance(entry, dict) else {}
        constraints = column.get("constraints", {}) or {}
        columns.append(
            {
                "name": column.get("name"),
                "type": column.get("type") or "",
                "nullable": not (
                    constraints.get("nullable") is False or constraints.get("primaryKey")
                ),
            }
        )
    change["columns"] = columns
    return change


def _qualified(schema: str | None, name: str | None) -> str:
    name = (name or "").strip('"').lower()
    return f"{schema.strip('\"').lower()}.{name}" if schema else name


def _rename_column(table: Table, old: str, new: str) -> None:
    """Rename in place, keeping the column where it was."""
    if old not in table.columns or not new:
        return
    renamed: dict[str, Column] = {}
    for name, column in table.columns.items():
        if name == old:
            column.name = new
            renamed[new] = column
        else:
            renamed[name] = column
    table.columns = renamed


def _apply_change(schema: Schema, change: dict, path: str) -> None:
    kind = change.get("type")
    table_name = _qualified(
        change.get("schemaName"), change.get("tableName") or change.get("viewName")
    )
    if kind == "createTable" or kind == "createView":
        table = schema.tables.setdefault(table_name, Table(table_name))
        table.kind = "view" if kind == "createView" else "table"
        for column in change.get("columns", []):
            if column.get("name"):
                table.columns[column["name"].lower()] = Column(
                    column["name"].lower(), column.get("type", ""), column.get("nullable", True)
                )
        table.touch(path)
    elif kind == "addColumn" and table_name in schema.tables:
        table = schema.tables[table_name]
        for column in change.get("columns", []):
            if column.get("name"):
                table.columns[column["name"].lower()] = Column(
                    column["name"].lower(), column.get("type", ""), column.get("nullable", True)
                )
        table.touch(path)
    elif kind == "dropColumn" and table_name in schema.tables:
        names = [change.get("columnName")] + [c.get("name") for c in change.get("columns", [])]
        for name in filter(None, names):
            schema.tables[table_name].columns.pop(name.lower(), None)
        schema.tables[table_name].touch(path)
    elif kind == "renameColumn" and table_name in schema.tables:
        table = schema.tables[table_name]
        _rename_column(
            table,
            (change.get("oldColumnName") or "").lower(),
            (change.get("newColumnName") or "").lower(),
        )
        table.touch(path)
    elif kind == "dropTable":
        schema.tables.pop(table_name, None)
    elif kind == "renameTable":
        old = _qualified(change.get("schemaName"), change.get("oldTableName"))
        new = _qualified(change.get("schemaName"), change.get("newTableName"))
        if old in schema.tables:
            table = schema.tables.pop(old)
            table.name = new
            schema.tables[new] = table
            table.touch(path)
    elif kind == "modifyDataType" and table_name in schema.tables:
        column = schema.tables[table_name].columns.get((change.get("columnName") or "").lower())
        if column:
            column.type = change.get("newDataType", column.type)
    elif kind in ("addNotNullConstraint", "dropNotNullConstraint") and table_name in schema.tables:
        column = schema.tables[table_name].columns.get((change.get("columnName") or "").lower())
        if column:
            column.nullable = kind == "dropNotNullConstraint"
    else:
        schema.statements_skipped += 1
        return
    schema.statements_applied += 1


# --- SQL DDL ----------------------------------------------------------------

_NAME = r'(?:"[^"]+"|[A-Za-z_][\w$]*)'
_QUALIFIED = rf"({_NAME}(?:\s*\.\s*{_NAME})?)"
_CREATE_TABLE = re.compile(
    rf"^create\s+(?:(?:global\s+|local\s+)?(?:temporary|temp|unlogged)\s+)?table\s+"
    rf"(?:if\s+not\s+exists\s+)?{_QUALIFIED}\s*\((.*)\)\s*[^)]*$",
    re.I | re.S,
)
_CREATE_TABLE_AS = re.compile(
    rf"^create\s+table\s+(?:if\s+not\s+exists\s+)?{_QUALIFIED}\s+as\b", re.I
)
_CREATE_VIEW = re.compile(
    rf"^create\s+(?:or\s+replace\s+)?(?:materialized\s+)?view\s+(?:if\s+not\s+exists\s+)?{_QUALIFIED}",
    re.I,
)
_ALTER_TABLE = re.compile(
    rf"^alter\s+table\s+(?:if\s+exists\s+)?(?:only\s+)?{_QUALIFIED}\s+(.*)$", re.I | re.S
)
_DROP_TABLE = re.compile(
    r"^drop\s+(?:table|view|materialized\s+view)\s+(?:if\s+exists\s+)?(.*)$", re.I | re.S
)
_TABLE_CONSTRAINT = re.compile(
    r"^(constraint|primary\s+key|foreign\s+key|unique|check|exclude|like)\b", re.I
)
_COLUMN_STOP = re.compile(
    r"\s+(not\s+null|null|default|constraint|references|primary\s+key|unique|check|"
    r"generated|collate)\b",
    re.I,
)
_REFERENCES = re.compile(rf"\breferences\s+{_QUALIFIED}", re.I)


def _unquote(name: str) -> str:
    return re.sub(r"\s+", "", name).replace('"', "").lower()


def split_statements(sql: str) -> list[str]:
    """Statements of a SQL script: split on `;` outside quotes, comments and
    $$-quoted bodies (a function body in a migration is full of `;`)."""
    statements: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    dollar: str | None = None
    while i < n:
        ch = sql[i]
        if dollar:
            if sql.startswith(dollar, i):
                buf.append(dollar)
                i += len(dollar)
                dollar = None
                continue
            buf.append(ch)
            i += 1
            continue
        if ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            i = n if end == -1 else end
            continue
        if ch == "/" and sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in ("'", '"'):
            end = i + 1
            while end < n:
                if sql[end] == ch:
                    if end + 1 < n and sql[end + 1] == ch:
                        end += 2
                        continue
                    break
                end += 1
            buf.append(sql[i : end + 1])
            i = end + 1
            continue
        if ch == "$":
            match = re.match(r"\$[A-Za-z_]*\$", sql[i:])
            if match:
                dollar = match.group(0)
                buf.append(dollar)
                i += len(dollar)
                continue
        if ch == ";":
            statement = "".join(buf).strip()
            if statement:
                statements.append(statement)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    quote: str | None = None
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == separator and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


def _column(definition: str) -> Column | None:
    match = re.match(rf"({_NAME})\s+(.*)$", definition.strip(), re.S)
    if not match:
        return None
    name = _unquote(match.group(1))
    rest = re.sub(r"\s+", " ", match.group(2)).strip()
    stop = _COLUMN_STOP.search(" " + rest)
    col_type = (rest[: stop.start() - 1] if stop else rest).strip().lower()
    nullable = not re.search(r"\bnot\s+null\b|\bprimary\s+key\b", rest, re.I)
    return Column(name, col_type, nullable)


def _apply_sql(schema: Schema, statement: str, path: str) -> None:
    statement = statement.strip()
    created = _CREATE_TABLE.match(statement)
    if created:
        name = _unquote(created.group(1))
        table = Table(name)
        for part in _split_top_level(created.group(2)):
            if _TABLE_CONSTRAINT.match(part):
                for ref in _REFERENCES.findall(part):
                    table.references.append(_unquote(ref))
                if re.match(r"primary\s+key\s*\(", part, re.I):
                    for col in _split_top_level(part[part.index("(") + 1 : part.rindex(")")]):
                        if _unquote(col) in table.columns:
                            table.columns[_unquote(col)].nullable = False
                continue
            column = _column(part)
            if column is not None:
                table.columns[column.name] = column
                for ref in _REFERENCES.findall(part):
                    table.references.append(_unquote(ref))
        table.touch(path)
        schema.tables[name] = table
        schema.statements_applied += 1
        return
    as_select = _CREATE_TABLE_AS.match(statement)
    view = _CREATE_VIEW.match(statement)
    if as_select or view:
        name = _unquote((as_select or view).group(1))
        table = schema.tables.setdefault(name, Table(name, kind="view" if view else "table"))
        table.touch(path)
        schema.statements_applied += 1
        return
    altered = _ALTER_TABLE.match(statement)
    if altered:
        name = _unquote(altered.group(1))
        if name not in schema.tables:
            schema.statements_skipped += 1
            return
        table = schema.tables[name]
        for action in _split_top_level(altered.group(2)):
            _apply_alter(schema, table, action.strip())
        table = schema.tables.get(table.name)
        if table is not None:
            table.touch(path)
        schema.statements_applied += 1
        return
    dropped = _DROP_TABLE.match(statement)
    if dropped:
        for target in _split_top_level(
            re.sub(r"\b(cascade|restrict)\b", "", dropped.group(1), flags=re.I)
        ):
            schema.tables.pop(_unquote(target), None)
        schema.statements_applied += 1
        return
    schema.statements_skipped += 1


def _apply_alter(schema: Schema, table: Table, action: str) -> None:
    lower = re.sub(r"\s+", " ", action).lower()
    if lower.startswith("add constraint") or lower.startswith("add foreign key"):
        for ref in _REFERENCES.findall(action):
            table.references.append(_unquote(ref))
        return
    add = re.match(r"add\s+(?:column\s+)?(?:if\s+not\s+exists\s+)?(.*)$", action, re.I | re.S)
    if add and not re.match(r"add\s+(primary|unique|check|exclude)\b", action, re.I):
        column = _column(add.group(1))
        if column is not None:
            table.columns[column.name] = column
        return
    drop = re.match(rf"drop\s+(?:column\s+)?(?:if\s+exists\s+)?({_NAME})", action, re.I)
    if drop and not re.match(r"drop\s+(constraint|default|not\s+null)\b", action, re.I):
        table.columns.pop(_unquote(drop.group(1)), None)
        return
    rename_column = re.match(rf"rename\s+(?:column\s+)?({_NAME})\s+to\s+({_NAME})", action, re.I)
    if rename_column and not lower.startswith("rename to"):
        _rename_column(table, _unquote(rename_column.group(1)), _unquote(rename_column.group(2)))
        return
    rename_table = re.match(rf"rename\s+to\s+({_NAME})", action, re.I)
    if rename_table:
        new_simple = _unquote(rename_table.group(1))
        schema_prefix = table.name.rsplit(".", 1)[0] + "." if "." in table.name else ""
        schema.tables.pop(table.name, None)
        table.name = schema_prefix + new_simple
        schema.tables[table.name] = table
        return
    set_schema = re.match(rf"set\s+schema\s+({_NAME})", action, re.I)
    if set_schema:
        schema.tables.pop(table.name, None)
        table.name = f"{_unquote(set_schema.group(1))}.{table.name.rsplit('.', 1)[-1]}"
        schema.tables[table.name] = table
        return
    alter = re.match(rf"alter\s+(?:column\s+)?({_NAME})\s+(.*)$", action, re.I | re.S)
    if alter:
        column = table.columns.get(_unquote(alter.group(1)))
        rest = alter.group(2).strip()
        if column is None:
            return
        type_change = re.match(
            r"(?:set\s+data\s+)?type\s+(.+?)(?:\s+using\b.*)?$", rest, re.I | re.S
        )
        if type_change:
            column.type = re.sub(r"\s+", " ", type_change.group(1)).strip().lower()
        elif re.match(r"set\s+not\s+null", rest, re.I):
            column.nullable = False
        elif re.match(r"drop\s+not\s+null", rest, re.I):
            column.nullable = True


def build_schema(sources: dict[str, str]) -> Schema:
    """Replay a service's migrations. `sources` holds every migration and
    changelog file's text, keyed by repo path."""
    schema = Schema()
    for step in migration_order(sources):
        if step[0] == "sql":
            path = step[1]
            schema.files_read.append(path)
            for statement in split_statements(sources.get(path, "")):
                _apply_sql(schema, statement, path)
        elif step[0] == "inline":
            for statement in split_statements(step[1]):
                _apply_sql(schema, statement, step[2])
        elif step[0] == "change":
            _apply_change(schema, step[1], step[2])
    for table in schema.tables.values():
        table.references = sorted(set(table.references))
    return schema


def match_table(name: str, tables: dict[str, Table]) -> str | None:
    """The table a name used in code refers to: exact, or the unique table
    whose unqualified name matches (code often omits the schema). None when
    no table — or more than one — fits."""
    name = name.lower().replace('"', "")
    if name in tables:
        return name
    bare = name.rsplit(".", 1)[-1]
    matches = [t for t in tables if t.rsplit(".", 1)[-1] == bare]
    return matches[0] if len(matches) == 1 else None
