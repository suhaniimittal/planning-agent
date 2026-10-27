from src.ingestion import db_schema


def _columns(schema, table):
    return [c.render() for c in schema.tables[table].columns.values()]


def test_create_then_alter_in_changelog_order():
    sources = {
        "liquibase/changelog-master.xml": """<databaseChangeLog>
  <changeSet id="2" author="b"><sqlFile path="liquibase/b.sql"/></changeSet>
  <changeSet id="1" author="a"><sqlFile path="liquibase/a.sql"/></changeSet>
</databaseChangeLog>""",
        # The changelog's order wins over file names.
        "liquibase/b.sql": "create table reports.reports (id uuid not null primary key, status varchar(50));",
        "liquibase/a.sql": """
alter table reports.reports add column archived boolean not null default false;
alter table reports.reports alter column status type varchar(100);
alter table reports.reports alter column status set not null;
alter table reports.reports rename column status to state;
""",
    }

    schema = db_schema.build_schema(sources)

    assert _columns(schema, "reports.reports") == [
        "id uuid not null",
        "state varchar(100) not null",
        "archived boolean not null",
    ]
    assert schema.tables["reports.reports"].defined_in == ["liquibase/b.sql", "liquibase/a.sql"]


def test_without_a_changelog_sql_files_apply_in_path_order():
    schema = db_schema.build_schema(
        {
            "db/V2__add.sql": "ALTER TABLE orders ADD COLUMN total numeric(10,2);",
            "db/V1__init.sql": 'CREATE TABLE IF NOT EXISTS "orders" ("id" uuid NOT NULL, note text);',
        }
    )

    assert _columns(schema, "orders") == ["id uuid not null", "note text", "total numeric(10,2)"]


def test_drop_and_rename_table_and_foreign_keys():
    schema = db_schema.build_schema(
        {
            "m.sql": """
create table a (id int primary key);
create table b (id int, a_id int references a(id), constraint fk foreign key (a_id) references a);
create table tmp (x int);
drop table if exists tmp cascade;
alter table b rename to bb;
""",
        }
    )

    assert set(schema.tables) == {"a", "bb"}
    assert schema.tables["bb"].references == ["a"]


def test_structured_liquibase_changes_in_xml_and_yaml():
    schema = db_schema.build_schema(
        {
            "db/changelog-master.yaml": """databaseChangeLog:
  - include:
      file: db/one.xml
  - changeSet:
      id: 2
      changes:
        - addColumn:
            tableName: orders
            columns:
              - column: {name: total, type: numeric}
""",
            "db/one.xml": """<databaseChangeLog>
  <changeSet id="1" author="a">
    <createTable tableName="orders" schemaName="sales">
      <column name="id" type="uuid"><constraints primaryKey="true"/></column>
    </createTable>
  </changeSet>
</databaseChangeLog>""",
        }
    )

    # addColumn names no schema, so it does not match sales.orders — the
    # change is skipped rather than applied to a guessed table.
    assert _columns(schema, "sales.orders") == ["id uuid not null"]
    assert schema.statements_skipped == 1


def test_statements_split_outside_quotes_comments_and_function_bodies():
    statements = db_schema.split_statements(
        """
-- a comment; with a semicolon
insert into t values ('a;b');
create function f() returns trigger as $$ begin x := 1; return new; end; $$ language plpgsql;
/* block; comment */ select 1
"""
    )

    assert statements == [
        "insert into t values ('a;b')",
        "create function f() returns trigger as $$ begin x := 1; return new; end; $$ language plpgsql",
        "select 1",
    ]


def test_match_table_by_exact_or_unique_unqualified_name():
    tables = {"reports.reports": None, "reports.clients": None, "audit.clients": None}

    assert db_schema.match_table("REPORTS.REPORTS", tables) == "reports.reports"
    assert db_schema.match_table("reports", tables) == "reports.reports"
    assert db_schema.match_table("clients", tables) is None  # two candidates
    assert db_schema.match_table("missing", tables) is None
