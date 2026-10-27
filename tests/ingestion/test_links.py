"""Stage 3 end to end: a small repo on disk, parsed by the real parser, then
linked. The UI fixture has the shape of the ticket that motivated pipeline
version 4 (Client UI: controls shift when the Archive button appears): the
screen imports a grid through a tsconfig alias whose folder is a barrel,
the button is rendered by the grid, and its type is a `type` alias.
"""

from pathlib import Path

from src.ingestion.code_parser import _CHUNK_TARGET_CHARS, parse_repo
from src.ingestion.links import compute_links, index_from_repo_graph, index_from_rows

GRID = "src/apps/client/components/InfiniteScrollGrid"
SCREEN = "src/apps/client/modules/Reports/pages/ReportsListScreen"

UI_FILES = {
    "tsconfig.json": """{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@client/*": ["src/apps/client/*"], // a comment, as the real one has
      "@design-system/v1": ["src/common/design/v1"],
    },
  },
}
""",
    f"{GRID}/index.ts": 'export { InfiniteScrollGrid } from "./view";\nexport * from "./types";\n',
    f"{GRID}/types.ts": """export type ActionButton = {
  id: string;
  onClick: () => void;
  title: string;
  disabled?: boolean;
};

export interface GridProps {
  title?: string;
  actionButton?: ActionButton;
}
""",
    f"{GRID}/styles.ts": """import styled from "styled-components";

export const SearchAndButtonWrap = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
`;
""",
    f"{GRID}/view.tsx": """import React from "react";
import { GridProps } from "./types";
import { SearchAndButtonWrap } from "./styles";

export const InfiniteScrollGrid = ({ title, actionButton }: GridProps) => {
  const submitAllFilters = () => {
    console.log("submit");
  };
  return (
    <div>
      {title}
      <SearchAndButtonWrap>
        {actionButton && (
          <button disabled={actionButton.disabled} onClick={actionButton.onClick}>
            {actionButton.title}
          </button>
        )}
      </SearchAndButtonWrap>
    </div>
  );
};
""",
    f"{SCREEN}/view.tsx": """import React, { useCallback, useState } from "react";
import { InfiniteScrollGrid, ActionButton } from "@client/components/InfiniteScrollGrid";

export const ReportsListScreen = () => {
  const [state, setState] = useState<ActionButton | null>(null);
  const updateArchiveButton = useCallback((rows: { archived: boolean }[]) => {
    const allArchived = rows.every((r) => r.archived);
    setState({ id: "archive", title: allArchived ? "Unarchive" : "Archive", onClick: () => {} });
  }, []);
  return <InfiniteScrollGrid title="Reports" actionButton={state ?? undefined} />;
};
""",
    f"{SCREEN}/view.test.tsx": """import { ReportsListScreen } from "./view";

describe("ReportsListScreen", () => {
  it("renders", () => {
    expect(ReportsListScreen).toBeDefined();
  });
});
""",
}


def _write(root: Path, files: dict[str, str]) -> None:
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def _ui(tmp_path):
    _write(tmp_path, UI_FILES)
    graph = parse_repo(tmp_path, "ui")
    return graph, compute_links(index_from_repo_graph("ui", graph))


def test_every_file_is_stored_whole(tmp_path):
    graph, _ = _ui(tmp_path)

    for path, text in UI_FILES.items():
        assert graph.file_source[path] == text


def test_the_button_type_is_a_symbol(tmp_path):
    graph, _ = _ui(tmp_path)

    action_button = next(c for c in graph.classes if c.name == "ActionButton")
    assert action_button.symbol_kind == "type"
    assert action_button.file_path == f"{GRID}/types.ts"


def test_a_usecallback_handler_is_a_symbol_named_after_its_parent(tmp_path):
    graph, _ = _ui(tmp_path)

    handler = next(f for f in graph.functions if f.name == "updateArchiveButton")
    assert handler.qualified_name == f"{SCREEN}/view.tsx::ReportsListScreen.updateArchiveButton"
    assert handler.parent_symbol == f"{SCREEN}/view.tsx::ReportsListScreen"
    assert "allArchived" in handler.source_snippet


def test_the_screen_imports_the_grid_through_the_alias_and_the_barrel(tmp_path):
    _, links = _ui(tmp_path)

    edges = {e.target: e for e in links.imports if e.source == f"{SCREEN}/view.tsx"}
    # Followed past index.ts to the files that define each name.
    assert edges[f"{GRID}/view.tsx"].names == ["InfiniteScrollGrid"]
    assert edges[f"{GRID}/view.tsx"].via == [f"{GRID}/index.ts"]
    assert edges[f"{GRID}/types.ts"].names == ["ActionButton"]
    assert f"{GRID}/index.ts" not in edges
    assert links.unresolved_imports == {}


def test_the_screen_references_the_component_it_renders(tmp_path):
    _, links = _ui(tmp_path)

    targets = {t for s, t in links.references if s == f"{SCREEN}/view.tsx::ReportsListScreen"}
    assert f"{GRID}/view.tsx::InfiniteScrollGrid" in targets
    assert f"{GRID}/types.ts::ActionButton" in targets
    grid_refs = {t for s, t in links.references if s == f"{GRID}/view.tsx::InfiniteScrollGrid"}
    assert f"{GRID}/styles.ts::SearchAndButtonWrap" in grid_refs
    assert f"{GRID}/types.ts::GridProps" in grid_refs


def test_the_test_file_is_linked_to_what_it_tests(tmp_path):
    _, links = _ui(tmp_path)

    assert (f"{SCREEN}/view.test.tsx", f"{SCREEN}/view.tsx") in links.tests


def test_no_chunk_is_far_over_the_target_and_none_repeats_a_signature(tmp_path):
    graph, _ = _ui(tmp_path)

    chunks = [c for s in graph.symbols for c in s.chunks]
    assert chunks
    assert all(len(c.text) <= 2 * _CHUNK_TARGET_CHARS for c in chunks)
    # The nested handler's lines are the handler's, not its parent's too.
    screen = next(f for f in graph.functions if f.name == "ReportsListScreen")
    assert not any("allArchived" in c.text for c in screen.chunks)


def test_test_files_are_not_embedded(tmp_path):
    graph, _ = _ui(tmp_path)

    assert not any(s.chunks for s in graph.symbols if s.file_path.endswith(".test.tsx"))


def test_links_are_identical_when_rebuilt_from_stored_rows(tmp_path):
    """An incremental sync recomputes links from what Neo4j holds, not from a
    fresh parse — both must give the same answer."""
    import json

    graph, from_parse = _ui(tmp_path)
    file_rows = [
        {
            "file_path": path,
            "language": record.language,
            "kind": record.kind,
            "import_specs": json.dumps(
                [s.to_dict() for s in graph.file_import_specs.get(path, [])]
            ),
            "package": graph.file_packages.get(path),
            "facts": json.dumps(graph.file_facts.get(path, {})),
        }
        for path, record in graph.files.items()
    ]
    labels = {"class": "Class", "function": "Function", "variable": "Variable"}
    symbol_rows = [
        {
            "qualified_name": s.qualified_name,
            "name": s.name,
            "label": labels[s.kind],
            "symbol_kind": s.symbol_kind,
            "file_path": s.file_path,
            "parent_symbol": s.parent_symbol,
            "identifiers": s.identifiers,
            "supertypes": s.supertypes,
            "facts": json.dumps(s.facts),
            "language": s.language,
        }
        for s in graph.symbols
    ]
    index = index_from_rows(
        "ui", file_rows, symbol_rows, lambda paths: {p: graph.file_source[p] for p in paths}
    )
    from_rows = compute_links(index)

    assert {(e.source, e.target) for e in from_rows.imports} == {
        (e.source, e.target) for e in from_parse.imports
    }
    assert set(from_rows.references) == set(from_parse.references)
    assert set(from_rows.tests) == set(from_parse.tests)


# --- a Kotlin service: Kafka, JPA, GraphQL, REST ------------------------------

PKG = "src/main/kotlin/com/asurint/rhino"

KOTLIN_FILES = {
    f"{PKG}/kafka/handlers/UpdateHighRiskHandler.kt": """package com.asurint.rhino.kafka.handlers

import com.asurint.kafka.configuration.Topics
import com.asurint.kafka.dispatching.KafkaMessageHandler
import com.asurint.models.orders.events.UpdateHighRisk

class UpdateHighRiskHandler : KafkaMessageHandler<UpdateHighRisk> {
    override val inputTopic = Topics.domain.orders
    override val messageType = UpdateHighRisk::class.java

    override fun handle(message: UpdateHighRisk) {
        println(message)
    }
}
""",
    f"{PKG}/kafka/publishers/AbstractPublisher.kt": """package com.asurint.rhino.kafka.publishers

import com.asurint.kafka.configuration.Topics
import com.asurint.kafka.spring.KafkaProducer

abstract class AbstractPublisher<MessageType>(private val producer: KafkaProducer<OrderKey, MessageType>) {
    fun handle(value: MessageType) {
        producer.synchronousSend(topicName = Topics.domain.rhino, key = key, value = value)
    }
}
""",
    f"{PKG}/kafka/publishers/ResultPublisher.kt": """package com.asurint.rhino.kafka.publishers

import com.asurint.kafka.spring.KafkaProducer
import com.asurint.models.keys.OrderKey
import com.asurint.models.rhino.RhinoResult

class ResultPublisher(
    producer: KafkaProducer<OrderKey, RhinoResult>,
) : AbstractPublisher<RhinoResult>(producer) {
    fun build(): RhinoResult = RhinoResult.newBuilder().build()
}
""",
    f"{PKG}/entities/ReportEntity.kt": """package com.asurint.rhino.entities

import javax.persistence.Entity
import javax.persistence.Table

@Entity
@Table(name = "reports", schema = "rhino")
data class ReportEntity(val id: String)
""",
    f"{PKG}/repositories/ReportRepository.kt": """package com.asurint.rhino.repositories

import com.asurint.rhino.entities.ReportEntity
import org.springframework.data.jpa.repository.JpaRepository

interface ReportRepository : JpaRepository<ReportEntity, String> {
    @Query(value = "select count(*) from rhino.audit_log where id = :id", nativeQuery = true)
    fun countAudit(id: String): Int
}
""",
    f"{PKG}/graphql/ReportsResolver.kt": """package com.asurint.rhino.graphql

import com.coxautodev.graphql.tools.GraphQLQueryResolver

class ReportsResolver : GraphQLQueryResolver {
    fun reports(): List<String> = listOf()
}
""",
    f"{PKG}/http/HealthController.kt": """package com.asurint.rhino.http

@RestController
@RequestMapping("/internal")
class HealthController {
    @GetMapping("/health/{id}")
    fun health(id: String): String = "ok"
}
""",
    "src/main/resources/graphql/rhino.graphqls": """type Query {
  reports: [String]
}
""",
    "liquibase/changelog-master.xml": """<databaseChangeLog xmlns="http://www.liquibase.org/xml/ns/dbchangelog">
    <changeSet id="1" author="a"><sqlFile path="liquibase/001_init.sql"/></changeSet>
    <changeSet id="2" author="a"><sqlFile path="liquibase/002_audit.sql"/></changeSet>
</databaseChangeLog>
""",
    "liquibase/001_init.sql": """CREATE SCHEMA IF NOT EXISTS rhino;
create table rhino.reports (
    id uuid not null constraint reports_pkey primary key,
    status varchar(50)
);
""",
    "liquibase/002_audit.sql": """create table rhino.audit_log (id uuid not null);
alter table rhino.reports add column archived boolean not null default false;
alter table rhino.reports rename column status to state;
""",
    "src/test/kotlin/com/asurint/rhino/repositories/ReportRepositoryTests.kt": """package com.asurint.rhino.repositories

class ReportRepositoryTests {
    fun works() {}
}
""",
}


def _kotlin(tmp_path):
    _write(tmp_path, KOTLIN_FILES)
    graph = parse_repo(tmp_path, "rhino-service")
    return graph, compute_links(index_from_repo_graph("rhino-service", graph))


def test_kafka_types_resolve_to_full_names_through_imports(tmp_path):
    _, links = _kotlin(tmp_path)

    handler = links.kafka[f"{PKG}/kafka/handlers/UpdateHighRiskHandler.kt::UpdateHighRiskHandler"]
    assert handler["handles"] == ["com.asurint.models.orders.events.UpdateHighRisk"]
    assert handler["input_topics"] == ["Topics.domain.orders"]
    publisher = links.kafka[f"{PKG}/kafka/publishers/ResultPublisher.kt::ResultPublisher"]
    assert publisher["publishes"] == ["com.asurint.models.rhino.RhinoResult"]
    # The topic is set where the base class sends, and is inherited.
    assert publisher["output_topics"] == ["Topics.domain.rhino"]
    # A type parameter resolves to nothing and is dropped, not guessed.
    assert f"{PKG}/kafka/publishers/AbstractPublisher.kt::AbstractPublisher" not in links.kafka


def test_tables_are_replayed_from_the_changelog_in_order(tmp_path):
    _, links = _kotlin(tmp_path)

    tables = {t.name: t for t in links.tables}
    assert set(tables) == {"rhino.reports", "rhino.audit_log"}
    columns = [c.render() for c in tables["rhino.reports"].columns.values()]
    assert columns == ["id uuid not null", "state varchar(50)", "archived boolean not null"]
    assert tables["rhino.reports"].defined_in == [
        "liquibase/001_init.sql",
        "liquibase/002_audit.sql",
    ]


def test_entities_repositories_and_sql_link_to_tables(tmp_path):
    _, links = _kotlin(tmp_path)

    assert (f"{PKG}/entities/ReportEntity.kt::ReportEntity", "rhino.reports") in links.maps_to
    repository = f"{PKG}/repositories/ReportRepository.kt::ReportRepository"
    assert (repository, "rhino.reports") in links.reads
    assert (repository, "rhino.reports") in links.writes
    assert (f"{repository}.countAudit", "rhino.audit_log") in links.reads


def test_graphql_operations_and_their_resolvers(tmp_path):
    _, links = _kotlin(tmp_path)

    assert [(o.type, o.name) for o in links.operations] == [("Query", "reports")]
    assert links.resolves == [
        (f"{PKG}/graphql/ReportsResolver.kt::ReportsResolver.reports", "Query", "reports")
    ]


def test_rest_routes_join_the_class_and_method_paths(tmp_path):
    _, links = _kotlin(tmp_path)

    assert links.endpoints == [
        (f"{PKG}/http/HealthController.kt::HealthController.health", "GET", "/internal/health/{id}")
    ]


def test_extends_and_jvm_test_naming(tmp_path):
    _, links = _kotlin(tmp_path)

    assert (
        f"{PKG}/kafka/publishers/ResultPublisher.kt::ResultPublisher",
        f"{PKG}/kafka/publishers/AbstractPublisher.kt::AbstractPublisher",
    ) in links.extends
    assert (
        "src/test/kotlin/com/asurint/rhino/repositories/ReportRepositoryTests.kt",
        f"{PKG}/repositories/ReportRepository.kt",
    ) in links.tests
