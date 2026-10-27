"""Facts read off real-shaped code: every pattern here is one found in the
services this agent ingests."""

from src.ingestion import code_facts
from src.ingestion.code_parser import parse_source_full


def _facts(source: str, path: str = "X.kt", language: str = "kotlin"):
    parsed = parse_source_full(source.encode(), path, language)
    return {s.name: s.facts for s in parsed.symbols}, parsed.facts


def test_jpa_entity_and_repository():
    facts, _ = _facts(
        """
@Entity
@Table(name = "abuse_nurse_registry", schema = "packages")
data class AbuseRegistryEntity(@Id val id: UUID)

interface AbuseRegistryRepository : JpaRepository<AbuseRegistryEntity, UUID>
"""
    )

    assert facts["AbuseRegistryEntity"] == {
        "db_table": "packages.abuse_nurse_registry",
        "jpa_entity": True,
    }
    assert facts["AbuseRegistryRepository"]["repository_entity"] == "AbuseRegistryEntity"


def test_kafka_handler_and_producer():
    facts, _ = _facts(
        """
class UpdateHighRiskSSNHandler(private val h: X) : KafkaMessageHandler<UpdateHighRisk> {
    override val inputTopic = Topics.domain.orders
    override fun handle(message: UpdateHighRisk) {}
}

class ReportAttachmentAddedPublisher(private val kafkaProducer: KafkaProducer<SpecificRecord, SpecificRecord>) {
    fun handle() {
        kafkaProducer.synchronousSend(
            topicName = Topics.domain.reports,
            value = ReportAttachmentAdded.newBuilder().setPayload(ReportsContext.newBuilder().build()).build(),
        )
    }
}
"""
    )

    assert facts["UpdateHighRiskSSNHandler"] == {
        "kafka_handles": ["UpdateHighRisk"],
        "kafka_input_topics": ["Topics.domain.orders"],
    }
    publisher = facts["ReportAttachmentAddedPublisher"]
    # A generic producer's message is where it is built — kept apart, since
    # its payload and context records are built the same way.
    assert "kafka_publishes" not in publisher
    assert publisher["kafka_publishes_built"] == ["ReportAttachmentAdded", "ReportsContext"]
    assert publisher["kafka_output_topics"] == ["Topics.domain.reports"]


def test_graphql_java_tools_resolvers_bind_methods_to_fields():
    facts, _ = _facts(
        """
class ReportsQuery : GraphQLQueryResolver {
    fun reports(): List<Report> = listOf()
    fun getArchivedReports(): List<Report> = listOf()
}
class ArchiveReportsResolver : GraphQLMutationResolver {
    fun archiveReports(input: ArchiveReportsInput): String = ""
}
"""
    )

    assert facts["reports"]["graphql_resolves"] == ["Query.reports"]
    assert facts["getArchivedReports"]["graphql_resolves"] == [
        "Query.archivedReports",
        "Query.getArchivedReports",
    ]
    assert facts["archiveReports"]["graphql_resolves"] == ["Mutation.archiveReports"]


def test_spring_graphql_and_dgs_annotations():
    facts, _ = _facts(
        """
class Controller {
    @QueryMapping
    fun orders(): List<Order> = listOf()

    @DgsData(parentType = "Mutation", field = "cancelOrder")
    fun cancel(): Boolean = true
}
"""
    )

    assert facts["orders"]["graphql_resolves"] == ["Query.orders"]
    assert facts["cancel"]["graphql_resolves"] == ["Mutation.cancelOrder"]


def test_rest_routes():
    facts, _ = _facts(
        """
@RestController
@RequestMapping("/api/v1")
class OrdersController {
    @GetMapping("/orders/{id}")
    fun get(id: String) = id

    @RequestMapping(value = "/orders", method = [RequestMethod.POST])
    fun create() = 1
}
"""
    )

    assert facts["get"]["routes"] == ["GET /api/v1/orders/{id}"]
    assert facts["create"]["routes"] == ["POST /api/v1/orders"]


def test_sql_in_strings_names_tables_read_and_written():
    reads, writes = code_facts.sql_tables(
        '''
        val q = """
            SELECT r.id FROM reports.reports r JOIN reports.clients c ON c.id = r.client_id
        """
        jdbc.update("update reports.reports set status = ? where id = ?")
        val notSql = "from here to there"
    '''
    )

    assert reads == ["reports.reports", "reports.clients"]
    assert writes == ["reports.reports"]


def test_graphql_documents_in_js_find_client_fields_and_server_schemas():
    uses, exposes = code_facts.graphql_documents(
        """
const GET = gql`
  query GetReports($id: ID) { reports(id: $id) { id ...F } __typename }
  ${FRAGMENT}
`;
const typeDefs = gql`
  type Query { orders: [Order] }
  extend type Mutation { cancel(id: ID!): Boolean }
`;
const broken = gql`query {`;
""",
        "typescript",
    )

    assert uses == ["query:reports"]
    assert exposes == ["Query.orders", "Mutation.cancel"]


def test_http_paths_in_client_code():
    paths = code_facts.http_paths(
        'fetch("/api/reports");\naxios.post(`/api/reports/${id}/archive`, body);\n', "typescript"
    )

    assert paths == ["ANY /api/reports", "POST /api/reports/{}/archive"]


def test_kotlin_constants_follow_object_aliases():
    constants = code_facts.kotlin_constants(
        """
object Topics {
    val domain = DomainTopics
    const val DEFAULT_TOPIC_PREFIX = "local"
}
object DomainTopics {
    const val orders = "orders"
    object nested { const val deep = "deep-topic" }
}
"""
    )

    assert constants["Topics.domain"] == {"alias": "DomainTopics"}
    assert constants["DomainTopics.orders"] == {"value": "orders"}
    assert constants["DomainTopics.nested.deep"] == {"value": "deep-topic"}
