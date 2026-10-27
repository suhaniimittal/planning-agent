from src.ingestion import service_links

MSG = "com.asurint.models.orders.events.UpdateHighRisk"
PAYLOAD = "com.asurint.models.orders.events.UpdateHighRiskPayload"


def _data(**overrides):
    data = {
        "avro": [],
        "kafka": [],
        "constants": [],
        "operations": [],
        "client_operations": [],
        "endpoints": [],
        "http_calls": [],
        "text_scan": [],
    }
    data.update(overrides)
    return data


def _kafka(service, qname, handles=(), publishes=(), built=(), inputs=(), outputs=()):
    return {
        "service": service,
        "qualified_name": qname,
        "handles": list(handles),
        "publishes": list(publishes),
        "publishes_built": list(built),
        "input_topics": list(inputs),
        "output_topics": list(outputs),
    }


TOPICS = {
    "service": "kafka",
    "file_path": "Topics.kt",
    "constants": {
        "Topics.domain": {"alias": "DomainTopics"},
        "DomainTopics.orders": {"value": "orders"},
    },
}


def test_kafka_links_publisher_to_handler_by_message_type_with_its_topic():
    result = service_links.compute(
        _data(
            avro=[
                {
                    "fqn": MSG,
                    "is_message": True,
                    "service": "models",
                    "file_path": "u.avdl",
                    "qualified_name": "u.avdl::UpdateHighRisk",
                }
            ],
            kafka=[
                _kafka("orders-service", "P", publishes=[MSG], outputs=["Topics.domain.orders"]),
                _kafka("rhino-service", "H", handles=[MSG], inputs=["Topics.domain.orders"]),
            ],
            constants=[TOPICS],
        )
    )

    assert result.service_links[("orders-service", "rhino-service")] == {
        "via": {"kafka"},
        "evidence": ["UpdateHighRisk"],
    }
    assert result.on_topic == [
        {"message": MSG, "topic": "orders", "constant": "Topics.domain.orders"}
    ]
    assert result.unresolved_topics == []


def test_built_records_count_only_when_they_are_messages():
    result = service_links.compute(
        _data(
            avro=[
                {
                    "fqn": MSG,
                    "is_message": True,
                    "service": "models",
                    "file_path": "u",
                    "qualified_name": "q1",
                },
                {
                    "fqn": PAYLOAD,
                    "is_message": False,
                    "service": "models",
                    "file_path": "u",
                    "qualified_name": "q2",
                },
            ],
            kafka=[_kafka("reports-api", "P", built=[MSG, PAYLOAD])],
        )
    )

    assert [p["message"] for p in result.publishes] == [MSG]


def test_graphql_links_only_an_unambiguous_operation():
    result = service_links.compute(
        _data(
            operations=[
                {"service": "reports-api", "type": "Query", "name": "reports"},
                {"service": "orders-service", "type": "Query", "name": "status"},
                {"service": "accounts-service", "type": "Query", "name": "status"},
            ],
            client_operations=[
                {
                    "service": "ui",
                    "file_path": "a.ts",
                    "operations": ["query:reports", "query:status"],
                },
            ],
        )
    )

    assert set(result.service_links) == {("ui", "reports-api")}
    ambiguous = [u for u in result.uses_operation if u["ambiguous"]]
    assert {u["target_service"] for u in ambiguous} == {"orders-service", "accounts-service"}
    assert result.ambiguous_operations == 1


def test_rest_links_a_client_path_to_the_one_matching_route():
    result = service_links.compute(
        _data(
            endpoints=[
                {"service": "rhino-service", "method": "GET", "path": "/internal/health/{id}"}
            ],
            http_calls=[
                {"service": "ui", "file_path": "x.ts", "paths": ["ANY /internal/health/{}"]}
            ],
        )
    )

    assert result.service_links[("ui", "rhino-service")]["via"] == {"rest"}


def test_text_scan_fills_in_only_what_concrete_links_do_not_explain():
    result = service_links.compute(
        _data(
            operations=[{"service": "reports-api", "type": "Query", "name": "reports"}],
            client_operations=[
                {"service": "ui", "file_path": "a.ts", "operations": ["query:reports"]}
            ],
            text_scan=[
                {"service": "ui", "target": "reports-api", "files": 3},
                {"service": "ui", "target": "accounts-service", "files": 1},
            ],
        )
    )

    assert result.service_links[("ui", "reports-api")]["via"] == {"graphql"}
    assert result.service_links[("ui", "accounts-service")]["via"] == {"text-scan"}
