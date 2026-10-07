from src.ingestion.edge_detector import find_service_references, find_service_references_in_text


def _union(results: dict[str, set[str]]) -> set[str]:
    found: set[str] = set()
    for refs in results.values():
        found |= refs
    return found


# --- find_service_references_in_text (the per-file core matcher) -----------


def test_in_text_finds_service_referenced_near_url_keyword():
    text = "PAYMENT_SERVICE_URL = 'http://payment-service:8080/graphql'"
    assert find_service_references_in_text(text, {"payment", "shipping"}) == {"payment"}


def test_in_text_ignores_bare_mention_with_no_networking_context():
    text = "TODO: ask the payment team about refund policy someday"
    assert find_service_references_in_text(text, {"payment"}) == set()


def test_in_text_detects_graphql_client_configuration():
    text = "const client = new ApolloClient({ uri: process.env.SHIPPING_GRAPHQL_ENDPOINT });"
    assert find_service_references_in_text(text, {"shipping"}) == {"shipping"}


def test_in_text_finds_multiple_distinct_services():
    text = "PAYMENT_URL = 'http://payment/graphql'\nSHIPPING_URL = 'http://shipping/graphql'"
    assert find_service_references_in_text(text, {"payment", "shipping", "user"}) == {"payment", "shipping"}


def test_in_text_no_candidates_returns_empty():
    assert find_service_references_in_text("payment url graphql", set()) == set()


def test_in_text_ignores_service_and_client_as_context_keywords():
    """Regression test: "service"/"client" are too generic to count as
    networking context — Spring's XxxService class-naming convention and
    ...service.auth-style package names put the literal word "service" next
    to almost every class in a typical Java app, with zero real network
    call. Observed for real: "user" matched nearly every file in an
    unrelated auth service purely via `UserService`/`import ...service.auth`."""
    text = (
        "package com.sathvik.CodingManager.service.auth;\n"
        "import com.sathvik.CodingManager.model.User;\n"
        "import com.sathvik.CodingManager.repository.UserRepository;\n\n"
        "public class LoginService {\n"
        "    private UserRepository userRepository;\n"
        "    private PaymentClient client;\n"
        "}\n"
    )
    assert find_service_references_in_text(text, {"user", "payment"}) == set()


def test_in_text_still_matches_url_scheme_directly():
    text = 'restTemplate.getForObject("http://payment-service/api", Response.class)'
    assert find_service_references_in_text(text, {"payment"}) == {"payment"}


def test_in_text_ignores_service_name_glued_inside_a_compound_identifier():
    """Regression test: "user" must not match as a mere substring of
    "UsernamePasswordAuthenticationFilter" (Spring Security's own class) —
    only a standalone token counts, not a fragment of a longer identifier."""
    text = "import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;"
    assert find_service_references_in_text(text, {"user"}) == set()


def test_in_text_ignores_keyword_glued_inside_an_unrelated_word():
    """Regression test: "uri" must not match as a substring of "security" —
    real observed false positive (every Spring Security import line)."""
    text = "org.springframework.security has nothing to do with payment"
    assert find_service_references_in_text(text, {"payment"}) == set()


def test_in_text_still_matches_service_name_with_underscore_or_hyphen_neighbors():
    """The token-boundary check must not be as strict as regex \\b, which
    would itself reject "payment" in "PAYMENT_SERVICE_URL" since underscore
    counts as a word character."""
    assert find_service_references_in_text("PAYMENT_SERVICE_URL = 'x'", {"payment"}) == {"payment"}
    assert find_service_references_in_text("payment-service.example.com/url", {"payment"}) == {"payment"}


# --- find_service_references (full-repo scan, per-file granularity) --------


def test_finds_service_referenced_near_url_keyword(tmp_path):
    (tmp_path / "client.py").write_text(
        "PAYMENT_SERVICE_URL = 'http://payment-service:8080/graphql'\n"
    )

    found = find_service_references(tmp_path, "orders", {"orders", "payment", "shipping"})

    assert found == {"client.py": {"payment"}}


def test_ignores_bare_mention_with_no_networking_context(tmp_path):
    (tmp_path / "notes.py").write_text(
        "# TODO: ask the payment team about refund policy someday\n" * 5
    )

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found == {}


def test_detects_graphql_client_configuration(tmp_path):
    (tmp_path / "gateway.js").write_text(
        "const client = new ApolloClient({ uri: process.env.SHIPPING_GRAPHQL_ENDPOINT });\n"
    )

    found = find_service_references(tmp_path, "orders", {"orders", "shipping"})

    assert found == {"gateway.js": {"shipping"}}


def test_scans_config_files_not_just_source(tmp_path):
    (tmp_path / "supergraph.yaml").write_text(
        "subgraphs:\n  - name: user\n    url: http://user-service/graphql\n"
    )

    found = find_service_references(tmp_path, "orders", {"orders", "user"})

    assert found == {"supergraph.yaml": {"user"}}


def test_excludes_self_from_candidates(tmp_path):
    (tmp_path / "app.py").write_text("ORDERS_SERVICE_URL = 'http://orders-service/graphql'\n")

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found == {}


def test_finds_multiple_distinct_services_across_different_files(tmp_path):
    (tmp_path / "payment_client.py").write_text("PAYMENT_URL = 'http://payment/graphql'\n")
    (tmp_path / "shipping_client.py").write_text("SHIPPING_URL = 'http://shipping/graphql'\n")

    found = find_service_references(tmp_path, "orders", {"orders", "payment", "shipping", "user"})

    assert found == {
        "payment_client.py": {"payment"},
        "shipping_client.py": {"shipping"},
    }


def test_multiple_services_referenced_in_the_same_file(tmp_path):
    (tmp_path / "clients.py").write_text(
        "PAYMENT_URL = 'http://payment/graphql'\n"
        "SHIPPING_URL = 'http://shipping/graphql'\n"
    )

    found = find_service_references(tmp_path, "orders", {"orders", "payment", "shipping"})

    assert found == {"clients.py": {"payment", "shipping"}}


def test_skips_denylisted_directories(tmp_path):
    nested = tmp_path / "node_modules" / "some_pkg"
    nested.mkdir(parents=True)
    (nested / "index.js").write_text("const url = 'http://payment-service/graphql';\n")

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found == {}


def test_skips_unscannable_extensions(tmp_path):
    (tmp_path / "photo.png").write_bytes(b"payment url graphql" * 10)

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found == {}


def test_scans_schema_and_process_definition_extensions(tmp_path):
    (tmp_path / "schema.graphqls").write_text('type Query { p: Payment } # url: http://payment-service')
    (tmp_path / "process.bpmn").write_text('<task url="http://payment-service/api" />')
    (tmp_path / "protocol.avdl").write_text('// http://payment-service/graphql endpoint url')

    for fname in ("schema.graphqls", "process.bpmn", "protocol.avdl"):
        found = find_service_references(tmp_path, "orders", {"orders", "payment"})
        assert found.get(fname) == {"payment"}, f"{fname} should have matched"


def test_scans_tracked_only_extensions_for_dependencies(tmp_path):
    (tmp_path / "index.html").write_text('<a href="http://payment-service/checkout">Pay</a>')
    (tmp_path / "style.css").write_text("/* url(http://payment-service/bg.png) */")
    (tmp_path / "data.csv").write_text("service_url\nhttp://payment-service/api\n")

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found.get("index.html") == {"payment"}
    assert found.get("style.css") == {"payment"}
    assert found.get("data.csv") == {"payment"}


def test_no_other_known_services_returns_empty_without_scanning(tmp_path):
    (tmp_path / "app.py").write_text("payment url graphql\n")

    found = find_service_references(tmp_path, "orders", {"orders"})

    assert found == {}


def test_keeps_scanning_every_file_even_after_all_candidates_seen(tmp_path):
    """No early exit: file b.py's own (lack of) references must still be
    recorded accurately, independent of what a.py already matched — needed
    since each file's contribution is stored on its own File node."""
    (tmp_path / "a.py").write_text("PAYMENT_URL = 'http://payment/graphql'\n")
    (tmp_path / "b.py").write_text("nothing relevant here\n")

    found = find_service_references(tmp_path, "orders", {"orders", "payment"})

    assert found == {"a.py": {"payment"}}
