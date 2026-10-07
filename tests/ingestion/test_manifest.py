import pytest

from src.ingestion.manifest import ManifestError, ServiceSpec, load_services, parse_service_specs


def _write(tmp_path, text):
    path = tmp_path / "services.yaml"
    path.write_text(text)
    return path


def test_load_services_parses_name_repo_and_optional_branch(tmp_path):
    path = _write(
        tmp_path,
        """
        services:
          - name: orders
            repo: org/orders-service
          - name: shipping
            repo: org/shipping-service
            branch: develop
        """,
    )

    specs = load_services(path)

    assert specs == [
        ServiceSpec(name="orders", repo="org/orders-service", branch=None),
        ServiceSpec(name="shipping", repo="org/shipping-service", branch="develop"),
    ]


def test_load_services_preserves_file_order(tmp_path):
    path = _write(
        tmp_path,
        """
        services:
          - name: zeta
            repo: org/zeta
          - name: alpha
            repo: org/alpha
        """,
    )

    specs = load_services(path)

    assert [s.name for s in specs] == ["zeta", "alpha"]


def test_load_services_missing_file_raises(tmp_path):
    with pytest.raises(ManifestError):
        load_services(tmp_path / "does-not-exist.yaml")


def test_load_services_invalid_yaml_raises(tmp_path):
    path = _write(tmp_path, "services: [this is not: valid: yaml")
    with pytest.raises(ManifestError):
        load_services(path)


def test_load_services_missing_services_key_raises(tmp_path):
    path = _write(tmp_path, "not_services:\n  - name: orders\n")
    with pytest.raises(ManifestError):
        load_services(path)


def test_load_services_entry_missing_name_raises(tmp_path):
    path = _write(tmp_path, "services:\n  - repo: org/orders-service\n")
    with pytest.raises(ManifestError):
        load_services(path)


def test_load_services_entry_missing_repo_raises(tmp_path):
    path = _write(tmp_path, "services:\n  - name: orders\n")
    with pytest.raises(ManifestError):
        load_services(path)


def test_load_services_duplicate_name_raises(tmp_path):
    path = _write(
        tmp_path,
        """
        services:
          - name: orders
            repo: org/orders-a
          - name: orders
            repo: org/orders-b
        """,
    )
    with pytest.raises(ManifestError):
        load_services(path)


def test_load_services_empty_list_returns_empty(tmp_path):
    path = _write(tmp_path, "services: []")
    assert load_services(path) == []


# --- parse_service_specs (the published agent's own input, no yaml file) ---


def test_parse_service_specs_parses_name_and_repo():
    specs = parse_service_specs("orders: myorg/orders-service")
    assert specs == [ServiceSpec(name="orders", repo="myorg/orders-service", branch=None)]


def test_parse_service_specs_parses_branch_after_at():
    specs = parse_service_specs("payment: myorg/payment-service@develop")
    assert specs == [ServiceSpec(name="payment", repo="myorg/payment-service", branch="develop")]


def test_parse_service_specs_multiple_lines_preserve_order():
    text = "orders: myorg/orders-service\npayment: myorg/payment-service@develop\n"
    specs = parse_service_specs(text)
    assert [s.name for s in specs] == ["orders", "payment"]


def test_parse_service_specs_accepts_equals_separator():
    specs = parse_service_specs("orders = myorg/orders-service")
    assert specs == [ServiceSpec(name="orders", repo="myorg/orders-service", branch=None)]


def test_parse_service_specs_skips_blank_lines_and_comments():
    text = "\n# a comment\norders: myorg/orders-service\n\n"
    specs = parse_service_specs(text)
    assert [s.name for s in specs] == ["orders"]


def test_parse_service_specs_malformed_line_raises_with_line_number():
    with pytest.raises(ManifestError, match="line 2"):
        parse_service_specs("orders: myorg/orders-service\nthis is not valid\n")


def test_parse_service_specs_duplicate_name_raises():
    text = "orders: myorg/orders-a\norders: myorg/orders-b\n"
    with pytest.raises(ManifestError, match="duplicate"):
        parse_service_specs(text)


def test_parse_service_specs_empty_text_raises():
    with pytest.raises(ManifestError, match="no services"):
        parse_service_specs("")


def test_parse_service_specs_only_comments_raises():
    with pytest.raises(ManifestError, match="no services"):
        parse_service_specs("# nothing here\n\n")
