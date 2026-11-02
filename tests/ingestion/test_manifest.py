import pytest

from src.ingestion.manifest import ManifestError, ServiceSpec, load_services


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
