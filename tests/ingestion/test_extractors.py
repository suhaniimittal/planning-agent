from src.ingestion.extractors import (
    build_entity_index,
    extract_owned_entities,
    extract_repo_slug,
    resolve_service_name,
)

USER_DOC_MD = """# User — Technical Documentation

> This service covers customer account storage, including cards and addresses.

*Generated from `org/user` at `abc123` — 27 source files, 246 symbols.*

---

## Project Overview

This service covers customer account storage, including cards and addresses.

|  |  |
|---|---|
| Domain | User account management |
| Languages | Go |

---

## File Reference

| File | Role | Symbols | Key symbols |
|---|---|---|---|
| `db.go` | src | 30 | `Init`, `New`, `MakeLoginEndpoint`, `AddCustomer`, `TestRegister` |
| `users.go` | src | 6 | `Login`, `Register`, `MaskCC`, `User`, `Address`, `Card`, `Customer` |

---
"""

ORDERS_DOC_MD = """# Orders — Technical Documentation

> Provides ordering capabilities for a microservices architecture.

*Generated from `org/orders` at `def456` — 38 source files, 410 symbols.*

---

## Project Overview

Provides ordering capabilities for a microservices architecture.

|  |  |
|---|---|
| Domain | E-commerce |
| Languages | Java |

---

## How the System Works

This service calls to customer, address, card, payment and shipping
services to complete an order.

---

## File Reference

| File | Role | Symbols | Key symbols | Uses | Used by | Documented above |
|---|---|---|---|---|---|---|
| `OrdersController.java` | source | 13 | `OrdersController`, `newOrder`, `parseId` | - | - | - |
| `CustomerOrder.java` | source | 32 | `CustomerOrder`, `toString`, `getId` | - | - | - |
| `Address.java` | source | 25 | `Address`, `toString`, `equals` | - | - | - |
| `Card.java` | source | 19 | `Card`, `toString`, `equals` | - | - | - |
| `Customer.java` | source | 25 | `Customer`, `toString`, `equals` | - | - | - |

---
"""

SHIPPING_DOC_MD = """# Shipping — Technical Documentation

> Provides shipping capabilities as part of a microservices architecture.

*Generated from `org/shipping` at `ghi789` — 20 source files, 100 symbols.*

---

## Project Overview

Provides shipping capabilities as part of a microservices architecture.

|  |  |
|---|---|
| Domain | Shipping |
| Languages | Java |

---

## File Reference

| File | Role | Symbols | Key symbols | Uses | Used by | Documented above |
|---|---|---|---|---|---|---|
| `ShippingController.java` | source | 5 | `ShippingController`, `Shipment` | - | - | - |

---
"""


def user_symbols() -> list[str]:
    return [
        "Init", "New", "MakeLoginEndpoint", "AddCustomer", "TestRegister",
        "Login", "Register", "MaskCC", "User", "Address", "Card", "Customer",
    ]


# --- extract_owned_entities ---------------------------------------------


def test_extract_owned_entities_user_doc_finds_expected_entities():
    result = extract_owned_entities(USER_DOC_MD, user_symbols())
    assert result == {"address", "card", "customer"}


def test_extract_owned_entities_excludes_verb_prefixed_symbols():
    result = extract_owned_entities(USER_DOC_MD, user_symbols())
    verbs = ("AddCustomer", "TestRegister", "MakeLoginEndpoint", "Init", "New", "Login", "Register")
    for verb_symbol in verbs:
        assert verb_symbol.lower() not in result


def test_extract_owned_entities_excludes_role_suffixed_symbols():
    symbols = ["UserController", "OrderRepository", "PaymentException", "Address"]
    result = extract_owned_entities(USER_DOC_MD, symbols)
    assert "usercontroller" not in result
    assert "orderrepository" not in result
    assert "paymentexception" not in result
    assert "address" in result


def test_extract_owned_entities_no_entities_found_returns_empty_set():
    symbols = ["AddCustomer", "GetOrder", "TestLogin", "OrdersController"]
    assert extract_owned_entities(USER_DOC_MD, symbols) == set()


def test_extract_owned_entities_no_overview_text_falls_back_to_symbol_shape():
    no_overview_md = "*Generated from `org/lib` at `xyz` — 1 source files.*\n"
    result = extract_owned_entities(no_overview_md, ["Address", "Card", "AddCustomer"])
    assert result == {"address", "card"}


def test_extract_owned_entities_singular_plural_overview_match():
    md = (
        "*Generated from `org/user` at `abc` — 1 source files.*\n\n"
        "## Project Overview\n\n"
        "This service covers cards and addresses.\n\n"
        "|  |  |\n|---|---|\n| Domain | User account management |\n\n---\n"
    )
    result = extract_owned_entities(md, ["Card", "Address"])
    assert result == {"card", "address"}


def test_extract_owned_entities_excludes_own_service_name():
    result = extract_owned_entities(USER_DOC_MD, user_symbols() + ["User"])
    assert "user" not in result


# --- build_entity_index --------------------------------------------------


def test_build_entity_index_single_owner_resolves():
    records = [{"name": "user", "full_doc": USER_DOC_MD, "key_symbols": user_symbols()}]
    index = build_entity_index(records)
    assert index == {"address": "user", "card": "user", "customer": "user"}


def test_build_entity_index_ambiguous_entity_dropped(capsys):
    doc_a = (
        "*Generated from `org/a` at `x` — 1 source files.*\n\n"
        "## Project Overview\n\nManages profiles.\n\n"
        "|  |  |\n|---|---|\n| Domain | Profile management |\n\n---\n"
    )
    doc_b = (
        "*Generated from `org/b` at `x` — 1 source files.*\n\n"
        "## Project Overview\n\nManages profiles too.\n\n"
        "|  |  |\n|---|---|\n| Domain | Profile management |\n\n---\n"
    )
    records = [
        {"name": "a", "full_doc": doc_a, "key_symbols": ["Profile"]},
        {"name": "b", "full_doc": doc_b, "key_symbols": ["Profile"]},
    ]
    index = build_entity_index(records)
    assert "profile" not in index
    assert "AMBIGUOUS" in capsys.readouterr().out


def test_build_entity_index_empty_when_no_records_own_anything():
    symbols = ["AddCustomer", "GetOrder"]
    records = [{"name": "orders", "full_doc": ORDERS_DOC_MD, "key_symbols": symbols}]
    assert build_entity_index(records) == {}


# --- resolve_service_name -------------------------------------------------


def test_resolve_service_name_exact_match_unchanged():
    known = {"orders", "shipping", "payment"}
    assert resolve_service_name("shipping", known) == "shipping"


def test_resolve_service_name_orders_to_shipping_regression():
    from src.ingestion.extractors import extract_narrative_list_targets

    known_services = {"orders", "shipping", "user"}
    entity_index = {"customer": "user", "address": "user", "card": "user"}

    mentions = extract_narrative_list_targets(ORDERS_DOC_MD)
    resolved = {resolve_service_name(m, known_services, entity_index) for m in mentions}
    assert "shipping" in resolved
    assert "user" in resolved  # via customer/address/card -> entity index
    assert None in resolved  # "payment" stays unresolved (no payment doc ingested)


def test_resolve_service_name_pluralization_cart_to_carts():
    known = {"carts", "user"}
    assert resolve_service_name("cart", known) == "carts"


def test_resolve_service_name_entity_index_customer_to_user():
    known = {"user", "orders"}
    entity_index = {"customer": "user", "address": "user", "card": "user"}
    assert resolve_service_name("customer", known, entity_index) == "user"


def test_resolve_service_name_unresolved_mention_returns_none():
    known = {"user", "orders"}
    assert resolve_service_name("payment", known, {"customer": "user"}) is None


def test_resolve_service_name_entity_index_none_defaults_gracefully():
    known = {"shipping"}
    assert resolve_service_name("shipping", known) == "shipping"


# --- extract_repo_slug ------------------------------------------------------


def test_extract_repo_slug_returns_full_org_repo():
    assert extract_repo_slug(ORDERS_DOC_MD) == "org/orders"


def test_extract_repo_slug_returns_none_when_no_generated_from_line():
    assert extract_repo_slug("# No provenance line here") is None
