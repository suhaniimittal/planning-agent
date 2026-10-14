"""Which files are test code, decided from the path alone — one definition
shared by ingestion (stamps `is_test` on every Class/Function/Chunk) and the
TDD code search (never shows test code to the planner). Test functions match
an issue's wording very closely ("throws when reportNumber is invalid") and
would otherwise crowd real code out of what the planner sees.
"""

from __future__ import annotations

import re

_PATTERN = (
    # a test/fixture folder anywhere in the path
    r"(^|/)(src/test|test|tests|__tests__|acceptance-tests|e2e|cypress|__mocks__|mocks|fixtures)/"
    # FooTest.kt, FooTests.java, FooSpec.scala
    r"|[A-Za-z0-9_](Test|Tests|Spec)\.(kt|java|scala)$"
    # foo.spec.ts, foo.test.tsx, foo.cy.js
    r"|\.(spec|test|cy)\.(ts|tsx|js|jsx)$"
)

_TEST_PATH = re.compile(_PATTERN)

# Cypher's `=~` must match the WHOLE string, so the same pattern is wrapped —
# used for nodes ingested before `is_test` existed.
CYPHER_TEST_PATH_RE = f".*({_PATTERN}).*"


def is_test_path(file_path: str | None) -> bool:
    return bool(file_path) and _TEST_PATH.search(file_path) is not None
