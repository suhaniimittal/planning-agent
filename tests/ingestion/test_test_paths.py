import re

import pytest

from src.ingestion.test_paths import CYPHER_TEST_PATH_RE, is_test_path


@pytest.mark.parametrize(
    "path, expected",
    [
        ("src/test/kotlin/unit/FooTests.kt", True),
        ("src/main/kotlin/FooTest.kt", True),
        ("src/test/resources/schema.graphqls", True),
        ("src/apps/client/cypress/support/x.ts", True),
        ("acceptance-tests/cypress/client/pages/a.ts", True),
        ("src/components/Button.spec.tsx", True),
        ("src/utils/date.test.ts", True),
        ("src/main/kotlin/Foo.kt", False),
        ("src/main/kotlin/latest.kt", False),
        ("src/main/kotlin/ContestService.kt", False),
        ("src/main/resources/graphql/reports.graphqls", False),
        ("src/apps/client/modules/Reports/components/SharedForms/employmentVerificationForm.ts", False),
        (None, False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


def test_cypher_pattern_agrees_when_matched_against_the_whole_path():
    """Cypher's =~ is a whole-string match — the wrapped pattern must give
    the same answer as is_test_path for nodes ingested before is_test."""
    for path in ["src/test/kotlin/A.kt", "src/main/kotlin/A.kt", "src/a/b.spec.ts", "src/a/b.ts"]:
        assert (re.fullmatch(CYPHER_TEST_PATH_RE, path) is not None) is is_test_path(path)
