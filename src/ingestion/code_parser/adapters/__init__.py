"""Registry of every LanguageAdapter this parser knows about.

Adding a new language: write one more `<language>_adapter.py` here (reusing
`shared.py`'s building blocks where the grammar allows), then register it in
ADAPTERS below and in `LANGUAGE_BY_EXTENSION` in the parent package's
`__init__.py`.
"""

from __future__ import annotations

from .java_adapter import JAVA_ADAPTER
from .kotlin_adapter import KOTLIN_ADAPTER
from .python_adapter import PYTHON_ADAPTER
from .typescript_adapter import JAVASCRIPT_ADAPTER, TSX_ADAPTER, TYPESCRIPT_ADAPTER

ADAPTERS = {
    "python": PYTHON_ADAPTER,
    "java": JAVA_ADAPTER,
    "typescript": TYPESCRIPT_ADAPTER,
    "tsx": TSX_ADAPTER,
    "javascript": JAVASCRIPT_ADAPTER,
    "kotlin": KOTLIN_ADAPTER,
}

__all__ = [
    "ADAPTERS",
    "PYTHON_ADAPTER",
    "JAVA_ADAPTER",
    "TYPESCRIPT_ADAPTER",
    "TSX_ADAPTER",
    "JAVASCRIPT_ADAPTER",
    "KOTLIN_ADAPTER",
]
