"""The OpenAI client for a run without the AI gateway (2026-10-26).

On the platform a gateway key is present and every model call goes through
the gateway, so nothing here runs. On a laptop there is no gateway key, and
the client is OpenAI's own, reading OPENAI_API_KEY from the environment.

A launcher may still install a client class of its own first (install());
otherwise one is built here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_factory: Callable[[], Any] | None = None


def install(factory: Callable[[], Any]) -> None:
    """Called by a launcher with the client class to use instead."""
    global _factory
    _factory = factory


def new_client() -> Any:
    """The installed client class's instance, else OpenAI's own client."""
    if _factory is not None:
        return _factory()
    from openai import AsyncOpenAI

    return AsyncOpenAI()  # reads OPENAI_API_KEY from the environment
