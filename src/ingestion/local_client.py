"""The laptop's OpenAI client, installed from outside the package (2026-10-26).

The platform refuses to publish an agent that builds an OpenAI client itself:
that would bypass the AI gateway. So the published code never does. On a
laptop, scripts/local_worker.py (through scripts/local_openai.py) installs the
OpenAI client class here before the agent starts; on the platform nothing is
installed and every model call goes through the gateway.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_factory: Callable[[], Any] | None = None


def install(factory: Callable[[], Any]) -> None:
    """Called by the laptop launcher with the OpenAI client class."""
    global _factory
    _factory = factory


def new_client() -> Any:
    """A client for a laptop run; an error saying what to do anywhere else."""
    if _factory is None:
        raise RuntimeError(
            "No AI gateway is configured (AI_GATEWAY_URL and AGENTS_GATEWAY_KEY), and no "
            "laptop OpenAI client is installed: start the agent through "
            "scripts/local_worker.py, which installs one."
        )
    return _factory()
