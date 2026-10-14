"""Where the platform's AI Gateway is, and its key (2026-10-13).

On Aetherion every agent is given the gateway the same way: its address in
``AI_GATEWAY_URL`` (or the SDK's default) and its key in ``BIFROST_VIRTUAL_KEY``
or ``AI_GATEWAY_KEY`` — that is how the Jira agents reach it with nothing in their
.env. This code used to look only for ``AGENTS_GATEWAY_KEY``, so once published it
missed the platform's key and fell back to needing an OpenAI key. Any of the
three names now counts; with none (a laptop) the callers use OpenAI directly.
"""

from __future__ import annotations

import os

_KEY_NAMES = ("AGENTS_GATEWAY_KEY", "BIFROST_VIRTUAL_KEY", "AI_GATEWAY_KEY")


def gateway_key() -> str:
    """The gateway key under whichever name it was given, or ""."""
    return next((os.environ[n] for n in _KEY_NAMES if os.environ.get(n, "").strip()), "")


def gateway_url() -> str:
    """The gateway's address: AI_GATEWAY_URL, else the SDK's own default."""
    url = os.environ.get("AI_GATEWAY_URL", "").strip() or os.environ.get("BIFROST_URL", "").strip()
    if url:
        return url
    from agent_lib.gateway.config import gateway_url as sdk_gateway_url

    return sdk_gateway_url()


def use_gateway() -> bool:
    """True on the platform (a gateway key is there); False on a laptop."""
    return bool(gateway_key())
