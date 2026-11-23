"""Secrets, from wherever the platform provides them (2026-10-15).

The same reader as the Jira agents' shared ``settings.secret``: a mounted file
(``<NAME>_FILE``, or ``<NAME>`` in ``SECRETS_DIR`` / /run/secrets / /etc/secrets)
before the environment variable, so production secrets need not sit in .env —
which is packed into the published agent.
"""

from __future__ import annotations

import os

# --- secrets -----------------------------------------------------------------

# Where a platform mounts secrets as files: Docker and Kubernetes use /run/secrets;
# SECRETS_DIR names another. A secret read from a file never sits in .env, so it is
# never packed into the published agent.
_DEFAULT_SECRETS_DIRS = ("/run/secrets", "/etc/secrets")


def secret(name: str, default: str = "") -> str:
    """A token, password or key, from wherever the platform provides it:

    1. the file named by ``<NAME>_FILE`` (the Docker / Kubernetes convention);
    2. the file ``<NAME>`` in ``SECRETS_DIR`` (else /run/secrets, /etc/secrets);
    3. the environment variable ``<NAME>`` — a laptop's .env, or a platform that
       injects secrets as variables.

    Surrounding whitespace is stripped; ``default`` when none holds a value."""
    from pathlib import Path

    candidates = []
    pointer = os.environ.get(f"{name}_FILE", "").strip()
    if pointer:
        candidates.append(Path(pointer))
    dirs = [os.environ.get("SECRETS_DIR", "").strip(), *_DEFAULT_SECRETS_DIRS]
    candidates += [Path(d) / name for d in dirs if d]
    for path in candidates:
        try:
            value = path.read_text().strip()
        except OSError:
            continue
        if value:
            return value
    return os.environ.get(name, "").strip() or default
