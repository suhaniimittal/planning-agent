"""Screenshots attached to a ticket, for the planner to see.

A ticket's two-line description ("controls shift when report selected")
left out what its screenshot showed at a glance: which row wrapped, and
that the empty space was beside the title. The images go to the model with
the first round of the lookup loop, next to the user prompt.

They arrive as storage keys — the form uploads each file to the platform's
storage and passes only its key, the same contract coding-agent uses for
its PDF — and are read here, checked, and turned into data URLs.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass

# More screenshots than this is a ticket that needs a description, not more
# images; each costs about 1-2k input tokens.
MAX_IMAGES = 5
# Well under the model's per-request limit; a screenshot is rarely 2 MB.
MAX_IMAGE_BYTES = 8_000_000
# Where the images go in the user prompt: right after the ticket, before the
# code. Sent after ~300k characters of code, they were read as a footnote to
# it; next to the ticket they are read the way a chat reads an attached
# screenshot.
SLOT = "(The screenshots follow.)"

_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


@dataclass
class TicketImage:
    name: str
    media_type: str
    data: bytes

    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{base64.b64encode(self.data).decode()}"


def around_slot(prompt: str) -> tuple[str, str]:
    """The prompt up to and including SLOT, and the rest after it. A prompt
    without SLOT is all head, so the images follow it whole."""
    head, slot, tail = prompt.partition(SLOT)
    return (head + slot, tail) if slot else (prompt, "")


def media_type(data: bytes) -> str | None:
    """The image type, from the file's own first bytes rather than its
    name, or None for anything the model does not take."""
    for signature, kind in _SIGNATURES:
        if data.startswith(signature):
            return kind
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def load(keys: list[str], read: Callable[[str], bytes]) -> list[TicketImage]:
    """The images behind `keys`, in order. A key that cannot be read, is
    not an image, or is too large is skipped with a line in the log — a
    screenshot never fails the run."""
    images: list[TicketImage] = []
    for key in dict.fromkeys(k for k in keys if k):
        if len(images) == MAX_IMAGES:
            print(f"  [TDD] images: only the first {MAX_IMAGES} are used")
            break
        name = key.rsplit("/", 1)[-1]
        try:
            data = read(key)
        except Exception as error:  # noqa: BLE001 - one unreadable upload is skipped
            print(f"  [TDD] images: skipped {name}: could not read it ({type(error).__name__})")
            continue
        kind = media_type(data)
        if kind is None:
            print(f"  [TDD] images: skipped {name}: not a PNG, JPEG, WebP or GIF")
            continue
        if len(data) > MAX_IMAGE_BYTES:
            print(f"  [TDD] images: skipped {name}: {len(data):,} bytes is over the limit")
            continue
        images.append(TicketImage(name=name, media_type=kind, data=data))
    if images:
        print(f"  [TDD] images: {len(images)} attached ({', '.join(i.name for i in images)})")
    return images
