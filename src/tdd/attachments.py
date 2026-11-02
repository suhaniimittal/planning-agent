"""The card's attachments, each kept with its own description (2026-11-02).

A card file is a description, text already extracted from it (`content`), an
uploaded file (`key`, a storage key), or any mix. The form uploads files on
their own and passes only their keys; `pair()` joins each one to the card
file that describes it. An uploaded file is read from storage and
recognised by its first bytes, not its name:

- an image (PNG, JPEG, WebP, GIF) is shown to the model next to its name
  and description on the OpenAI path; the AI Gateway's chat takes text, so
  on the platform `describe()` has the gateway's vision call put the image
  into words first, next to the description someone wrote;
- a PDF gives its text (pypdf), up to a page and character limit; a scanned
  PDF has none, and its description stands in;
- a plain-text file gives its text.

A file that cannot be read is noted and its description still used: an
attachment never stops a run. Attachments are the ticket's, not source code;
they are still held in memory only, for this run.
"""

from __future__ import annotations

import io
import re
from collections.abc import Callable
from dataclasses import dataclass

from src.query import images as ticket_images
from src.query import planner

from .inputs import CardFile

MAX_TEXT_CHARS = 20_000  # per attachment
MAX_PDF_PAGES = 30


@dataclass
class Attachment:
    name: str
    description: str
    text: str
    image: ticket_images.TicketImage | None = None
    note: str = ""  # why the uploaded file could not be used, when it could not


def pair(files: list[CardFile], uploads: list[dict]) -> tuple[list[CardFile], list[str]]:
    """(the card's files with the uploaded ones joined in, a note per guess).

    ``uploads`` are {key, name?, description?}. One that brings its own
    description (Orchestration) is a file as it is. Otherwise it joins the
    card file of the same name that is waiting for one — a description with
    no text and no key yet. A name that matches none takes the first such
    file left, in order, and a note says to check they belong together; with
    none left, the upload is kept without a description, and a note says so."""
    out = [f.model_copy() for f in files]
    notes: list[str] = []
    bare: list[tuple[str, str]] = []
    for upload in uploads:
        key = str(upload.get("key") or "").strip()
        if not key:
            continue
        name = str(upload.get("name") or "").strip() or key.rsplit("/", 1)[-1]
        description = str(upload.get("description") or "").strip()
        if description:
            out.append(CardFile(name=name, key=key, description=description))
        else:
            bare.append((key, name))
    waiting = [f for f in out if not f.key and not f.content]
    unmatched = []
    for key, name in bare:
        match = _same_name(name, waiting)
        if match is None:
            unmatched.append((key, name))
            continue
        match.key = key
        waiting.remove(match)
    for key, name in unmatched:
        if waiting:
            match = waiting.pop(0)
            match.key = key
            notes.append(
                f"the uploaded file {name} was paired by order with the description of "
                f"{match.name}; check they belong together"
            )
        else:
            out.append(CardFile(name=name, key=key))
            notes.append(
                f"the uploaded file {name} has no description; add one under card.files"
            )
    return out, notes


def _same_name(name: str, files: list[CardFile]) -> CardFile | None:
    """The one waiting file named like ``name``: the same name, else one name
    inside the other (a stored key may add a prefix). None when no or two match."""
    wanted = _stem(name)
    exact = [f for f in files if _stem(f.name) == wanted]
    if len(exact) == 1:
        return exact[0]
    loose = [
        f for f in files
        if _stem(f.name) and (_stem(f.name) in wanted or wanted in _stem(f.name))
    ]
    return loose[0] if len(loose) == 1 else None


def _stem(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().rsplit(".", 1)[0])


def load(files: list[CardFile], read: Callable[[str], bytes] | None) -> list[Attachment]:
    """Every card file, in order. ``read`` gets a storage key's bytes; None
    when there is no storage to read from."""
    out: list[Attachment] = []
    images = 0
    for f in files:
        item = Attachment(name=f.name, description=f.description, text=f.content[:MAX_TEXT_CHARS])
        if f.key:
            if read is None:
                item.note = "the uploaded file could not be read here"
            else:
                try:
                    data = read(f.key)
                except Exception as error:  # noqa: BLE001 - one unreadable upload is skipped
                    item.note = f"the uploaded file could not be read ({type(error).__name__})"
                    data = b""
                if data:
                    images += _use(item, data, images)
        out.append(item)
        if item.note:
            print(f"  [TDD] attachment {item.name}: {item.note}")
    if out:
        print(f"  [TDD] attachments: {len(out)} ({images} image(s))")
    return out


_DESCRIBE_SYSTEM = (
    "You describe a screenshot attached to a Jira ticket, for a developer who will fix it. "
    "Say which screen it is, what controls and content are where, and anything marked or "
    "visibly wrong (overlaps, gaps, shifted rows, errors). Plain sentences, under 150 words. "
    "Describe only what is visible; no guesses about code."
)


async def describe(items: list[Attachment]) -> None:
    """On the gateway, each image put into words by its vision call, added to the
    attachment's text. A description that fails is noted; the run goes on."""
    if not planner._use_gateway():
        return  # the OpenAI path sends the images themselves
    for item in items:
        if item.image is None:
            continue
        prompt = f"The ticket's attachment {item.name}."
        if item.description:
            prompt += f" Its author describes it as: {item.description}"
        try:
            seen = (await planner.describe_image(item.image, prompt, _DESCRIBE_SYSTEM)).strip()
        except Exception as error:  # noqa: BLE001 - the written description still stands
            item.note = f"the image could not be described ({type(error).__name__})"
            print(f"  [TDD] attachment {item.name}: {item.note}")
            continue
        if seen:
            item.text = (f"{item.text}\n" if item.text else "") + f"The image shows: {seen}"
            print(f"  [TDD] attachment {item.name}: described by the model "
                  f"({len(seen):,} chars)")


def _use(item: Attachment, data: bytes, images: int) -> int:
    """Fills ``item`` from the uploaded bytes; 1 when it became an image."""
    kind = ticket_images.media_type(data)
    if kind:
        if images >= ticket_images.MAX_IMAGES:
            item.note = f"only the first {ticket_images.MAX_IMAGES} images are shown"
        elif len(data) > ticket_images.MAX_IMAGE_BYTES:
            item.note = f"the image is over {ticket_images.MAX_IMAGE_BYTES // 1_000_000} MB"
        else:
            item.image = ticket_images.TicketImage(name=item.name, media_type=kind, data=data)
            return 1
        return 0
    if data.startswith(b"%PDF"):
        text, note = _pdf_text(data)
        item.text = (item.text + "\n" + text).strip()[:MAX_TEXT_CHARS]
        item.note = note
        return 0
    if b"\x00" not in data[:8000]:
        item.text = (item.text + "\n" + data.decode("utf-8", errors="replace")).strip()
        item.text = item.text[:MAX_TEXT_CHARS]
        return 0
    item.note = "not an image, a PDF or text; only its description is used"
    return 0


def _pdf_text(data: bytes) -> tuple[str, str]:
    """(text, note) of a PDF: the note says what was left out, or why nothing came out."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return "", "PDF text not extracted: pypdf is not installed"
    try:
        reader = PdfReader(io.BytesIO(data))
        pages = reader.pages[:MAX_PDF_PAGES]
        text = "\n".join((page.extract_text() or "").strip() for page in pages).strip()
    except Exception as error:  # noqa: BLE001 - a broken PDF is skipped, never fatal
        return "", f"the PDF could not be read ({type(error).__name__})"
    notes = []
    if len(reader.pages) > MAX_PDF_PAGES:
        notes.append(f"only the first {MAX_PDF_PAGES} of {len(reader.pages)} pages were read")
    if not text:
        notes.append("the PDF has no text (scanned?); only its description is used")
    elif len(text) > MAX_TEXT_CHARS:
        notes.append(f"its text was cut at {MAX_TEXT_CHARS:,} characters")
    return text, "; ".join(notes)
