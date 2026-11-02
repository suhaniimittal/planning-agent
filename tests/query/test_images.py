import types

import pytest

from src.query import context as ctx
from src.query import images, planner
from src.query import prompt as prompts

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 20
WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "


def test_the_type_comes_from_the_bytes_not_the_name():
    assert images.media_type(PNG) == "image/png"
    assert images.media_type(JPEG) == "image/jpeg"
    assert images.media_type(WEBP) == "image/webp"
    assert images.media_type(b"%PDF-1.7") is None


def test_unreadable_unsupported_and_repeated_uploads_are_skipped_not_fatal():
    store = {"u/a.png": PNG, "u/b.pdf": b"%PDF-1.7", "u/c.jpg": JPEG}

    def read(key):
        return store[key]  # "u/missing.png" raises KeyError

    loaded = images.load(["u/a.png", "u/b.pdf", "u/missing.png", "u/a.png", "u/c.jpg"], read)

    assert [(i.name, i.media_type) for i in loaded] == [("a.png", "image/png"),
                                                         ("c.jpg", "image/jpeg")]
    assert loaded[0].data_url().startswith("data:image/png;base64,iVBORw0KGgo")


def test_at_most_five_images_are_used():
    loaded = images.load([f"k{i}.png" for i in range(8)], lambda key: PNG)

    assert len(loaded) == images.MAX_IMAGES


def test_the_screenshots_sit_right_after_the_issue_before_the_code():
    prompt = prompts.build_user_prompt(
        "Controls shift.", [{"name": "ui", "summary": "s"}], ctx.Context(),
        image_names=["before.png", "after.png"],
    )

    assert "## Screenshots (2, in this order)\n1. before.png\n2. after.png" in prompt
    assert "the before screen is how it should stay" in prompt
    head, tail = images.around_slot(prompt)
    assert head.startswith("## Issue\nControls shift.") and head.endswith(images.SLOT)
    assert "## Code" in tail and "## Code" not in head
    assert "## Screenshots" not in prompts.build_user_prompt(
        "i", [{"name": "ui", "summary": "s"}], ctx.Context())


def test_a_prompt_without_the_slot_is_all_head():
    assert images.around_slot("just text") == ("just text", "")


@pytest.mark.asyncio
async def test_the_single_call_sends_images_at_the_slot(monkeypatch):
    sent = {}

    class Completions:
        async def create(self, **kwargs):
            sent.update(kwargs)
            message = types.SimpleNamespace(content="{}")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    class FakeOpenAI:
        def __init__(self):
            self.chat = types.SimpleNamespace(completions=Completions())

    from src.ingestion import local_client

    monkeypatch.setattr(local_client, "_factory", FakeOpenAI)
    shot = images.TicketImage(name="a.png", media_type="image/png", data=PNG)

    await planner._plan_via_openai("system", f"issue\n{images.SLOT}\ncode", [shot])

    content = sent["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": f"issue\n{images.SLOT}"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[2] == {"type": "text", "text": "\ncode"}
