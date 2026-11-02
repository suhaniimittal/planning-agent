"""The TDD conversation in Memsy (2026-10-26): recalled before, saved after."""

import types

import pytest

from src.memory import conversation


class FakeMemsy:
    def __init__(self, results=(), fail=False):
        self.results, self.fail = list(results), fail
        self.searches, self.ingested, self.closed = [], [], 0

    async def search(self, query, *, actor_id=None, limit=10, **_):
        if self.fail:
            raise RuntimeError("memsy down")
        self.searches.append((query, actor_id, limit))
        return types.SimpleNamespace(
            results=[types.SimpleNamespace(content=c) for c in self.results]
        )

    async def ingest(self, events):
        if self.fail:
            raise RuntimeError("memsy down")
        self.ingested.extend(events)

    async def close(self):
        self.closed += 1


@pytest.fixture
def memsy(monkeypatch):
    def install(**kwargs):
        fake = FakeMemsy(**kwargs)
        monkeypatch.setattr(conversation, "_client", lambda: fake)
        return fake

    return install


@pytest.mark.asyncio
async def test_earlier_discussion_is_recalled_for_the_story(memsy):
    fake = memsy(results=["BigDecimal was rejected: it changes stored types."])
    found = await conversation.recall("bts-33", "why not BigDecimal?")
    assert found == ["BigDecimal was rejected: it changes stored types."]
    assert fake.searches == [("why not BigDecimal?", "tdd:BTS-33", 5)]
    assert fake.closed == 1


@pytest.mark.asyncio
async def test_a_question_and_its_answer_are_saved_as_one_turn(memsy):
    fake = memsy()
    saved = await conversation.remember_exchange(
        "BTS-33", "28180482", "why paise?", "To avoid rounding.", "question", "answer_only"
    )
    assert saved is True
    user, agent = fake.ingested
    assert (user.actor_id, user.session_id, user.kind) == (
        "tdd:BTS-33",
        "page:28180482",
        "user_message",
    )
    assert agent.kind == "assistant_message" and agent.content == "To avoid rounding."
    assert user.team_id == "BTS" and '"verdict": "answer_only"' in user.metadata


@pytest.mark.asyncio
async def test_a_question_on_the_story_uses_the_story_session(memsy):
    fake = memsy()
    await conversation.remember_exchange("BTS-33", "", "why paise?", "")
    assert [e.session_id for e in fake.ingested] == ["story:BTS-33"]  # no empty answer saved


@pytest.mark.asyncio
async def test_a_published_version_is_noted_with_its_decisions(memsy):
    fake = memsy()
    await conversation.remember_version(
        "BTS-33",
        "v1.2",
        "28180482",
        [{"id": "paise", "title": "Integer paise", "choice": "Math.round(amount*100)"}],
        ["limit-order"],
    )
    (note,) = fake.ingested
    assert note.kind == "app_event" and note.session_id == "versions:BTS-33"
    assert "BTS-33 TDD v1.2 was published" in note.content
    assert "Integer paise: Math.round(amount*100)" in note.content and "limit-order" in note.content


@pytest.mark.asyncio
async def test_memsy_down_never_blocks_an_answer(memsy):
    memsy(fail=True)
    assert await conversation.recall("BTS-33", "why?") == []
    assert await conversation.remember_exchange("BTS-33", "1", "why?", "because") is False


@pytest.mark.asyncio
async def test_without_a_key_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("MEMSY_API_KEY", raising=False)
    monkeypatch.setattr(conversation, "secret", lambda name, default="": "")
    assert conversation._client() is None
    assert await conversation.recall("BTS-33", "why?") == []
