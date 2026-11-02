"""The planner's input (2026-11-02), checked before any GitHub call.

    card         key, type, title, description,
                 files[]  {name, description?, content?, key?}
    services[]   service, repo, branch, rank, reason, prs[] {card, pr, branch, url}, path?
    questions[]  question, answers[]      (an empty list = still open)
    issue_key    optional; the card's key when left out (the memory's key)

A card file is an attachment: a description someone wrote, text already
extracted from it (`content`), or an uploaded file's storage key — at least
one of the three. The form's attachment slots arrive as entries with a key
and a description side by side.

`rank` 1 is the most relevant service. `branch` empty means the repo's
default branch. `path` is the service's folder inside a shared repo. A pull
request's `url` is never trusted: links are rebuilt from the repo and number.

Only the SHAPE is checked here, and every problem is listed at once in plain
words. Whether a repo, branch or pull request exists is GitHub's to say
(prepare.py), and a service that fails there is skipped with a warning while
the others go on — so a repo that is not even owner/name is a warning too
(`repo_problem`), not a reason to reject the whole input.
"""

from __future__ import annotations

import json
import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# owner/name, as GitHub allows them: an owner of letters, digits and single
# hyphens; a name of letters, digits, '.', '_' and '-'.
_REPO = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}$")
_HOST_PREFIX = re.compile(r"^https?://[^/]+/")


class InputError(ValueError):
    """The input is not usable; ``problems`` lists every reason, one per line."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            "The planning input is not valid:\n" + "\n".join(f"- {p}" for p in problems)
        )


class _Model(BaseModel):
    # Unknown fields are ignored: Orchestration may send more than this needs.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class CardFile(_Model):
    name: str = Field(min_length=1)
    description: str = ""
    content: str = ""
    key: str = ""

    @model_validator(mode="after")
    def _has_something(self) -> CardFile:
        if not (self.description or self.content or self.key):
            raise ValueError("needs a description, its text (content) or an uploaded file's key")
        return self


class Card(_Model):
    key: str = Field(min_length=1)
    type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = ""
    files: list[CardFile] = Field(default_factory=list)

    @field_validator("key")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @property
    def is_bug(self) -> bool:
        return self.type.lower() in ("bug", "defect", "incident")


class PullRef(_Model):
    card: str = ""
    pr: int | None = Field(default=None, gt=0)
    branch: str = ""
    url: str = ""  # kept as given, never used

    @field_validator("pr", mode="before")
    @classmethod
    def _number(cls, value: object) -> object:
        # "142" and "#142" are both pull request 142; "" is none.
        if isinstance(value, str):
            return value.strip().lstrip("#") or None
        return value


class ServiceInput(_Model):
    service: str = Field(min_length=1)
    repo: str = Field(min_length=1)
    branch: str = ""
    rank: int = Field(ge=1)
    reason: str = ""
    prs: list[PullRef] = Field(default_factory=list)
    path: str = ""

    @field_validator("repo")
    @classmethod
    def _slug(cls, value: str) -> str:
        # A web address or a clone URL names the same repo as owner/name.
        return _HOST_PREFIX.sub("", value).removesuffix(".git").strip("/")

    @field_validator("path")
    @classmethod
    def _folder(cls, value: str) -> str:
        return value.strip("/")


class Question(_Model):
    question: str = Field(min_length=1)
    answers: list[str] = Field(default_factory=list)

    @field_validator("answers", mode="before")
    @classmethod
    def _as_list(cls, value: object) -> object:
        if value is None:
            return []
        return [value] if isinstance(value, str) else value

    @field_validator("answers")
    @classmethod
    def _no_blanks(cls, value: list[str]) -> list[str]:
        return [answer for answer in value if answer]

    @property
    def answered(self) -> bool:
        return bool(self.answers)


class PlanInput(_Model):
    card: Card
    services: list[ServiceInput] = Field(min_length=1)
    questions: list[Question] = Field(default_factory=list)
    issue_key: str = ""

    @model_validator(mode="after")
    def _unique_services(self) -> PlanInput:
        names = [s.service for s in self.services]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(f"each service may be listed once; repeated: {', '.join(repeated)}")
        return self

    @property
    def memory_key(self) -> str:
        """The ticket the memory reads and saves: issue_key, else the card's key."""
        return (self.issue_key or self.card.key).upper()

    def ranked_services(self) -> list[ServiceInput]:
        """Best rank first; services of equal rank keep their input order."""
        return sorted(self.services, key=lambda s: s.rank)


def repo_problem(repo: str) -> str | None:
    """Why ``repo`` cannot be a GitHub repository, or None when it can."""
    if _REPO.match(repo):
        return None
    return f"'{repo}' is not a GitHub repository in the form owner/name"


def parse(payload: dict | str) -> PlanInput:
    """The validated input, or InputError listing every problem.

    ``payload`` is the JSON object, or its text (the form's textarea)."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as error:
            raise InputError(
                [f"not valid JSON: {error.msg} (line {error.lineno}, column {error.colno})"]
            ) from None
    if not isinstance(payload, dict):
        raise InputError(["the input must be a JSON object with card, services and questions"])
    try:
        return PlanInput.model_validate(payload)
    except ValidationError as error:
        # `from None`: the pydantic error repeats the input values, ticket
        # text included; the problems below say where and what, nothing more.
        raise InputError([_problem(e) for e in error.errors()]) from None


def _where(loc: tuple) -> str:
    """("services", 0, "rank") -> "services[0].rank"."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else str(part)
    return out or "input"


def _problem(error: dict) -> str:
    where = _where(error["loc"])
    if error["type"] == "missing":
        return f"{where} is missing"
    return f"{where}: {error['msg'].removeprefix('Value error, ')}"
