"""The agreed input for a TDD (2026-10-29), as the text the planner reads.

    {"card": {"key", "type", "title", "description", "comments",
              "files": [{"name", "content"}]},
     "services": [{"service", "repo", "branch", "rank", "reason",
                   "prs": [{"card", "pr", "branch", "url"}]}],
     "questions": [{"question", "answers": ["..."]}]}

Services are listed by name, highest rank first, so the planner always
designs them (a service the text names is always selected). A question with
answers is a requirement; one without is an open question the TDD lists and
does not guess. Pure: no I/O.
"""

from __future__ import annotations

from typing import Any

MAX_FILE_CHARS = 8_000
MAX_FILES_CHARS = 40_000


def _files(files: list[dict[str, Any]]) -> list[str]:
    lines, left = [], MAX_FILES_CHARS
    for f in files:
        name = str(f.get("name") or "file")
        content = str(f.get("content") or "").strip()
        if not content:
            lines.append(f"=== {name} === (not read)")
            continue
        content = content[: min(MAX_FILE_CHARS, max(left, 0))]
        left -= len(content)
        lines += [f"=== {name} ===", content or "(not included: the files' space was used up)"]
    return lines


def _service(s: dict[str, Any]) -> list[str]:
    where = f"{s.get('repo')}@{s.get('branch') or 'main'}" if s.get("repo") else ""
    line = f"- {s.get('service')} (rank {s.get('rank')}/10)"
    line += f" in {where}" if where else ""
    line += f": {s['reason']}" if s.get("reason") else ""
    out = [line]
    for pr in s.get("prs") or []:
        state = f", {pr['state']}" if pr.get("state") else ""
        branch = f" on branch {pr['branch']}" if pr.get("branch") else ""
        out.append(
            f"  earlier: {pr.get('card')} PR #{pr.get('pr')}{branch}{state} {pr.get('url', '')}"
        )
    return out


def issue_text(payload: dict[str, Any]) -> str:
    card = payload.get("card") or {}
    title = str(card.get("title") or "").strip()
    head = f"{card.get('key', '')}"
    if card.get("type"):
        head += f" ({card['type']})"
    lines = [f"{head}: {title}".strip(": "), str(card.get("description") or "").strip()]
    comments = [str(c).strip() for c in card.get("comments") or [] if str(c).strip()]
    if comments:
        lines += ["", "Comments on the card:", *[f"- {c}" for c in comments]]
    if card.get("files"):
        lines += ["", "Attached files:", *_files(list(card["files"]))]

    services = sorted(
        [s for s in payload.get("services") or [] if s.get("service")],
        key=lambda s: -int(s.get("rank") or 0),
    )
    if services:
        lines += [
            "",
            "Services this card touches, ranked by the review (10 = the centre of the change, "
            "6-8 = must change, 3-5 = check whether it changes). Design them in this order; "
            "an earlier pull request shows the code that already does something similar — "
            "reuse it, and keep what it built working:",
        ]
        for s in services:
            lines += _service(s)

    answered, open_ = [], []
    for q in payload.get("questions") or []:
        question = str(q.get("question") or "").strip()
        answers = [str(a).strip() for a in q.get("answers") or [] if str(a).strip()]
        if not question:
            continue
        if answers:
            answered.append(f"- {question}\n  Answer: " + " / ".join(answers))
        else:
            open_.append(f"- {question}")
    if answered:
        lines += [
            "",
            "Questions the review asked, with the answers given (requirements):",
            *answered,
        ]
    if open_:
        lines += [
            "",
            "Open questions — not answered yet. List each in the design's open questions and "
            "do not decide it silently:",
            *open_,
        ]
    # What the caller adds after the card: a revision's instructions and the
    # current design, the project's earlier designs.
    notes = str(payload.get("notes") or "").strip()
    if notes:
        lines += ["", notes]
    return "\n".join(lines).strip()


def repos_text(payload: dict[str, Any]) -> str:
    """The ranked services' repositories as Planning's repos lines:
    "imps-service: org/bank@main"."""
    lines = []
    for s in payload.get("services") or []:
        if s.get("service") and s.get("repo"):
            branch = f"@{s['branch']}" if s.get("branch") else ""
            lines.append(f"{s['service']}: {s['repo']}{branch}")
    return "\n".join(dict.fromkeys(lines))
