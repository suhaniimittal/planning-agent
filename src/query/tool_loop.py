"""The planner's lookup loop: before it answers, the model may ask for code
it was not shown.

The first call is the same prompt the single-call planner sends — the ticket
and the code context chose — plus a list of read-only lookups. The model
either answers with the TDD, or asks for lookups; their results are added to
the same conversation and it is called again, until it answers or a limit is
reached, when it is told to answer with what it has. Nothing earlier in the
conversation is rewritten: each round only adds the model's requests and
their results to the end.

What it fixes is the one thing a chosen context cannot: a file that was
needed and not chosen. On the Archive-button ticket the grid holding the
`md={4}` white space was left out, and the TDD could only say "inspect
InfiniteScrollGrid" — one lookup would have shown it.

Every lookup is read-only, limited to the services of the run, and printed
to the log. Only the OpenAI path runs the loop: the gateway's
chat_with_tools takes no output limit.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field

from src.ingestion.embedder import embed

from . import graph_reader, images as ticket_images, planner, prompt
from .context import CONTEXT_BUDGET_CHARS, MAX_FILE_SHARE, Context, ShownFile

# The limits, all of them. Safety caps on time, cost and size — the same
# for every ticket, not tuned to one. A ticket the starting code covers
# ends in round 1 and costs what the single call did.
#
# Calls to the model, the last of which must answer. Each is 30-90 s with
# reasoning on; a developer checking a bug opens two to four more files.
MAX_ROUNDS = 6
# Lookups across all rounds; several may be asked for in one round.
MAX_LOOKUPS = 15
# One result: the same cap as one file in the starting code. A longer file
# comes back as its symbols, and can then be asked for by line range.
MAX_RESULT_CHARS = int(CONTEXT_BUDGET_CHARS * MAX_FILE_SHARE)
# All results together: half again the starting code.
MAX_LOOKUP_CHARS = CONTEXT_BUDGET_CHARS // 2
# Wall clock, after which the next call must answer. Checked between
# rounds, so a run can go one round (~1.5 min) past it, plus any send-backs —
# all of which must fit inside the agent's TDD step limit (agent.py
# _TDD_TIMEOUT, 15 min) with the context and the PDF, or the step is killed.
MAX_SECONDS = 360
# Times an answer that leaves ticket points unanswered is sent back
# (ticket_points.py). Each adds a round on top of MAX_ROUNDS: an answer
# forced out on the last round is the one most likely to have skipped
# something, and must still be checkable.
MAX_SEND_BACKS = 2
# What a code search returns.
_SEARCH_RESULTS = 10
_SEARCH_PER_SERVICE = 8
# What an exact-text search returns, and the shortest text it takes: two
# characters match half of every file.
_FIND_TEXT_FILES = 20
_MIN_FIND_TEXT = 3

_FINAL_NOTE = "No more lookups are available. Respond now with the JSON object."


def enabled() -> bool:
    return os.environ.get("PLANNER_LOOP", "").strip().lower() in ("1", "on", "true", "yes")


def _string(description: str) -> dict:
    return {"type": "string", "description": description}


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function", "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


TOOLS = [
    _tool(
        "read_file",
        "Read one file of a service, whole, or lines start_line-end_line of it. Use the "
        "path as a file header shows it; an import path such as \"@client/x\" is answered "
        "with the files it may mean.",
        {"service": _string("service name"), "path": _string("file path within the service"),
         "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
        ["service", "path"],
    ),
    _tool(
        "find_importers",
        "List every file that imports one file: who a change to it reaches.",
        {"service": _string("service name"), "path": _string("file path within the service")},
        ["service", "path"],
    ),
    _tool(
        "find_usages",
        "List where a class, function or constant is used: the symbols that reference it "
        "and the files that import it by name.",
        {"service": _string("the service that defines it"), "symbol": _string("its name")},
        ["service", "symbol"],
    ),
    _tool(
        "find_text",
        "Find exact text in the code — an identifier, a prop, a string, a CSS rule — and the "
        "lines it is on. Case-sensitive.",
        {"text": _string("the exact text"),
         "service": _string("limit the search to one service (optional)")},
        ["text"],
    ),
    _tool(
        "search_code",
        "Search the code by meaning, for something you can only describe in words.",
        {"query": _string("what you are looking for, in words"),
         "service": _string("limit the search to one service (optional)")},
        ["query"],
    ),
    _tool(
        "get_table",
        "A database table's columns, foreign keys, migrations and the code that uses it.",
        {"name": _string("table name, with or without its schema")},
        ["name"],
    ),
    _tool(
        "get_message",
        "A Kafka message: its schema file, topic, and the classes that publish and handle it.",
        {"name": _string("message name, full or short")},
        ["name"],
    ),
]

_TOOL_NAMES = {t["name"] for t in TOOLS}


@dataclass
class Lookups:
    """Runs the model's lookups against the graph, within the limits."""

    services: list[str]
    context: Context
    reader: object = graph_reader
    count: int = 0
    chars: int = 0
    # Files read, for verification: a change to one of them is checked
    # against it like a change to a file of the starting code.
    files: list[ShownFile] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    @property
    def spent(self) -> bool:
        return self.count >= MAX_LOOKUPS or self.chars >= MAX_LOOKUP_CHARS

    async def run(self, name: str, args: dict) -> str:
        signature = name + json.dumps(args, sort_keys=True)
        if signature in self._seen:
            result = "Same lookup as before; its result is above."
        elif self.spent:
            result = "The lookup limit is reached. Answer with what you have."
        else:
            self._seen.add(signature)
            self.count += 1
            handler = getattr(self, f"_{name}") if name in _TOOL_NAMES else None
            try:
                result = await handler(**args) if handler else f"No lookup named {name}."
            except TypeError as error:
                result = f"Bad arguments for {name}: {error}"
            if len(result) > MAX_RESULT_CHARS:
                result = result[:MAX_RESULT_CHARS] + "\n[… cut at the size limit for one lookup]"
            self.chars += len(result)
        shown = ", ".join(f"{k}={v}" for k, v in args.items())
        self.log.append(f"{name}({shown}) -> {result.splitlines()[0][:100] if result else ''}"
                        f" [{len(result):,} chars]")
        return result

    def _outside(self, service: str) -> str | None:
        if service in self.services:
            return None
        return f"{service} is not a service of this run. Services: {', '.join(self.services)}."

    async def _read_file(self, service: str, path: str, start_line: int | None = None,
                         end_line: int | None = None) -> str:
        if outside := self._outside(service):
            return outside
        path = path.lstrip("/")
        key = (service, path)
        loaded = (await asyncio.to_thread(self.reader.load_file_sources, [key])).get(key)
        if loaded is None:
            # The last folder and the name: "RowCell/view", not "view" — and
            # "components/InfiniteScrollGrid" for an import alias.
            fragment = "/".join(path.split("/")[-2:]).split(".", 1)[0]
            near = await asyncio.to_thread(self.reader.find_files, service, fragment)
            if not near:
                return f"No file {path} in {service}, and none whose path contains '{fragment}'."
            return (f"No file {path} in {service}. Files whose path contains '{fragment}':\n"
                    + "\n".join(near))
        ranged = start_line is not None or end_line is not None
        whole_shown = any(
            f.key == key and f.status == "WHOLE" for f in self.context.files + self.files
        )
        if whole_shown and not ranged:
            return f"{path} is already shown whole above."
        symbols = (await asyncio.to_thread(self.reader.load_file_symbols, [key])).get(key, [])
        text = loaded["source"] or ""
        lines = text.split("\n")
        shown = ShownFile(service=service, path=path, tier="lookup", reason="read on request",
                          line_count=loaded.get("line_count") or len(lines), symbols=symbols)
        if ranged:
            start = max(1, start_line or 1)
            end = min(len(lines), end_line or len(lines))
            shown.text = "\n".join(lines[start - 1 : end])
            if (start, end) != (1, len(lines)):
                shown.status, shown.shown_lines = "PARTIAL", [(start, end)]
        else:
            shown.text = text
        if len(shown.text) > MAX_RESULT_CHARS:
            shown.text, shown.status = "", "OUTLINE"
            shown.outline_note = (f"{len(text):,} characters is over the limit for one lookup; "
                                  "ask for a line range")
        else:
            self.files.append(shown)
        return prompt._file_block(shown)

    async def _find_importers(self, service: str, path: str) -> str:
        if outside := self._outside(service):
            return outside
        key = (service, path.lstrip("/"))
        found = (await asyncio.to_thread(self.reader.importers_of, [key], 30)).get(key)
        if not found:
            return f"No file imports {key[1]}."
        more = found["count"] - len(found["files"])
        return (f"{found['count']} files import {key[1]}:\n" + "\n".join(found["files"])
                + (f"\n… {more} more" if more > 0 else ""))

    async def _find_usages(self, service: str, symbol: str) -> str:
        if outside := self._outside(service):
            return outside
        found = await asyncio.to_thread(self.reader.usages_of, service, symbol)
        if not found:
            return f"Nothing references or imports {symbol} in the graph."
        return f"{symbol} is used in:\n" + "\n".join(
            f"{u['service']} · {u['file_path']}"
            + (" (imports it)" if u["by"] == "import" else f" (by {u['by']})")
            for u in found
        )

    async def _find_text(self, text: str, service: str | None = None) -> str:
        if service and (outside := self._outside(service)):
            return outside
        if len(text.strip()) < _MIN_FIND_TEXT:
            return f"Give at least {_MIN_FIND_TEXT} characters to find."
        services = [service] if service else self.services
        found = await asyncio.to_thread(self.reader.find_text, services, text, _FIND_TEXT_FILES)
        if not found:
            return f"{text!r} is not in the code of {', '.join(services)}."
        out = []
        for f in found:
            out.append(f"{f['service']} · {f['file_path']}")
            out += [f"    {h['line']}: {h['text'].strip()[:200]}" for h in f["hits"]]
        more = (" (the first files; narrow it by service or with longer text)"
                if len(found) == _FIND_TEXT_FILES else "")
        return f"{text!r} found in {len(found)} files{more}:\n" + "\n".join(out)

    async def _search_code(self, query: str, service: str | None = None) -> str:
        if service and (outside := self._outside(service)):
            return outside
        vector = await embed(query)
        services = [service] if service else self.services
        found = await asyncio.gather(*[
            asyncio.to_thread(self.reader.search_service_chunks, s, [vector], [],
                              _SEARCH_PER_SERVICE)
            for s in services
        ])
        rows = sorted(
            ((row["score"], s, row) for s, hits in zip(services, found, strict=True)
             for row in hits),
            key=lambda r: -r[0],
        )[:_SEARCH_RESULTS]
        if not rows:
            return "Nothing found."
        out = []
        for score, s, row in rows:
            path, _, symbol = row["function_qualified_name"].partition("::")
            snippet = " ".join((row.get("text") or "").split())[:200]
            out.append(f"{score:.2f} {s} · {path}" + (f" · {symbol}" if symbol else "")
                       + f"\n    {snippet}")
        return "\n".join(out)

    async def _get_table(self, name: str) -> str:
        found = await asyncio.to_thread(self.reader.tables_named, name, self.services)
        if not found:
            return f"No table named {name} in the services of this run."
        return "\n\n".join(
            f"{t['service']} · {t['name']}\n"
            f"Columns: {', '.join(t.get('columns') or []) or 'not known'}\n"
            f"Foreign keys to: {', '.join(t.get('foreign_keys') or []) or 'none'}\n"
            f"Defined in: {', '.join(t.get('defined_in') or [])}\n"
            f"Used by: {', '.join(u for u in t.get('used_by') or [] if u)}"
            for t in found
        )

    async def _get_message(self, name: str) -> str:
        found = await asyncio.to_thread(self.reader.messages_named, name)
        if not found:
            return f"No Kafka message named {name}."
        out = []
        for m in found:
            users = [u for u in m["users"] if u.get("file_path")]
            out.append(
                f"{m['name']}\nSchema: {m['schema_service']} · {m['schema_file']}\n"
                f"Topics: {', '.join(m['topics']) or 'not known'}\n"
                + "\n".join(f"{u['how']}: {u['service']} · {u['file_path']} ({u['class']})"
                            for u in users)
            )
        return "\n\n".join(out)


@dataclass
class LoopResult:
    raw: str
    rounds: int
    lookups: Lookups
    send_backs: int = 0
    rechecked: bool = False


def _apply_fixes(answer: str, reply: str) -> str | None:
    """The re-check's reply applied to the answer it reviewed: the answer
    itself for {"unchanged": true}, the answer with each fix applied for
    {"fixes": [...]}, the reply as it is when it is a whole new design, and
    None when the fixes cannot be applied. A fix is {"path", "value"} or
    {"path", "op": "remove"}, `path` a JSON Pointer into the answer ("-" as
    the last segment adds to a list).

    Fixes rather than the whole design again: rewriting every field to
    change two of them was the slowest call of a run."""
    try:
        data = json.loads(reply)
        if data == {"unchanged": True}:
            return answer
        if not (isinstance(data, dict) and set(data) == {"fixes"}):
            return reply
        doc = json.loads(answer)
        for fix in data["fixes"]:
            *parents, last = [
                part.replace("~1", "/").replace("~0", "~")
                for part in fix["path"].split("/")[1:]
            ]
            target = doc
            for part in parents:
                target = target[int(part)] if isinstance(target, list) else target[part]
            if isinstance(target, list):
                if fix.get("op") == "remove":
                    del target[int(last)]
                elif last == "-":
                    target.append(fix["value"])
                else:
                    target[int(last)] = fix["value"]
            elif fix.get("op") == "remove":
                del target[last]
            else:
                target[last] = fix["value"]
        return json.dumps(doc)
    except (json.JSONDecodeError, KeyError, IndexError, ValueError, TypeError):
        return None


async def _ask(client, rounds: int, why: str, **request):
    """One call to the model, with a line before and after it: a call with
    reasoning on takes up to minutes, and a log that says nothing for that
    long looks stuck."""
    print(f"  [loop] round {rounds}: waiting for the model ({why})…")
    started = time.monotonic()
    response = await client.responses.create(**request)
    asked = sum(1 for item in response.output if item.type == "function_call")
    print(f"  [loop] round {rounds}: model replied in {time.monotonic() - started:.0f}s — "
          + (f"{asked} lookups" if asked else f"an answer ({len(response.output_text):,} chars)"))
    return response


def _first_input(user_prompt: str, images) -> str | list[dict]:
    """Round 1's input: the user prompt, with any screenshots from the
    ticket (images.TicketImage), in upload order, at images.SLOT."""
    if not images:
        return user_prompt
    head, tail = ticket_images.around_slot(user_prompt)
    content = [{"type": "input_text", "text": head}] + [
        {"type": "input_image", "image_url": image.data_url(), "detail": "high"}
        for image in images
    ]
    if tail:
        content.append({"type": "input_text", "text": tail})
    return [{"role": "user", "content": content}]


async def run(
    system_prompt: str, user_prompt: str, lookups: Lookups, client=None, check=None,
    recheck: str | None = None, images=None,
) -> LoopResult:
    """Call the model, run the lookups it asks for, and call it again with
    their results, until it answers. `check(answer)` returns what to tell
    the model about an answer that is not finished yet, or None to accept
    it; it is asked at most MAX_SEND_BACKS times. `recheck` is sent once,
    when an answer is otherwise accepted and time is left, for the model to
    review its own design; its reply is fixes to that design (see
    _apply_fixes). Each send-back and the re-check add a round on top of
    MAX_ROUNDS. `images` (screenshots from the ticket) go with round 1;
    the later rounds continue that conversation, so the model keeps them.
    Returns the answer, raw."""
    if client is None:
        from src.ingestion import local_client

        # The loop runs only off the gateway: a laptop's installed client.
        client = local_client.new_client()
    params = {
        "model": planner._OPENAI_MODEL,
        "instructions": system_prompt,
        "tools": TOOLS,
        "reasoning": {"effort": planner._OPENAI_REASONING_EFFORT},
        "max_output_tokens": planner._OPENAI_REASONING_MAX_TOKENS,
        "text": {"format": {"type": "json_object"}},
    }
    started = time.monotonic()
    # Each round continues the stored conversation (previous_response_id),
    # so the model's reasoning from earlier rounds carries over and only
    # the new results are sent.
    why = "reading the ticket and the code" + (f" and {len(images)} images" if images else "")
    response = await _ask(client, 1, why, input=_first_input(user_prompt, images), **params)
    rounds = 1
    send_backs = 0
    rechecked = False
    reviewed: str | None = None  # the answer the re-check is reviewing
    answer: str | None = None
    while True:
        calls = [item for item in response.output if item.type == "function_call"]
        inputs: list[dict] = []
        if not calls:
            text = response.output_text
            if reviewed is not None:
                fixed = _apply_fixes(reviewed, text)
                if fixed is None:
                    print(f"  [loop] round {rounds}: the re-check's fixes could not be applied; "
                          "keeping the design it reviewed")
                    fixed = reviewed
                elif fixed != reviewed:
                    print(f"  [loop] round {rounds}: re-check fixes applied")
                else:
                    print(f"  [loop] round {rounds}: re-check found nothing to fix")
                text, reviewed = fixed, None
            answer = text
            note = None
            why = "answer sent back"
            if check is not None and send_backs < MAX_SEND_BACKS:
                note = check(text)
                if note is not None:
                    send_backs += 1
                    print(f"  [loop] round {rounds}: answer sent back — {note.splitlines()[0]}")
            if (note is None and recheck and not rechecked
                    and time.monotonic() - started < MAX_SECONDS):
                note, rechecked, reviewed, why = recheck, True, text, "re-check"
                print(f"  [loop] round {rounds}: answer accepted; re-check sent")
            if note is None:
                return LoopResult(raw=answer, rounds=rounds, lookups=lookups,
                                  send_backs=send_backs, rechecked=rechecked)
            inputs.append({"role": "user", "content": note})
        else:
            if rounds >= MAX_ROUNDS + send_backs + rechecked:
                raise RuntimeError("the model asked for lookups after being told to answer")
            start = len(lookups.log)
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                inputs.append({"type": "function_call_output", "call_id": call.call_id,
                               "output": await lookups.run(call.name, args)})
            for line in lookups.log[start:]:
                print(f"  [loop] round {rounds}: {line}")
            why = f"{len(calls)} lookup results"
        last = (rounds + 1 >= MAX_ROUNDS + send_backs + rechecked or lookups.spent
                or time.monotonic() - started >= MAX_SECONDS)
        if last:
            inputs.append({"role": "user", "content": _FINAL_NOTE})
            why += ", must answer now"
        rounds += 1
        response = await _ask(
            client, rounds, why, previous_response_id=response.id, input=inputs,
            tool_choice="none" if last else "auto", **params,
        )
