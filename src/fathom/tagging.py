"""Tagging a rollout's turns: a reader reads the attempt and says, in a few words for each part of
each turn, what the agent meant to do, what its call did, and what came back.

A tag names the kind of action and leaves the task's details out, so that the same action in
another task reads the same: tags are for finding turns across rollouts, which `rank` does by
their vectors, and for grouping them (`fathom.clustering`). The vectors of a turn's own text mostly say which task it is, and found 0 to 3 of
22 known turns where the tags' vectors found 18 (measured 16 Nov 2026). The reader is
one as in `fathom.reading`; with none named it is Claude Haiku through the Claude API, at high
effort: at low and medium it left 3 to 4% of the calls of long rollouts untagged, and put some
tags on the step after their own (measured on ten rollouts, 16 Nov 2026). At high effort it still
left every result untagged in 1 of 80 rollouts, so a reader that gives a part no tag at all is
asked once more.

A reader that can count a job's tokens and names a limit (`count`, `limit`, as `ClaudeAPI` does) has
a long rollout split into several calls, each with the task and a run of the turns, and each planned
to fill half the limit: a reader leaves more turns untagged in a long call than in a short one.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fathom import embedding
from fathom.reading import ClaudeAPI, Reader, clip
from fathom.reading import resolve as _reader

if TYPE_CHECKING:
    from fathom.rollout import Rollout

PARTS = embedding.PARTS
# How much of a reader's limit a call is planned to fill. Half: calls of 64,000 to 90,000 tokens left
# 3 to 5% of the calls and results of three long rollouts untagged, calls under 50,000 at most 2%
# (Claude Haiku 5.5 at high effort, limit 100,000, 17 Nov 2026).
ROOM = 0.5
PER_TAG = 70  # tokens Claude Haiku 5.5 wrote for each tag at high effort, thinking included (80 rollouts, 16 Nov 2026)
NOTHING = {"none", "n/a", "-"}  # what a reader writes for a part it has nothing to say of: no tag

SCHEMA = {
    "type": "object",
    "properties": {"turns": {"type": "array", "items": {
        "type": "object",
        "properties": {"step": {"type": "integer"}, **{p: {"type": "string"} for p in PARTS}},
        "required": ["step"],
    }}},
    "required": ["turns"],
}

PROMPT = """You are tagging the turns of a coding agent's attempt at {task}, so that turns can later be found and counted across thousands of attempts at other tasks. This is a tagging task: read the transcript, answer.

## File
transcript.txt: what the agent was asked, in `<task>`, then its turns in order. A turn is one step of the agent, in `<turn step="id">`, with up to three parts: what it thought and said (`<thought>`), the tools it called (`<call>`), and what came back (`<result>`, long output cut in the middle). A part's text may itself hold headings, lists or code: it all belongs to that part, up to the part's closing tag.

## What to answer
For every turn, by its step id, a tag for each part the turn has:
- `thought`: what the agent means to do or decides in this turn.
- `call`: what the call of this same turn does, whatever the thought before it says.
- `result`: what came back from that call: what it showed, or how it failed.

A tag is a few words, at most eight, in lower case, starting with a verb. It names the kind of action and leaves this task's details out: no file names, table names, identifiers or values. The same action in another task, on other files and data, should get the same tag. For example: "reads the skill's instructions", "decides to ask the user which branch", "runs the tests", "lists the tables", "fails: module not found", "shows two tests failing", "asks the user to confirm before going on". These show the form only; say what this turn does. A turn gets a tag for every part it has, and no field for a part it lacks."""


def resolve(reader: str | Reader | None) -> Reader:
    """The reader to ask: Claude Haiku through the Claude API, Claude Code with the model of that
    alias, or the function given."""
    return ClaudeAPI(effort="high") if reader is None else _reader(reader)


def plan(rollout: Rollout, ask: Reader) -> list[dict]:
    """The calls tagging a rollout takes, in order: for each, the step ids of the turns it tags
    and the tokens it sends (None when the reader cannot count them). One call, unless the reader
    names a limit and the whole transcript passes the share of it a call is planned to fill
    (`ROOM`): then as few calls as keep each under that."""
    trajectory = rollout.trajectory
    ids, by = embedding.texts(trajectory)
    count, limit = getattr(ask, "count", None), getattr(ask, "limit", None)
    if count is None:
        return [{"steps": ids, "tokens": None}]
    whole = count(*job(rollout, ids, trajectory=trajectory)[:3])
    if limit is None or whole <= ROOM * limit or len(ids) < 2:
        return [{"steps": ids, "tokens": whole}]
    limit = ROOM * limit
    size = [sum(len(by[p][i]) for p in PARTS) for i in range(len(ids))]
    for pieces in range(min(len(ids), max(2, math.ceil(whole / (0.9 * limit)))), len(ids) + 1):
        cuts, run, share = [[]], 0, sum(size) / pieces
        for i, step in enumerate(ids):  # runs of turns of about equal length
            if cuts[-1] and run + size[i] / 2 > share * len(cuts) and len(cuts) < pieces:
                cuts.append([])
            cuts[-1].append(step)
            run += size[i]
        made = [{"steps": c, "tokens": count(*job(rollout, c, (j, len(cuts)), trajectory)[:3])} for j, c in enumerate(cuts)]
        if all(m["tokens"] <= limit for m in made) or pieces == len(ids):
            return made
    return made


def job(rollout: Rollout, steps: list[int], place: tuple[int, int] = (0, 1), trajectory: dict | None = None) -> tuple[str, dict, dict[str, str], list]:
    """What a reader is asked to tag the turns `steps` of a rollout: the prompt, the schema of the
    answer, the files and the folders (none). `place` says which of how many calls this is."""
    task = rollout.labels.get("task")
    transcript = render(trajectory if trajectory is not None else rollout.trajectory, steps, place)
    return PROMPT.format(task=f"the task {task}" if task else "a task"), SCHEMA, {"transcript.txt": transcript}, []


def tag(rollout: Rollout, reader: str | Reader | None = None, answered: list[dict] | None = None) -> dict:
    """The tags layer for a rollout, as `reader` answers: who tagged it, when and at what cost, the
    turns' step ids, and for each part the tag of each turn, None where the turn has no such part
    or the reader gave it none. `answered` is the calls already made some other way, as through a
    batch: for each, the `steps` it tagged, the `answer` and what the reader told `about` it."""
    ask = resolve(reader)
    trajectory = rollout.trajectory
    ids, by = embedding.texts(trajectory)
    row = {s: i for i, s in enumerate(ids)}
    parts: dict[str, list] = {p: [None] * len(ids) for p in PARTS}
    about: dict = {}
    calls = answered if answered is not None else [{"steps": c["steps"]} for c in plan(rollout, ask)]
    for j, call in enumerate(calls):
        mine = [row[s] for s in call["steps"] if s in row]
        for again in range(2):  # a reader now and then leaves a whole part out: it is asked once more, for what is missing
            if again == 0 and "answer" in call:
                answer, told = call["answer"], call.get("about") or {}
            else:
                answer, told = ask(*job(rollout, call["steps"], (j, len(calls)), trajectory))
            about = _sum(about, told)
            for t in answer.get("turns") or []:
                if isinstance(t, dict) and t.get("step") in row and row[t["step"]] in mine:
                    at = row[t["step"]]
                    for p in PARTS:
                        said = " ".join(t[p].split()) if isinstance(t.get(p), str) else ""
                        if by[p][at] and parts[p][at] is None and said and said.lower() not in NOTHING:
                            parts[p][at] = said
            if not any(any(by[p][i] for i in mine) and not any(parts[p][i] for i in mine) for p in PARTS):
                break
    return {"reader": getattr(ask, "name", getattr(ask, "__name__", "reader")),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **about, "steps": ids, "parts": parts}


def cost(rollouts: list[Rollout], ask: Reader, per_tag: float = PER_TAG, plans: list[list[dict]] | None = None) -> dict:
    """What tagging `rollouts` would take, without tagging any: the calls, the rollouts split over
    several, the tokens sent, as the reader counts them, the tags to write and the tokens that
    takes at `per_tag` each, and the cost, by call and through a batch. Fails if the reader cannot
    count tokens."""
    if getattr(ask, "count", None) is None:
        raise ValueError("this reader cannot count a job's tokens, so the cost of tagging cannot be told ahead: a ClaudeAPI reader can")
    plans = plans if plans is not None else [plan(r, ask) for r in rollouts]
    price = getattr(ask, "price", None)
    out = {"rollouts": len(rollouts), "calls": 0, "split": 0, "input_tokens": 0, "tags": 0, "per_tag": round(per_tag, 1),
           "output_tokens": 0, "cost_usd": 0.0, "batch_usd": 0.0}
    for r, calls in zip(rollouts, plans):
        ids, by = embedding.texts(r.trajectory)
        row = {s: i for i, s in enumerate(ids)}
        out["calls"] += len(calls)
        out["split"] += len(calls) > 1
        for call in calls:
            tags = sum(bool(by[p][row[s]]) for s in call["steps"] for p in PARTS)
            written = round(tags * per_tag)
            out["input_tokens"] += call["tokens"]
            out["tags"] += tags
            out["output_tokens"] += written
            if price is not None and out["cost_usd"] is not None:
                each, half = price(call["tokens"], written), price(call["tokens"], written, True)
                out["cost_usd"], out["batch_usd"] = (None, None) if each is None else (out["cost_usd"] + each, out["batch_usd"] + half)
    if price is None:
        out["cost_usd"] = out["batch_usd"] = None
    elif out["cost_usd"] is not None:
        out["cost_usd"], out["batch_usd"] = round(out["cost_usd"], 4), round(out["batch_usd"], 4)
    return out


def _sum(before: dict, now: dict) -> dict:
    """What two answers of a reader cost together: its numbers added, the rest as the later one said."""
    out = {**before, **now}
    for k in before.keys() & now.keys():
        a, b = before[k], now[k]
        if isinstance(a, dict) and isinstance(b, dict):
            out[k] = _sum(a, b)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            out[k] = round(a + b, 6)
    return out


def render(trajectory: dict, steps: list[int] | None = None, place: tuple[int, int] = (0, 1)) -> str:
    """What the reader is handed: what the agent was asked, then each turn's parts as
    `turn.text` has them (its own text whole, each tool result cut to its first and last characters). With
    `steps`, those turns only, and a word that the file is one part (`place`) of several."""
    asked = next((st.get("message") or "" for st in trajectory.get("steps") or [] if st.get("source") == "user"), "")
    out = [f"<task>\n{clip(asked, 3000, 1000)}\n</task>"]
    ids, by = embedding.texts(trajectory)
    keep = set(ids if steps is None else steps)
    if place[1] > 1:
        mine = [s for s in ids if s in keep]
        out.append(f"<note>\nThe attempt is too long for one file. This is part {place[0] + 1} of {place[1]}: steps {mine[0]} to {mine[-1]}. "
                   "Tag the turns in this file, and no others.\n</note>")
    for i, step in enumerate(ids):
        if step in keep:
            out.append(f'<turn step="{step}">\n' + "\n".join(f"<{p}>\n{by[p][i]}\n</{p}>" for p in PARTS if by[p][i]) + "\n</turn>")
    return "\n\n".join(out)


def rank(rollouts: list[Rollout], text: str, part: str, embedder: embedding.Embedder) -> list[tuple[float, int, int]]:
    """Every turn of `rollouts` with a tag for `part`, by how like `text` its tag reads, best first:
    the cosine of the tag's vector with the text's, the rollout's place, and the turn's step id.
    The tags' vectors are the rollouts' own, as `embed` made them; only `text` is embedded here."""
    import numpy as np

    query = np.asarray(embedder([text])[0], dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    out: list[tuple[float, int, int]] = []
    for mi, r in enumerate(rollouts):
        _, steps, rows = r._part(part)
        if steps:
            out += [(float(x), mi, step) for x, step in zip(rows.astype(np.float32) @ query, steps)]
    return sorted(out, key=lambda x: (-x[0], x[1], x[2]))
