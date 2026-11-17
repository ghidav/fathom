"""Analysing a group's flagged mistakes: an analyst reads what the readers flagged in its rollouts
and names the kinds of mistake that recur, each pointing at the mistakes it holds.

Two questions are asked. `within` a group: the kinds among its rollouts' mistakes, and what
separates the rollouts that solved their task from those that failed. `across` the sub-groups of a
group split by a label: which of their kinds are the same mistake. The analyst is a reader, as in
`fathom.reading`: a `claude` model alias, or a function `(prompt, schema, files, dirs)`.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from fathom.reading import CAUSES, SEVERITIES, Reader

if TYPE_CHECKING:
    from fathom.group import Group
    from fathom.rollout import Rollout

SEVERITY_TEXT = ("critical: still in the final patch and makes a hidden test fail; major: would have failed the task "
                 "had it not been caught, or cost 10 or more main-agent calls or 15% of them; minor: any other waste")
CAUSE_TEXT = ("agent: its own error; environment: the sandbox or the harness forced it; verifier: a hidden test asks for "
              "what the task never names, or rejects a correct result")

WITHIN_SCHEMA = {
    "type": "object",
    "properties": {
        "kinds": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "severity": {"enum": list(SEVERITIES)},
                "what": {"type": "string"},
                "members": {"type": "array", "items": {
                    "type": "object",
                    "properties": {"rollout": {"type": "integer"}, "mistake": {"type": "integer"}},
                    "required": ["rollout", "mistake"],
                }},
            },
            "required": ["name", "severity", "what", "members"],
        }},
        "difference": {"type": "string"},
        "unsure": {"type": "string"},
    },
    "required": ["kinds", "difference"],
}
ACROSS_SCHEMA = {
    "type": "object",
    "properties": {
        "kinds": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "severity": {"enum": list(SEVERITIES)},
                "what": {"type": "string"},
                "members": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "severity", "what", "members"],
        }},
        "unsure": {"type": "string"},
    },
    "required": ["kinds"],
}

WITHIN_PROMPT = """You are analysing the mistakes readers flagged in a group of rollouts, each one attempt by a coding agent at a task; {solved} of {n} solved theirs.{shared} Hidden tests the agent never saw decided whether a rollout solved its task. This is an analysis task: read files, judge, answer. Run no code and change nothing.

## Files you may read (only these)
- group.txt, in your working directory: each rollout, numbered from 1, with its labels, whether it solved its task, and the mistakes its reader flagged, numbered from 0. Each mistake has a severity ({severities}), a cause ({causes}), the main-agent steps it covers, and what went wrong.
- When a flag doesn't say enough: the files group.txt names for a rollout, its whole trajectory (ATIF; find a step by its `step_id`) and the folder of what its hidden tests checked and printed.

## What to answer
- `kinds`: the kinds of mistake in this group: the same wrong idea in the fix, or the same wasted effort for the same reason. A kind made by two or more rollouts is one; so is every critical or major mistake, even when one rollout alone made it. Mistakes with different causes are different kinds. Give each a short name (at most six words), its usual severity, and in `what` two sentences: what the agent does, and what it leads to. Its members are each rollout (its number) and the number of its mistake; a mistake belongs to at most one kind.
- `difference`: two or three sentences on what separates the rollouts that solved their task from those that failed. If they all ended the same way, what they share.
- `unsure`: what you were unsure about, in a sentence or two."""

ACROSS_PROMPT = """You are sorting the mistakes an agent made across groups of rollouts into the mistakes that recur. The rollouts, each one attempt by a coding agent at a task, were put in groups by their {label}. For each group, a reader flagged every rollout's mistakes and an analyst named the group's kinds of mistake. This is an analysis task: read the file, judge, answer. Run no code and change nothing.

## File you may read (only this)
- kinds.txt, in your working directory: each group, with its {label} and how many of its rollouts solved their task, then its kinds of mistake. Each kind has an id (`<group>.<kind>`), a severity ({severities}), a cause ({causes}), how many of the group's rollouts made it and how many of those failed, its name, and what the agent did.

## What to answer
- `kinds`: the mistakes across the groups. Kinds of different groups that are the same mistake are one: the same wrong habit, or the same wasted effort for the same reason, whatever the {label}. Kinds with different causes are different mistakes. Describe each by what the agent does, not by the group it happened in, unless it belongs to one group alone. A kind that matches no other stands alone. Give each a short name (at most six words), its usual severity, two sentences in `what` (what the agent does, and what it leads to), and its `members`: the ids of the group kinds it holds. Place every id exactly once.
- `unsure`: what you were unsure about, in a sentence or two."""


def within(group: Group, ask: Reader) -> dict:
    """The kinds of mistake among the group's flagged mistakes, as `ask` names them. Each kind's
    members point at a mistake: the rollout's place in the layer's own `rollouts`, and the
    mistake's place among that rollout's. `rollouts` keeps a mark of each rollout's mistakes as
    read, so that it shows when they were flagged again. A group with no mistake flagged has no
    kinds, and the analyst is not asked."""
    members = group.rollouts
    about = {"analyst": _name(ask), "at": _now()}
    table = [{"rollout": group._ref(r, i), "flagged": digest(r.mistakes["mistakes"])} for i, r in enumerate(members)]
    if not any(r.mistakes["mistakes"] for r in members):
        return {**about, "cost_usd": 0.0, "rollouts": table, "kinds": [],
                "difference": "No mistake is flagged in this group.", "unsure": None}
    text, dirs = _group_text(members)
    shared = _shared(members)
    prompt = WITHIN_PROMPT.format(solved=sum(r.solved is True for r in members), n=len(members), severities=SEVERITY_TEXT,
                                  causes=CAUSE_TEXT, shared=f" The rollouts share: {shared}." if shared else "")
    answer, told = ask(prompt, WITHIN_SCHEMA, {"group.txt": text}, dirs)
    kinds, taken = [], set()
    for k in answer.get("kinds") or []:
        held = []
        for x in k.get("members") or []:
            at = (x.get("rollout", 0) - 1, x.get("mistake", -1))
            if 0 <= at[0] < len(members) and 0 <= at[1] < len(members[at[0]].mistakes["mistakes"]) and at not in taken:
                taken.add(at)
                held.append({"rollout": at[0], "mistake": at[1]})
        if held and k.get("severity") in SEVERITIES:
            causes = Counter(members[m["rollout"]].mistakes["mistakes"][m["mistake"]].get("cause") for m in held)
            kinds.append({"name": (k.get("name") or "").strip(), "severity": k["severity"], "cause": _most(causes),
                          "what": (k.get("what") or "").strip(), "members": held})
    return {**about, **told, "rollouts": table, "kinds": kinds, "difference": answer.get("difference"), "unsure": answer.get("unsure")}


def across(label: str, subs: dict[str, Group], ask: Reader) -> dict:
    """The mistakes across a group's sub-groups by `label`, as `ask` sorts the sub-groups' kinds.
    Each kind's members point at a sub-group's kind: the label's value, and the kind's place in
    that sub-group's analysis. A kind the analyst leaves out stands as its own."""
    about = {"analyst": _name(ask), "at": _now(), "by": label}
    table = [{"group": value, "kinds": digest(sub.mistakes["kinds"])} for value, sub in subs.items()]
    known, lines = {}, []
    for n, (value, sub) in enumerate(subs.items(), 1):
        lines.append(f"## Group {n}: {label} {value}: {sum(r.solved is True for r in sub)} of {len(sub)} solved")
        read = sub._members()
        for i, k in enumerate(sub.mistakes["kinds"]):
            made = {m["rollout"] for m in k["members"]}
            failed = sum(read[r].solved is False for r in made)
            known[f"{n}.{i}"] = {"group": value, "kind": i}
            lines.append(f"{n}.{i}  {k['severity']}; cause {k['cause']}; {len(made)} of {len(sub)} rollouts, {failed} of them failed. {k['name']}: {k['what']}")
        lines.append("" if sub.mistakes["kinds"] else "(no kinds of mistake)\n")
    if not known:
        return {**about, "cost_usd": 0.0, "groups": table, "kinds": [], "unsure": None}
    answer, told = ask(ACROSS_PROMPT.format(label=label, severities=SEVERITY_TEXT, causes=CAUSE_TEXT), ACROSS_SCHEMA,
                       {"kinds.txt": "\n".join(lines)}, [])

    def own(ref: dict) -> dict:
        return subs[ref["group"]].mistakes["kinds"][ref["kind"]]

    kinds, placed = [], set()
    for k in answer.get("kinds") or []:
        ids = [x for x in dict.fromkeys(k.get("members") or []) if x in known and x not in placed]
        if ids and k.get("severity") in SEVERITIES:
            placed.update(ids)
            refs = [known[x] for x in ids]
            causes: Counter = Counter()
            for ref in refs:
                causes[own(ref)["cause"]] += len(own(ref)["members"])
            kinds.append({"name": (k.get("name") or "").strip(), "severity": k["severity"], "cause": _most(causes),
                          "what": (k.get("what") or "").strip(), "members": refs})
    for x, ref in known.items():
        if x not in placed:
            kinds.append({**{f: own(ref)[f] for f in ("name", "severity", "cause", "what")}, "members": [ref]})
    return {**about, **told, "groups": table, "kinds": kinds, "unsure": answer.get("unsure")}


def digest(what: object) -> str:
    """A short mark of some JSON: a layer keeps one of what it read, to tell whether that has changed since."""
    return hashlib.sha1(json.dumps(what, sort_keys=True).encode()).hexdigest()[:12]


def _group_text(members: list[Rollout]) -> tuple[str, list[Path]]:
    """The members' flagged mistakes as text, and the folders the analyst may look things up in."""
    out, dirs = [], []
    for n, r in enumerate(members, 1):
        outcome = {True: "solved", False: "failed", None: "no verdict"}[r.solved]
        labels = ", ".join(f"{k} = {v}" for k, v in r.labels.items())
        out.append(f"### Rollout {n}: {outcome}, {r.mistakes.get('calls', '?')} main-agent calls" + (f" ({labels})" if labels else ""))
        if r.path:
            out.append(f"trajectory: {r.path}")
            dirs.append(r.path.parent)
        if r.verifier:
            out.append(f"hidden tests: {r.verifier}/")
            dirs.append(r.verifier)
        for i, m in enumerate(r.mistakes["mistakes"]):
            out.append(f"{i}. {m.get('severity')}, cause {m.get('cause')}, steps {_ranges(m.get('steps'))}: {m.get('what')}"
                       + (f"\n   evidence: {m['evidence']}" if m.get("evidence") else ""))
        out.append("(no mistakes flagged)\n" if not r.mistakes["mistakes"] else "")
    return "\n".join(out), list(dict.fromkeys(dirs))


def _shared(members: list[Rollout]) -> str:
    """The labels every member has with one value: what the group's rollouts have in common."""
    first = members[0].labels if members else {}
    return ", ".join(f"{k} = {v}" for k, v in first.items() if all(k in r.labels and r.labels[k] == v for r in members))


def _ranges(steps: list | None) -> str:
    spans = [steps] if steps and not isinstance(steps[0], list) else steps or []
    return ", ".join(str(r[0]) if r[0] == r[-1] else f"{r[0]}-{r[-1]}" for r in spans if r)


def _most(causes: Counter) -> str:
    """The cause most of a kind's mistakes have; the agent's before the environment's before the verifier's on a tie."""
    return max(CAUSES, key=lambda c: causes.get(c, 0))


def _name(ask: Reader) -> str:
    return getattr(ask, "name", getattr(ask, "__name__", "analyst"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
