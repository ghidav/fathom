"""A group: the rollouts a caller chose to look at together.

    group = Group.from_job(job)        # every trial of a Harbor job; or Group(rollouts)
    group.flag_mistakes()              # each member's, a few at a time; the ones flagged already are left
    group.tag()                        # a reader says in a few words what each part of each turn does
    group.embed()                      # each of those tags, as a vector
    group.analyze_mistakes(by="task")  # an analyst names the kinds of mistake that recur
    group.search("asks the user")      # the turns whose tag reads most like a description, best first
    group.cluster("thought")           # its turns grouped by what their tags read like
    group.describe()                   # counts over the group, its kinds of mistake and its clusters, with no model
    group.table()                      # a row for each turn, with all that is known of it: for any other count
    group.by("task")                   # a sub-group for each task

Which rollouts belong together is the caller's to say: a group knows nothing of runs or steps, only
its members and their labels.
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from fathom import analysis, clustering, embedding, reading, tagging
from fathom.rollout import PARTS, Rollout, Turn, clock, covered

WORKERS = 16
POLL = 30  # seconds between looks at a batch that is being worked on
_UNREAD: Any = object()


@dataclass(frozen=True)
class Batch:
    """What a group's `flag_mistakes`, `embed` or `tag` did to its members."""

    made: list[Rollout]  # got the layer now
    skipped: list[Rollout]  # had it already
    failed: list[tuple[Rollout, Exception]]  # could not get it, and why
    cost_usd: float  # what the layers made cost, as far as their makers said


@dataclass(frozen=True)
class Hit:
    """A turn a group's `search` found."""

    rollout: Rollout
    turn: Turn
    score: float  # how like the description its tag reads: the cosine of their vectors


class Group:
    """Rollouts looked at together, and the folder its own results are kept in, if it has one."""

    def __init__(self, rollouts: Iterable[Rollout], *, path: str | Path | None = None) -> None:
        self.rollouts = list(rollouts)
        self.path = Path(path) if path else None
        self._mistakes: Any = _UNREAD if self.path else None
        self._clusters: dict[str, dict] | None = None if self.path else {}
        self._subs: dict[str, dict[str, Group]] = {}  # the sub-groups an analysis by a label was made on

    @classmethod
    def from_job(cls, folder: str | Path, labels: dict | None = None) -> Group:
        """The group of a Harbor job folder: the rollout of every trial in it that left a
        trajectory, by name, each labelled with `labels` too. The job's folder is the group's."""
        folder = Path(folder)
        trials = sorted(p.parent.parent for p in folder.glob("*/agent/trajectory.json"))
        return cls([Rollout.from_trial(t, labels) for t in trials], path=folder)

    def __len__(self) -> int:
        return len(self.rollouts)

    def __iter__(self) -> Iterator[Rollout]:
        return iter(self.rollouts)

    def by(self, label: str) -> dict[Any, Group]:
        """The group split by a label: a sub-group for each value its members have, in order of
        value, the members without the label under None, last. Each sub-group's folder is
        `<label>/<value>` under the group's, if the group has one."""
        split: dict[Any, list[Rollout]] = {}
        for r in self.rollouts:
            split.setdefault(r.labels.get(label), []).append(r)
        order = sorted(split, key=lambda v: (v is None, isinstance(v, str), v if v is not None else 0))
        return {v: Group(split[v], path=self.path and self.path / label / str(v).replace("/", "__")) for v in order}

    @property
    def mistakes(self) -> dict | None:
        """The layer `analyze_mistakes` makes: the analyst, when and at what cost, and the group's
        kinds of mistake, each with a name, a severity, a cause, what the agent does, and members
        that point at what it holds: flagged mistakes (a rollout's place in the layer's `rollouts`
        and the mistake's among that rollout's), or, for an analysis `by` a label, kinds of the
        sub-groups (the label's value and the kind's place there)."""
        if self._mistakes is _UNREAD:
            file = self.path / "mistakes.json"
            self._mistakes = json.loads(file.read_text()) if file.exists() else None
        return self._mistakes

    @property
    def clusters(self) -> dict[str, dict]:
        """The layers `cluster` makes, by the part clustered: when it was made, the embedder, each
        rollout's tagged turns (their step ids) with the cluster each fell in (-1 for none), and
        each cluster's name, the tag nearest its middle, with the few nearest."""
        if self._clusters is None:
            self._clusters = {f.name.split(".")[1]: json.loads(f.read_text()) for f in sorted(self.path.glob("clusters.*.json"))}
        return self._clusters

    def flag_mistakes(self, reader: str | reading.Reader = "sonnet", *, save: bool = True, overwrite: bool = False,
                      workers: int = WORKERS) -> Batch:
        """Have `reader` flag the mistakes of every member, `workers` at a time. Members flagged
        already are left as they are, unless `overwrite` is on. A member that cannot be read is
        reported and the others go on. Fails, before anything is spent, if the reader cannot run or
        `save` is on and a member to flag has no file."""
        ask = reading.resolve(reader)
        todo = [r for r in self.rollouts if overwrite or r.mistakes is None]
        _may_write(todo, save)
        return self._each(todo, lambda r: r.flag_mistakes(ask, save=save, overwrite=overwrite).get("cost_usd"), workers)

    def embed(self, parts: Sequence[str] = PARTS, embedder: str | embedding.Embedder | None = None, *,
              save: bool = True, overwrite: bool = False, workers: int = WORKERS) -> Batch:
        """Embed the tags of every member's turns for `parts` with `embedder`, `workers` members at
        a time. A member gets only the parts it lacks, or whose tags have changed since, unless
        `overwrite` is on. A member that cannot be embedded is reported and the others go on.
        Fails, before anything is spent, if the embedder cannot run, a member is not tagged, a
        member holds another embedder's vectors or vectors of its turns' own text and `overwrite`
        is off, or `save` is on and a member to embed has no file."""
        if unknown := [p for p in parts if p not in PARTS]:
            raise ValueError(f"no part {unknown[0]}: a turn's parts are {', '.join(PARTS)}")
        embedder = embedding.resolve(embedder)
        name = embedding.name(embedder)
        if untagged := sum(r.tags is None for r in self.rollouts):
            raise ValueError(f"{untagged} of the group's {len(self)} rollouts are not tagged: tag them first, with group.tag()")
        todo: list[tuple[Rollout, list[str], float]] = []
        for r in self.rollouts:
            had = r._embedded()
            if had is not None and not overwrite:
                if had["of"] != "tags":
                    raise FileExistsError(f"{r._name()} holds vectors of its turns' own text, not of their tags: "
                                          "pass overwrite=True to replace them, or leave such members out")
                if had["embedder"] != name:
                    raise FileExistsError(f"the tags of {r._name()} are embedded by {had['embedder']}, whose vectors would not compare "
                                          f"with {name}'s: pass overwrite=True to embed them again, or leave such members out")
            same = had is not None and had["of"] == "tags" and had["embedder"] == name
            fresh = r._fresh(had) if same else []
            lacking = list(parts) if overwrite else [p for p in parts if p not in fresh]
            if lacking:
                todo.append((r, lacking, (had.get("cost_usd") or 0.0) if same else 0.0))
        _may_write([r for r, _, _ in todo], save)
        asked = {id(r): (lacking, before) for r, lacking, before in todo}

        def work(r: Rollout) -> float | None:
            lacking, before = asked[id(r)]
            cost = r.embed(lacking, embedder, save=save, overwrite=overwrite).get("cost_usd")
            if save:
                r._release()  # a run's vectors are gigabytes: they are on disk now
            return None if cost is None else cost - before

        return self._each([r for r, _, _ in todo], work, workers)

    def tag(self, reader: str | reading.Reader | None = None, *, save: bool = True, overwrite: bool = False,
            workers: int = WORKERS, batch: bool = False) -> Batch:
        """Have `reader` tag the turns of every member, `workers` at a time: a few words for each
        part of each turn. With no reader named it is Claude Haiku through the Claude API. Members
        tagged already are left as they are, unless `overwrite` is on, so a run that stopped is
        picked up by calling again. A member that cannot be tagged is reported and the others go
        on. Fails, before anything is spent, if the reader cannot run or `save` is on and a member
        to tag has no file.

        With `batch`, the members are handed to the reader's batch (for `ClaudeAPI`, the Message
        Batches API: half the price, answered within a day and mostly within the hour), and the
        call waits for it. The batch is noted in `tags.batch.json` in the group's folder until its
        answers are in, so that a call that was cut short is picked up by calling again, and
        nothing is paid twice. It needs a group with a folder, `save` on, and a reader with a batch."""
        ask = tagging.resolve(reader)
        todo = [r for r in self.rollouts if overwrite or r.tags is None]
        _may_write(todo, save)
        if batch:
            return self._tag_in_a_batch(ask, todo, save, workers)
        return self._each(todo, lambda r: r.tag(ask, save=save, overwrite=overwrite).get("cost_usd"), workers)

    def tag_cost(self, reader: str | reading.Reader | None = None, *, overwrite: bool = False, workers: int = WORKERS) -> dict:
        """What tagging the members not tagged yet (all of them with `overwrite`) would take,
        without tagging any: the `rollouts`, the `calls` and how many rollouts are `split` over
        several, the `input_tokens`, counted by the reader and so exact, the `tags` to write and
        the `output_tokens` that takes at `per_tag` each, and the cost, by call (`cost_usd`) and
        through a batch (`batch_usd`). `per_tag` is what this reader wrote for each tag in the
        `measured_on` members it has tagged already, or, with none, a figure from one earlier run:
        the output is the estimated half. Counting costs nothing. Fails if the reader cannot count."""
        ask = tagging.resolve(reader)
        name = getattr(ask, "name", getattr(ask, "__name__", "reader"))
        todo = [r for r in self.rollouts if overwrite or r.tags is None]
        done = [r.tags for r in self.rollouts if r.tags is not None and r.tags.get("reader") == name
                and isinstance((r.tags.get("tokens") or {}).get("output"), (int, float))]
        written = sum(bool(t) for layer in done for tags in layer["parts"].values() for t in tags)
        per_tag = sum(layer["tokens"]["output"] for layer in done) / written if written else tagging.PER_TAG
        if getattr(ask, "count", None) is None:
            raise ValueError("this reader cannot count a job's tokens, so the cost of tagging cannot be told ahead: a ClaudeAPI reader can")
        with ThreadPoolExecutor(max(1, min(workers, len(todo) or 1))) as pool:
            plans = list(pool.map(lambda r: tagging.plan(r, ask), todo))
        return {**tagging.cost(todo, ask, per_tag, plans), "measured_on": len(done) if written else 0}

    def _tag_in_a_batch(self, ask: reading.Reader, todo: list[Rollout], save: bool, workers: int) -> Batch:
        """Tag `todo` through the reader's batch: hand the calls in, or find the batch a call cut
        short had handed in, wait, and make each member's layer from its answers."""
        name = getattr(ask, "name", getattr(ask, "__name__", "reader"))
        if getattr(ask, "submit", None) is None or getattr(ask, "collect", None) is None:
            raise ValueError("this reader has no batch to hand its calls to: a ClaudeAPI reader has")
        if self.path is None or not save:
            raise ValueError("a batch can take hours, and is picked up again from a note in the group's folder: "
                             "it needs a group with a folder, and save on")
        note = self.path / "tags.batch.json"
        mine = {self._ref(r, i): r for i, r in enumerate(self.rollouts)}
        doing = {id(r) for r in todo}
        if note.exists():
            state = json.loads(note.read_text())
            if state["reader"] != name:
                raise FileExistsError(f"a batch by {state['reader']} is being worked on for this group ({note}): "
                                      "call again with that reader to pick it up, or remove the note to leave it")
        else:
            if not todo:
                return Batch(made=[], skipped=list(self.rollouts), failed=[], cost_usd=0.0)
            with ThreadPoolExecutor(max(1, min(workers, len(todo)))) as pool:
                plans = list(pool.map(lambda r: tagging.plan(r, ask), todo))
            refs = {id(r): ref for ref, r in mine.items()}
            calls, jobs = {}, {}
            for n, (r, plan) in enumerate(zip(todo, plans)):
                trajectory = r.trajectory
                for j, call in enumerate(plan):
                    calls[f"r{n}-c{j}"] = {"rollout": refs[id(r)], "steps": call["steps"], "place": [j, len(plan)]}
                    jobs[f"r{n}-c{j}"] = tagging.job(r, call["steps"], (j, len(plan)), trajectory)[:3]
            state = {"reader": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "batches": ask.submit(jobs), "calls": calls}
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = note.with_name(note.name + ".tmp")
            tmp.write_text(json.dumps(state))
            tmp.replace(note)
        while (answers := ask.collect(state["batches"])) is None:
            time.sleep(POLL)
        per: dict[str, list[tuple[int, dict, Any]]] = {}
        for key, call in state["calls"].items():
            per.setdefault(call["rollout"], []).append((call["place"][0], call, answers.get(key, RuntimeError("the batch gave no answer to this call"))))
        made, failed, cost = [], [], 0.0
        for ref, got in per.items():
            r = mine.get(ref)
            if r is None or id(r) not in doing:  # no longer a member, or tagged since
                continue
            try:
                if errors := [a for _, _, a in got if isinstance(a, Exception)]:
                    raise errors[0]
                answered = [{"steps": call["steps"], "answer": a[0], "about": a[1]} for _, call, a in sorted(got, key=lambda x: x[0])]
                layer = r._tagged(tagging.tag(r, ask, answered), save)
            except Exception as e:  # noqa: BLE001 - one failure is reported, the others go on
                failed.append((r, e))
                continue
            made.append(r)
            cost += layer.get("cost_usd") or 0.0
        note.unlink()
        return Batch(made=made, skipped=[r for r in self.rollouts if id(r) not in doing], failed=failed, cost_usd=cost)

    def analyze_mistakes(self, analyst: str | reading.Reader = "opus", by: str | None = None, *, save: bool = True,
                         overwrite: bool = False, workers: int = WORKERS) -> dict:
        """Have `analyst` name the kinds of mistake that recur among the members' flagged mistakes,
        and say what separates the rollouts that solved their task from those that failed. With
        `by`, a label, each sub-group is analyzed first, `workers` at a time, and their kinds are
        then sorted into the mistakes that recur across them; a sub-group analyzed already is
        left as it is, unless `overwrite` is on. Returns the layer, keeps it as `mistakes`, and
        saves it as `mistakes.json` in the group's folder (a sub-group's in its own) unless
        `save` is off. Fails, before anything is spent, if a member is not flagged, if the group
        is analyzed already and `overwrite` is off, or if `save` is on and the group has no folder."""
        ask = reading.resolve(analyst)
        if not self.rollouts:
            raise ValueError("an empty group has no mistakes to analyze")
        if unflagged := sum(r.mistakes is None for r in self.rollouts):
            raise ValueError(f"{unflagged} of the group's {len(self)} rollouts are not flagged: flag them first, with group.flag_mistakes()")
        if save and self.path is None:
            raise ValueError("this group has no folder to save in: pass save=False, or give it a path")
        if self.mistakes is not None and not overwrite:
            stale = "" if self._current() else ", before its rollouts were last flagged"
            raise FileExistsError(f"this group's mistakes are analyzed already{stale}: pass overwrite=True to analyze them again")
        if by is None:
            layer = analysis.within(self, ask)
        else:
            subs = {str(v): g for v, g in self.by(by).items()}
            if not overwrite:
                for v, g in subs.items():
                    if g.mistakes is not None and ("by" in g.mistakes or not g._current()):
                        raise FileExistsError(f"the analysis of {by} {v} is not one of its rollouts' mistakes as they are now: "
                                              "pass overwrite=True to analyze the sub-groups again")
            todo = [g for g in subs.values() if overwrite or g.mistakes is None]
            _, failed, spent = _each(todo, lambda g: g.analyze_mistakes(ask, save=save, overwrite=overwrite).get("cost_usd"), workers)
            if failed:
                raise RuntimeError(f"{len(failed)} of the {len(subs)} sub-groups by {by} could not be analyzed; "
                                   f"the others are kept. The first: {failed[0][1]}")
            layer = analysis.across(by, subs, ask)
            layer["total_usd"] = (layer.get("cost_usd") or 0.0) + spent  # with the sub-groups analyzed now
            self._subs[by] = subs
        self._mistakes = layer
        if save:
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = self.path / "mistakes.json.tmp"
            tmp.write_text(json.dumps(layer, indent=1))
            tmp.replace(self.path / "mistakes.json")
        return layer

    def search(self, text: str, part: str = "thought", *, top: int | None = 20,
               embedder: str | embedding.Embedder | None = None) -> list[Hit]:
        """The turns whose tag for `part` reads most like `text`, a description of what the agent
        does ("asks the user a question", "runs the tests"), best first: the `top` of them, or all
        with `top=None`. The description is embedded with `embedder`, which must be the one the
        members' tags were embedded with, and compared with their vectors. Nothing is saved.
        Fails, before anything is spent, if a member's tags for the part are not embedded, or by
        another embedder."""
        if part not in PARTS:
            raise ValueError(f"no part {part}: a turn's parts are {', '.join(PARTS)}")
        if not text.strip():
            raise ValueError("nothing to search for: say what the agent does in the turns you want")
        by = self._embedded_by(part)
        embedder = embedding.resolve(embedder)
        if by is not None and embedding.name(embedder) != by:
            raise ValueError(f"the group's {part} tags are embedded by {by}, and a description embedded by "
                             f"{embedding.name(embedder)} would not compare: search with that embedder")
        turns: dict[int, dict[int, Turn]] = {}
        hits = []
        for score, mi, step in tagging.rank(self.rollouts, text, part, embedder)[:top]:
            if mi not in turns:
                turns[mi] = {t.id: t for t in self.rollouts[mi].turns}
            hits.append(Hit(self.rollouts[mi], turns[mi][step], score))
        return hits

    def cluster(self, part: str = "thought", *, save: bool = True, overwrite: bool = False) -> dict:
        """Cluster the members' turns by what their tag for `part` reads like, across all the
        rollouts together: the tags' vectors are grouped, and each cluster is named by the tag
        nearest its middle. No model is asked and nothing is spent. Returns the layer, keeps it in
        `clusters` under the part, and saves it as `clusters.<part>.json` in the group's folder
        unless `save` is off. Fails if a member's tags for the part are not embedded, or by
        another embedder than the others', if there are too few tagged turns, if the part is
        clustered already and `overwrite` is off, or if `save` is on and the group has no folder."""
        if part not in PARTS:
            raise ValueError(f"no part {part}: a turn's parts are {', '.join(PARTS)}")
        if why := clustering.missing():
            raise RuntimeError(why)
        if save and self.path is None:
            raise ValueError("this group has no folder to save in: pass save=False, or give it a path")
        if part in self.clusters and not overwrite:
            raise FileExistsError(f"this group's turns are clustered by their {part} tags already: pass overwrite=True to cluster them again")
        self._embedded_by(part)
        if (turns := sum(len(r._part(part, rows=False)[1]) for r in self.rollouts)) < clustering.FEWEST:
            raise ValueError(f"only {turns} turns of the group have a {part} tag: too few to cluster ({clustering.FEWEST} are needed)")
        layer = clustering.cluster(self, part)
        self.clusters[part] = layer
        if save:
            self.path.mkdir(parents=True, exist_ok=True)
            tmp = self.path / f"clusters.{part}.json.tmp"
            tmp.write_text(json.dumps(layer))
            tmp.replace(self.path / f"clusters.{part}.json")
        return layer

    def describe(self, by: str | None = None) -> dict:
        """Counts over the group, read from its members and their mistakes as they are, with no
        model: how many rollouts solved their task, failed and are flagged; whose failure each
        failed rollout is (`unknown`: not flagged, or no critical mistake flagged); the mistakes
        by severity and cause; the seconds of the members' turns; and the seconds of the turns a
        mistake covers, each turn once, under the worst mistake that covers it.

        Once the group's mistakes are analyzed (`analyzed` says by whom, when, and whether that
        still fits the rollouts' mistakes as they are), `kinds` has each kind of mistake, worst
        first, with the mistakes and rollouts it holds, how those rollouts ended, the seconds of
        the turns it covers and, for an analysis by a label, the sub-groups it shows up in. With
        `by`, a label, `per` counts the rollouts and their outcomes for each of its values, and
        each kind's `per` the rollouts that made it: how a mistake changes from step to step.

        Once the group's turns are clustered, `clusters` has, for each part clustered, when and
        with which embedder, whether the clusters still fit the members' tags as they are, and
        then each cluster, longest first, with its name and nearest tags, its turns and their
        seconds, the rollouts it shows up in and how they ended, how many of its turns a flagged
        mistake covers, by the worst, of those in flagged rollouts, and, with `by`, the rollouts
        it shows up in per value."""
        severities, causes = reading.SEVERITIES, reading.CAUSES
        table = lambda: {s: dict.fromkeys(causes, 0) for s in severities}  # noqa: E731
        out: dict = {"rollouts": len(self.rollouts), "solved": 0, "failed": 0, "flagged": 0,
                     "failed_by": dict.fromkeys((*causes, "unknown"), 0),
                     "mistakes": table(), "seconds": 0.0, "seconds_in_mistakes": table()}
        times: dict[int, dict[int, float]] = {}  # each member's seconds by step id
        worst_at: dict[int, dict[int, str]] = {}  # the severity of the worst mistake that covers each of a member's turns
        for r in self.rollouts:
            out["solved"] += r.solved is True
            out["failed"] += r.solved is False
            out["flagged"] += r.mistakes is not None
            if r.solved is False:
                out["failed_by"][r.failed_by or "unknown"] += 1
            for m in (r.mistakes or {}).get("mistakes") or []:
                if m.get("severity") in severities and m.get("cause") in causes:
                    out["mistakes"][m["severity"]][m["cause"]] += 1
            covering = r._covering()
            took, bad = times.setdefault(id(r), {}), worst_at.setdefault(id(r), {})
            for st, seconds in clock(r.trajectory):
                took[int(st["step_id"])] = seconds
                out["seconds"] += seconds
                known = [m for m in covering.get(int(st["step_id"]), ()) if m.get("severity") in severities and m.get("cause") in causes]
                if known:
                    worst = min(known, key=lambda m: severities.index(m["severity"]))
                    bad[int(st["step_id"])] = worst["severity"]
                    out["seconds_in_mistakes"][worst["severity"]][worst["cause"]] += seconds
        out["seconds"] = round(out["seconds"], 1)
        out["seconds_in_mistakes"] = {s: {c: round(v, 1) for c, v in row.items()} for s, row in out["seconds_in_mistakes"].items()}
        if by:
            out["per"] = {str(v): {"rollouts": len(g), "solved": sum(r.solved is True for r in g), "failed": sum(r.solved is False for r in g)}
                          for v, g in self.by(by).items()}
        layer = self.mistakes
        out["analyzed"] = layer and {"analyst": layer.get("analyst"), "at": layer.get("at"), "by": layer.get("by"), "current": self._current()}
        if layer and out["analyzed"]["current"]:
            rows = []
            for k in layer["kinds"]:
                held = self._held(k)
                made = {id(r): r for r, _ in held}
                steps: dict[int, set[int]] = {}
                for r, i in held:  # the turns the kind covers in a rollout, each once
                    steps.setdefault(id(r), set()).update(covered(r.mistakes["mistakes"][i]))
                row = {**{f: k[f] for f in ("name", "severity", "cause", "what")}, "mistakes": len(held), "rollouts": len(made),
                       "failed": sum(r.solved is False for r in made.values()), "solved": sum(r.solved is True for r in made.values()),
                       "seconds": round(sum(times[rid].get(s, 0.0) for rid, on in steps.items() for s in on), 1)}
                if layer.get("by"):
                    row["groups"] = len({ref["group"] for ref in k["members"]})
                if by:
                    row["per"] = dict(Counter(str(r.labels.get(by)) for r in made.values()))
                rows.append(row)
            out["kinds"] = sorted(rows, key=lambda x: (severities.index(x["severity"]), -x["rollouts"], x["name"]))
        if self.clusters:
            out["clusters"] = {part: self._clustered(part, by, times, worst_at) for part in self.clusters}
        return out

    def table(self) -> list[dict]:
        """A row for each turn of each member, in order, with all that is known of it, for counts
        `describe` does not make: sum, filter and group the rows as the question asks.

        A row has the turn's `rollout` (as the group's layers name it), the rollout's labels, each
        under its own name, and its outcome (`solved`, `reward`, `failed_by`); the `turn`'s step
        id, its `seconds` and the tokens it `sampled` and was given as `context`; its tag for each
        part (`thought`, `call`, `result`), None where it has none; for each part clustered, as
        long as the clusters still fit the tags, the name of the turn's cluster (`thought_cluster`
        and so on), None where it fell in none; and the worst flagged mistake that covers it: its
        `severity`, its `cause` and what it is (`mistake`), None when none does. A label named as
        one of these fields is given as `label_<name>`."""
        severities = reading.SEVERITIES
        named: dict[str, dict[str, dict[int, str | None]]] = {}  # for each part clustered, each member's cluster names by step id
        for part, layer in self.clusters.items():
            if self._fits(part):
                names = [c["name"] for c in layer["clusters"]]
                named[part] = {x["rollout"]: {s: names[c] if c >= 0 else None for s, c in zip(x["steps"], x["clusters"])} for x in layer["rollouts"]}
        fields = {"rollout", "solved", "reward", "failed_by", "turn", "seconds", "sampled", "context", *PARTS,
                  *(f"{p}_cluster" for p in PARTS), "severity", "cause", "mistake"}
        rows = []
        for i, r in enumerate(self.rollouts):
            ref = self._ref(r, i)
            about = {"rollout": ref, **{(f"label_{k}" if k in fields else k): v for k, v in r.labels.items()},
                     "solved": r.solved, "reward": r.reward, "failed_by": r.failed_by}
            for t in r.turns:
                known = [m for m in t.mistakes if m.get("severity") in severities]
                worst = min(known, key=lambda m: severities.index(m["severity"])) if known else {}
                rows.append({**about, "turn": t.id, "seconds": t.seconds, "sampled": t.sampled, "context": t.context,
                             **{p: t.tags.get(p) for p in PARTS},
                             **{f"{part}_cluster": by_step.get(ref, {}).get(t.id) for part, by_step in named.items()},
                             "severity": worst.get("severity"), "cause": worst.get("cause"), "mistake": worst.get("what")})
        return rows

    def _fits(self, part: str) -> bool:
        """Whether the clusters by a part are of the members' tags as they are now: the same
        rollouts, each with the tags the clustering read."""
        layer = self.clusters[part]
        mine = {self._ref(r, i): r for i, r in enumerate(self.rollouts)}
        read = [mine.get(x["rollout"]) for x in layer["rollouts"]]
        return len(read) == len(self.rollouts) and all(
            r is not None and r.tags is not None and analysis.digest(r.tags["parts"][part]) == x.get("tags")
            for r, x in zip(read, layer["rollouts"]))

    def _clustered(self, part: str, by: str | None, times: dict, worst_at: dict) -> dict:
        """What `describe` says of the clusters by a part: when they were made, whether they still
        fit the members' tags (the same rollouts, each with the tags the clustering read), and, if
        so, each cluster's numbers."""
        layer = self.clusters[part]
        mine = {self._ref(r, i): r for i, r in enumerate(self.rollouts)}
        read = [mine.get(x["rollout"]) for x in layer["rollouts"]]
        current = self._fits(part)
        out = {"at": layer.get("at"), "embedder": layer["embedder"], "current": current, "turns": layer["turns"]}
        if not current:
            return out
        rows = [{"name": c["name"], "tags": c["tags"], "turns": 0, "seconds": 0.0, "in_mistakes": {**dict.fromkeys(reading.SEVERITIES, 0), "of": 0}}
                for c in layer["clusters"]]
        made: list[dict[int, Rollout]] = [{} for _ in rows]
        none = {"turns": 0, "seconds": 0.0}
        for r, x in zip(read, layer["rollouts"]):
            for step, c in zip(x["steps"], x["clusters"]):
                row = rows[c] if c >= 0 else none
                row["turns"] += 1
                row["seconds"] += times[id(r)].get(step, 0.0)
                if c >= 0:
                    made[c][id(r)] = r
                    if r.mistakes is not None:
                        row["in_mistakes"]["of"] += 1
                        if step in worst_at[id(r)]:
                            row["in_mistakes"][worst_at[id(r)][step]] += 1
        for row, has in zip(rows, made):
            row.update(seconds=round(row["seconds"], 1), rollouts=len(has), solved=sum(r.solved is True for r in has.values()),
                       failed=sum(r.solved is False for r in has.values()))
            if by:
                row["per"] = dict(Counter(str(r.labels.get(by)) for r in has.values()))
        out["unclustered"] = {"turns": none["turns"], "seconds": round(none["seconds"], 1)}
        out["rows"] = sorted(rows, key=lambda x: (-x["seconds"], x["name"]))
        return out

    def _embedded_by(self, part: str) -> str | None:
        """The embedder of the members' tag vectors for a part, None for a group with no members.
        Fails if a member is not tagged, if its tags for the part, as they are now, are not
        embedded, or if the members' vectors are by more than one embedder."""
        if untagged := sum(r.tags is None for r in self.rollouts):
            raise ValueError(f"{untagged} of the group's {len(self)} rollouts are not tagged: tag them first, with group.tag()")
        held = [r._part(part, rows=False) for r in self.rollouts]
        if lacking := sum(h is None for h in held):
            raise ValueError(f"{lacking} of the group's {len(self)} rollouts have no vectors of their {part} tags as they are now: "
                             f"embed them first, with group.embed()")
        if len(by := {h[0] for h in held}) > 1:
            raise ValueError(f"the group's {part} tags are embedded by {len(by)} embedders ({', '.join(sorted(by))}): their vectors do not compare")
        return next(iter(by), None)

    def _ref(self, rollout: Rollout, place: int) -> str:
        """How a layer of this group names a member: its trajectory's file, from the group's folder
        if it has one, or its place in the group if the rollout has no file."""
        if rollout.path is None:
            return f"#{place}"
        return os.path.relpath(rollout.path.resolve(), self.path.resolve()) if self.path else str(rollout.path)

    def _members(self) -> list[Rollout | None]:
        """The members the group's analysis read, in its order; None for one the group no longer has."""
        mine = {self._ref(r, i): r for i, r in enumerate(self.rollouts)}
        return [mine.get(x["rollout"]) for x in self.mistakes["rollouts"]]

    def _sub(self, label: str) -> dict[str, Group]:
        """The sub-groups by a label, by the value as text: the ones an analysis was just made on, or as split now."""
        return self._subs.get(label) or {str(v): g for v, g in self.by(label).items()}

    def _current(self) -> bool:
        """Whether the group's analysis is of its rollouts' mistakes as they are now: the same
        members, each with the mistakes the analysis read; and so for each sub-group, if it is by a label."""
        layer = self.mistakes
        if layer.get("by"):
            subs = self._sub(layer["by"])
            read = {x["group"]: x["kinds"] for x in layer["groups"]}
            return set(read) == set(subs) and all(
                g.mistakes is not None and "by" not in g.mistakes and analysis.digest(g.mistakes["kinds"]) == read[v] and g._current()
                for v, g in subs.items())
        members = self._members()
        return len(members) == len(self.rollouts) and all(
            r is not None and r.mistakes is not None and analysis.digest(r.mistakes["mistakes"]) == x["flagged"]
            for r, x in zip(members, layer["rollouts"]))

    def _held(self, kind: dict) -> list[tuple[Rollout, int]]:
        """The flagged mistakes a kind of the group's analysis holds, each as its rollout and its
        place among that rollout's mistakes, through the sub-groups' kinds if it points at those."""
        layer = self.mistakes
        if layer.get("by"):
            subs = self._sub(layer["by"])
            return [x for ref in kind["members"] for x in subs[ref["group"]]._held(subs[ref["group"]].mistakes["kinds"][ref["kind"]])]
        members = self._members()
        return [(members[m["rollout"]], m["mistake"]) for m in kind["members"]]

    def _each(self, todo: list[Rollout], work: Callable[[Rollout], float | None], workers: int) -> Batch:
        doing = {id(r) for r in todo}
        made, failed, cost = _each(todo, work, workers)
        return Batch(made=made, skipped=[r for r in self.rollouts if id(r) not in doing], failed=failed, cost_usd=cost)


def _each(todo: list, work: Callable[[Any], float | None], workers: int) -> tuple[list, list[tuple[Any, Exception]], float]:
    """Do `work`, which says what it cost, to each of `todo`, `workers` at a time: the ones done,
    the ones that failed and why, and what it all cost."""
    made, failed, cost = [], [], 0.0
    with ThreadPoolExecutor(max(1, min(workers, len(todo) or 1))) as pool:
        futures = [(x, pool.submit(work, x)) for x in todo]
        for x, f in futures:
            try:
                spent = f.result()
            except Exception as e:  # noqa: BLE001 - one failure is reported, the others go on
                failed.append((x, e))
                continue
            made.append(x)
            cost += spent if isinstance(spent, (int, float)) else 0.0
    return made, failed, cost


def _may_write(todo: list[Rollout], save: bool) -> None:
    if save and (without := [r for r in todo if r.path is None]):
        raise ValueError(f"{len(without)} of the rollouts have no file to save beside: pass save=False, or load them from files")
