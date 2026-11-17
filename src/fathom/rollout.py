"""A rollout: one agent's attempt at one task, as an ATIF trajectory, and what analysing it added.

    rollout = Rollout.from_trial(trial)   # a Harbor trial folder; Rollout.load(file) for any ATIF file
    rollout.flag_mistakes()               # a reader flags its mistakes
    rollout.tag()                         # a reader says in a few words what each part of each turn does
    rollout.embed()                       # an embedder turns each of those tags into a vector
    rollout.turns                         # each turn with its time, tokens, mistakes, vectors and tags

What analysing adds is kept in layers beside the trajectory file, never in it, joined to it by step
id: `<name>.mistakes.json`, `<name>.vectors.npz` and `<name>.tags.json` next to `<name>.json`. A layer costs money to
make, so it is saved as soon as it is made, and one that is there is never replaced unless asked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections.abc import Iterable, Sequence
from typing import Any

from fathom import embedding, reading, tagging
from fathom.analysis import digest

PARTS = embedding.PARTS
_UNREAD: Any = object()


@dataclass(frozen=True)
class Turn:
    """One turn of the main agent: the ATIF step it took, and what is known of it."""

    step: dict  # as in the trajectory
    seconds: float  # from its reply to the next turn's, or to the trajectory's end: its tools, then the next model call
    sampled: int  # tokens the model wrote
    context: int  # tokens the model was given
    text: dict[str, str] = field(default_factory=dict)  # its own text by part: what it thought and said, its calls, what came back (long results cut)
    mistakes: tuple[dict, ...] = ()  # the flagged mistakes that cover it
    tags: dict[str, str] = field(default_factory=dict)  # its tag by part, for the parts tagged
    vectors: dict[str, Any] = field(default_factory=dict)  # the vector of its tag, by part, for the tags embedded

    @property
    def id(self) -> int:
        return self.step["step_id"]


class Rollout:
    """An ATIF trajectory with what else is known of the attempt: the `reward` it got, the folder
    of the `verifier`'s output, the folders of what the agent was `offered` (skills), and `labels`
    (task, step, run), which are carried and never interpreted, but for `task`, which the reader
    is told."""

    def __init__(self, trajectory: dict, *, reward: float | None = None, verifier: str | Path | None = None,
                 offered: Iterable[str | Path] = (), labels: dict | None = None) -> None:
        self._trajectory: dict | None = trajectory
        self.reward = reward
        self.verifier = Path(verifier) if verifier else None
        self.offered = [Path(p) for p in offered]
        self.labels = dict(labels or {})
        self.path: Path | None = None  # the trajectory's file, when it has one
        self.mistakes: dict | None = None  # the layer `flag_mistakes` makes
        self._vectors: Any = None
        self._tags: Any = None

    @classmethod
    def load(cls, path: str | Path, **known: Any) -> Rollout:
        """The rollout of an ATIF file, with the layers saved beside it. `known` as for `Rollout`."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"no trajectory at {path}")
        rollout = cls({}, **known)
        rollout._trajectory = None  # read from the file when asked for, and not kept: a run's trajectories are gigabytes
        rollout.path = path
        if (file := rollout._beside("mistakes.json")).exists():
            rollout.mistakes = json.loads(file.read_text())
        rollout._vectors = _UNREAD  # read when asked for: it needs numpy
        rollout._tags = _UNREAD  # read when asked for: a run's tags are many
        return rollout

    @classmethod
    def from_trial(cls, trial: str | Path, labels: dict | None = None) -> Rollout:
        """The rollout of a Harbor trial folder: its trajectory, the reward in its result, its
        verifier's folder and the skills its agent was offered. It is labelled with its trial and
        its task; `labels` adds to those."""
        trial = Path(trial)
        result = _json(trial / "result.json") or {}
        rewards = (result.get("verifier_result") or {}).get("rewards") or {}
        skills = trial / "agent" / "sessions" / "skills"
        return cls.load(trial / "agent" / "trajectory.json",
                        reward=rewards.get("reward", next(iter(rewards.values()), None)),
                        verifier=trial / "verifier" if (trial / "verifier").is_dir() else None,
                        offered=[skills] if skills.is_dir() else [],
                        labels={"trial": trial.name, "task": _task(trial, result), **(labels or {})})

    @property
    def trajectory(self) -> dict:
        """The ATIF trajectory, as its harness wrote it."""
        return self._trajectory if self._trajectory is not None else json.loads(self.path.read_text())

    @property
    def solved(self) -> bool | None:
        """Whether the attempt solved its task; None when it has no reward."""
        return None if not isinstance(self.reward, (int, float)) else self.reward >= 1

    @property
    def failed_by(self) -> str | None:
        """Whose failure a failed rollout is, by its flagged mistakes: the agent's when a critical
        mistake is the agent's, else the environment's when one is, else the verifier's, which
        then failed it alone. None for a rollout that did not fail, or with no critical mistake
        flagged."""
        if self.solved is not False:
            return None
        causes = {m.get("cause") for m in (self.mistakes or {}).get("mistakes") or [] if m.get("severity") == "critical"}
        return next((c for c in reading.CAUSES if c in causes), None)

    @property
    def vectors(self) -> dict | None:
        """The layer `embed` makes: the embedder's name, the turns' step ids, and for each part
        embedded a matrix with a row per turn, which turns have a tag for that part, and a mark of
        the tags it was embedded from. `of` says what was embedded: "tags", or "text" for a layer
        made from the turns' own text, which nothing here reads."""
        if self._vectors is _UNREAD:
            file = self._beside("vectors.npz")
            self._vectors = embedding.load(file) if file.exists() else None
        return self._vectors

    @property
    def tags(self) -> dict | None:
        """The layer `tag` makes: the reader, when and at what cost, the turns' step ids, and for
        each part a tag per turn, None where the turn has no such part."""
        if self._tags is _UNREAD:
            self._tags = _json(self._beside("tags.json"))
        return self._tags

    def flag_mistakes(self, reader: str | reading.Reader = "sonnet", *, save: bool = True, overwrite: bool = False) -> dict:
        """Have `reader` flag the rollout's mistakes, each with a severity, a cause and the turns it
        covers. Returns the layer, keeps it as `mistakes`, and saves it beside the trajectory unless
        `save` is off. Fails, before anything is spent, if the mistakes are flagged already and
        `overwrite` is off, or if `save` is on and the rollout has no file."""
        self._may_write(save)
        if self.mistakes is not None and not overwrite:
            raise FileExistsError(f"the mistakes of {self._name()} are flagged already: pass overwrite=True to flag them again")
        layer = reading.read(self, reader)
        self.mistakes = layer
        if save:
            _replace(self._beside("mistakes.json"), json.dumps(layer, indent=1))
        return layer

    def embed(self, parts: Sequence[str] = PARTS, embedder: str | embedding.Embedder | None = None, *,
              save: bool = True, overwrite: bool = False) -> dict:
        """Embed the tag of each turn's `parts` (its thought, its call, its result) with `embedder`.
        Returns the layer, keeps it as `vectors`, and saves it beside the trajectory unless `save`
        is off. A part not embedded yet is added to the layer, and so is one whose tags have
        changed since. Fails, before anything is spent, if the turns are not tagged; if a part
        asked for is there already, the layer is by another embedder, whose vectors would not
        compare, or it is of the turns' own text, and `overwrite` is off; or if `save` is on and
        the rollout has no file."""
        if unknown := [p for p in parts if p not in PARTS]:
            raise ValueError(f"no part {unknown[0]}: a turn's parts are {', '.join(PARTS)}")
        self._may_write(save)
        if self.tags is None:
            raise ValueError(f"the turns of {self._name()} are not tagged: tag them first, with tag()")
        embedder = embedding.resolve(embedder)
        had = self.vectors
        same = had is not None and had.get("of") == "tags" and had["embedder"] == embedding.name(embedder)
        fresh = self._fresh(had) if same else []
        if had is not None and not overwrite:
            if had.get("of") != "tags":
                raise FileExistsError(f"{self._name()} holds vectors of its turns' own text, not of their tags: pass overwrite=True to replace them")
            if not same:
                raise FileExistsError(f"the tags of {self._name()} are embedded already, by {had['embedder']}: pass overwrite=True to embed them again")
            if there := [p for p in parts if p in fresh]:
                raise FileExistsError(f"the tags of {self._name()} are embedded already, with their {', '.join(there)}: "
                                      "pass overwrite=True to embed them again")
        keep = {**had, **{k: {p: had[k][p] for p in fresh} for k in ("parts", "has", "tagged")}} if same else None
        layer = embedding.embed(self.tags["steps"], self.tags["parts"], parts, embedder, keep)
        self._vectors = layer
        if save:
            embedding.save(self._beside("vectors.npz"), layer)
        return layer

    def tag(self, reader: str | reading.Reader | None = None, *, save: bool = True, overwrite: bool = False) -> dict:
        """Have `reader` say in a few words what each part of each turn does: what the agent meant,
        what its call did, what came back. With no reader named it is Claude Haiku through the
        Claude API. Returns the layer, keeps it as `tags`, and saves it beside the trajectory
        unless `save` is off. Fails, before anything is spent, if the turns are tagged already and
        `overwrite` is off, or if `save` is on and the rollout has no file."""
        self._may_write(save)
        if self.tags is not None and not overwrite:
            raise FileExistsError(f"the turns of {self._name()} are tagged already: pass overwrite=True to tag them again")
        return self._tagged(tagging.tag(self, reader), save)

    def _tagged(self, layer: dict, save: bool) -> dict:
        """Take a tags layer as the rollout's, and save it beside the trajectory if asked."""
        self._tags = layer
        if save:
            _replace(self._beside("tags.json"), json.dumps(layer))
        return layer

    @property
    def turns(self) -> list[Turn]:
        """The enriched trajectory: the main agent's turns in order, each with its own text by part,
        its time and tokens and, where the layers have them, the mistakes that cover it, its tags
        and their vectors."""
        trajectory = self.trajectory
        _, by = embedding.texts(trajectory)  # in the order of the main agent's steps, as `clock` is
        covering = self._covering()
        t = self.tags
        at = {s: i for i, s in enumerate(t["steps"])} if t else {}
        v = self.vectors
        fresh = self._fresh(v)  # the parts embedded from the tags as they are now
        row = {s: i for i, s in enumerate(v["steps"])} if fresh else {}
        out = []
        for i, (st, seconds) in enumerate(clock(trajectory)):
            sid = int(st["step_id"])
            metrics = st.get("metrics") or {}
            out.append(Turn(
                step=st,
                text={p: by[p][i] for p in PARTS if by[p][i]},
                seconds=seconds,
                sampled=int(metrics.get("completion_tokens") or 0),
                context=int(metrics.get("prompt_tokens") or 0),
                mistakes=tuple(covering.get(sid, ())),
                tags={p: tags[at[sid]] for p, tags in t["parts"].items() if tags[at[sid]]} if sid in at else {},
                vectors={p: v["parts"][p][row[sid]] for p in fresh if v["has"][p][row[sid]]} if sid in row else {},
            ))
        return out

    def _covering(self) -> dict[int, list[dict]]:
        """The flagged mistakes that cover each step, by step id."""
        out: dict[int, list[dict]] = {}
        for m in (self.mistakes or {}).get("mistakes") or []:
            for s in covered(m):
                out.setdefault(s, []).append(m)
        return out

    def _embedded(self) -> dict | None:
        """The vectors layer without its matrices, read no further than that: its embedder, its
        cost, what it is of, the marks of the tags it was made from, and the names of its parts."""
        if self._vectors is _UNREAD:
            file = self._beside("vectors.npz")
            return embedding.peek(file) if file.exists() else None
        v = self._vectors
        return v and {**{k: v.get(k) for k in ("embedder", "at", "cost_usd", "tagged")}, "of": v.get("of", "text"), "parts": list(v["parts"])}

    def _fresh(self, layer: dict | None) -> list[str]:
        """The parts of a vectors layer, whole or without its matrices, that were embedded from the
        rollout's tags as they are now."""
        if layer is None or layer.get("of") != "tags" or self.tags is None:
            return []
        return [p for p in layer["parts"] if (layer.get("tagged") or {}).get(p) == digest(self.tags["parts"][p])]

    def _part(self, part: str, rows: bool = True) -> tuple[str, list[int], Any] | None:
        """The vectors of the tags of one part, without reading or keeping the others: their
        embedder, the step ids of the turns tagged for the part and, unless `rows` is off, their
        vectors. None if the part's tags, as they are now, are not embedded."""
        if self._vectors is _UNREAD:
            file = self._beside("vectors.npz")
            got = embedding.load_part(file, part, rows) if file.exists() else None
            if got is None or part not in self._fresh({**got[0], "parts": [part]}):
                return None
            return got[0]["embedder"], got[1], got[2]
        v = self._vectors
        if part not in self._fresh(v):
            return None
        has = v["has"][part]
        return v["embedder"], [s for s, h in zip(v["steps"], has) if h], v["parts"][part][has] if rows else None

    def _release(self) -> None:
        """Let go of the vectors held in memory, if they are saved: they are read again when asked for."""
        if self.path and self._beside("vectors.npz").exists():
            self._vectors = _UNREAD

    def save(self, path: str | Path | None = None, *, overwrite: bool = False) -> None:
        """Save the layers the rollout holds beside its trajectory file, or beside `path`, which
        becomes its file; the trajectory is written there if no file is. Fails if a layer is saved
        there already and `overwrite` is off."""
        if path is None and self.path is None:
            raise ValueError("this rollout has no file: say where to save it")
        target = Path(path) if path else self.path
        files = {"mistakes.json": self.mistakes, "vectors.npz": self.vectors, "tags.json": self.tags}
        taken = [f for name, layer in files.items() if layer is not None and (f := _beside(target, name)).exists()]
        if taken and not overwrite:
            raise FileExistsError(f"{taken[0]} is there already: pass overwrite=True to replace it")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            _replace(target, json.dumps(self.trajectory))
        if self.mistakes is not None:
            _replace(_beside(target, "mistakes.json"), json.dumps(self.mistakes, indent=1))
        if self.vectors is not None:
            embedding.save(_beside(target, "vectors.npz"), self.vectors)
        if self.tags is not None:
            _replace(_beside(target, "tags.json"), json.dumps(self.tags))
        self.path = target

    def _beside(self, name: str) -> Path:
        return _beside(self.path, name)

    def _may_write(self, save: bool) -> None:
        if save and self.path is None:
            raise ValueError("this rollout has no file to save beside: pass save=False and save it later, or load it from a file")

    def _name(self) -> str:
        return str(self.path) if self.path else "this rollout"


def clock(trajectory: dict) -> list[tuple[dict, float]]:
    """The main agent's turns in order, each with its seconds: from its reply to the next turn's,
    or to the trajectory's end. ATIF gives a step one timestamp, so that is a turn's tools and
    then the next model call."""
    steps = trajectory.get("steps") or []
    main = list(embedding.main_steps(trajectory))
    end = _at(steps[-1]) if steps else None
    out = []
    for i, st in enumerate(main):
        a, b = _at(st), (_at(main[i + 1]) if i + 1 < len(main) else end)
        out.append((st, round(b - a, 1) if a is not None and b is not None and b >= a else 0.0))
    return out


def covered(mistake: dict) -> set[int]:
    """The step ids a flagged mistake covers: its `steps` are ranges, `[[first, last], ...]`."""
    spans = mistake.get("steps") or []
    if spans and not isinstance(spans[0], list):  # one range, written flat
        spans = [spans]
    return {s for r in spans if r for s in range(int(r[0]), int(r[-1]) + 1)}


def _beside(path: Path, name: str) -> Path:
    return path.with_name(f"{path.stem}.{name}")


def _replace(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _at(step: dict) -> float | None:
    try:
        return datetime.fromisoformat(str(step.get("timestamp")).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _task(trial: Path, result: dict) -> str:
    """The trial's task: its name in Harbor, unless that name does not say which task it is, when
    it is the name of the task's folder. Some datasets give all their tasks one name."""
    name = result.get("task_name") or ""
    given = result.get("task_id")
    folder = Path(given["path"]).name if isinstance(given, dict) and given.get("path") else ""
    if folder and name != folder and not name.endswith("/" + folder):
        return folder
    return name or trial.name.rsplit("__", 1)[0]
