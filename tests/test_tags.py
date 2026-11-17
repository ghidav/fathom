"""Tagging through the interface: a rollout's turns and a group's members, with a reader that
costs nothing, and what the Claude API reader refuses before it asks anything."""

import json
import re

import pytest

from fathom import ClaudeAPI, Group, Rollout
from test_group import job, make  # noqa: F401 - the job fixture
from test_rollout import TRAJECTORY, trial  # noqa: F401 - the trial fixture


class Tagger:
    """A reader that tags every part it is asked about by its step, and one part no turn has."""

    name = "fake-tagger"

    def __init__(self):
        self.asked = []

    def __call__(self, prompt, schema, files, dirs):
        self.asked.append({"prompt": prompt, "files": files, "dirs": dirs})
        steps = [int(s) for s in re.findall(r'^<turn step="(\d+)">$', files["transcript.txt"], re.M)]
        turns = [{"step": s, "thought": f"thinks at {s}", "call": f"calls   at {s}", "result": f"gets at {s}"} for s in steps]
        turns[-1]["call"] = "None"  # what a reader writes when it has nothing to say
        return {"turns": turns + [{"step": 99, "thought": "no such turn"}]}, {"model": "fake-1", "cost_usd": 0.25}


def test_tagging_saves_the_layer_and_gives_each_turn_a_tag_for_each_part_it_has(trial):
    reader = Tagger()
    r = Rollout.from_trial(trial)
    layer = r.tag(reader)
    assert layer["reader"] == "fake-tagger" and layer["cost_usd"] == 0.25 and layer["steps"] == [2, 4, 5]
    asked = reader.asked[0]
    assert "set/parse-bug" in asked["prompt"] and asked["dirs"] == []  # it needs no folder
    assert "Fix the bug in parse()." in asked["files"]["transcript.txt"] and "sub-agent work" not in asked["files"]["transcript.txt"]
    turns = Rollout.from_trial(trial).turns  # read back from beside the trajectory
    assert turns[0].tags == {"thought": "thinks at 2", "call": "calls at 2", "result": "gets at 2"}
    assert turns[2].tags == {"thought": "thinks at 5"}  # the last turn called nothing: a tag for a part it lacks is dropped
    assert Rollout(TRAJECTORY).tag(lambda *a: ({"turns": [{"step": 4, "call": "None", "result": "edits"}]}, {}), save=False)["parts"] == {
        "thought": [None, None, None], "call": [None, None, None], "result": [None, "edits", None]}  # "none" is no tag
    assert json.loads((trial / "agent" / "trajectory.json").read_text()) == TRAJECTORY  # the trajectory is not touched


def test_a_reader_that_leaves_a_whole_part_out_is_asked_once_more_for_it(trial):
    asked = []

    def reader(prompt, schema, files, dirs):  # no result the first time, and other words for the calls the second
        asked.append(prompt)
        first = len(asked) == 1
        return {"turns": [{"step": s, "call": "calls" if first else "calls otherwise", **({} if first else {"result": "gets"})} for s in (2, 4)]}, {
            "cost_usd": 0.25, "tokens": {"input": 10, "output": 2}, "model": "fake-1"}

    layer = Rollout.from_trial(trial).tag(reader)
    assert len(asked) == 2 and layer["parts"]["result"] == ["gets", "gets", None]
    assert layer["parts"]["call"] == ["calls", "calls", None]  # what the first answer gave stays
    assert layer["parts"]["thought"] == [None, None, None]  # asked twice, never given: left as it is
    assert layer["cost_usd"] == 0.5 and layer["tokens"] == {"input": 20, "output": 4} and layer["model"] == "fake-1"


def test_tagging_again_fails_before_the_reader_is_asked_unless_overwrite(trial):
    reader = Tagger()
    Rollout.from_trial(trial).tag(reader)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        Rollout.from_trial(trial).tag(reader)
    with pytest.raises(ValueError, match="no file"):
        Rollout(TRAJECTORY).tag(reader)
    assert len(reader.asked) == 1
    Rollout.from_trial(trial).tag(reader, overwrite=True)
    assert len(reader.asked) == 2 and Rollout(TRAJECTORY).tag(reader, save=False)["steps"] == [2, 4, 5]


def test_tagging_a_group_tags_the_members_that_lack_it(job):
    reader = Tagger()
    g = Group.from_job(job)
    g.rollouts[0].tag(reader)
    done = g.tag(reader)
    assert [r.labels["trial"] for r in done.made] == ["parse__b", "split__c"] and len(done.skipped) == 1
    assert done.failed == [] and done.cost_usd == 0.5 and len(reader.asked) == 3
    assert all(r.tags for r in Group.from_job(job))
    assert g.tag(reader).made == [] and len(reader.asked) == 3


def test_the_api_reader_refuses_a_job_that_names_folders_before_it_asks(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ClaudeAPI()
    with pytest.raises(RuntimeError, match="cannot open folders"):
        ClaudeAPI(key="not-a-key")("read", {"type": "object"}, {}, [tmp_path])


class Topics:
    """An embedder that puts a text by the words it has: asking, testing, reading."""

    name = "topics"

    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        self.texts += texts
        return [[float("ask" in t), float("test" in t), float("read" in t), 0.01] for t in texts]

    spent = 0.0


def says(by_step):
    """A reader that tags the thought of each step as `by_step` says."""
    return lambda prompt, schema, files, dirs: ({"turns": [{"step": s, "thought": t} for s, t in by_step.items()]}, {})


def test_search_finds_the_turns_whose_tag_reads_like_the_description_best_first(job):  # noqa: F811
    g = Group.from_job(job)
    g.rollouts[0].tag(says({2: "reads the file", 4: "decides to ask the user", 5: "runs the tests"}))
    g.rollouts[1].tag(says({2: "asks the user which file", 4: "reads the file", 5: "reads the file"}))
    g.rollouts[2].tag(says({2: "runs the tests"}))
    embedder = Topics()
    with pytest.raises(ValueError, match="3 of the group's 3 rollouts have no vectors of their thought tags"):
        g.search("asks the user a question", embedder=embedder)
    assert g.embed(parts=["thought"], embedder=embedder).cost_usd == 0.0 and len(embedder.texts) == 7  # a tag of each tagged turn
    hits = g.search("asks the user a question", embedder=embedder, top=2)
    assert [(h.rollout.labels["trial"], h.turn.id, h.turn.tags["thought"]) for h in hits] == [
        ("parse__a", 4, "decides to ask the user"), ("parse__b", 2, "asks the user which file")]
    assert hits[0].score == pytest.approx(1.0, abs=1e-3) and hits[0].turn.seconds == 5.0
    assert len(embedder.texts) == 8  # the search embedded its description, and no tag
    assert [h.turn.id for h in g.search("runs the tests", embedder=embedder, top=None)][:2] == [5, 2] and len(embedder.texts) == 9
    assert len(g.search("reads", embedder=embedder, top=None)) == 7  # every turn with a tag for the part, and no other
    assert Group.from_job(job).search("asks the user", embedder=Topics(), top=1)[0].turn.id == 4  # the vectors are saved: a new group reads them
    other = Topics()
    other.name = "another"
    with pytest.raises(ValueError, match="would not compare"):
        g.search("reads", embedder=other)
    assert other.texts == []


def test_search_refuses_before_it_embeds_anything(job):  # noqa: F811
    g = Group.from_job(job)
    embedder = Topics()
    with pytest.raises(ValueError, match="not tagged"):
        g.search("asks the user", embedder=embedder)
    g.tag(Tagger())
    with pytest.raises(ValueError, match="embed them first"):
        g.search("asks the user", embedder=embedder)
    with pytest.raises(ValueError, match="no part"):
        g.search("asks the user", "message", embedder=embedder)
    with pytest.raises(ValueError, match="nothing to search"):
        g.search("  ", embedder=embedder)
    assert embedder.texts == []
