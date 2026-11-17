"""A rollout through its interface: flagging, tagging and embedding with readers and an embedder
that cost nothing, what is saved beside the trajectory, and what is refused before anything is spent."""

import json
import re

import numpy as np
import pytest

from fathom import Rollout, embedding
from fathom.reading import render


def step(i, at, source="agent", **more):
    return {"step_id": i, "timestamp": f"2026-11-16T10:00:{at:02d}Z", "source": source, **more}


def call(i, name, command, result):
    return {"tool_calls": [{"tool_call_id": f"c{i}", "function_name": name, "arguments": {"command": command}}],
            "observation": {"results": [{"source_call_id": f"c{i}", "content": result}]}}


TRAJECTORY = {
    "schema_version": "ATIF-v1.7",
    "agent": {"name": "claude-code", "model_name": "some-model"},
    "steps": [
        step(1, 0, "user", message="Fix the bug in parse()."),
        step(2, 5, reasoning_content="Look at the file first.", message="", metrics={"prompt_tokens": 100, "completion_tokens": 10},
             **call(2, "Bash", "cat parse.py", "def parse(): ...")),
        step(3, 6, message="sub-agent work", extra={"is_sidechain": True}),
        step(4, 15, message="I'll edit it.", metrics={"prompt_tokens": 150, "completion_tokens": 20},
             **call(4, "Edit", "parse.py", "ok")),
        step(5, 20, message="Done.", metrics={"prompt_tokens": 180, "completion_tokens": 5}),
    ],
}
ANSWER = {"calls": 2, "unsure": "nothing", "mistakes": [
    {"severity": "critical", "cause": "agent", "steps": [[4, 5]], "what": "Edited the wrong branch.", "evidence": "assert parse() == 1"},
    {"severity": "minor", "cause": "environment", "steps": [[2, 2]], "what": "Read the whole file."},
    {"severity": "odd", "cause": "agent", "steps": [[2, 2]], "what": "Not a severity."},
]}


class Reader:
    """A reader that answers ANSWER and keeps what it was asked."""

    name = "fake"

    def __init__(self):
        self.asked = []

    def __call__(self, prompt, schema, files, dirs):
        self.asked.append({"prompt": prompt, "files": files, "dirs": dirs})
        return ANSWER, {"model": "fake-1", "cost_usd": 0.0}


def tag_all(prompt, schema, files, dirs, said=""):
    """A reader that tags every part of every turn by its step; a rollout keeps the tags of the parts a turn has."""
    steps = [int(s) for s in re.findall(r'^<turn step="(\d+)">$', files["transcript.txt"], re.M)]
    return {"turns": [{"step": s, "thought": f"thinks{said} at {s}", "call": f"calls{said} at {s}", "result": f"gets{said} at {s}"} for s in steps]}, {}


class Embedder:
    """An embedder whose vectors say how long each text is, and which counts the texts it got."""

    def __init__(self, name="fake-embedder"):
        self.name, self.texts, self.spent = name, [], 0.0

    def __call__(self, texts):
        self.texts += texts
        self.spent += 0.5
        return [[float(len(t)), 1.0, 0.0] for t in texts]


@pytest.fixture
def trial(tmp_path):
    """A Harbor trial folder: a trajectory, a failed result, a verifier's output and one skill."""
    (tmp_path / "agent" / "sessions" / "skills" / "a-skill").mkdir(parents=True)
    (tmp_path / "verifier").mkdir()
    (tmp_path / "agent" / "trajectory.json").write_text(json.dumps(TRAJECTORY))
    (tmp_path / "verifier" / "test-stdout.txt").write_text("assert parse() == 1")
    (tmp_path / "result.json").write_text(json.dumps({"task_name": "set/parse-bug", "verifier_result": {"rewards": {"reward": 0.0}}}))
    return tmp_path


def test_a_trial_folder_gives_the_rollout_what_is_known_of_it(trial):
    r = Rollout.from_trial(trial, labels={"step": 3})
    assert r.solved is False and r.reward == 0.0
    assert r.verifier == trial / "verifier" and r.offered == [trial / "agent" / "sessions" / "skills"]
    assert r.labels == {"trial": trial.name, "task": "set/parse-bug", "step": 3}
    assert r.mistakes is None and r.vectors is None


def test_turns_are_the_main_agents_steps_with_their_time_and_tokens(trial):
    turns = Rollout.from_trial(trial).turns
    assert [t.id for t in turns] == [2, 4, 5]  # not the user's step, nor the sub-agent's
    assert [t.seconds for t in turns] == [10.0, 5.0, 0.0]  # to the next turn's reply; the last ends the trajectory
    assert [(t.sampled, t.context) for t in turns] == [(10, 100), (20, 150), (5, 180)]
    assert all(t.mistakes == () and t.vectors == {} for t in turns)


def test_a_turn_has_its_own_text_by_part():
    turns = Rollout(TRAJECTORY).turns
    assert turns[0].text == {"thought": "Look at the file first.", "call": "Bash: cat parse.py", "result": "def parse(): ..."}
    assert turns[2].text == {"thought": "Done."}  # a part the turn lacks is not there


def test_flagging_saves_the_layer_beside_the_trajectory_and_puts_mistakes_on_their_turns(trial):
    reader = Reader()
    r = Rollout.from_trial(trial)
    layer = r.flag_mistakes(reader)
    assert layer is r.mistakes and layer["reader"] == "fake" and layer["model"] == "fake-1" and layer["at"]
    assert [m["severity"] for m in layer["mistakes"]] == ["critical", "minor"]  # the one with no severity is dropped
    assert json.loads((trial / "agent" / "trajectory.mistakes.json").read_text()) == layer
    assert json.loads((trial / "agent" / "trajectory.json").read_text()) == TRAJECTORY  # the trajectory is untouched
    by = {t.id: [m["what"] for m in t.mistakes] for t in r.turns}
    assert by == {2: ["Read the whole file."], 4: ["Edited the wrong branch."], 5: ["Edited the wrong branch."]}
    again = Rollout.from_trial(trial)  # read back from the folder
    assert again.mistakes == layer and [len(t.mistakes) for t in again.turns] == [1, 1, 1]


def test_the_reader_is_told_the_outcome_and_given_the_verifier_and_the_skills(trial):
    reader = Reader()
    Rollout.from_trial(trial).flag_mistakes(reader)
    asked = reader.asked[0]
    assert "got the task set/parse-bug and failed it" in asked["prompt"]
    assert str(trial / "verifier") in asked["prompt"] and "test-stdout.txt" in asked["prompt"]
    assert str(trial / "agent" / "sessions" / "skills") in asked["prompt"]
    assert asked["dirs"] == [trial / "agent", trial / "verifier", trial / "agent" / "sessions" / "skills"]
    assert "[1] USER: Fix the bug in parse()." in asked["files"]["transcript.txt"]


def test_flagging_again_fails_before_the_reader_is_asked_unless_overwrite(trial):
    reader = Reader()
    r = Rollout.from_trial(trial)
    r.flag_mistakes(reader)
    with pytest.raises(FileExistsError, match="overwrite=True"):
        r.flag_mistakes(reader)
    with pytest.raises(FileExistsError):
        Rollout.from_trial(trial).flag_mistakes(reader)  # a layer on disk counts as much as one in memory
    assert len(reader.asked) == 1
    first = r.mistakes["at"]
    r.flag_mistakes(reader, overwrite=True)
    assert len(reader.asked) == 2 and r.mistakes["at"] >= first
    assert not list((trial / "agent").glob("*.bak")) and not list((trial / "agent").glob("*.tmp"))


def test_a_rollout_in_memory_cannot_save_and_fails_before_the_reader_is_asked(tmp_path):
    reader = Reader()
    r = Rollout(TRAJECTORY, reward=1.0)
    with pytest.raises(ValueError, match="save=False"):
        r.flag_mistakes(reader)
    assert reader.asked == []
    layer = r.flag_mistakes(reader, save=False)
    assert r.mistakes is layer and "solved it" in reader.asked[0]["prompt"]
    assert "trajectory.json" in reader.asked[0]["files"] and reader.asked[0]["dirs"] == []  # it has no folder to read in
    with pytest.raises(FileExistsError):
        r.flag_mistakes(reader, save=False)  # the layer in memory is not replaced quietly either
    r.save(tmp_path / "kept" / "run.json")
    back = Rollout.load(tmp_path / "kept" / "run.json")
    assert back.trajectory == TRAJECTORY and back.mistakes == layer


def test_embedding_gives_each_turn_a_vector_for_each_tag_it_has(trial):
    embedder = Embedder()
    r = Rollout.from_trial(trial)
    with pytest.raises(ValueError, match="not tagged"):
        r.embed(embedder=embedder)
    assert embedder.texts == []
    r.tag(tag_all)
    layer = r.embed(embedder=embedder)
    assert (layer["embedder"], layer["of"], layer["steps"], layer["cost_usd"]) == ("fake-embedder", "tags", [2, 4, 5], 1.5)
    assert set(layer["parts"]) == {"thought", "call", "result"} and layer["parts"]["thought"].shape == (3, 3)
    assert sorted(embedder.texts) == sorted(["thinks at 2", "thinks at 4", "thinks at 5", "calls at 2", "calls at 4", "gets at 2", "gets at 4"])
    turns = r.turns  # it is the tags that were embedded: the last turn calls nothing, and has no tag for it
    assert [sorted(t.vectors) for t in turns] == [["call", "result", "thought"], ["call", "result", "thought"], ["thought"]]
    assert np.allclose(np.linalg.norm(turns[0].vectors["thought"].astype(float)), 1.0, atol=1e-3)  # rows are of unit length
    back = Rollout.from_trial(trial)  # read back from the folder
    assert (trial / "agent" / "trajectory.vectors.npz").exists() and back.vectors["embedder"] == "fake-embedder"
    assert np.array_equal(back.turns[1].vectors["call"], turns[1].vectors["call"])
    bare = Rollout(TRAJECTORY)  # a part no turn has a tag for is embedded as nothing, and does not fail
    bare.tag(lambda *a: ({"turns": [{"step": 2, "thought": "thinks"}]}, {}), save=False)
    assert bare.embed(embedder=Embedder(), save=False)["parts"]["call"].shape == (3, 0) and [sorted(t.vectors) for t in bare.turns] == [["thought"], [], []]


def test_a_missing_part_is_added_and_a_part_that_is_there_is_not_replaced(trial):
    embedder = Embedder()
    r = Rollout.from_trial(trial)
    r.tag(tag_all)
    r.embed(parts=["thought"], embedder=embedder)
    asked = len(embedder.texts)
    r.embed(parts=["call"], embedder=embedder)  # not an overwrite: the part was missing
    assert set(Rollout.from_trial(trial).vectors["parts"]) == {"thought", "call"}
    with pytest.raises(FileExistsError, match="thought"):
        r.embed(parts=["thought", "result"], embedder=embedder)
    assert len(embedder.texts) == asked + 2  # nothing was embedded for the call that failed
    r.embed(parts=["thought", "result"], embedder=embedder, overwrite=True)
    assert set(r.vectors["parts"]) == {"thought", "call", "result"}


def test_vectors_of_tags_since_replaced_are_not_read_and_are_embedded_again(trial):
    embedder = Embedder()
    r = Rollout.from_trial(trial)
    r.tag(tag_all)
    r.embed(embedder=embedder)
    r.tag(lambda *a: tag_all(*a, said=" again"), overwrite=True)
    back = Rollout.from_trial(trial)
    assert back.turns[0].tags["thought"] == "thinks again at 2" and back.turns[0].vectors == {}  # the vectors are of the old tags
    asked = len(embedder.texts)
    back.embed(parts=["thought"], embedder=embedder)  # not an overwrite: these tags were never embedded
    assert embedder.texts[asked:] == ["thinks again at 2", "thinks again at 4", "thinks again at 5"]
    assert [sorted(t.vectors) for t in Rollout.from_trial(trial).turns] == [["thought"]] * 3 and set(back.vectors["parts"]) == {"thought"}


def test_another_embedders_vectors_and_vectors_of_the_turns_own_text_do_not_mix(trial):
    r = Rollout.from_trial(trial)
    r.tag(tag_all)
    r.embed(parts=["thought"], embedder=Embedder())
    other = Embedder("another")
    with pytest.raises(FileExistsError, match="fake-embedder"):
        r.embed(parts=["call"], embedder=other)
    assert other.texts == []
    r.embed(parts=["call"], embedder=other, overwrite=True)
    assert r.vectors["embedder"] == "another" and set(r.vectors["parts"]) == {"call"}  # the old ones are gone
    embedding.save(trial / "agent" / "trajectory.vectors.npz", {  # a layer as it was made before tags: of the turns' own text
        "embedder": "another", "steps": [2, 4, 5], "parts": {"thought": np.ones((3, 3), np.float16)}, "has": {"thought": np.ones(3, bool)}})
    old = Rollout.from_trial(trial)
    assert old.vectors["of"] == "text" and old.turns[0].vectors == {}
    with pytest.raises(FileExistsError, match="own text"):
        old.embed(parts=["call"], embedder=other)
    old.embed(parts=["call"], embedder=other, overwrite=True)
    assert old.vectors["of"] == "tags" and set(Rollout.from_trial(trial).vectors["parts"]) == {"call"}


def test_an_unknown_part_and_a_rollout_with_no_file_are_refused_before_embedding(trial):
    embedder = Embedder()
    with pytest.raises(ValueError, match="no part"):
        Rollout.from_trial(trial).embed(parts=["reply"], embedder=embedder)
    with pytest.raises(ValueError, match="save=False"):
        Rollout(TRAJECTORY).embed(embedder=embedder)
    assert embedder.texts == []


def test_the_transcript_keeps_what_the_agent_was_told_whole_and_cuts_long_outputs():
    told, out = "x" * 5000, "y" * 5000
    text = render({"steps": [step(1, 0, "user", message=told), step(2, 1, message="", **call(2, "Bash", "ls", out))]})
    assert told in text and out not in text and "[...4300 chars...]" in text
