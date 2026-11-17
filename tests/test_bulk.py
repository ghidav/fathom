"""Tagging at the size of a run, through the interface: what it would cost before it is run, a
rollout too long for one call, and a batch that is picked up after being cut short. The reader
counts, prices and batches as `ClaudeAPI` does, and costs nothing."""

import json
import re

import pytest

import fathom.group
from fathom import Group, Rollout, tagging
from test_cluster import trajectory
from test_group import job, make  # noqa: F401 - the job fixture
from test_rollout import TRAJECTORY


class Bulk:
    """A reader that tags every part of the turns it is shown, counts a job by its length, and has a batch."""

    name = "bulk"

    def __init__(self, limit=None, wrong=()):
        self.limit, self.wrong = limit, wrong  # `wrong`: the ids of the batch jobs it fails to answer
        self.asked, self.handed, self.looks = [], [], 0

    def answer(self, files, parts=("thought", "call", "result")):
        steps = [int(s) for s in re.findall(r'^<turn step="(\d+)">$', files["transcript.txt"], re.M)]
        return {"turns": [{"step": s, **{p: f"{p} at {s}" for p in parts}} for s in steps]}

    def __call__(self, prompt, schema, files, dirs):
        self.asked.append(files["transcript.txt"])
        return self.answer(files), {"cost_usd": 0.5, "tokens": {"input": self.count(prompt, schema, files), "output": 14}}

    def count(self, prompt, schema, files):
        return (len(prompt) + len(files["transcript.txt"])) // 4

    def price(self, sent, written, batch=False):
        return (sent + written) / 1000 * (0.5 if batch else 1)

    def submit(self, jobs):
        self.handed.append(jobs)
        return [f"batch-{len(self.handed)}"]

    def collect(self, batches):
        self.looks += 1
        if self.looks == 1:
            return None  # still being worked on
        jobs = self.handed[int(batches[0].split("-")[1]) - 1]
        return {k: RuntimeError("the batch did not answer this job: errored") if k in self.wrong
                else (self.answer(files, ("thought",) if k in getattr(self, "thin", ()) else ("thought", "call", "result")), {"cost_usd": 0.25})
                for k, (prompt, schema, files) in jobs.items()}


def test_a_rollout_too_long_for_one_call_is_split_and_every_turn_is_still_tagged():
    r = Rollout(trajectory(0), labels={"task": "set/parse-bug"})
    whole = tagging.plan(r, Bulk())
    assert len(whole) == 1 and whole[0]["steps"] == list(range(2, 11)) and whole[0]["tokens"] > 0
    reader = Bulk(limit=int(whole[0]["tokens"] * 0.7))
    calls = tagging.plan(r, reader)
    assert len(calls) > 1 and all(c["tokens"] <= reader.limit for c in calls)
    assert [s for c in calls for s in c["steps"]] == list(range(2, 11))  # every turn once, in order
    layer = r.tag(reader, save=False)
    assert layer["parts"]["thought"] == [f"thought at {s}" for s in range(2, 11)] and len(reader.asked) == len(calls)
    assert layer["cost_usd"] == 0.5 * len(calls) and layer["tokens"]["output"] == 14 * len(calls)  # the calls' costs, added
    first = reader.asked[0]
    assert f"This is part 1 of {len(calls)}: steps 2 to {calls[0]['steps'][-1]}" in first and "<task>\nFix the bug.\n</task>" in first
    assert f'<turn step="{calls[1]["steps"][0]}">' not in first  # a call is shown its own turns only
    assert len(tagging.plan(r, lambda *a: ({}, {}))) == 1  # a reader that cannot count is asked in one call


def test_the_cost_of_tagging_is_told_without_tagging(job):  # noqa: F811
    reader = Bulk()
    g = Group.from_job(job)
    told = g.tag_cost(reader)
    sent = sum(reader.count(*tagging.job(r, [2, 4, 5])[:3]) for r in g)
    assert (told["rollouts"], told["calls"], told["split"], told["input_tokens"]) == (3, 3, 0, sent)
    assert (told["tags"], told["per_tag"], told["output_tokens"], told["measured_on"]) == (21, 70, 21 * 70, 0)  # seven parts a rollout
    assert told["cost_usd"] == pytest.approx((sent + 21 * 70) / 1000, abs=1e-3) and told["batch_usd"] == pytest.approx(told["cost_usd"] / 2, abs=1e-3)
    assert reader.asked == [] and all(r.tags is None for r in g)  # nothing was tagged
    g.rollouts[0].tag(reader)  # seven tags for 14 tokens: what this reader writes for a tag is now measured
    after = g.tag_cost(reader)
    assert (after["rollouts"], after["tags"], after["per_tag"], after["output_tokens"], after["measured_on"]) == (2, 14, 2.0, 28, 1)
    assert g.tag_cost(reader, overwrite=True)["rollouts"] == 3
    with pytest.raises(ValueError, match="cannot count"):
        g.tag_cost(lambda *a: ({}, {}))


def test_a_batch_tags_the_members_and_is_picked_up_after_being_cut_short(job, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fathom.group, "POLL", 0)
    reader = Bulk()
    reader.collect = lambda batches: (_ for _ in ()).throw(KeyboardInterrupt())  # the call is cut short while it waits
    with pytest.raises(KeyboardInterrupt):
        Group.from_job(job).tag(reader, batch=True)
    note = json.loads((job / "tags.batch.json").read_text())
    assert note["reader"] == "bulk" and note["batches"] == ["batch-1"] and len(note["calls"]) == 3 and len(reader.handed) == 1
    assert note["calls"]["r0-c0"] == {"rollout": "parse__a/agent/trajectory.json", "steps": [2, 4, 5], "place": [0, 1]}
    del reader.collect
    with pytest.raises(FileExistsError, match="a batch by bulk is being worked on"):
        Group.from_job(job).tag(Other(), batch=True)
    done = Group.from_job(job).tag(reader, batch=True)  # called again: the same batch, not a new one
    assert len(reader.handed) == 1 and reader.looks == 2 and reader.asked == []
    assert [r.labels["trial"] for r in done.made] == ["parse__a", "parse__b", "split__c"] and done.failed == [] and done.cost_usd == 0.75
    assert not (job / "tags.batch.json").exists()
    turns = Group.from_job(job).rollouts[0].turns
    assert turns[0].tags == {"thought": "thought at 2", "call": "call at 2", "result": "result at 2"} and turns[2].tags == {"thought": "thought at 5"}
    again = Group.from_job(job).tag(reader, batch=True)
    assert again.made == [] and len(again.skipped) == 3 and len(reader.handed) == 1  # nothing left to hand in


class Other(Bulk):
    name = "another"


def test_a_job_the_batch_fails_leaves_its_member_untagged_and_the_others_go_on(job, monkeypatch):  # noqa: F811
    monkeypatch.setattr(fathom.group, "POLL", 0)
    reader = Bulk(wrong={"r1-c0"})
    reader.thin = {"r2-c0"}  # and one answer comes back with thoughts only
    done = Group.from_job(job).tag(reader, batch=True)
    assert [r.labels["trial"] for r in done.made] == ["parse__a", "split__c"]
    assert [(r.labels["trial"], str(e)) for r, e in done.failed] == [("parse__b", "the batch did not answer this job: errored")]
    assert len(reader.asked) == 1  # the member whose calls and results got no tag was asked once more, by a call
    g = Group.from_job(job)
    assert [r.tags is not None for r in g] == [True, False, True] and g.rollouts[2].turns[0].tags["call"] == "call at 2"
    assert g.rollouts[2].tags["cost_usd"] == 0.75  # the batch's answer and the call after it
    assert len(g.tag(reader).made) == 1  # the one left is tagged by calling again, batch or not


def test_a_batch_is_refused_before_anything_is_handed_in_when_it_could_not_be_picked_up(job):  # noqa: F811
    reader = Bulk()
    with pytest.raises(ValueError, match="no batch"):
        Group.from_job(job).tag(lambda *a: ({}, {}), batch=True)
    with pytest.raises(ValueError, match="needs a group with a folder"):
        Group(Group.from_job(job).rollouts).tag(reader, batch=True)
    with pytest.raises(ValueError, match="needs a group with a folder"):
        Group.from_job(job).tag(reader, batch=True, save=False)
    with pytest.raises(ValueError, match="1 of the rollouts"):
        Group([*Group.from_job(job), Rollout(TRAJECTORY)], path=job).tag(reader, batch=True)
    assert reader.handed == [] and reader.asked == []
