"""A group through its interface: what it does to its members, what it leaves alone, and what it
counts, with a reader and an embedder that cost nothing."""

import json

import pytest

from fathom import Group, Rollout
from test_rollout import ANSWER, TRAJECTORY, Embedder, Reader, tag_all


def make(job, name, reward, task="set/parse-bug"):
    trial = job / name
    (trial / "agent").mkdir(parents=True)
    (trial / "agent" / "trajectory.json").write_text(json.dumps(TRAJECTORY))
    (trial / "result.json").write_text(json.dumps({"task_name": task, "verifier_result": {"rewards": {"reward": reward}}}))
    return trial


@pytest.fixture
def job(tmp_path):
    """A Harbor job folder: three trials of two tasks, one solved, and a folder that is no trial."""
    make(tmp_path, "parse__a", 0.0)
    make(tmp_path, "parse__b", 1.0)
    make(tmp_path, "split__c", 0.5, task="set/split-bug")
    (tmp_path / "modules").mkdir()
    return tmp_path


def test_a_job_folder_gives_a_group_of_its_trials(job):
    g = Group.from_job(job, labels={"step": 7})
    assert len(g) == 3 and g.path == job
    assert [r.labels["trial"] for r in g] == ["parse__a", "parse__b", "split__c"]
    assert all(r.labels["step"] == 7 for r in g)
    assert [r.solved for r in g] == [False, True, False]


def test_by_splits_on_a_label_and_gives_each_sub_group_a_folder(job):
    g = Group.from_job(job)
    tasks = g.by("task")
    assert list(tasks) == ["set/parse-bug", "set/split-bug"]
    assert [len(x) for x in tasks.values()] == [2, 1]
    assert tasks["set/parse-bug"].path == job / "task" / "set__parse-bug"
    assert list(Group(g.rollouts).by("task").values())[0].path is None  # a group with no folder gives none
    for i, r in enumerate(g):
        r.labels["step"] = [10, 9, None][i]
    assert list(g.by("step")) == [9, 10, None]  # in order of value, not of its spelling; the unlabelled last


def test_flagging_a_group_flags_the_members_that_lack_it(job):
    reader = Reader()
    g = Group.from_job(job)
    g.rollouts[0].flag_mistakes(reader)
    done = g.flag_mistakes(reader)
    assert [r.labels["trial"] for r in done.made] == ["parse__b", "split__c"]
    assert [r.labels["trial"] for r in done.skipped] == ["parse__a"] and done.failed == [] and done.cost_usd == 0.0
    assert len(reader.asked) == 3
    assert all(r.mistakes for r in Group.from_job(job))  # each saved beside its trajectory
    again = g.flag_mistakes(reader)
    assert again.made == [] and len(again.skipped) == 3 and len(reader.asked) == 3
    assert len(g.flag_mistakes(reader, overwrite=True).made) == 3 and len(reader.asked) == 6


def test_a_member_that_cannot_be_read_is_reported_and_the_others_go_on(job):
    def reader(prompt, schema, files, dirs):
        if "split-bug" in prompt:
            raise RuntimeError("the reader timed out")
        return ANSWER, {"cost_usd": 0.25}

    done = Group.from_job(job).flag_mistakes(reader)
    assert len(done.made) == 2 and done.cost_usd == 0.5
    assert [(r.labels["trial"], str(e)) for r, e in done.failed] == [("split__c", "the reader timed out")]


def test_a_group_refuses_before_spending_when_a_member_has_nowhere_to_save(job):
    reader, embedder = Reader(), Embedder()
    g = Group([*Group.from_job(job), Rollout(TRAJECTORY)])
    with pytest.raises(ValueError, match="1 of the rollouts"):
        g.flag_mistakes(reader)
    with pytest.raises(ValueError, match="4 of the group's 4 rollouts are not tagged"):
        g.embed(embedder=embedder)
    g.tag(tag_all, save=False)
    with pytest.raises(ValueError, match="1 of the rollouts"):
        g.embed(embedder=embedder)
    assert reader.asked == [] and embedder.texts == []
    assert len(g.flag_mistakes(reader, save=False).made) == 4


def test_embedding_a_group_gives_each_member_the_parts_it_lacks(job):
    embedder = Embedder()
    g = Group.from_job(job)
    g.tag(tag_all)
    g.rollouts[0].embed(parts=["thought"], embedder=embedder)
    asked = len(embedder.texts)
    done = g.embed(parts=["thought", "call"], embedder=embedder, workers=1)  # one at a time: this embedder counts for all threads
    assert len(done.made) == 3 and done.skipped == []
    assert len(embedder.texts) == asked + 2 + 5 + 5  # the first lacked only its calls
    assert done.cost_usd == 0.5 + 1.0 + 1.0  # this call's batches, not what the first had cost before
    assert all(set(r.vectors["parts"]) == {"thought", "call"} for r in Group.from_job(job))
    again = g.embed(parts=["thought", "call"], embedder=embedder)
    assert again.made == [] and len(again.skipped) == 3
    assert sorted(g.rollouts[1].turns[0].vectors) == ["call", "thought"]  # still there to read after the batch let go of them
    g.rollouts[2].tag(lambda *a: tag_all(*a, said=" again"), overwrite=True)
    later = g.embed(parts=["thought", "call"], embedder=embedder)  # the one tagged again is embedded again, and no other
    assert [r.labels["trial"] for r in later.made] == ["split__c"] and len(later.skipped) == 2


def test_a_group_refuses_to_mix_embedders_before_embedding_anything(job):
    g = Group.from_job(job)
    g.tag(tag_all)
    g.rollouts[2].embed(parts=["thought"], embedder=Embedder())
    other = Embedder("another")
    with pytest.raises(FileExistsError, match="fake-embedder"):
        g.embed(parts=["thought"], embedder=other)
    assert other.texts == []
    assert len(g.embed(parts=["thought"], embedder=other, overwrite=True).made) == 3
    assert {r.vectors["embedder"] for r in Group.from_job(job)} == {"another"}


def test_describe_counts_outcomes_mistakes_and_time(job):
    g = Group.from_job(job)
    empty = g.describe()
    assert (empty["rollouts"], empty["solved"], empty["failed"], empty["flagged"]) == (3, 1, 2, 0)
    assert empty["failed_by"] == {"agent": 0, "environment": 0, "verifier": 0, "unknown": 2} and empty["seconds"] == 45.0

    def reader(prompt, schema, files, dirs):  # the split task fails on the verifier alone; the others as ANSWER
        if "split-bug" in prompt:
            return {"calls": 2, "mistakes": [{"severity": "critical", "cause": "verifier", "steps": [[5, 5]], "what": "Unnamed column."}]}, {}
        return ANSWER, {}

    g.flag_mistakes(reader)
    t = g.describe()
    assert t["flagged"] == 3 and t["failed_by"] == {"agent": 1, "environment": 0, "verifier": 1, "unknown": 0}
    assert t["mistakes"]["critical"] == {"agent": 2, "environment": 0, "verifier": 1}
    assert t["mistakes"]["minor"] == {"agent": 0, "environment": 2, "verifier": 0}
    # turns take 10, 5 and 0 s: ANSWER's critical covers the second and third, its minor the first
    assert t["seconds_in_mistakes"]["critical"]["agent"] == 10.0 and t["seconds_in_mistakes"]["minor"]["environment"] == 20.0
    assert t["seconds_in_mistakes"]["critical"]["verifier"] == 0.0
    assert g.by("task")["set/split-bug"].describe()["failed_by"]["verifier"] == 1


# ---------- analyzing a group's mistakes

import re  # noqa: E402


class Analyst:
    """An analyst that puts each rollout's first mistake in one kind and the first rollout's second
    in another, with some members it should not have named; across groups, it joins the critical kinds."""

    name = "fake-analyst"

    def __init__(self):
        self.asked = []

    def __call__(self, prompt, schema, files, dirs):
        self.asked.append({"prompt": prompt, "files": files, "dirs": dirs})
        if "group.txt" in files:
            flagged = [int(n) for n, rest in re.findall(r"### Rollout (\d+):.*\n(?:.*\n)*?(\(no mistakes|0\. )", files["group.txt"]) if rest == "0. "]
            return {"kinds": [
                {"name": "Edits the wrong branch", "severity": "critical", "what": "It edits the wrong branch. The test fails.",
                 "members": [{"rollout": n, "mistake": 0} for n in flagged] + [{"rollout": 99, "mistake": 0}]},
                {"name": "Reads the whole file", "severity": "minor", "what": "It reads it all. Calls are wasted.",
                 "members": [{"rollout": flagged[0], "mistake": 1}, {"rollout": flagged[0], "mistake": 0}, {"rollout": flagged[0], "mistake": 7}]},
                {"name": "Not a kind", "severity": "odd", "what": "", "members": [{"rollout": flagged[0], "mistake": 1}]},
            ], "difference": "The solved ones edited the right branch."}, {"model": "fake-2", "cost_usd": 0.1}
        critical = re.findall(r"^(\d+\.\d+)  critical", files["kinds.txt"], re.M)
        return {"kinds": [{"name": "Edits the wrong place", "severity": "critical", "what": "Across tasks. Tests fail.",
                           "members": [*critical, critical[0], "9.9"]}], "unsure": "little"}, {"model": "fake-2", "cost_usd": 0.2}


@pytest.fixture
def flagged(job):
    g = Group.from_job(job)
    g.flag_mistakes(Reader())
    return g


def test_analyzing_needs_every_member_flagged_and_a_folder_before_the_analyst_is_asked(job):
    analyst = Analyst()
    g = Group.from_job(job)
    g.rollouts[0].flag_mistakes(Reader())
    with pytest.raises(ValueError, match="2 of the group's 3 rollouts are not flagged"):
        g.analyze_mistakes(analyst)
    g.flag_mistakes(Reader())
    with pytest.raises(ValueError, match="no folder"):
        Group(g.rollouts).analyze_mistakes(analyst)
    assert analyst.asked == []
    assert Group(g.rollouts).analyze_mistakes(analyst, save=False)["kinds"]


def test_analyzing_names_the_kinds_and_points_them_at_the_flagged_mistakes(flagged, job):
    analyst = Analyst()
    layer = flagged.analyze_mistakes(analyst)
    assert layer is flagged.mistakes and layer["analyst"] == "fake-analyst" and layer["model"] == "fake-2" and layer["cost_usd"] == 0.1
    assert [x["rollout"] for x in layer["rollouts"]] == [f"{n}/agent/trajectory.json" for n in ("parse__a", "parse__b", "split__c")]
    wrong, reads = layer["kinds"]  # the kind with no severity is dropped
    assert wrong["members"] == [{"rollout": i, "mistake": 0} for i in range(3)] and wrong["cause"] == "agent"  # rollout 99 is not there
    assert reads["members"] == [{"rollout": 0, "mistake": 1}] and reads["cause"] == "environment"  # a mistake is in one kind only
    assert layer["difference"] == "The solved ones edited the right branch."
    asked = analyst.asked[0]
    assert "1 of 3 solved theirs" in asked["prompt"] and "### Rollout 2: solved" in asked["files"]["group.txt"]
    assert str(job / "parse__a" / "agent" / "trajectory.json") in asked["files"]["group.txt"] and job / "parse__a" / "agent" in asked["dirs"]
    assert Group.from_job(job).mistakes == layer  # saved in the group's folder, and read back
    with pytest.raises(FileExistsError, match="overwrite=True"):
        Group.from_job(job).analyze_mistakes(analyst)
    assert len(analyst.asked) == 1
    flagged.analyze_mistakes(analyst, overwrite=True)
    assert len(analyst.asked) == 2


def test_a_group_whose_rollouts_share_a_label_tells_the_analyst(flagged):
    analyst = Analyst()
    flagged.by("task")["set/parse-bug"].analyze_mistakes(analyst)
    assert "The rollouts share: task = set/parse-bug." in analyst.asked[0]["prompt"]


def test_analyzing_by_a_label_analyzes_each_sub_group_then_sorts_their_kinds(flagged, job):
    analyst = Analyst()
    flagged.by("task")["set/split-bug"].analyze_mistakes(analyst)  # one sub-group is analyzed already: it is left
    layer = flagged.analyze_mistakes(analyst, by="task")
    assert len(analyst.asked) == 3  # the other sub-group, then across the two
    assert layer["by"] == "task" and [x["group"] for x in layer["groups"]] == ["set/parse-bug", "set/split-bug"]
    assert layer["cost_usd"] == 0.2 and layer["total_usd"] == pytest.approx(0.3)  # with the one sub-group analyzed now
    joined, alone = layer["kinds"][0], layer["kinds"][1:]
    assert joined["name"] == "Edits the wrong place" and joined["cause"] == "agent"
    assert joined["members"] == [{"group": "set/parse-bug", "kind": 0}, {"group": "set/split-bug", "kind": 0}]  # each id once
    assert [(k["name"], k["members"]) for k in alone] == [("Reads the whole file", [{"group": "set/parse-bug", "kind": 1}]),
                                                         ("Reads the whole file", [{"group": "set/split-bug", "kind": 1}])]  # left out: they stand alone
    assert (job / "mistakes.json").exists() and (job / "task" / "set__parse-bug" / "mistakes.json").exists()
    assert "were put in groups by their task" in analyst.asked[-1]["prompt"] and analyst.asked[-1]["dirs"] == []
    assert "1.0  critical; cause agent; 2 of 2 rollouts, 1 of them failed. Edits the wrong branch" in analyst.asked[-1]["files"]["kinds.txt"]


def test_describe_gives_each_kind_its_rollouts_failures_time_and_spread(flagged, job):
    for r, step in zip(flagged, [1, 1, 2]):
        r.labels["step"] = step
    assert flagged.describe()["analyzed"] is None
    flagged.analyze_mistakes(Analyst(), by="task")
    d = Group.from_job(job).describe()  # read back from the folders
    assert d["analyzed"] == {"analyst": "fake-analyst", "at": d["analyzed"]["at"], "by": "task", "current": True}
    wrong = d["kinds"][0]
    assert (wrong["name"], wrong["severity"], wrong["cause"]) == ("Edits the wrong place", "critical", "agent")
    assert (wrong["mistakes"], wrong["rollouts"], wrong["failed"], wrong["solved"], wrong["groups"]) == (3, 3, 2, 1, 2)
    assert wrong["seconds"] == 15.0  # the second and third turns of each: 5 s and 0 s, three times
    assert [(k["name"], k["rollouts"], k["seconds"], k["groups"]) for k in d["kinds"][1:]] == [("Reads the whole file", 1, 10.0, 1)] * 2
    per = flagged.describe(by="step")
    assert per["per"] == {"1": {"rollouts": 2, "solved": 1, "failed": 1}, "2": {"rollouts": 1, "solved": 0, "failed": 1}}
    assert per["kinds"][0]["per"] == {"1": 2, "2": 1}


def test_an_analysis_of_mistakes_since_flagged_again_shows_as_out_of_date(flagged):
    analyst = Analyst()
    flagged.analyze_mistakes(analyst)
    assert flagged.describe()["analyzed"]["current"] is True

    def other(prompt, schema, files, dirs):
        return {"calls": 1, "mistakes": [{"severity": "minor", "cause": "agent", "steps": [[2, 2]], "what": "Another reading."}]}, {}

    flagged.rollouts[0].flag_mistakes(other, overwrite=True)
    d = flagged.describe()
    assert d["analyzed"]["current"] is False and "kinds" not in d  # its pointers no longer fit: no numbers from them
    with pytest.raises(FileExistsError, match="before its rollouts were last flagged"):
        flagged.analyze_mistakes(analyst)
    flagged.analyze_mistakes(analyst, overwrite=True)
    assert flagged.describe()["analyzed"]["current"] is True


def test_a_sub_group_that_cannot_be_analyzed_stops_the_analysis_and_keeps_the_others(flagged, job):
    def analyst(prompt, schema, files, dirs):
        if "split-bug" in files.get("group.txt", ""):
            raise RuntimeError("the analyst timed out")
        return Analyst()(prompt, schema, files, dirs)

    with pytest.raises(RuntimeError, match="1 of the 2 sub-groups by task could not be analyzed.*timed out"):
        flagged.analyze_mistakes(analyst, by="task")
    assert (job / "task" / "set__parse-bug" / "mistakes.json").exists() and not (job / "mistakes.json").exists()
    assert Group.from_job(job).analyze_mistakes(Analyst(), by="task")["kinds"]  # the next try asks only for what is missing
