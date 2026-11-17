"""Clustering a group's turns through its interface, on rollouts whose turns are of three plain
kinds, with a tagger that says the kind in a few wordings and an embedder that puts each kind in
its own corner. Neither costs anything."""

import json
import re
import zlib

import numpy as np
import pytest

from fathom import Group

THEMES = {"read": "I should read the file parse.py to see how the function works",
          "test": "Now run the tests with pytest to verify the behaviour",
          "edit": "Edit the code and fix the branch condition in the function"}
TAGS = {"read": "reads the file", "test": "runs the tests", "edit": "edits the code"}
ORDER = ["read", "test", "edit"] * 3  # a rollout's nine turns


def trajectory(n):
    steps = [{"step_id": 1, "timestamp": "2026-11-16T10:00:00Z", "source": "user", "message": "Fix the bug."}]
    for i, theme in enumerate(ORDER):
        steps.append({"step_id": i + 2, "timestamp": f"2026-11-16T10:00:{(i + 1) * 5:02d}Z", "source": "agent",
                      "reasoning_content": f"{THEMES[theme]} (attempt {n}, turn {i})", "message": ""})
    return {"schema_version": "ATIF-v1.7", "agent": {"name": "a", "version": "1"}, "steps": steps}


def tagger(wording=""):
    """A reader that tags each turn's thought by its kind, each in a wording of its own."""
    def read(prompt, schema, files, dirs):
        turns = re.findall(r'<turn step="(\d+)">\n<thought>\n(.*?)\n</thought>', files["transcript.txt"], re.S)
        kind = lambda text: next(k for k, s in THEMES.items() if text.startswith(s))  # noqa: E731
        return {"turns": [{"step": int(s), "thought": f"{TAGS[kind(t)]}{wording}, way {zlib.crc32(t.encode()) % 100000}"} for s, t in turns]}, {}
    return read


class Corners:
    """An embedder that puts a tag near the corner of its kind, and keeps what it was asked."""

    name = "corners"

    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        self.texts += texts
        out = []
        for t in texts:
            v = np.random.default_rng(zlib.crc32(t.encode())).normal(0, 0.05, 16)
            v[[t.startswith(s) for s in TAGS.values()].index(True)] += 1.0
            out.append(v.tolist())
        return out


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    """Sixty trials of one task, a third of them solved, each of nine turns: 540 turns in all."""
    folder = tmp_path_factory.mktemp("job")
    for n in range(60):
        trial = folder / f"parse__{n:02d}"
        (trial / "agent").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text(json.dumps(trajectory(n)))
        (trial / "result.json").write_text(json.dumps({"task_name": "set/parse-bug", "verifier_result": {"rewards": {"reward": float(n % 3 == 0)}}}))
    return folder


def test_clustering_is_refused_before_anything_is_embedded_when_it_cannot_be_right(job):
    embedder = Corners()
    g = Group.from_job(job)
    with pytest.raises(ValueError, match="60 of the group's 60 rollouts are not tagged"):
        g.cluster("thought")
    with pytest.raises(ValueError, match="no part reply"):
        g.cluster("reply")
    g.tag(tagger())
    with pytest.raises(ValueError, match="60 of the group's 60 rollouts have no vectors of their thought tags"):
        g.cluster("thought")
    g.embed(parts=["thought"], embedder=embedder)
    assert 500 < len(embedder.texts) <= 540  # a tag of each turn
    with pytest.raises(ValueError, match="only 27 turns"):
        Group(g.rollouts[:3], path=job / "few").cluster("thought")
    with pytest.raises(ValueError, match="have no vectors of their call tags"):
        g.cluster("call")  # no call was tagged, so none was embedded
    with pytest.raises(ValueError, match="no folder"):
        Group(g.rollouts).cluster("thought")


def test_clustering_groups_the_turns_whose_tags_read_alike_and_names_each_by_a_tag(job):
    g = Group.from_job(job)  # tagged and embedded by the test above
    g.rollouts[7].tag(lambda *a: ({"turns": []}, {}), overwrite=True)  # one member whose turns got no tag at all
    g.embed(parts=["thought"], embedder=Corners())
    assert Group.from_job(job).cluster("thought", save=False)["turns"] == 531  # the others cluster without it
    assert len(g.search("runs the tests", embedder=Corners(), top=None)) == 531
    g.rollouts[7].tag(tagger(), overwrite=True)
    g.embed(parts=["thought"], embedder=Corners())
    layer = g.cluster("thought")
    assert layer is g.clusters["thought"] and "analyst" not in layer  # no model is asked, and nothing is embedded
    assert (layer["part"], layer["embedder"], layer["turns"], layer["sampled"]) == ("thought", "corners", 540, 540)
    assert len(layer["clusters"]) == 3 and len(layer["rollouts"]) == 60
    for x, themes in zip(layer["clusters"], _themes(g)):  # each cluster holds the turns of one kind, and is named by a tag of that kind
        (kind,) = themes
        assert x["name"] == x["tags"][0] and all(t.startswith(TAGS[kind]) for t in x["tags"]) and 1 <= len(x["tags"]) <= 5
    assert set().union(*_themes(g)) == set(THEMES)
    assert all(x["steps"] == list(range(2, 11)) and min(x["clusters"]) >= 0 for x in layer["rollouts"])
    assert Group.from_job(job).clusters["thought"] == layer  # saved in the group's folder, and read back
    with pytest.raises(FileExistsError, match="overwrite=True"):
        Group.from_job(job).cluster("thought")
    assert g.search("runs the tests", embedder=Corners(), top=1)[0].turn.tags["thought"].startswith("runs the tests")


def test_describe_gives_each_cluster_its_time_its_rollouts_and_its_mistakes(job):
    g = Group.from_job(job)  # clustered by the test above

    def reader(prompt, schema, files, dirs):  # the first edit of a rollout is a critical mistake, its first read a minor one
        return {"calls": 0, "mistakes": [{"severity": "critical", "cause": "agent", "steps": [[4, 4]], "what": "Wrong edit."},
                                         {"severity": "minor", "cause": "agent", "steps": [[2, 2]], "what": "Slow read."}]}, {}

    Group(g.rollouts[:10]).flag_mistakes(reader)
    for r in g:
        r.labels["step"] = 1 if r.labels["trial"] < "parse__30" else 2
    d = g.describe(by="step")
    about = d["clusters"]["thought"]
    assert about["current"] is True and about["embedder"] == "corners" and about["turns"] == 540
    assert about["unclustered"] == {"turns": 0, "seconds": 0.0}
    by_kind = {next(iter(t)): next(row for row in about["rows"] if row["name"] == c["name"]) for t, c in zip(_themes(g), g.clusters["thought"]["clusters"])}
    edit, read, test = (by_kind[k] for k in ("edit", "read", "test"))
    assert all((row["turns"], row["rollouts"], row["solved"], row["failed"]) == (180, 60, 20, 40) for row in (edit, read, test))
    assert edit["in_mistakes"] == {"critical": 10, "major": 0, "minor": 0, "of": 30}  # ten flagged rollouts, three edits each, one flagged
    assert read["in_mistakes"] == {"critical": 0, "major": 0, "minor": 10, "of": 30} and test["in_mistakes"]["of"] == 30
    assert edit["per"] == {"1": 30, "2": 30} and edit["tags"][0] == edit["name"]
    assert sum(row["seconds"] for row in about["rows"]) == pytest.approx(d["seconds"])  # every turn is in a cluster
    assert edit["seconds"] == 60 * 10.0  # a rollout's edits: 5 s, 5 s and the last turn's 0 s
    rows = g.table()  # the same numbers, from the rows
    assert len(rows) == 540 and sum(x["seconds"] for x in rows) == pytest.approx(d["seconds"])
    mine = [x for x in rows if x["thought_cluster"] == edit["name"]]
    assert len(mine) == 180 and sum(x["seconds"] for x in mine) == edit["seconds"] and len({x["rollout"] for x in mine}) == 60
    assert sum(x["severity"] == "critical" for x in mine) == 10 and {x["step"] for x in rows} == {1, 2}  # a label, under its own name
    g.rollouts[0].labels["turn"] = "a label named as a field"
    assert g.table()[0]["label_turn"] == "a label named as a field" and g.table()[0]["turn"] == 2
    del g.rollouts[0].labels["turn"]
    first = rows[2]
    assert (first["rollout"], first["turn"], first["task"], first["solved"]) == ("parse__00/agent/trajectory.json", 4, "set/parse-bug", True)
    assert first["thought"].startswith("edits the code") and (first["severity"], first["cause"], first["mistake"]) == ("critical", "agent", "Wrong edit.")
    assert first["call"] is None and "call_cluster" not in first and rows[1]["mistake"] is None


def test_clusters_of_turns_tagged_again_show_as_out_of_date(job):
    g = Group.from_job(job)
    g.rollouts[0].tag(tagger(" again"), overwrite=True)
    about = g.describe()["clusters"]["thought"]
    assert about["current"] is False and "rows" not in about
    with pytest.raises(ValueError, match="1 of the group's 60 rollouts have no vectors of their thought tags as they are now"):
        g.cluster("thought", overwrite=True)
    assert len(g.embed(parts=["thought"], embedder=Corners()).made) == 1  # the one tagged again
    assert len(g.cluster("thought", overwrite=True)["clusters"]) == 3 and g.describe()["clusters"]["thought"]["current"]


def _themes(g):
    themes = [set() for _ in g.clusters["thought"]["clusters"]]
    for x in g.clusters["thought"]["rollouts"]:
        for step, c in zip(x["steps"], x["clusters"]):
            themes[c].add(ORDER[step - 2])
    return themes
