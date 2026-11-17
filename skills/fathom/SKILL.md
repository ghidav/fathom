---
name: fathom
description: "Fathom: answer questions about what agents did across a run's rollouts (ATIF trajectories, Harbor jobs) from their turns, tagged, searched and clustered, and their flagged mistakes. Use when asked what a run's agents did, such as where the time went or what goes with solving, or to tag, embed or flag a run's rollouts."
---

# Fathom

Fathom is a Python package you use from your own session. It makes the **layers** (mistakes, tags, vectors, clusters), joins them into a **table** with a row per turn, and finds turns by a description. The judging is yours: you decide which turns are the behaviour asked about, read some to check, and compute the numbers in code.

Run it where fathom is installed (`uv run python`), one script per question, kept so every number can be computed again. Each method's docstring says what it saves and what it refuses: `help(Group.tag)`.

## 1. Load the group

```python
from pathlib import Path
from fathom import Group

group = Group.from_job("jobs/my-job")              # one Harbor job; the job's folder holds the group's layers

jobs = sorted(Path("jobs").glob("my-run-*"))       # a run of several jobs: one group, with a folder of its own
group = Group([r for j in jobs for r in Group.from_job(j, labels={"step": int(j.name.rsplit("-", 1)[1])})],
              path="analysis/my-run")
```

Every rollout is labelled with its `trial` and `task`; add the labels the question splits on, such as `step`. A trial that left no trajectory is left out. The group is loaded when `len(group)` and `group.describe()`'s `solved` and `failed` are the run's own counts.

## 2. Make the layers the question needs

| The question is about | Needs | Made by | Costs |
| --- | --- | --- | --- |
| time, tokens and outcome, by rollout, task or step | nothing | | |
| what the agent does in its turns | tags, then vectors | `group.tag(batch=True)`, `group.embed()` | tags: what `group.tag_cost()` says, about $0.09 per 1,000 turns through a batch and twice that without; vectors: cents |
| where the time goes across tasks | clusters as well | `group.cluster(part)` for `thought`, `call`, `result` | nothing |
| what went wrong, and whose failure it was | mistakes | `group.flag_mistakes()`, then `group.analyze_mistakes(by="task")` | flagging: $0.14 a rollout with Sonnet (80 rollouts of 24 turns); analyzing: about $0.09 a task with Opus, and one more call across the tasks |

- **Paid layers wait for a yes.** `group.tag_cost()` counts for free: the rollouts not tagged yet, the calls, and `batch_usd`. Give the user the cost and the time, and make the layer on their yes.
- **Measure the rate on a slice of a large run.** The tokens `tag_cost` says will be sent are exact; the tokens written are an estimate until some members are tagged. Tag a slice first, `Group(group.rollouts[::5], path=group.path).tag(batch=True)`, and `group.tag_cost()` then prices the rest at the rate this run measured.
- **A batch waits.** `tag(batch=True)` returns when the answers are in: mostly within the hour, at most a day. Start it detached, and call again to pick up one that was cut: nothing is paid twice.
- **Where things land.** A rollout's layers are written beside its `trajectory.json`, inside the job's folder, and the group's in `group.path`. Vectors take about 6 MB per 1,000 turns. A member that has a layer keeps it, and a layer is replaced only with `overwrite=True`.
- **What must be at hand.** Tagging: `ANTHROPIC_API_KEY` and the `api` extra. Embedding and search: `OPENROUTER_API_KEY` and the `embed` extra. Clustering: the `cluster` extra. Flagging and analyzing: Claude Code, signed in.

The layers are made when the returned `Batch.failed` is empty, or each failure is named in your report. In the table a tag is `None` where the turn has no such part (many turns have no thought), and for the 0.5 to 3% of parts the reader skipped.

## 3. Find the turns

```python
rows = group.table()                                          # a row per turn: labels, outcome, seconds, tokens, tags, clusters, mistake
hits = group.search("runs the tests", part="call", top=200)   # Hit: .rollout, .turn, .score
turn = hits[0].turn                                           # .id, .tags, .text (its own words, by part), .seconds, .step
```

Write a **rule**: a test on a row that says whether the turn is the behaviour. Search first, to learn how the reader words the behaviour, then make the rule from what you read: words in a tag, a set of clusters, the tool in `turn.step`, or the literal command in `turn.text["call"]` when the question is about one.

Then **read a sample** of the turns the rule takes, 30 to 40 of them in `turn.text`, and count how many are the behaviour. Read some turns it leaves out as well: search hits just outside it. The rule is ready when the sample's share on topic is one you can state beside the result ("38 of 40 on topic").

What the layers will mislead you on:

- **Search follows the topic of a tag, not which way it went.** "tests fail" ranks "tests pass" high. A word in the tag settles it: `"fail" in row["result"]`.
- **Search ranks every turn and has no cutoff.** Real matches score 0.6 to 0.8, and the nearest tag to something that never happened about 0.5, so a top hit is no proof the behaviour occurs.
- **A short description of the act finds more than a sentence about it.**
- **Tags are free text and nearly every one is worded differently.** Count by a rule, never by equal strings.
- **A cluster is named by one tag.** Read its five `tags` in `describe()` and a sample of its turns before you name it; two clusters that read alike can point opposite ways.
- **Part of the time is in no cluster**: 7 to 27% on the runs measured. `describe()` gives it as `unclustered`; say it with any share of time.
- **Stale clusters vanish from the table.** After members are tagged again, or the group's members change, the `<part>_cluster` columns are absent, and `describe()["clusters"][part]["current"]` is false. Cluster again with `overwrite=True`.
- **A turn is a step of the main agent.** Sub-agents' steps are not turns, and a turn's `seconds` run from its reply to the next turn's: its tools, then the next model call.
- **A mistake can cover a whole rollout**, as a verifier's often does, which inflates the share of any cluster's turns "in mistakes".

## 4. Count and compare

- **Time**: the rule's `seconds` over all rows' `seconds`. **Rollouts**: those with any row the rule takes.
- **Solved against failed**: compare **within a task**, then average the differences over the tasks that have both. Tasks differ in difficulty and length, so a pooled comparison mostly compares tasks.
- **Over training**: by the `step` label, or by a task's **pass**, its k-th appearance in the run, which keeps the mix of tasks out of the trend.

These are associations. A behaviour that goes with solving is a lead for an experiment, and the report says so.

## 5. Report

Lead with the number. With it give the rule, the turns and rollouts it took, the sample you read, and what share of the time sat in no cluster when clusters were used. The answer is done when every number in it comes from a script you kept.
