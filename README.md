# fathom

Analyse an agent's rollouts: flag the mistakes in each, and tag and embed its turns. The input is an
[ATIF](https://harborframework.com/docs/trajectory-format) trajectory; what the analysis adds is
kept in files beside it.

```python
from fathom import Rollout

rollout = Rollout.from_trial("jobs/my-job/some-task__AbC123")   # a Harbor trial folder
rollout.flag_mistakes()        # a reader tags the mistakes: severity, cause, the turns each covers
rollout.tag()                  # a reader says in a few words what each turn's thought, call and result do
rollout.embed()                # an embedder turns each of those tags into a vector

for turn in rollout.turns:     # the enriched trajectory
    turn.id, turn.text, turn.seconds, turn.sampled, turn.context, turn.mistakes, turn.tags, turn.vectors
```

The distribution is named `fathom-rollouts`, since `fathom` on PyPI is another library; the module
is `fathom`. Its extras are `api` (tagging), `embed` (embedding and search) and `cluster`.

An agent that answers questions about a run with fathom has a skill for it:
[`skills/fathom/SKILL.md`](skills/fathom/SKILL.md), which says how to load a run, which layers a
question needs and what they cost, and how to count from the table without being misled.

## A rollout

`Rollout.from_trial(folder)` reads a Harbor trial folder. `Rollout.load(file, ...)` reads any ATIF
file, and `Rollout(trajectory, ...)` takes one already in memory. Besides the trajectory, a rollout
knows:

- `reward`: what the attempt got. It solved its task at 1 or more. The reader is told the outcome.
- `verifier`: the folder of the hidden tests' output. The reader quotes it as evidence, and needs it
  to tell a mistake of the verifier from one of the agent.
- `offered`: folders of the skills the agent was offered.
- `labels`: task, step, run, or anything else. They are carried and never interpreted, except that
  the reader is told the task.

A turn is one step of the main agent; sub-agents' steps are not turns. `turn.text` has its own text
by part: what the agent thought and said (`thought`), its tool calls (`call`) and what came back
(`result`, long output cut in the middle). That is what a tag is a few words about, and what to
read when a tag is in doubt. ATIF gives each step one
timestamp, so a turn's `seconds` run from its reply to the next turn's reply: its tools, then the
next model call.

## Layers

Each analysis makes a layer, joined to the trajectory by step id and saved next to it. The
trajectory file is never changed.

| Method | Layer | File beside `trajectory.json` |
|---|---|---|
| `flag_mistakes(reader="sonnet")` | `rollout.mistakes` | `trajectory.mistakes.json` |
| `tag(reader=None)` | `rollout.tags` | `trajectory.tags.json` |
| `embed(parts=("thought", "call", "result"), embedder=None)` | `rollout.vectors` | `trajectory.vectors.npz` |

A mistake has a `severity` (critical, major, minor), a `cause` (agent, environment, verifier), the
`steps` it covers as ranges of step ids, `what` went wrong, and, when critical, the `evidence` and
where it could have been caught (`missed_catches`). The mistakes layer also says who read the
rollout, when, and what it cost. Anyone may write that file by hand or with another tool.

A tag is a few words on one part of one turn: what the agent meant to do (`thought`), what its call
did (`call`), what came back (`result`). It names the kind of action and leaves the task's details
out, so that the same action in another task reads alike. `turn.tags` has a turn's tags by part.
Tags are free text, and nearly every one is worded differently, so they are for finding turns (by a
word, or with a group's `search`), not for counting equal strings.

`embed` embeds the tags, so a rollout is tagged first. It does not embed the turns' own text: those
vectors mostly say which task a turn belongs to, and neither find nor group turns by what the agent
does (the numbers are under search and clusters below). A vectors layer keeps a mark of the tags it
was made from. When the turns are tagged again its vectors are no longer read, and `embed` makes
them again.

Layers cost money to make, so the methods take two flags:

- `save=True`: the layer is saved as soon as it is made. A rollout with no file must pass
  `save=False`, and can be saved later with `rollout.save(path)`.
- `overwrite=False`: if the layer is there already, on disk or in memory, the call fails before
  anything is spent. Embedding a part that is missing, or whose tags have changed, is not an
  overwrite. Embedding with another embedder is, because its vectors would not compare with the
  ones there, and so is replacing a layer made from the turns' own text by an earlier version.

## A group

A group is the rollouts a caller chose to look at together. Which rollouts belong together is the
caller's to say: a group knows only its members and their labels.

```python
from fathom import Group

group = Group.from_job("jobs/my-job", labels={"step": 3})   # every trial of a Harbor job; or Group(rollouts)
done = group.flag_mistakes()       # each member's, 16 at a time
group.tag()                        # each member's turns, tagged
group.embed()
group.analyze_mistakes(by="task")  # an analyst names the kinds of mistake that recur
group.search("asks the user a question")   # the turns whose tag reads most like that, best first
group.cluster("thought")           # its turns grouped by what their tags read like
group.describe()                   # counts over the group, its kinds of mistake and its clusters, with no model
group.table()                      # a row for each turn with all that is known of it, for any other count
group.by("task")                   # {task: Group}
```

- `flag_mistakes`, `tag` and `embed` take the same arguments as a rollout's, and `workers`. A member
  that has the layer already is left as it is, unless `overwrite` is on; `embed` gives a member only
  the parts it lacks. They return a `Batch`: the members `made`, `skipped` and `failed` (each with its
  error; the others go on), and `cost_usd`.
- They refuse before anything is spent if the reader or embedder cannot run, if `save` is on and a
  member has no file, if a member to embed is not tagged, or if a member holds another embedder's
  vectors, or vectors of its turns' own text, and `overwrite` is off.
- `describe()` counts the rollouts solved, failed and flagged, whose failure each failed one is
  (`failed_by`: the agent's when a critical mistake is the agent's, else the environment's, else the
  verifier's alone), the mistakes by severity and cause, the seconds of the members' turns, and the
  seconds a mistake covers. `rollout.failed_by` says it for one rollout.
- `by(label)` splits the group into a sub-group for each value of a label. A group has a folder for
  its own results (`path`; a job's group has the job's folder), and a sub-group gets
  `<label>/<value>` under it.
- `table()` gives a row for each turn of each member: its `rollout`, the rollout's labels and outcome
  (`solved`, `reward`, `failed_by`), the `turn`'s step id, its `seconds` and tokens, its three tags,
  its cluster for each part clustered (`thought_cluster`, ...), and the worst flagged mistake that
  covers it (`severity`, `cause`, `mistake`). It is for the counts `describe()` does not make: a
  question about a run is mostly a filter and a sum over these rows. A label named as one of the
  fields is given as `label_<name>`. On 80 rollouts the 1,902 rows take 0.2 s.

### Tagging a whole run

```python
group.tag_cost()            # what tagging would take, counted and not run
group.tag(batch=True)       # through the Message Batches API: half the price
```

- `tag_cost(reader=None, overwrite=False)` says what tagging the members not tagged yet would take:
  the calls, the tokens sent, the tags to write, and the cost by call and through a batch. The
  tokens sent are counted by the API, which costs nothing, and are exact: on 80 rollouts the count
  was 1,830,297 and the bill 1,830,297. The tokens written are an estimate: the tags to write
  times what this reader wrote for a tag in the members it has tagged already (`measured_on`), or,
  with none tagged, a figure from one earlier run. Tag a few members first and the estimate for the
  rest is measured on the run itself.
- A long rollout is split over several calls, each with the task and a run of the turns. Claude
  Haiku 5.5 costs five times more for a prompt past 100,000 tokens, so the reader names that as its
  limit, and a call is planned to fill half of it: in calls of 64,000 to 90,000 tokens the reader
  left 3 to 5% of the calls and results of three long rollouts untagged, in calls under 50,000 at
  most 2%. `tag_cost` counts the calls as they will be made.
- `tag(batch=True)` hands every call to the Message Batches API and waits: half the price, answered
  within a day and mostly within the hour. The batch is noted in `tags.batch.json` in the group's
  folder until its answers are in. A call that is cut short is picked up by calling again, and
  nothing is paid twice. A member the batch could not answer is reported and left untagged.
- Without a batch, a run that stopped is picked up the same way: members tagged already are left.

Measured on one run (17 Nov 2026): `tag_cost` on 80 rollouts, a call each, took 3 s and said $0.356
where tagging them had cost $0.354. A 95-turn rollout split on purpose into three calls was tagged
as fully as in one (94 of 94 calls and results), for $0.015 against $0.016. Five rollouts through a
batch took 92 s and cost $0.0045, against $0.0047 estimated.

A run of 2,575 longer rollouts (186,488 turns) was counted in six minutes: 4,753 calls (1,492
rollouts split, no call over 50,000 tokens), 169.7M tokens to send, 554,000 tags to write. At the
46 to 70 tokens a tag measured on nine of its rollouts and on the 80 above, that was $15 to $18
through a batch. Tagged through batches of 18, 908 and 3,827 calls, it cost $15.75, at 53 tokens a
tag: each batch was answered within 15 minutes, no call failed, and 99.5% of the calls and results
and 99.7% of the thoughts got a tag (17 rollouts had a part under 90% tagged). The tags are 22 MB.
Embedding them took 32 minutes and $0.04, and their vectors are 1.15 GB. Clustering took 70 to 85 s
a part (76, 59 and 79 clusters, most of them in all 40 tasks), `table()` 11 s and a search 3 s.

### The kinds of mistake

`analyze_mistakes(analyst="opus", by=None)` has an analyst read what the readers flagged and name
the kinds of mistake that recur: each with a name, a severity, a cause, what the agent does, and
members that point at the flagged mistakes it holds. It also says what separates the rollouts that
solved their task from those that failed. The layer is `group.mistakes`, saved as `mistakes.json`
in the group's folder, under the same `save` and `overwrite` flags as a rollout's layers.

- Every member must be flagged first; it does not flag them itself.
- With `by="task"`, each task's sub-group is analyzed first, 16 at a time, and their kinds are then
  sorted into the mistakes that recur across tasks. A sub-group analyzed already is left as it is.
  One level of splitting is enough for a run of thousands of rollouts.
- `describe()` then lists the kinds, worst first, each with the rollouts that made it, how they
  ended, the seconds of the turns it covers and the sub-groups it shows up in. `describe(by="step")`
  also counts each kind's rollouts per step, which is how a mistake changes over training, with no
  model asked.
- An analysis points at mistakes by their place, so it goes out of date when a member is flagged
  again with a different result. `describe()` then says `analyzed.current` is false and gives no
  kinds, and `analyze_mistakes` refuses until `overwrite=True`.

On 16 rollouts of four tasks, analyzing by task took 30 s and $0.46 with Opus: four calls of about
$0.09 and one of $0.10 across them (16 Nov 2026).

### Searching the turns

`search(text, part="thought", top=20, embedder=None)` gives the turns whose tag for a part reads
most like a description of what the agent does, best first, each a `Hit` with its `rollout`, its
`turn` (time, tokens, mistakes, tags) and a `score`. `top=None` gives every tagged turn, ranked.

```python
for hit in group.search("asks the user a question"):
    hit.rollout.labels["task"], hit.turn.id, hit.turn.tags["thought"], hit.score
```

- Every member must be tagged and embedded first. A search embeds the description only, with the
  embedder the tags were embedded with, and compares it with the saved vectors. Nothing is saved.
- It searches the tags and not the turns' own text. The vectors of a turn's own text mostly say
  which task it is: on one 80-rollout run they put 0 to 3 of 22 known turns in the top 22, and the
  tags' vectors 18, with all 22 in the top 31.
- It ranks, it does not count: where the matches end is the caller's to say. A real match scored
  0.6 to 0.8 on that run and the nearest tag to something that never happened about 0.5.
- It follows what a tag is about more than which way it went: "the pipeline run fails" also ranks
  "pipeline run succeeds" high. Tags are plain text, so a word can settle it:
  `[h for h in hits if "fail" in h.turn.tags["result"]]`.
- A short description of the act works better than a long sentence about it.

On those 80 rollouts (1,902 turns) a search took about 1 s, most of it the call that embeds the
description (17 Nov 2026).

### Clusters of turns

`cluster(part="thought")` groups the turns of all the group's rollouts by what their tag for a
part reads like. Where a kind of mistake says what went wrong, a cluster says what the agent was
doing, wrong or not. No model is asked and nothing is spent: the tags' saved vectors are grouped,
and a cluster is named by the tag nearest its middle and keeps the five nearest. The layer is
`group.clusters[part]`, saved as `clusters.<part>.json` in the group's folder under the same `save`
and `overwrite` flags as the other layers.

- Every member must be tagged and embedded first, and the group needs at least 500 tagged turns for
  the part. It needs the `cluster` extra (numpy, scikit-learn, umap-learn).
- The method is UMAP to ten dimensions, then HDBSCAN, which finds its own number of clusters and
  leaves some turns in none. Past 40,000 turns the clusters are found on a sample and each other
  turn takes the cluster of the sampled turns nearest to it.
- `describe()` then lists, for each part clustered, the clusters, longest first: the turns and their
  seconds, the rollouts it shows up in and how they ended, and how many of its turns a flagged
  mistake covers, by the worst. `describe(by="task")` adds the rollouts per task, which says how
  widely a cluster is shared.
- Clusters go out of date when a member is tagged again with other tags; `describe()` then says so
  and gives no rows.

It is the tags that are clustered, not the turns' own text. On one 80-rollout run of 20 tasks, with
only the order of the turns changed, the clusters of the thought tags agreed with themselves at
0.95 (adjusted Rand index, five orders) and those of the call tags at 0.79, where clusters of the
vectors of the thinking itself agreed at 0.46. The tag clusters also followed the task less
(adjusted mutual information 0.17 and 0.10, against 0.30) and the act, as the tool call gives it,
more (0.36 and 0.51, against 0.20). Not measured: how far the clusters move when the same run is
tagged again.

On that run, clustering a part took 3 to 5 s and gave 13 clusters of the thought tags, 27 of the call
tags and 24 of the result tags, with 7 to 10% of the time in turns of no cluster. The same tags
embedded another time had given 14, 25 and 28: vectors embedded again are not quite the same, and
the count moves with them (17 Nov 2026).

A rollout with a file reads its trajectory when asked and does not keep it, so a group can hold a
whole run: 80 rollouts load and are described in 0.1 s.

## Readers and embedders

A reader is a `claude` model alias, which runs Claude Code (`claude -p`, signed in, on the PATH)
with read-only access to the trajectory, the verifier's folder and the offered skills. It can also
be any function `(prompt, schema, files, dirs) -> (answer, about)`.

`ClaudeAPI(model="claude-haiku-5-5", effort=None)` is a reader that asks the Claude API directly, in
one call, with the files it is handed in the prompt. It costs a fraction of what Claude Code does
for the same reading, and it cannot open folders, so it suits tagging and not flagging mistakes. It
needs the `api` extra (the `anthropic` package) and `ANTHROPIC_API_KEY`. `tag` uses it when no
reader is named, at high effort. It can also `count` a job's tokens before the job is run, say what
a call costs (`price`), name the prompt length past which a call costs more (`limit`), and hand
jobs to a batch (`submit`, `collect`). A reader of your own that has these is counted, split and
batched the same way; one without them is asked one call a rollout.

An embedder is a function from texts to vectors with a `name`. The default is
Qwen3-Embedding-8B at 1024 dimensions through OpenRouter, which needs `OPENROUTER_API_KEY`.
Embedding needs numpy: install with the `embed` extra. An analyst, for a group, is a reader too.

On one 29-turn rollout, a Sonnet reading took 16 s and cost $0.18 (16 Nov 2026). Embedding the tags
of 80 rollouts (4,900 tags) took 25 s and cost $0.00035 (17 Nov 2026).

Tagging the 80 rollouts of one run (1,902 turns) with Claude Haiku 5.5 took 107 s and cost $0.35,
16 at a time. It tagged 99% of the calls, 98% of the results and 97% of the thoughts, and of 214
calls whose tool says what they do, 212 had a tag that said so. At low effort the same run cost
$0.23, but 9% of the calls got no tag or the tag of the call before. Even at high effort it left
every result untagged in 1 of the 80 rollouts, so a reader that gives a part no tag at all is asked
once more. On that run a description searched among the embedded thought tags put 18 of 22 known
turns in the top 22, where the vectors of the thinking itself put 0 to 3 (16 Nov 2026).

## Tests

`uv run pytest` runs the module through its interface with a reader and an embedder that cost
nothing.
