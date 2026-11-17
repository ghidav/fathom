"""Clustering a group's turns by their tags: the turns of all its rollouts, grouped by what the
tag of one of their parts reads like (its vector), and each cluster known by the tags nearest its
middle.

The tags are clustered and not the turns' own text: a tag says what the agent does and leaves the
task out, so the clusters follow the act and not the task, and they hold when only the order of the
turns changes (measured on one 80-rollout run, 16 Nov 2026: see the README). The method is UMAP to
ten dimensions, then HDBSCAN, which finds its own number of clusters. Past SAMPLE turns the
clusters are found on a sample, and each other turn takes the cluster most common among the
sampled turns nearest to it. It needs numpy, scikit-learn and umap-learn: the `cluster` extra.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from fathom.analysis import digest

if TYPE_CHECKING:
    from fathom.group import Group

SAMPLE = 40_000  # turns the clusters are found on
FEWEST = 500  # tagged turns below which there is too little to cluster
SHOWN = 5  # tags a cluster is known by


def missing() -> str | None:
    """Why clustering cannot run here, if it cannot."""
    if any(importlib.util.find_spec(m) is None for m in ("numpy", "sklearn", "umap")):
        return "clustering needs numpy, scikit-learn and umap-learn: install `fathom-rollouts[cluster]`"
    return None


def cluster(group: Group, part: str) -> dict:
    """The clusters of the group's turns by their tag for `part`, from the vectors `embed` made of
    those tags: their embedder, each rollout's tagged turns (their step ids) with the cluster each
    fell in (-1 for none) and a mark of its tags as read, and each cluster's name, the tag nearest
    its middle, with the few nearest."""
    import numpy as np
    import umap
    from sklearn.cluster import HDBSCAN

    members = group.rollouts
    vecs, own, ids, tags, embedder = [], [], [], [], None
    for mi, r in enumerate(members):
        embedder, steps, rows = r._part(part)
        if not steps:  # none of its turns has a tag for the part
            continue
        said = dict(zip(r.tags["steps"], r.tags["parts"][part]))
        vecs.append(rows)
        own += [mi] * len(steps)
        ids += steps
        tags += [said[s] for s in steps]
    n = len(ids)
    V = np.concatenate(vecs).astype(np.float32)
    rng = np.random.default_rng(0)
    S = np.sort(rng.choice(n, SAMPLE, replace=False)) if n > SAMPLE else np.arange(n)
    XS = V[S]
    # the layout the method was measured with, said outright: umap-learn is changing its default
    low = umap.UMAP(n_components=10, metric="cosine", n_neighbors=30, min_dist=0.0, random_state=0, n_jobs=1,
                    compatibility_layout=True).fit_transform(XS)
    lab_s = HDBSCAN(min_cluster_size=max(25, len(S) // 250), min_samples=10, copy=True).fit_predict(low)
    k = int(lab_s.max()) + 1
    if k == 0:
        raise RuntimeError(f"no clusters among the {n} tagged turns: their tags do not group")
    lab = np.full(n, -1)
    lab[S] = lab_s
    rest = np.setdiff1d(np.arange(n), S)
    for a in range(0, len(rest), 1024):  # the turns not sampled: the cluster most common among their 15 nearest sampled turns
        rows = rest[a:a + 1024]
        near = np.argpartition(-(V[rows] @ XS.T), 15, axis=1)[:, :15]
        lab[rows] = [np.bincount(lab_s[x] + 1, minlength=k + 1).argmax() - 1 for x in near]

    clusters = []
    for c in range(k):
        rows = S[lab_s == c]
        middle = V[rows].mean(0)
        nearest = rows[np.argsort(-(V[rows] @ middle), kind="stable")]
        shown = list(dict.fromkeys(tags[i] for i in nearest))[:SHOWN]
        clusters.append({"name": shown[0], "tags": shown})
    table = [{"rollout": group._ref(r, mi), "tags": digest(r.tags["parts"][part]), "steps": [], "clusters": []} for mi, r in enumerate(members)]
    for i in range(n):
        table[own[i]]["steps"].append(int(ids[i]))
        table[own[i]]["clusters"].append(int(lab[i]))
    return {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "part": part, "embedder": embedder,
            "turns": n, "sampled": int(len(S)), "rollouts": table, "clusters": clusters}
