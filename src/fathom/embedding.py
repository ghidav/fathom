"""Embedding a rollout's turns, part by part: the tag of what the agent thought and said, of what it
called, and of what came back, each as its own vector.

It is the tags that are embedded, not the turns' own text: the vectors of a turn's own text mostly
say which task it is, and neither find nor group turns by what the agent does (measured 16 Nov
2026: see the README). A layer keeps a mark of the tags each part was embedded from, so that it
shows when the turns were tagged again.

An embedder is a function from texts to vectors with a `name`; vectors made by embedders of
different names are not comparable. If it keeps a running `spent` in USD, the layer records what
the embedding cost; one shared by threads should count each thread's own, as OpenRouter does. A
model id, or nothing, means that model through OpenRouter, which needs `OPENROUTER_API_KEY` and
numpy.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from fathom.analysis import digest
from fathom.reading import clip

PARTS = ("thought", "call", "result")
OWN = (24_000, 6_000)  # the agent's own text is embedded whole, short of this
BACK = (1_000, 1_000)  # of each tool result, its first and last characters
BATCH = 256  # tags embedded in one call

Embedder = Callable[[list[str]], Sequence[Sequence[float]]]


def main_steps(trajectory: dict):
    """The main agent's own steps: its turns. Sub-agents' steps are not among them."""
    for st in trajectory.get("steps") or []:
        if st.get("source") == "agent" and not (st.get("extra") or {}).get("is_sidechain"):
            yield st


def texts(trajectory: dict) -> tuple[list[int], dict[str, list[str]]]:
    """Each turn's step id, and its text by part, '' where the turn has no such part: its thinking
    and message, its tool calls, and what the tools gave back."""
    ids, by = [], {p: [] for p in PARTS}
    for st in main_steps(trajectory):
        results = {r.get("source_call_id"): r for r in (st.get("observation") or {}).get("results") or []}
        calls, back = [], []
        for tc in st.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            body = (args.get("command") or args.get("prompt") or json.dumps(args)) if isinstance(args, dict) else args
            calls.append(f"{tc.get('function_name')}: {clip(body, *OWN)}")
            if res := (results.get(tc.get("tool_call_id")) or {}).get("content"):
                back.append(clip(res, *BACK))
        ids.append(int(st["step_id"]))
        by["thought"].append("\n".join(clip(st[k], *OWN) for k in ("reasoning_content", "message") if st.get(k)).strip())
        by["call"].append("\n".join(calls).strip())
        by["result"].append("\n".join(back).strip())
    return ids, by


class OpenRouter:
    """An embedder that asks OpenRouter. The default is the one Diver's maps were measured with:
    Qwen3-Embedding-8B at 1024 dimensions, served by Nebius."""

    def __init__(self, model: str = "qwen/qwen3-embedding-8b", dimensions: int | None = 1024, provider: str | None = "nebius",
                 key: str | None = None) -> None:
        self.model, self.dimensions, self.provider = model, dimensions, provider
        self.name = f"{model}@{dimensions}" if dimensions else model
        self.key = key or os.environ.get("OPENROUTER_API_KEY", "").strip()
        self._spent = threading.local()
        if not self.key:
            raise RuntimeError("no OPENROUTER_API_KEY: set it, or pass an embedder of your own")

    @property
    def spent(self) -> float:
        """What the calling thread's calls have cost so far, in USD."""
        return getattr(self._spent, "usd", 0.0)

    def __call__(self, texts: list[str]) -> list[list[float]]:
        body: dict = {"model": self.model, "input": texts}
        if self.dimensions:
            body["dimensions"] = self.dimensions
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        for attempt in range(7):
            try:
                req = urllib.request.Request("https://openrouter.ai/api/v1/embeddings", data=json.dumps(body).encode(),
                                             headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=180) as resp:
                    r = json.load(resp)
                rows = [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
                if len(rows) != len(texts):
                    raise ValueError(f"got {len(rows)} vectors for {len(texts)} texts")
                self._spent.usd = self.spent + ((r.get("usage") or {}).get("cost") or 0)
                return rows
            except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as e:
                if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 402, 403):
                    raise RuntimeError(f"OpenRouter answered {e.code}: {e.read()[:300].decode(errors='replace')}") from None
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError("OpenRouter kept failing to embed; try again later")


def resolve(embedder: str | Embedder | None) -> Embedder:
    """The embedder to call: the default, the OpenRouter model of that id, or the function given."""
    if embedder is None:
        return OpenRouter()
    if isinstance(embedder, str):
        return OpenRouter(embedder, dimensions=None, provider=None)
    return embedder


def name(embedder: Embedder) -> str:
    return getattr(embedder, "name", getattr(embedder, "__name__", "embedder"))


def embed(steps: list[int], tags: dict[str, list], parts: Sequence[str], embedder: Embedder, keep: dict | None = None) -> dict:
    """The vectors layer: `embedder` on each turn's tag for each of `parts`, as rows of unit length
    in step order (zeros where a turn has no tag for the part, which `has` tells), with a mark of
    the tags each part was embedded from (`tagged`), and with the parts of `keep`, a layer by the
    same embedder for the same turns, that were not asked for again."""
    import numpy as np

    before = getattr(embedder, "spent", None)
    made, has = {}, {}
    for part in parts:
        rows = [i for i, tag in enumerate(tags[part]) if tag]
        got: list = []
        for a in range(0, len(rows), BATCH):
            got.extend(embedder([tags[part][i] for i in rows[a:a + BATCH]]))
        if rows:
            v = np.asarray(got, dtype=np.float32).reshape(len(rows), -1)
            made[part] = np.zeros((len(steps), v.shape[1]), np.float16)
            made[part][rows] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        else:  # no turn has a tag for the part: a matrix with no columns, and no row to read
            made[part] = np.zeros((len(steps), 0), np.float16)
        has[part] = np.isin(np.arange(len(steps)), rows)
    marks = {part: digest(tags[part]) for part in parts}
    cost = getattr(embedder, "spent", None)
    cost = cost - before if isinstance(cost, (int, float)) and isinstance(before, (int, float)) else None
    if keep is not None and keep["steps"] == steps:
        made = {**keep["parts"], **made}
        has = {**keep["has"], **has}
        marks = {**keep["tagged"], **marks}
        if cost is not None or keep.get("cost_usd") is not None:
            cost = (cost or 0) + (keep.get("cost_usd") or 0)
    return {"embedder": name(embedder), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cost_usd": cost,
            "of": "tags", "steps": steps, "parts": made, "has": has, "tagged": marks}


def _about(z: Any) -> dict:
    """What a saved layer says of itself. One with no word on what it is of was made from the turns' own text."""
    about = json.loads(str(z["about"]))
    return {**about, "of": about.get("of") or "text", "tagged": about.get("tagged") or {}}


def save(path: Path, layer: dict) -> None:
    import numpy as np

    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, steps=np.array(layer["steps"], dtype=np.int64),
             about=np.array(json.dumps({k: layer.get(k) for k in ("embedder", "at", "cost_usd", "of", "tagged")})),
             **layer["parts"], **{f"has_{p}": h for p, h in layer["has"].items()})
    tmp.replace(path)


def peek(path: Path) -> dict:
    """A saved layer without its matrices: its embedder, when it was made, its cost, what it is of,
    the marks of the tags it was made from, and its parts."""
    import numpy as np

    with np.load(path) as z:
        return {**_about(z), "parts": [k for k in z.files if k in PARTS]}


def load_part(path: Path, part: str, rows: bool = True) -> tuple[dict, list[int], Any] | None:
    """One part of a saved layer, read no further than that: what the layer says of itself, the
    step ids of the turns that have the part and, unless `rows` is off, their vectors. None if the
    part is not there."""
    import numpy as np

    with np.load(path) as z:
        if part not in z.files:
            return None
        has = z[f"has_{part}"]
        return _about(z), [int(s) for s in z["steps"][has]], z[part][has] if rows else None


def load(path: Path) -> dict:
    import numpy as np

    with np.load(path) as z:
        parts = [k for k in z.files if k in PARTS]
        return {**_about(z), "steps": [int(s) for s in z["steps"]],
                "parts": {p: z[p] for p in parts}, "has": {p: z[f"has_{p}"] for p in parts}}
