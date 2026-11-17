"""Flagging a rollout's mistakes: a reader reads the trajectory and says what went wrong, at which
steps, how badly, and whose mistake it was.

A reader is a `claude` model alias ("sonnet", "opus", ...), read through `claude -p`, or any
function `(prompt, schema, files, dirs) -> (answer, about)`: `files` are texts put in its working
directory, `dirs` folders it may look things up in, `answer` fits `schema`, and `about` says who
answered and what it cost (`model`, `seconds`, `cost_usd`; all optional). `ClaudeAPI` is a reader
that asks the Claude API in one call, for a job that needs only the files it is handed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from fathom.rollout import Rollout

SEVERITIES = ("critical", "major", "minor")
CAUSES = ("agent", "environment", "verifier")  # whose a mistake is
TIMEOUT = 1800

Reader = Callable[[str, dict, dict[str, str], list[Path]], tuple[dict, dict]]

RANGES = {"type": "array", "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}
SCHEMA = {
    "type": "object",
    "properties": {
        "calls": {"type": "integer"},
        "mistakes": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "severity": {"enum": list(SEVERITIES)},
                "steps": RANGES,
                "cause": {"enum": list(CAUSES)},
                "what": {"type": "string"},
                "evidence": {"type": "string"},
                "missed_catches": RANGES,
            },
            "required": ["severity", "steps", "cause", "what"],
        }},
        "unsure": {"type": "string"},
    },
    "required": ["calls", "mistakes"],
}

PROMPT = """You are tagging mistakes in one coding-agent trajectory. This is an analysis task: read files, judge, answer. Run no code and change nothing.

## Context
An agent{agent} got {task} and {outcome}. Hidden tests the agent never saw decided the outcome.

## Files you may read (only these)
- transcript.txt, in your working directory: the trajectory rendered as text. Each `[n]` marks the main agent's step n. `USER` steps are what the agent was told, given whole: the task statement first, then texts it was given along the way, such as a skill's instructions. `SAYS` is the agent's text, then a tool call and its output after `->`. Sub-agent steps are left out. Long outputs are cut and shown as `[...N chars...]`; that comes from the rendering, not the agent. The whole trajectory, cut outputs included, is {trajectory} (ATIF; find a step by its `step_id`).{given}

Read nothing else.

## Tagging rules
Tag every mistake with one severity:
- **critical**: still in the final patch and makes a hidden test fail.{quote} A solved trial has no critical mistakes, and a failed trial has at least one. Also list the steps where the agent could have caught it and didn't (`missed_catches`).
- **major**: either a mistake that would have failed the task if it hadn't been caught, or one that cost 10 or more main-agent tool calls, or 15% or more of the trial's calls.
- **minor**: any other wasted call or wasted context.

Give each mistake a `cause`, whose mistake it is:
- `agent`: the agent's own error, given what it was told (the task, and instructions it was given along the way, such as a skill's) and what it could find out.
- `environment`: the sandbox or the harness forced it, for example a missing package, no network, a tool that fails or prints nothing, or an instruction it was given that cannot be followed there, such as to ask a user when nobody is there to answer.
- `verifier`: the hidden tests' own. A test fails although the agent's result meets the task as stated: it asks for something the task never names, or rejects a correct result. Check the task statement first: what it names anywhere, it states. Such a mistake is critical when it failed the trial; in `what`, say what the test wanted that the task did not ask for.

What the agent was told counts for it: before you blame the agent for following a text it was given, or for passing over a skill it was offered, read what that text says.

Group consecutive calls that serve one purpose into one mistake and cite its main-agent steps as ranges, `[[first, last], ...]`. `what` is one sentence. Count the main agent's tool calls and give the total as `calls`. In `unsure`, say in a sentence or two what you were unsure about."""


def read(rollout: Rollout, reader: str | Reader) -> dict:
    """The rollout's mistakes as `reader` flags them: who read it, when and at what cost, the main
    agent's tool calls counted, each mistake with its severity, cause, steps (ranges of ATIF step
    ids), what went wrong, the evidence and where it could have been caught, and what the reader
    was unsure about."""
    ask = resolve(reader)
    trajectory = rollout.trajectory
    agent = trajectory.get("agent") or {}
    who = " and ".join(filter(None, [agent.get("model_name"), agent.get("name") and f"running inside {agent['name']}"]))
    outcome = {True: "solved it", False: "failed it", None: "has no verdict from the tests"}[rollout.solved]
    files, dirs, given = {"transcript.txt": render(trajectory)}, [], ""
    if rollout.path:
        where = str(rollout.path)
        dirs.append(rollout.path.parent)
    else:  # a rollout held in memory only: the reader gets the trajectory as a file of its own
        where = "trajectory.json, in your working directory"
        files["trajectory.json"] = json.dumps(trajectory)
    if rollout.verifier:
        held = sorted(p.name for p in rollout.verifier.glob("*") if p.is_file())
        given += f"\n- {rollout.verifier}/: what the hidden tests checked, which failed, and their output ({', '.join(held) or 'empty'})."
        dirs.append(rollout.verifier)
    for folder in rollout.offered:
        given += f"\n- {folder}/: skills the agent was offered, each a folder whose SKILL.md says when to use it and what it tells the agent to do."
        dirs.append(folder)
    task = rollout.labels.get("task")
    prompt = PROMPT.format(agent=f" ({who})" if who else "", task=f"the task {task}" if task else "a task", outcome=outcome,
                           trajectory=where, given=given,
                           quote=" You must quote the failing assertion from the verifier's files as `evidence`." if rollout.verifier else "")
    answer, about = ask(prompt, SCHEMA, files, dirs)
    return {
        "reader": getattr(ask, "name", getattr(ask, "__name__", "reader")),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **about,
        "calls": answer.get("calls"),
        "mistakes": [m for m in answer.get("mistakes") or [] if m.get("severity") in SEVERITIES],
        "unsure": answer.get("unsure"),
    }


def resolve(reader: str | Reader) -> Reader:
    """The reader to ask: Claude Code with the model of that alias, or the function given."""
    return Claude(reader) if isinstance(reader, str) else reader


def render(trajectory: dict) -> str:
    """The main agent's steps as text: each `[step_id]` with what it said, its tool calls and their
    outputs, long ones cut in the middle. What the agent was told (the task statement, a skill's
    instructions) is whole: whose a mistake is depends on it."""
    out, side = [], 0
    for st in trajectory.get("steps") or []:
        if (st.get("extra") or {}).get("is_sidechain"):
            side += 1
            continue
        if st.get("source") != "agent":
            out.append(f"[{st['step_id']}] {str(st.get('source')).upper()}: {clip(st.get('message') or '', 60000, 2000)}")
            continue
        if st.get("message"):
            out.append(f"[{st['step_id']}] SAYS: {clip(st['message'], 700, 300)}")
        results = {r.get("source_call_id"): r for r in (st.get("observation") or {}).get("results") or []}
        for tc in st.get("tool_calls") or []:
            args = tc.get("arguments") or {}
            body = args.get("command") or args.get("prompt") or json.dumps(args) if isinstance(args, dict) else args
            res = (results.get(tc.get("tool_call_id")) or {}).get("content")
            out.append(f"[{st['step_id']}] {tc.get('function_name')}: {clip(body, 900, 200)}\n  -> {clip(res or '', 350, 350)}")
    return f"sub-agent steps omitted: {side}\n" + "\n".join(out)


def clip(text: object, head: int, tail: int) -> str:
    """`text` whole when it is short, else its first `head` and last `tail` characters."""
    text = text if isinstance(text, str) else json.dumps(text)
    return text if len(text) <= head + tail else f"{text[:head]}\n[...{len(text) - head - tail} chars...]\n{text[-tail:]}"


class Claude:
    """A reader that asks `claude -p`: Claude Code, signed in, on the PATH. It may only read: the
    files it is handed, and the folders named."""

    def __init__(self, model: str = "sonnet", timeout: int = TIMEOUT) -> None:
        if shutil.which("claude") is None:
            raise RuntimeError("no `claude` on the PATH: a reader named by a model alias runs Claude Code")
        self.name, self.timeout = model, timeout

    def __call__(self, prompt: str, schema: dict, files: dict[str, str], dirs: list[Path]) -> tuple[dict, dict]:
        # This may run inside a Claude Code session; the reader is not part of it.
        env = {k: v for k, v in os.environ.items()
               if not (k.startswith("CLAUDE_CODE_") or k in {"CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_AGENT_SDK_VERSION"})}
        began = time.time()
        with tempfile.TemporaryDirectory(prefix="fathom-read-") as tmp:
            for name, text in files.items():
                (Path(tmp) / name).write_text(text)
            done = subprocess.run(
                ["claude", "-p", "--model", self.name, "--output-format", "json", "--json-schema", json.dumps(schema),
                 "--tools", "Read,Grep,Glob", "--allowedTools", "Read", "Grep", "Glob", *[a for d in dirs for a in ("--add-dir", str(d))],
                 "--strict-mcp-config", "--setting-sources", "project", "--no-session-persistence", "--disable-slash-commands",
                 prompt],
                cwd=tmp, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=self.timeout)
        try:
            reply = json.loads(done.stdout)
        except ValueError:
            raise RuntimeError(f"claude exited {done.returncode}: {(done.stderr or done.stdout).strip()[-500:]}") from None
        if reply.get("is_error") or not isinstance(reply.get("structured_output"), dict):
            raise RuntimeError(f"claude: {reply.get('subtype')}: {str(reply.get('result'))[:500]}")
        return reply["structured_output"], {
            "model": next(iter(reply.get("modelUsage") or {}), self.name),
            "seconds": round(time.time() - began),
            "cost_usd": reply.get("total_cost_usd"),
        }


# USD per million tokens, input and output, on 3 Nov 2026; a model not here gets no cost
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
LONG = {"claude-haiku-5-5": (100_000, 5.0)}  # a prompt of more tokens than this costs so many times more
BATCH_BYTES, BATCH_JOBS = 50_000_000, 50_000  # what one batch is given at most: the API takes 256 MB and 100,000 requests, a home uplink less


class ClaudeAPI:
    """A reader that asks the Claude API directly, in one call, with the files it is handed in the
    prompt. It costs a fraction of what Claude Code does for the same reading, and cannot open
    folders: it refuses a job that names any. It needs the `anthropic` package (the `api` extra)
    and a key, `ANTHROPIC_API_KEY` or `key`. `effort` is how hard the model works on the answer
    (low, medium, high, ...); with none it is the model's own default.

    Beyond answering, it can say what a job would take before it is run (`count`, `price`,
    `limit`), and hand many jobs to the Message Batches API (`submit`, `collect`): half the price,
    answered within a day and mostly within the hour."""

    def __init__(self, model: str = "claude-haiku-5-5", effort: str | None = None, key: str | None = None, max_tokens: int = 64_000) -> None:
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("the Claude API reader needs the anthropic package: install fathom with its `api` extra") from None
        if not (key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError("no ANTHROPIC_API_KEY: set it, or pass a key, or a reader of your own")
        self.name, self.effort, self.max_tokens = model, effort, max_tokens
        self.limit = LONG.get(model, (None, 1.0))[0]  # the tokens a prompt may have before it costs more: a longer job is better split
        self._client = anthropic.Anthropic(api_key=key, max_retries=6) if key else anthropic.Anthropic(max_retries=6)

    def __call__(self, prompt: str, schema: dict, files: dict[str, str], dirs: list[Path]) -> tuple[dict, dict]:
        if dirs:
            raise RuntimeError("the Claude API reader cannot open folders, and this job names some: use a `claude` model alias for it")
        began = time.time()
        with self._client.messages.stream(**self._params(prompt, schema, files)) as stream:
            reply = stream.get_final_message()
        answer, about = self._read(reply)
        return answer, {**about, "seconds": round(time.time() - began)}

    def count(self, prompt: str, schema: dict, files: dict[str, str]) -> int:
        """The tokens the model would be sent for a job, as the API counts them. It costs nothing."""
        params = self._params(prompt, schema, files)
        return self._client.messages.count_tokens(**{k: params[k] for k in ("model", "messages", "output_config")}).input_tokens

    def price(self, sent: int, written: int, batch: bool = False) -> float | None:
        """What one call costs in USD, by the tokens sent and written; half through a batch. None for a model with no known price."""
        if self.name not in PRICES:
            return None
        over, times = LONG.get(self.name, (None, 1.0))
        rate = (times if over is not None and sent > over else 1.0) * (0.5 if batch else 1.0)
        return round((sent * PRICES[self.name][0] + written * PRICES[self.name][1]) * rate / 1e6, 6)

    def submit(self, jobs: dict[str, tuple[str, dict, dict[str, str]]]) -> list[str]:
        """Hand jobs to the Message Batches API, each `(prompt, schema, files)` under an id of its
        own (letters, digits, `_` and `-`). Returns the ids of the batches made: more than one when
        the jobs are too many or too large for one."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        made: list[str] = []
        requests, size = [], 0
        try:
            for name, (prompt, schema, files) in jobs.items():
                params = self._params(prompt, schema, files)
                weight = len(json.dumps(params))
                if requests and (size + weight > BATCH_BYTES or len(requests) >= BATCH_JOBS):
                    made.append(self._client.messages.batches.create(requests=requests).id)
                    requests, size = [], 0
                requests.append(Request(custom_id=name, params=MessageCreateParamsNonStreaming(**params)))
                size += weight
            if requests:
                made.append(self._client.messages.batches.create(requests=requests).id)
        except BaseException:  # not all could be handed in: the ones that were would be paid for and never read
            for b in made:
                try:
                    self._client.messages.batches.cancel(b)
                except Exception:  # noqa: BLE001, S110 - the first failure is the one to report
                    pass
            raise
        return made

    def collect(self, batches: list[str]) -> dict[str, tuple[dict, dict] | Exception] | None:
        """The answers to the jobs of some batches, by job id: `(answer, about)`, or the error that
        kept a job from being answered. None while a batch is still being worked on."""
        if any(self._client.messages.batches.retrieve(b).processing_status != "ended" for b in batches):
            return None
        out: dict[str, tuple[dict, dict] | Exception] = {}
        for b in batches:
            for row in self._client.messages.batches.results(b):
                try:
                    if row.result.type != "succeeded":
                        why = getattr(getattr(getattr(row.result, "error", None), "error", None), "message", None)
                        raise RuntimeError(f"the batch did not answer this job: {row.result.type}" + (f": {why}" if why else ""))
                    out[row.custom_id] = self._read(row.result.message, batch=True)
                except Exception as e:  # noqa: BLE001 - one job's failure is its own
                    out[row.custom_id] = e
        return out

    def _params(self, prompt: str, schema: dict, files: dict[str, str]) -> dict:
        handed = "".join(f'<file name="{name}">\n{text}\n</file>\n\n' for name, text in files.items())
        return {"model": self.name, "max_tokens": self.max_tokens,
                "output_config": {**({"effort": self.effort} if self.effort else {}), "format": {"type": "json_schema", "schema": _closed(schema)}},
                "messages": [{"role": "user", "content": handed + prompt}]}

    def _read(self, reply: Any, batch: bool = False) -> tuple[dict, dict]:
        if reply.stop_reason != "end_turn":
            raise RuntimeError(f"{self.name} stopped with {reply.stop_reason} before it had answered")
        answer = json.loads(next(b.text for b in reply.content if b.type == "text"))
        sent = reply.usage.input_tokens + (reply.usage.cache_read_input_tokens or 0) + (reply.usage.cache_creation_input_tokens or 0)
        return answer, {"model": reply.model, "cost_usd": self.price(sent, reply.usage.output_tokens, batch),
                        "tokens": {"input": sent, "output": reply.usage.output_tokens}}


def _closed(schema: object) -> object:
    """`schema` with every object closed to other properties, as the API's JSON output asks."""
    if isinstance(schema, list):
        return [_closed(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _closed(v) for k, v in schema.items()}
    if out.get("type") == "object":
        out.setdefault("additionalProperties", False)
    return out
