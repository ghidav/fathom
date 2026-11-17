"""Analysing rollouts: an agent's attempt at a task, with its mistakes flagged and its turns tagged,
and groups of them."""

from fathom.embedding import PARTS, OpenRouter
from fathom.group import Batch, Group, Hit
from fathom.reading import CAUSES, SEVERITIES, Claude, ClaudeAPI
from fathom.rollout import Rollout, Turn

__all__ = ["CAUSES", "PARTS", "SEVERITIES", "Batch", "Claude", "ClaudeAPI", "Group", "Hit", "OpenRouter", "Rollout", "Turn"]
