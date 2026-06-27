"""Salience scoring — how important is each item to the current task.

This drives everything downstream: low-salience items get compressed or dropped
first, high-salience facts get lifted to the edges of the window. The score is a
transparent, weighted blend of signals (no model call required), so a team can
audit *why* something was trimmed.
"""

from __future__ import annotations

from typing import List, Optional

from .textutil import keywords, coverage
from .tokens import count_tokens
from .types import ContextItem

# Role priors — system instructions and the user's words matter more than
# verbose tool dumps by default.
_ROLE_WEIGHT = {
    "system": 0.95,
    "user": 0.75,
    "assistant": 0.55,
    "tool": 0.45,
}
_KIND_WEIGHT = {
    "fact": 0.95,
    "memory": 0.8,
    "doc": 0.6,
    "message": 0.6,
    "tool_result": 0.45,
}


def score_salience(
    items: List[ContextItem], task: Optional[str] = None
) -> List[ContextItem]:
    """Set ``item.salience`` in place for every item, returning the list.

    Signals blended:
      * task relevance  — keyword overlap with the task (the strongest signal)
      * recency         — later items are likelier to be load-bearing
      * role / kind     — priors on inherent importance
      * brevity         — huge low-relevance blobs are penalized (rot risk)
    Pinned items are forced to 1.0; explicitly provided salience is respected.
    """
    n = len(items)
    task_vocab = keywords(task) if task else set()

    for idx, it in enumerate(items):
        if it.pinned:
            it.salience = 1.0
            continue
        if it.salience is not None:
            continue

        recency = (idx + 1) / n if n else 1.0  # 0..1, newest ~1.0
        role = _ROLE_WEIGHT.get(it.role, 0.5)
        kind = _KIND_WEIGHT.get(it.kind, 0.5)
        relevance = coverage(it.content, task_vocab) if task_vocab else 0.5

        toks = count_tokens(it.content)
        # Penalize long items that don't earn their length via relevance.
        size_penalty = 0.0
        if toks > 800 and relevance < 0.15:
            size_penalty = 0.2

        score = (
            0.40 * relevance
            + 0.25 * recency
            + 0.20 * role
            + 0.15 * kind
            - size_penalty
        )
        it.salience = max(0.0, min(1.0, score))

    return items
