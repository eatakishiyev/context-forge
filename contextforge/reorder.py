"""Reordering & edge-anchoring — fight "lost in the middle".

Reordering a *conversation* would break it (turn 12 may reference turn 11), so we
don't shuffle dialogue. Instead we:

  1. extract the highest-salience facts into a compact "current state" block and
     anchor it at the *front* of the window, and
  2. repeat a terse recap of those facts at the *back*, just before the live turn.

Both edges are where attention is strongest. Free-floating context items (docs,
memories, facts — things with no conversational order) *can* be reordered, and
are sorted by salience toward the edges.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from .textutil import keywords
from .tokens import count_tokens
from .types import Action, ContextItem

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Sentences that look like durable state worth anchoring.
_FACTY = re.compile(
    r"\b(is|are|was|were|must|should|will|named|called|equals?|=|prefers?|"
    r"requires?|deadline|budget|id|key|email|account|version|chose|decided|"
    r"because|note that|remember)\b",
    re.I,
)


def extract_salient_facts(
    items: List[ContextItem], task: str = None, max_facts: int = 12
) -> List[str]:
    """Pull terse, durable-looking statements from the highest-salience items."""
    task_kw = keywords(task) if task else set()
    scored: List[Tuple[float, str]] = []
    for it in items:
        sal = it.salience if it.salience is not None else 0.5
        if sal < 0.4 and not it.pinned:
            continue
        for sent in _SENT_SPLIT.split(it.content.strip()):
            s = sent.strip()
            if not (20 <= len(s) <= 240):
                continue
            score = sal
            if _FACTY.search(s):
                score += 0.3
            if task_kw and keywords(s) & task_kw:
                score += 0.3
            if it.pinned:
                score += 0.5
            scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    facts, seen = [], set()
    for _, s in scored:
        key = s.lower()[:60]
        if key in seen:
            continue
        seen.add(key)
        facts.append(s)
        if len(facts) >= max_facts:
            break
    return facts


def _anchor_item(facts: List[str], position: str) -> ContextItem:
    if position == "header":
        head = "## Current state (ContextForge — load-bearing facts)\n"
        body = "\n".join(f"- {f}" for f in facts)
    else:
        head = "## Key facts recap (ContextForge — do not lose these)\n"
        body = "\n".join(f"- {f}" for f in facts)
    return ContextItem(
        content=head + body,
        role="system",
        kind="fact",
        id=f"cf_anchor_{position}",
        pinned=True,
        salience=1.0,
        meta={"cf_generated": True},
    )


def edge_anchor(
    items: List[ContextItem], task: str = None, max_facts: int = 12
) -> Tuple[List[ContextItem], List[Action]]:
    """Insert a state header at the front and a recap footer at the back."""
    facts = extract_salient_facts(items, task, max_facts)
    if not facts:
        return items, []

    header = _anchor_item(facts, "header")
    footer = _anchor_item(facts[: max(3, max_facts // 2)], "footer")

    # Header goes after any leading pinned system prompt; footer goes at the very
    # end (closest to the model's "what do I do now" attention peak).
    lead = 0
    while lead < len(items) and items[lead].role == "system" and items[lead].pinned:
        lead += 1
    new_items = items[:lead] + [header] + items[lead:] + [footer]

    actions = [
        Action("anchor", f"state header ({len(facts)} facts)",
               target="cf_anchor_header", tokens_saved=-count_tokens(header.content)),
        Action("anchor", "recap footer at window edge",
               target="cf_anchor_footer", tokens_saved=-count_tokens(footer.content)),
    ]
    return new_items, actions


def reorder_free_items(items: List[ContextItem]) -> Tuple[List[ContextItem], List[Action]]:
    """Sort reorderable (non-dialogue) items so the most salient sit at the edges.

    Conversational items (user/assistant messages, tool results) keep their
    order; only docs/memories/facts get rearranged into an edges-in pattern.
    """
    reorderable_kinds = {"doc", "memory", "fact"}
    idxs = [i for i, it in enumerate(items)
            if it.kind in reorderable_kinds and not it.pinned]
    if len(idxs) < 3:
        return items, []

    block = [items[i] for i in idxs]
    block.sort(key=lambda it: (it.salience or 0.5), reverse=True)
    # edges-in: best, 3rd, 5th ... at front; ... 6th, 4th, 2nd at back
    front, back = [], []
    for k, it in enumerate(block):
        (front if k % 2 == 0 else back).append(it)
    arranged = front + list(reversed(back))

    out = list(items)
    for slot, it in zip(idxs, arranged):
        out[slot] = it
    return out, [Action("reorder", f"{len(block)} free items arranged edges-in")]
