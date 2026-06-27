"""Token budgeting — the hard ceiling.

After compression there may still be more context than the target budget. We drop
the lowest-salience, non-pinned items until we fit, newest-and-most-relevant
first to survive. Every drop is logged so the team can see exactly what the model
did *not* see.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .tokens import count_items_tokens, count_tokens
from .types import Action, ContextItem


def enforce_budget(
    items: List[ContextItem],
    target_tokens: Optional[int],
    model: Optional[str] = None,
) -> Tuple[List[ContextItem], List[Action]]:
    if not target_tokens:
        return items, []

    current = count_items_tokens(items, model)
    if current <= target_tokens:
        return items, []

    actions: List[Action] = []
    # Candidate drop order: lowest salience first; ties broken by oldest.
    order = sorted(
        range(len(items)),
        key=lambda i: (items[i].salience if items[i].salience is not None else 0.5, i),
    )
    drop: set = set()
    for i in order:
        if current <= target_tokens:
            break
        it = items[i]
        if it.pinned:
            continue
        saved = count_tokens(it.content) + 4
        drop.add(i)
        current -= saved
        actions.append(
            Action("drop", f"over budget (sal={it.salience:.2f})",
                   target=it.id, tokens_saved=saved)
        )

    kept = [it for i, it in enumerate(items) if i not in drop]
    return kept, actions
