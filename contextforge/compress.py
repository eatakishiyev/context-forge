"""Compression passes — extractive, deterministic, auditable.

The v0 compiler does *lossless-ish* compression only: it removes redundancy and
trims provably stale, low-salience material. It never paraphrases (which risks
silently dropping the one fact that mattered — the exact failure ContextForge
exists to prevent). LLM-backed abstractive summarization is a pluggable upgrade
(see ``Policy.summarizer``), kept off the default path on purpose.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .textutil import similarity
from .tokens import count_tokens
from .types import Action, ContextItem

Summarizer = Callable[[str], str]


def normalize_whitespace(items: List[ContextItem]) -> Tuple[List[ContextItem], List[Action]]:
    """Collapse runs of whitespace/blank lines. Cheap, safe, surprisingly large
    on tool dumps and pasted logs."""
    actions: List[Action] = []
    for it in items:
        before = count_tokens(it.content)
        lines = [ln.rstrip() for ln in it.content.splitlines()]
        # collapse 3+ blank lines to 1
        out, blanks = [], 0
        for ln in lines:
            if ln == "":
                blanks += 1
                if blanks <= 1:
                    out.append(ln)
            else:
                blanks = 0
                out.append(" ".join(ln.split()) if "  " in ln else ln)
        new = "\n".join(out).strip()
        after = count_tokens(new)
        if after < before:
            it.content = new
            actions.append(
                Action("normalize", f"whitespace ({before-after} tok)",
                       target=it.id, tokens_saved=before - after)
            )
    return items, actions


def dedup_items(
    items: List[ContextItem], threshold: float = 0.9
) -> Tuple[List[ContextItem], List[Action]]:
    """Drop near-duplicate items, keeping the *latest* occurrence (it usually has
    the freshest framing) and never dropping pinned items."""
    actions: List[Action] = []
    keep: List[ContextItem] = []
    # iterate newest -> oldest so the survivor is the most recent copy
    seen: List[ContextItem] = []
    for it in reversed(items):
        if it.pinned or not it.content.strip():
            seen.append(it)
            continue
        dup_of = None
        for kept in seen:
            if kept.kind != it.kind:
                continue
            if similarity(it.content, kept.content) >= threshold:
                dup_of = kept
                break
        if dup_of is not None:
            actions.append(
                Action("dedup", f"near-duplicate of {dup_of.id or '∅'}",
                       target=it.id, tokens_saved=count_tokens(it.content))
            )
        else:
            seen.append(it)
    kept_ids = {id(x) for x in seen}
    keep = [it for it in items if id(it) in kept_ids]
    return keep, actions


def truncate_stale(
    items: List[ContextItem],
    *,
    salience_floor: float = 0.35,
    keep_recent: int = 6,
    head_chars: int = 600,
    tail_chars: int = 300,
    summarizer: Optional[Summarizer] = None,
) -> Tuple[List[ContextItem], List[Action]]:
    """Trim long, old, low-salience items.

    The most recent ``keep_recent`` items are left untouched (recency is
    load-bearing for agents). Older items below ``salience_floor`` that are large
    get either summarized (if a summarizer is supplied) or head/tail-clipped with
    an explicit elision marker so nothing is *silently* removed.
    """
    actions: List[Action] = []
    n = len(items)
    for idx, it in enumerate(items):
        if it.pinned:
            continue
        if idx >= n - keep_recent:
            continue
        sal = it.salience if it.salience is not None else 0.5
        before = count_tokens(it.content)
        if sal >= salience_floor or before < 250:
            continue

        if summarizer is not None:
            new = summarizer(it.content).strip()
            kind = "summary"
        else:
            body = it.content.strip()
            if len(body) <= head_chars + tail_chars:
                continue
            elided = len(body) - head_chars - tail_chars
            new = (
                body[:head_chars].rstrip()
                + f"\n… [ContextForge elided {elided} chars — low salience] …\n"
                + body[-tail_chars:].lstrip()
            )
            kind = "clip"

        after = count_tokens(new)
        if after < before:
            it.content = new
            it.meta["cf_compressed"] = kind
            actions.append(
                Action("truncate", f"{kind} stale item (sal={sal:.2f})",
                       target=it.id, tokens_saved=before - after)
            )
    return items, actions
