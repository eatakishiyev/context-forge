"""The ContextCompiler — orchestrates the passes into one compile().

Pipeline (each stage is independently toggleable via Policy):

    salience  ->  normalize  ->  dedup  ->  truncate stale  ->
    reorder free items  ->  edge-anchor salient facts  ->  enforce budget

Before/after rot scores and a full action log are attached to the result so the
transformation is fully auditable — the "what the model actually saw" story.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional

from . import compress, reorder, budget as budget_mod
from .rot import rot_score, DEFAULT_DANGER_START, DEFAULT_DANGER_FULL
from .salience import score_salience
from .tokens import count_items_tokens
from .types import Action, CompileResult, ContextItem, Trace


@dataclass
class Policy:
    """Knobs for a compile. Defaults are conservative (loss-averse)."""

    target_tokens: Optional[int] = None  # None = no hard budget
    # passes
    normalize: bool = True
    dedup: bool = True
    truncate: bool = True
    reorder: bool = True
    anchor: bool = True
    budget: bool = True
    # tuning
    dedup_threshold: float = 0.9
    salience_floor: float = 0.35
    keep_recent: int = 6
    max_facts: int = 12
    # optional abstractive summarizer (str -> str); off by default on purpose
    summarizer: Optional[object] = None
    # per-model rot calibration
    danger_start: int = DEFAULT_DANGER_START
    danger_full: int = DEFAULT_DANGER_FULL
    weights: Optional[dict] = None

    def rot_kwargs(self):
        kw = {"danger_start": self.danger_start, "danger_full": self.danger_full}
        if self.weights:
            kw["weights"] = self.weights
        return kw

    def apply_profile(self, profile) -> "Policy":
        """Calibrate this policy's rot knee from a ModelProfile (in place)."""
        self.danger_start = profile.danger_start
        self.danger_full = profile.danger_full
        if profile.weights:
            self.weights = profile.weights
        return self


class ContextCompiler:
    def __init__(self, target_tokens: Optional[int] = None, policy: Optional[Policy] = None):
        if policy is None:
            policy = Policy(target_tokens=target_tokens)
        elif target_tokens is not None:
            policy.target_tokens = target_tokens
        self.policy = policy

    def compile(self, trace: Trace) -> CompileResult:
        p = self.policy
        model = trace.model
        task = trace.task

        # Work on a deep copy so the caller's trace is never mutated.
        items: List[ContextItem] = copy.deepcopy(trace.items)
        for i, it in enumerate(items):
            if it.id is None:
                it.id = f"item_{i}"

        score_salience(items, task)
        tokens_before = count_items_tokens(items, model)
        rot_before = rot_score(items, task, model, **p.rot_kwargs())

        actions: List[Action] = []

        if p.normalize:
            items, a = compress.normalize_whitespace(items)
            actions += a
        if p.dedup:
            items, a = compress.dedup_items(items, threshold=p.dedup_threshold)
            actions += a
        if p.truncate:
            items, a = compress.truncate_stale(
                items,
                salience_floor=p.salience_floor,
                keep_recent=p.keep_recent,
                summarizer=p.summarizer,
            )
            actions += a
        if p.reorder:
            items, a = reorder.reorder_free_items(items)
            actions += a
        if p.anchor:
            items, a = reorder.edge_anchor(items, task, max_facts=p.max_facts)
            actions += a
        if p.budget and p.target_tokens:
            items, a = budget_mod.enforce_budget(items, p.target_tokens, model)
            actions += a

        # Re-score salience on the final set (anchors are pinned 1.0 already).
        score_salience(items, task)
        tokens_after = count_items_tokens(items, model)
        rot_after = rot_score(items, task, model, **p.rot_kwargs())

        return CompileResult(
            items=items,
            task=task,
            model=model,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            rot_before=rot_before,
            rot_after=rot_after,
            actions=actions,
        )
