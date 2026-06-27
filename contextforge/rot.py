"""Context-rot risk scoring.

A 0–100 score (higher = more likely the model silently degrades on this context),
broken into auditable components. Grounded in the empirically observed failure
modes of long-context models (Chroma's context-rot work, 2025):

  * load          — token pressure. Degradation begins *well before* the
                    advertised window and has a sharp knee; we model that with a
                    ramp between ``danger_start`` and ``danger_full``.
  * redundancy    — repeated / near-duplicate spans dilute attention.
  * middle_burial — "lost in the middle": salient items stranded in the center
                    of a long context are attended to least.
  * fragmentation — many small disjoint items are harder to integrate than a few
                    coherent ones.

The score is a *risk estimate*, not a guarantee — the benchmark harness is how a
team calibrates it against their own accuracy numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .salience import score_salience
from .textutil import char_ngrams
from .tokens import count_items_tokens, count_tokens
from .types import ContextItem

# Defaults reflect the observed knee on "long context" frontier models: real
# degradation ramps up far below the nominal 1M-token ceiling.
DEFAULT_DANGER_START = 180_000
DEFAULT_DANGER_FULL = 400_000


@dataclass
class RotReport:
    total: float
    level: str
    components: Dict[str, float] = field(default_factory=dict)
    tokens: int = 0
    n_items: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _level(score: float) -> str:
    if score < 25:
        return "low"
    if score < 50:
        return "moderate"
    if score < 75:
        return "high"
    return "severe"


def _load_component(tokens: int, danger_start: int, danger_full: int) -> float:
    if tokens <= danger_start:
        # Gentle pre-ramp so big-but-safe contexts still register some pressure.
        return 100.0 * 0.25 * (tokens / danger_start) if danger_start else 0.0
    if tokens >= danger_full:
        return 100.0
    frac = (tokens - danger_start) / (danger_full - danger_start)
    # Convex ramp — the "knee". Starts at 25 (end of pre-ramp), climbs to 100.
    return 25.0 + 75.0 * (frac ** 0.7)


def _redundancy_component(items: List[ContextItem]) -> float:
    """Mean pairwise near-duplication, weighted toward the worst offenders."""
    texts = [i.content for i in items if i.content.strip()]
    if len(texts) < 2:
        return 0.0
    grams = [char_ngrams(t, 5) for t in texts]
    # Cap pairwise comparisons for very long traces — sample a window.
    max_pairs = 4000
    pairs = 0
    acc = 0.0
    n = len(grams)
    step = 1
    if n * (n - 1) // 2 > max_pairs:
        step = max(1, (n * (n - 1) // 2) // max_pairs)
    counter = 0
    for i in range(n):
        for j in range(i + 1, n):
            counter += 1
            if counter % step:
                continue
            a, b = grams[i], grams[j]
            if not a or not b:
                continue
            inter = len(a & b)
            union = len(a | b)
            if union:
                acc += inter / union
                pairs += 1
    if not pairs:
        return 0.0
    mean_sim = acc / pairs
    # Redundancy hurts fast; small amounts of overlap are normal in dialogue.
    return min(100.0, mean_sim * 180.0)


def _middle_burial_component(items: List[ContextItem]) -> float:
    """How much salient content is stranded in the middle third."""
    n = len(items)
    if n < 6:
        return 0.0
    third = n / 3.0
    mid_sal = 0.0
    edge_sal = 0.0
    for idx, it in enumerate(items):
        sal = it.salience if it.salience is not None else 0.5
        weight = sal * (count_tokens(it.content) + 1)
        if third <= idx < 2 * third:
            mid_sal += weight
        else:
            edge_sal += weight
    total = mid_sal + edge_sal
    if total == 0:
        return 0.0
    # If most salience sits in the middle, that's bad. Baseline middle share is
    # ~1/3; penalize the excess above that.
    mid_share = mid_sal / total
    excess = max(0.0, mid_share - 0.33) / 0.67
    return min(100.0, excess * 100.0)


def _fragmentation_component(n_items: int) -> float:
    # Logistic-ish: few items -> low, hundreds of items -> high.
    if n_items <= 12:
        return 0.0
    return min(100.0, 100.0 * (1 - (1 / (1 + (n_items - 12) / 60.0))))


def rot_score(
    items: List[ContextItem],
    task: Optional[str] = None,
    model: Optional[str] = None,
    *,
    danger_start: int = DEFAULT_DANGER_START,
    danger_full: int = DEFAULT_DANGER_FULL,
    weights: Optional[Dict[str, float]] = None,
) -> RotReport:
    """Compute the context-rot risk report for a list of items."""
    if any(i.salience is None for i in items):
        score_salience(items, task)

    tokens = count_items_tokens(items, model)
    n = len(items)

    w = {"load": 0.40, "redundancy": 0.20, "middle_burial": 0.20, "fragmentation": 0.20}
    if weights:
        w.update(weights)

    comp = {
        "load": round(_load_component(tokens, danger_start, danger_full), 1),
        "redundancy": round(_redundancy_component(items), 1),
        "middle_burial": round(_middle_burial_component(items), 1),
        "fragmentation": round(_fragmentation_component(n), 1),
    }
    total = sum(comp[k] * w[k] for k in comp)

    notes = []
    if comp["load"] >= 50:
        notes.append(
            f"{tokens:,} tokens is past the degradation knee "
            f"({danger_start:,}–{danger_full:,})."
        )
    if comp["redundancy"] >= 40:
        notes.append("High near-duplicate content — dedup will help.")
    if comp["middle_burial"] >= 40:
        notes.append("Salient facts are buried mid-context — reorder to edges.")
    if comp["fragmentation"] >= 50:
        notes.append(f"{n} fragments — consolidation will reduce attention load.")

    return RotReport(
        total=round(total, 1),
        level=_level(total),
        components=comp,
        tokens=tokens,
        n_items=n,
        notes=notes,
    )
