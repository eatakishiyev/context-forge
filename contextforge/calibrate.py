"""Fit a per-model rot profile from accuracy-vs-context-size measurements.

Input: a list of (tokens, accuracy) points — how a given model's accuracy on a
recall task falls as the context grows. Output: a ``ModelProfile`` whose
``danger_start`` / ``danger_full`` bracket the degradation knee, so the rot score
matches that model's real behavior.

The fit is deliberately simple and robust to noisy points (no SciPy needed):
  * plateau  = max accuracy (small-context performance)
  * floor    = min accuracy (long-context performance)
  * danger_start = where accuracy has dropped 10% of the way to the floor
  * danger_full  = where accuracy has dropped 90% of the way to the floor
Crossings are linearly interpolated between bracketing points.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .profiles import ModelProfile
from .rot import DEFAULT_DANGER_START, DEFAULT_DANGER_FULL

Point = Tuple[float, float]  # (tokens, accuracy)


def _crossing(points: List[Point], threshold: float) -> Optional[float]:
    """First token value (ascending) where accuracy drops to/through threshold."""
    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if y0 >= threshold >= y1 and y0 != y1:
            frac = (y0 - threshold) / (y0 - y1)
            return x0 + frac * (x1 - x0)
        if y1 <= threshold and y0 <= threshold:
            return x0
    return None


def _r2(points: List[Point], start: float, full: float) -> float:
    """Coarse goodness-of-fit of a piecewise-linear knee to the points."""
    if full <= start:
        return 0.0
    ys = [y for _, y in points]
    plateau, floor = max(ys), min(ys)
    if plateau == floor:
        return 1.0
    ss_res = ss_tot = 0.0
    mean = sum(ys) / len(ys)
    for x, y in points:
        if x <= start:
            pred = plateau
        elif x >= full:
            pred = floor
        else:
            t = (x - start) / (full - start)
            pred = plateau - t * (plateau - floor)
        ss_res += (y - pred) ** 2
        ss_tot += (y - mean) ** 2
    return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot else 1.0


def fit_knee(measurements: Sequence[Point]) -> Tuple[int, int, float]:
    """Return (danger_start, danger_full, r2) fitted to the measurements."""
    pts = sorted((float(t), float(a)) for t, a in measurements)
    if len(pts) < 2:
        return DEFAULT_DANGER_START, DEFAULT_DANGER_FULL, 0.0

    ys = [y for _, y in pts]
    plateau, floor = max(ys), min(ys)
    span = plateau - floor

    # Effectively no degradation observed → no meaningful knee; keep defaults but
    # push the knee out past the largest measured size.
    if span < 0.05:
        big = int(pts[-1][0])
        return max(DEFAULT_DANGER_START, big), max(DEFAULT_DANGER_FULL, big * 2), 0.0

    hi = plateau - 0.10 * span  # degradation begins
    lo = floor + 0.10 * span    # degradation (nearly) complete

    start = _crossing(pts, hi)
    full = _crossing(pts, lo)
    if start is None:
        start = pts[0][0]
    if full is None or full <= start:
        full = pts[-1][0]
    r2 = _r2(pts, start, full)
    return int(round(start)), int(round(full)), round(r2, 3)


def fit_profile(
    model: str,
    measurements: Sequence[Point],
    weights: Optional[dict] = None,
    notes: str = "",
) -> ModelProfile:
    start, full, r2 = fit_knee(measurements)
    return ModelProfile(
        name=model,
        danger_start=start,
        danger_full=full,
        weights=weights,
        n_samples=len(measurements),
        fit_r2=r2,
        notes=notes or f"Fitted from {len(measurements)} measurements (R²={r2}).",
    )
