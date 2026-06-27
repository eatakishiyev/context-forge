"""Per-model rot calibration profiles.

The rot score's degradation knee (``danger_start`` / ``danger_full``) is a
*property of the model*, not a universal constant. A profile captures that knee —
fitted from real accuracy-vs-context-size measurements (see ``calibrate.py``) —
so the score reflects how *your* model actually degrades.

Profiles live in a JSON registry. Resolution order for the registry path:
  1. explicit path argument
  2. $CONTEXTFORGE_PROFILES
  3. <repo>/profiles/profiles.json
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

from .rot import DEFAULT_DANGER_START, DEFAULT_DANGER_FULL


@dataclass
class ModelProfile:
    name: str
    danger_start: int = DEFAULT_DANGER_START
    danger_full: int = DEFAULT_DANGER_FULL
    # optional per-component weight overrides for rot_score
    weights: Optional[Dict[str, float]] = None
    # provenance
    n_samples: int = 0
    fit_r2: Optional[float] = None
    notes: str = ""

    def rot_kwargs(self) -> dict:
        kw = {"danger_start": self.danger_start, "danger_full": self.danger_full}
        if self.weights:
            kw["weights"] = self.weights
        return kw

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelProfile":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# A conservative built-in default reflecting the observed knee on current
# frontier "long-context" models — used until a model is calibrated.
DEFAULT_PROFILE = ModelProfile(
    name="default",
    danger_start=DEFAULT_DANGER_START,
    danger_full=DEFAULT_DANGER_FULL,
    notes="Uncalibrated default. Run `contextforge calibrate` for per-model knees.",
)


def default_registry_path() -> str:
    env = os.environ.get("CONTEXTFORGE_PROFILES")
    if env:
        return env
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo, "profiles", "profiles.json")


def load_registry(path: Optional[str] = None) -> Dict[str, ModelProfile]:
    path = path or default_registry_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: ModelProfile.from_dict(d) for name, d in raw.items()}


def save_profile(profile: ModelProfile, path: Optional[str] = None) -> str:
    path = path or default_registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    reg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            reg = json.load(f)
    reg[profile.name] = profile.to_dict()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2)
    return path


def get_profile(model: Optional[str], path: Optional[str] = None) -> ModelProfile:
    """Resolve a profile for ``model``; fall back to the built-in default.

    Tries an exact match first, then a prefix match (so ``claude-opus-4-8`` can
    match a registered ``claude-opus`` family profile).
    """
    if not model:
        return DEFAULT_PROFILE
    reg = load_registry(path)
    if model in reg:
        return reg[model]
    for name, prof in reg.items():
        if model.startswith(name) or name.startswith(model):
            return prof
    return DEFAULT_PROFILE
