"""Core data types: ContextItem, Trace, CompileResult, Action."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ContextItem:
    """A single unit entering the context window.

    A trace is an ordered list of these — chat turns, tool results, retrieved
    docs, memories. The compiler treats them uniformly but uses ``role``/``kind``
    to decide what is safe to reorder, compress, or drop.
    """

    content: str
    role: str = "user"  # system | user | assistant | tool
    kind: str = "message"  # message | tool_result | doc | memory | fact
    id: Optional[str] = None
    # If True the item is never compressed, dropped, or moved out of order.
    pinned: bool = False
    # 0..1 importance. Computed by the salience pass if left None.
    salience: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ContextItem":
        if isinstance(d, str):
            return cls(content=d)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        meta = dict(d.get("meta", {}))
        for k, v in d.items():
            if k not in known:
                meta[k] = v
        return cls(
            content=d.get("content", ""),
            role=d.get("role", "user"),
            kind=d.get("kind", "message"),
            id=d.get("id"),
            pinned=bool(d.get("pinned", False)),
            salience=d.get("salience"),
            meta=meta,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Trace:
    """An ordered context, plus the task it is meant to serve.

    ``task`` (the user's current question/objective) anchors salience scoring —
    items relevant to the task survive compression; tangents get trimmed.
    """

    items: List[ContextItem] = field(default_factory=list)
    task: Optional[str] = None
    model: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Trace":
        # Accept either {"items": [...]} or a bare list of messages.
        if isinstance(d, list):
            d = {"items": d}
        return cls(
            items=[ContextItem.from_dict(i) for i in d.get("items", [])],
            task=d.get("task"),
            model=d.get("model"),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "meta": self.meta,
            "items": [i.to_dict() for i in self.items],
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


@dataclass
class Action:
    """One transformation the compiler applied — the audit trail."""

    type: str  # dedup | truncate | drop | anchor | reorder | normalize
    detail: str
    target: Optional[str] = None
    tokens_saved: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CompileResult:
    """Output of a compile: the new context plus the full before/after report."""

    items: List[ContextItem]
    task: Optional[str]
    model: Optional[str]
    tokens_before: int
    tokens_after: int
    rot_before: "Any"  # RotReport (avoid import cycle)
    rot_after: "Any"
    actions: List[Action] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def savings_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return 100.0 * self.tokens_saved / self.tokens_before

    def to_messages(self) -> List[Dict[str, str]]:
        """Render as a plain chat-style message list for handing to a model."""
        return [{"role": i.role, "content": i.content} for i in self.items]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task,
            "model": self.model,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "savings_pct": round(self.savings_pct, 1),
            "rot_before": self.rot_before.to_dict(),
            "rot_after": self.rot_after.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
            "items": [i.to_dict() for i in self.items],
        }

    def summary(self) -> str:
        lines = [
            "ContextForge compile report",
            "=" * 40,
            f"tokens   {self.tokens_before:>10,} -> {self.tokens_after:>10,}"
            f"  ({self.savings_pct:+.1f}%)",
            f"rot risk {self.rot_before.total:>10.0f} -> {self.rot_after.total:>10.0f}"
            f"  ({self.rot_before.level} -> {self.rot_after.level})",
            f"items    {len(self.actions)} actions applied",
        ]
        return "\n".join(lines)
