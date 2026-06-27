"""Token counting.

Uses ``tiktoken`` when available for accurate counts; otherwise falls back to a
conservative character-based heuristic. The heuristic is deliberately model
agnostic — for relative deltas (the thing the product reports) it is more than
good enough, and it keeps the core dependency-free so the CLI runs anywhere.
"""

from __future__ import annotations

from typing import Iterable, Optional

# Average characters per token across English + code. Real tokenizers land
# around 3.5–4.5 chars/token; 4.0 is a stable midpoint for *relative* deltas.
_CHARS_PER_TOKEN = 4.0

_ENCODER_CACHE: dict = {}


def _get_encoder(model: Optional[str]):
    try:
        import tiktoken  # type: ignore
    except Exception:
        return None

    key = model or "cl100k_base"
    if key in _ENCODER_CACHE:
        return _ENCODER_CACHE[key]

    enc = None
    try:
        if model:
            enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = None
    if enc is None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            enc = None

    _ENCODER_CACHE[key] = enc
    return enc


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Return the token count of ``text`` for ``model`` (best effort)."""
    if not text:
        return 0
    enc = _get_encoder(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return int(round(len(text) / _CHARS_PER_TOKEN))


def count_items_tokens(items: Iterable, model: Optional[str] = None) -> int:
    """Total tokens across a sequence of ``ContextItem`` (or strings)."""
    total = 0
    for it in items:
        content = it.content if hasattr(it, "content") else str(it)
        # Small per-message overhead (role + delimiters), mirroring chat formats.
        total += count_tokens(content, model) + 4
    return total


def using_accurate_tokenizer() -> bool:
    """True if tiktoken is installed and usable."""
    return _get_encoder(None) is not None
