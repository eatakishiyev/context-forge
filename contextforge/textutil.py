"""Small, dependency-free text helpers shared across passes."""

from __future__ import annotations

import re
from typing import List, Set

_WORD_RE = re.compile(r"[a-z0-9]+")
# Light stopword set — enough to stop common words from dominating overlap.
_STOP = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "this", "that", "it", "i", "you", "we", "they", "he", "she", "do", "does",
    "did", "can", "will", "would", "should", "could", "have", "has", "had",
    "not", "no", "yes", "so", "from", "your", "our", "my", "me", "us",
}


def tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def keywords(text: str) -> Set[str]:
    return {t for t in tokens(text) if t not in _STOP and len(t) > 2}


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def similarity(text_a: str, text_b: str, n: int = 4) -> float:
    """Char-ngram Jaccard similarity in [0, 1]. Cheap near-duplicate detector."""
    return jaccard(char_ngrams(text_a, n), char_ngrams(text_b, n))


def overlap(text: str, vocab: Set[str]) -> float:
    """Precision: fraction of ``text``'s keywords that appear in ``vocab``."""
    kw = keywords(text)
    if not kw or not vocab:
        return 0.0
    return len(kw & vocab) / len(kw)


def coverage(text: str, vocab: Set[str]) -> float:
    """Recall: fraction of ``vocab`` (the task's terms) that ``text`` covers.

    Better than precision for salience — it rewards items that *address the task*
    regardless of their length, instead of penalizing long, informative facts.
    """
    if not vocab:
        return 0.0
    kw = keywords(text)
    if not kw:
        return 0.0
    return len(kw & vocab) / len(vocab)
