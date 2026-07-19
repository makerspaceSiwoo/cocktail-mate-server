"""Trigram similarity primitives for cocktail search autocomplete.

Pure module — no FastAPI/DB imports.
"""

import math

THRESHOLDS = {"ko": 0.45, "en": 0.35}


def trigrams(s: str) -> set:
    """Compute the set of trigrams for a string (spec §6-6).

    Uses padding "  " + s + " " so any non-empty string yields ≥1 trigram.
    Returns empty set for falsy input.
    """
    if not s:
        return set()
    padded = "  " + s + " "
    return {padded[i : i + 3] for i in range(len(padded) - 2)}


def dice(a: str, b: str) -> float:
    """Sørensen–Dice coefficient over trigram sets (spec §6-6).

    Guards against denom == 0 → returns 0.0 (prevents NaN).
    """
    A = trigrams(a)
    B = trigrams(b)
    denom = len(A) + len(B)
    if denom == 0:
        return 0.0
    inter = len(A & B)
    return 2 * inter / denom


def safe(n: float) -> float:
    """Return n if finite, else 0.0 (defense-in-depth for sort comparators)."""
    return n if math.isfinite(n) else 0.0


def rank_score(sim: float, field: str) -> float:
    """Normalize similarity to [0, 1] relative to the field threshold (spec §5-4).

    Maps sim == threshold → 0.0 and sim == 1.0 → 1.0.
    """
    t = THRESHOLDS[field]
    return (sim - t) / (1 - t)
