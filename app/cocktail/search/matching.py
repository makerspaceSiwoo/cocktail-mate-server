"""Fuzzy similarity primitives for the cocktail name search core.

Uses rapidfuzz.fuzz.partial_ratio (0–100 scale) instead of hand-rolled
trigram Dice. Asymmetric window eliminates the length penalty that caused
short queries to miss longer names (e.g. '깔루아' missing '칼루아 밀크').

Pure module — no FastAPI/DB imports.
"""

from rapidfuzz import fuzz

# Thresholds on the 0–100 scale (tuned against §10-2 spec cases).
# ko: Korean jamo-normalized similarity; en: English similarity.
THRESHOLDS = {"ko": 70, "en": 78}

# Minimum length of the normalized query to run fuzzy matching.
# Short queries (< 3 jamo/chars) over-match at partial_ratio=100 against
# any name that contains them (spec §9/§6-6).
MIN_FUZZY_LEN = 3


def similarity(a: str, b: str) -> float:
    """Return partial_ratio similarity between two strings (0–100).

    rapidfuzz.fuzz.partial_ratio slides the shorter string across the longer
    one and returns the best window score — eliminates length penalty.
    Never returns NaN; always finite.
    """
    return fuzz.partial_ratio(a, b)


def rank_score(sim: float, field: str) -> float:
    """Normalize similarity to [0, 1] relative to the field threshold (spec §5-4).

    Maps sim == threshold → 0.0 and sim == 100 → 1.0.
    """
    t = THRESHOLDS[field]
    return (sim - t) / (100 - t)
