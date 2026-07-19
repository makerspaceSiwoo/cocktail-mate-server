"""In-memory cocktail search index with tiered item-first scoring.

Scoring tiers (spec §6-3/§6-4/§6-6 대응3):
  3 = exact match (normalized)  → score 1.0
  2 = prefix match              → score len(q) / len(target)
  1 = fuzzy (partial_ratio ≥ threshold)  → score rank_score(sim, field)
  1 = choseong-prefix           → score len(q) / len(cho)
  0 = no match (excluded)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.cocktail.autocomplete.matching import (
    MIN_FUZZY_LEN,
    THRESHOLDS,
    rank_score,
    similarity,
)
from app.cocktail.autocomplete.normalize import (
    CHOSEONG,
    extract_choseong,
    norm,
    sanitize_keyword,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMPTY: tuple[int, float, str | None] = (0, 0.0, None)

_CHOSEONG_SET: set[str] = set(CHOSEONG)

# ---------------------------------------------------------------------------
# IndexItem
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexItem:
    id: int
    name: str
    name_en: str | None
    ko_norm: str
    en_norm: str | None
    cho: str


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_field(
    q_norm: str,
    target_norm: str | None,
    field: str,
) -> tuple[int, float, str | None]:
    """Score a single normalized field against the query (spec §6-3)."""
    if not target_norm:
        return EMPTY
    if target_norm == q_norm:
        return (3, 1.0, field)
    if target_norm.startswith(q_norm):
        return (2, len(q_norm) / len(target_norm), field)
    if len(q_norm) >= MIN_FUZZY_LEN:
        sim = similarity(q_norm, target_norm)
        if sim >= THRESHOLDS[field]:
            return (1, rank_score(sim, field), field)
    return EMPTY


def _is_choseong_query(q_norm: str) -> bool:
    """True when every character in the (already-normed) query is a choseong jamo."""
    return bool(q_norm) and all(ch in _CHOSEONG_SET for ch in q_norm)


def _score_choseong(
    q_norm: str,
    item_cho: str,
) -> tuple[int, float, str | None]:
    """Score a choseong-prefix match (spec §6-5)."""
    if item_cho and item_cho.startswith(q_norm):
        return (1, len(q_norm) / len(item_cho), "cho")
    return EMPTY


def _best(
    a: tuple[int, float, str | None],
    b: tuple[int, float, str | None],
) -> tuple[int, float, str | None]:
    """Return the better candidate (higher tier, then higher score)."""
    if (b[0], b[1]) > (a[0], a[1]):
        return b
    return a


def score_item(
    item: IndexItem,
    q_norm: str,
    choseong_query: bool,
) -> tuple[int, float, str | None]:
    """Score one cocktail against the query across all applicable fields."""
    best = _score_field(q_norm, item.ko_norm, "ko")
    if item.en_norm:
        best = _best(best, _score_field(q_norm, item.en_norm, "en"))
    if choseong_query:
        best = _best(best, _score_choseong(q_norm, item.cho))
    return best


# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------


class CocktailSearchIndex:
    def __init__(self, items: list[IndexItem]) -> None:
        self._items = items

    @classmethod
    def build(cls, rows) -> "CocktailSearchIndex":
        """Build an index from an iterable of (id, name, name_en) tuples.

        DB-free: tests can call this with plain fixtures.
        """
        items: list[IndexItem] = []
        for row_id, row_name, row_name_en in rows:
            ko_norm = norm(row_name)
            en_norm = norm(row_name_en) if row_name_en else None
            cho = extract_choseong(row_name)
            items.append(
                IndexItem(
                    id=row_id,
                    name=row_name,
                    name_en=row_name_en,
                    ko_norm=ko_norm,
                    en_norm=en_norm,
                    cho=cho,
                )
            )
        return cls(items)

    @classmethod
    def load_from_db(cls, db) -> "CocktailSearchIndex":
        """Query the DB and build the index. Cocktail import is deferred here."""
        from cocktail_mate_db.models import Cocktail  # noqa: PLC0415

        rows = db.query(Cocktail.id, Cocktail.name, Cocktail.name_en).all()
        return cls.build(rows)

    def search(self, keyword: str, limit: int) -> list[dict]:
        """Return up to `limit` scored results for `keyword`.

        Returned dicts include original name/name_en (never normalized)
        plus tier, score, and matched_field for optional debug use.
        """
        kw = sanitize_keyword(keyword)
        if not kw:
            return []
        q = norm(kw)
        if not q:
            return []

        cq = _is_choseong_query(q)

        candidates: list[dict] = []
        for item in self._items:
            tier, score, matched_field = score_item(item, q, cq)
            if tier > 0:
                candidates.append(
                    {
                        "id": item.id,
                        "name": item.name,
                        "name_en": item.name_en,
                        "tier": tier,
                        "score": score,
                        "matched_field": matched_field,
                    }
                )

        candidates.sort(
            key=lambda c: (-c["tier"], -c["score"], len(c["name"]), c["id"])
        )
        return candidates[:limit]


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------

_index: CocktailSearchIndex | None = None
_lock = threading.Lock()


def set_index(idx: CocktailSearchIndex) -> None:
    """Set the global index (used by lifespan warm-up and tests)."""
    global _index
    _index = idx


def get_index() -> CocktailSearchIndex | None:
    """Return the current global index (may be None if not yet built)."""
    return _index


def ensure_index(db) -> CocktailSearchIndex:
    """Return the global index, building from DB on first call (double-checked locking)."""
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                set_index(CocktailSearchIndex.load_from_db(db))
    assert _index is not None
    return _index
