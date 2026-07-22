"""In-memory cocktail name search index with tiered item-first scoring.

Scoring tiers (task-v2-1-brief.md is the authoritative source):
  4 = exact match (normalized)                       -> score 1.0
  3 = prefix match                                    -> score len(q) / len(target)
  3 = choseong-prefix                                 -> score len(q) / len(cho)
  2 = contains match (judged independently of fuzzy)  -> score len(q) / len(target)
  1 = fuzzy (partial_ratio >= threshold)              -> score rank_score(sim, field)
  0 = no match (excluded)

DB-free, FastAPI-free: this module only ever sees SearchDocuments handed to it
by a caller (Task 2's loader owns fetching rows from the DB).
"""

from __future__ import annotations

from app.cocktail.search.matching import (
    MIN_FUZZY_LEN,
    THRESHOLDS,
    rank_score,
    similarity,
)
from app.cocktail.search.normalize import (
    CHOSEONG,
    extract_choseong,
    norm,
    sanitize_keyword,
)
from app.cocktail.search.types import SearchDocument, SearchHit, SearchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (tier, score, match_type, matched_field)
_Candidate = tuple[int, float, str, str]

EMPTY: _Candidate = (0, 0.0, "", "")

_CHOSEONG_SET: set[str] = set(CHOSEONG)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_field(q_norm: str, target_norm: str | None, field: str) -> _Candidate:
    """Score a single normalized field against the query.

    First matching condition wins, in order: exact, prefix, contains, fuzzy.
    Contains is judged independently of fuzzy -- it is never derived from a
    fuzzy score.
    """
    if not target_norm:
        return EMPTY
    if target_norm == q_norm:
        return (4, 1.0, "exact", field)
    if target_norm.startswith(q_norm):
        return (3, len(q_norm) / len(target_norm), "prefix", field)
    if q_norm in target_norm:
        return (2, len(q_norm) / len(target_norm), "contains", field)
    if len(q_norm) >= MIN_FUZZY_LEN:
        sim = similarity(q_norm, target_norm)
        if sim >= THRESHOLDS[field]:
            return (1, rank_score(sim, field), "fuzzy", field)
    return EMPTY


def _is_choseong_query(q_norm: str) -> bool:
    """True when every character in the (already-normed) query is a choseong jamo."""
    return bool(q_norm) and all(ch in _CHOSEONG_SET for ch in q_norm)


def _score_choseong(q_norm: str, item_cho: str) -> _Candidate:
    """Score a choseong-prefix match."""
    if item_cho and item_cho.startswith(q_norm):
        return (3, len(q_norm) / len(item_cho), "choseong_prefix", "cho")
    return EMPTY


def _best(a: _Candidate, b: _Candidate) -> _Candidate:
    """Return the better candidate (higher tier, then higher score).

    On an exact tie of both tier and score, keeps `a`. Callers evaluate ko
    before en (and before cho), so a ko/en tie resolves in favor of ko.
    """
    if (b[0], b[1]) > (a[0], a[1]):
        return b
    return a


def score_document(
    document: SearchDocument,
    q_norm: str,
    choseong_query: bool,
) -> _Candidate:
    """Score one cocktail against the query across all applicable fields."""
    best = _score_field(q_norm, document.ko_norm, "ko")
    if document.en_norm:
        best = _best(best, _score_field(q_norm, document.en_norm, "en"))
    if choseong_query:
        best = _best(best, _score_choseong(q_norm, document.cho))
    return best


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------


def make_document(id: int, name: str, name_en: str | None) -> SearchDocument:
    """Build a SearchDocument from raw catalog fields.

    DB-free: callers (e.g. Task 2's loader) supply plain values, so this can
    also be used directly from fixtures.
    """
    ko_norm = norm(name)
    en_norm = norm(name_en) if name_en else None
    cho = extract_choseong(name)
    return SearchDocument(
        id=id,
        name=name,
        name_en=name_en,
        ko_norm=ko_norm,
        en_norm=en_norm,
        cho=cho,
    )


# ---------------------------------------------------------------------------
# Index class
# ---------------------------------------------------------------------------


class CocktailNameSearchIndex:
    """In-memory, DB-free search index over a fixed list of SearchDocuments.

    Construction takes an already-built list of documents; this class never
    queries a DB or builds documents itself (see `make_document` for that).
    """

    def __init__(self, documents: list[SearchDocument]) -> None:
        self._documents = documents

    def search(self, keyword: str, limit: int, offset: int = 0) -> SearchResult:
        """Return a page of scored hits for `keyword`.

        `total` is the full match count before `offset`/`limit` slicing;
        `hits` is already sorted and sliced to the requested page.
        """
        kw = sanitize_keyword(keyword)
        if not kw:
            return SearchResult(hits=[], total=0)
        q = norm(kw)
        if not q:
            return SearchResult(hits=[], total=0)

        cq = _is_choseong_query(q)

        matches: list[SearchHit] = []
        for document in self._documents:
            tier, score, match_type, matched_field = score_document(document, q, cq)
            if tier > 0:
                matches.append(
                    SearchHit(
                        id=document.id,
                        name=document.name,
                        name_en=document.name_en,
                        match_type=match_type,
                        matched_field=matched_field,
                        score=score,
                        tier=tier,
                    )
                )

        matches.sort(key=lambda hit: (-hit.tier, -hit.score, len(hit.name), hit.id))

        total = len(matches)
        hits = matches[offset : offset + limit]
        return SearchResult(hits=hits, total=total)
