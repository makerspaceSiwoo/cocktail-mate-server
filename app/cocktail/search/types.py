"""Data types for the shared cocktail name search core.

Pure dataclasses -- no FastAPI/DB imports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchDocument:
    """One cocktail's identity plus its precomputed normalized forms.

    Deliberately holds ONLY id/name/name_en (+ derived normalized forms).
    It must NOT carry image_url/base_tag/description -- those are catalog
    concerns, not search concerns, and the search core stays free of them.
    """

    id: int
    name: str
    name_en: str | None
    ko_norm: str
    en_norm: str | None
    cho: str


@dataclass(frozen=True)
class SearchHit:
    """One scored match, ready to surface to a caller."""

    id: int
    name: str
    name_en: str | None
    match_type: str  # "exact"|"prefix"|"choseong_prefix"|"contains"|"fuzzy"
    matched_field: str  # "ko"|"en"|"cho"
    score: float
    tier: int  # 4 exact, 3 prefix/choseong_prefix, 2 contains, 1 fuzzy


@dataclass(frozen=True)
class SearchResult:
    """A page of hits plus the total match count before slicing."""

    hits: list[SearchHit]  # already sorted AND sliced to the requested page
    total: int  # total matches BEFORE limit/offset
