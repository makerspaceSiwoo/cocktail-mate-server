"""Evaluation-set tests for the cocktail autocomplete feature (spec §10-2).

Layer A: Direct CocktailSearchIndex.build(FIXTURE).search(...) assertions.
Layer B: API tests via FastAPI TestClient (no live DB).

Spec refs: §6-3 (item-first scoring), §6-4 (ranking), §6-5 (choseong),
           §6-6 (edge cases), §8-1 (sanitize_keyword), §10-2 (eval set).

OUT OF SCOPE (spec §7): mixed-language / keyboard-layout typo cases.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.cocktail.autocomplete.index import (
    CocktailSearchIndex,
    set_index,
)

# ---------------------------------------------------------------------------
# Fixture — ~30 cocktails as (id, name, name_en) tuples
# ---------------------------------------------------------------------------
# Includes at least three Korean-creation cocktails with name_en=None for
# null-safety coverage.

FIXTURE: list[tuple[int, str, str | None]] = [
    (1, "칼루아 밀크", "Kahlua Milk"),
    (2, "마가리타", "Margarita"),
    (3, "다이키리", "Daiquiri"),
    (4, "진토닉", "Gin Tonic"),
    (5, "모히토", "Mojito"),
    (6, "블랙 러시안", "Black Russian"),
    (7, "화이트 러시안", "White Russian"),
    (8, "칼루아 브라질리언 커피", "Kahlua Brazilian Coffee"),
    (9, "슬로우 진 피즈", "Sloe Gin Fizz"),
    (10, "코스모폴리탄", "Cosmopolitan"),
    (11, "섹스 온 더 비치", "Sex on the Beach"),
    (12, "피나 콜라다", "Pina Colada"),
    (13, "롱 아일랜드 아이스티", "Long Island Iced Tea"),
    (14, "마티니", "Martini"),
    (15, "아마레토 사워", "Amaretto Sour"),
    (16, "싱가포르 슬링", "Singapore Sling"),
    (17, "에그노그", "Eggnog"),
    (18, "네그로니", "Negroni"),
    (19, "위스키 사워", "Whiskey Sour"),
    (20, "데킬라 선라이즈", "Tequila Sunrise"),
    # Korean-creation cocktails — no English name (null-safety)
    (21, "클라우드 나인", None),
    (22, "해장 칵테일", None),
    (23, "비티스 진", None),
    # More cocktails to round out fixture
    (24, "망고 마가리타", "Mango Margarita"),
    (25, "블루 라군", "Blue Lagoon"),
    (26, "아페롤 스프리츠", "Aperol Spritz"),
    (27, "클로버 클럽", "Clover Club"),
    (28, "파인애플 다이키리", "Pineapple Daiquiri"),
    (29, "민트 줄렙", "Mint Julep"),
    (30, "올드 패션드", "Old Fashioned"),
]


@pytest.fixture(scope="module")
def idx() -> CocktailSearchIndex:
    return CocktailSearchIndex.build(FIXTURE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def names(results: list[dict]) -> list[str]:
    return [r["name"] for r in results]


def ids(results: list[dict]) -> list[int]:
    return [r["id"] for r in results]


def rank_of(results: list[dict], target_name: str) -> int:
    """0-indexed rank of target_name in results, or raises AssertionError."""
    ns = names(results)
    assert target_name in ns, f"{target_name!r} not found in results {ns}"
    return ns.index(target_name)


# ---------------------------------------------------------------------------
# Layer A — Direct index.search() evaluation cases (spec §10-2)
# ---------------------------------------------------------------------------


class TestSpellingVariants:
    """Korean transliteration + spacing variants (§10-2 row 1 & 2)."""

    def test_kahlua_milk_spacing_and_transliteration(self, idx):
        """깔루아밀크 (wrong choseong + no space) finds 칼루아 밀크."""
        results = idx.search("깔루아밀크", limit=10)
        assert "칼루아 밀크" in names(results)

    def test_kahlua_milk_single_jamo_diff(self, idx):
        """깔루아밀크 score is tier 1 (fuzzy ko match) and above threshold."""
        results = idx.search("깔루아밀크", limit=10)
        row = next(r for r in results if r["name"] == "칼루아 밀크")
        assert row["tier"] == 1
        assert row["matched_field"] == "ko"
        assert row["score"] > 0

    def test_kahlua_ko_prefix_returns_kahlua_milk(self, idx):
        """칼루아 (exact Korean prefix) finds 칼루아 밀크 — tier 2 prefix."""
        results = idx.search("칼루아", limit=10)
        assert "칼루아 밀크" in names(results)

    def test_kahlua_ko_prefix_tier(self, idx):
        """칼루아 → 칼루아 밀크 is a tier 2 (prefix) match."""
        results = idx.search("칼루아", limit=10)
        row = next(r for r in results if r["name"] == "칼루아 밀크")
        assert row["tier"] == 2

    def test_kahlua_milk_ranks_above_brazilian_coffee(self, idx):
        """칼루아 → 칼루아 밀크 ranks above 칼루아 브라질리언 커피 (spec §6-4 length tie-break)."""
        results = idx.search("칼루아", limit=10)
        assert rank_of(results, "칼루아 밀크") < rank_of(
            results, "칼루아 브라질리언 커피"
        )

    def test_margarita_spelling_variant(self, idx):
        """마르가리타 (wrong consonant) finds 마가리타 — fuzzy ko match."""
        results = idx.search("마르가리타", limit=10)
        assert "마가리타" in names(results)

    def test_margarita_no_space(self, idx):
        """마가리타 without space finds 마가리타."""
        results = idx.search("마가리타", limit=10)
        assert "마가리타" in names(results)

    def test_pina_colada_no_space(self, idx):
        """피나콜라다 (no space) finds 피나 콜라다 — norm strips separators."""
        results = idx.search("피나콜라다", limit=10)
        assert "피나 콜라다" in names(results)

    def test_pina_colada_exact_match_tier(self, idx):
        """피나콜라다 → 피나 콜라다 is a tier 3 (exact normalized) match."""
        results = idx.search("피나콜라다", limit=10)
        row = next(r for r in results if r["name"] == "피나 콜라다")
        assert row["tier"] == 3


class TestChoseongSearch:
    """Choseong (initial consonant) search (spec §6-5 / §10-2 row 4)."""

    def test_choseong_kahlua_milk(self, idx):
        """ㅋㄹㅇㅁㅋ (choseong of 칼루아 밀크) finds 칼루아 밀크."""
        results = idx.search("ㅋㄹㅇㅁㅋ", limit=10)
        assert "칼루아 밀크" in names(results)

    def test_choseong_kahlua_milk_matched_field(self, idx):
        """ㅋㄹㅇㅁㅋ → 칼루아 밀크 is matched via cho field."""
        results = idx.search("ㅋㄹㅇㅁㅋ", limit=10)
        row = next(r for r in results if r["name"] == "칼루아 밀크")
        assert row["matched_field"] == "cho"

    def test_choseong_mojito(self, idx):
        """ㅁㅎㅌ (choseong of 모히토) finds 모히토."""
        results = idx.search("ㅁㅎㅌ", limit=10)
        assert "모히토" in names(results)

    def test_choseong_martini(self, idx):
        """ㅁㅌㄴ (choseong of 마티니) finds 마티니."""
        results = idx.search("ㅁㅌㄴ", limit=10)
        assert "마티니" in names(results)

    def test_choseong_gin_tonic(self, idx):
        """ㅈㅌㄴ (choseong of 진토닉) finds 진토닉."""
        results = idx.search("ㅈㅌㄴ", limit=10)
        assert "진토닉" in names(results)

    def test_choseong_daiquiri(self, idx):
        """ㄷㅇㅋㄹ (choseong of 다이키리) finds 다이키리."""
        results = idx.search("ㄷㅇㅋㄹ", limit=10)
        assert "다이키리" in names(results)


class TestMidCompositionInput:
    """IME mid-composition input (spec §5-3 / §10-2 row 5)."""

    def test_mid_composition_kahlua_milk(self, idx):
        """칼루아 밀ㅋ (composing 크) finds 칼루아 밀크 — prefix match after norm."""
        results = idx.search("칼루아 밀ㅋ", limit=10)
        assert "칼루아 밀크" in names(results)

    def test_mid_composition_tier(self, idx):
        """칼루아 밀ㅋ → 칼루아 밀크 is tier 2 (prefix match)."""
        results = idx.search("칼루아 밀ㅋ", limit=10)
        row = next(r for r in results if r["name"] == "칼루아 밀크")
        assert row["tier"] == 2

    def test_mid_composition_mojito(self, idx):
        """모히ㅌ (composing 토) finds 모히토."""
        results = idx.search("모히ㅌ", limit=10)
        assert "모히토" in names(results)

    def test_mid_composition_martini(self, idx):
        """마티ㄴ (composing 니) finds 마티니."""
        results = idx.search("마티ㄴ", limit=10)
        assert "마티니" in names(results)


class TestEnglishTypos:
    """English typo tolerance (spec §5-4 / §10-2 rows 6 & 7)."""

    def test_margharita_finds_margarita(self, idx):
        """margharita (inserted h) finds 마가리타 — en fuzzy match."""
        results = idx.search("margharita", limit=10)
        assert "마가리타" in names(results)

    def test_margharita_matched_field(self, idx):
        """margharita → 마가리타 is matched via en field."""
        results = idx.search("margharita", limit=10)
        row = next(r for r in results if r["name"] == "마가리타")
        assert row["matched_field"] == "en"

    def test_daquiri_finds_daiquiri(self, idx):
        """daquiri (missing i) finds 다이키리 — en fuzzy match."""
        results = idx.search("daquiri", limit=10)
        assert "다이키리" in names(results)

    def test_daquiri_matched_field(self, idx):
        """daquiri → 다이키리 is matched via en field."""
        results = idx.search("daquiri", limit=10)
        row = next(r for r in results if r["name"] == "다이키리")
        assert row["matched_field"] == "en"

    def test_margarita_exact_english_match(self, idx):
        """margarita (correct spelling) finds 마가리타 — tier 3 exact."""
        results = idx.search("margarita", limit=10)
        assert "마가리타" in names(results)
        row = next(r for r in results if r["name"] == "마가리타")
        assert row["tier"] == 3

    def test_daiquiri_exact_english_match(self, idx):
        """daiquiri (correct spelling) finds 다이키리 — tier 3 exact."""
        results = idx.search("daiquiri", limit=10)
        assert "다이키리" in names(results)
        row = next(r for r in results if r["name"] == "다이키리")
        assert row["tier"] == 3

    def test_mojito_exact_english_match(self, idx):
        """mojito (correct spelling) finds 모히토 — tier 3 exact."""
        results = idx.search("mojito", limit=10)
        assert "모히토" in names(results)
        row = next(r for r in results if r["name"] == "모히토")
        assert row["tier"] == 3


class TestEnglishPrefix:
    """English prefix search (spec §10-2 row 8)."""

    def test_kahlua_en_prefix_finds_kahlua_milk(self, idx):
        """kahlua (en prefix of 'Kahlua Milk') finds 칼루아 밀크 — tier 2."""
        results = idx.search("kahlua", limit=10)
        assert "칼루아 밀크" in names(results)

    def test_kahlua_en_prefix_tier(self, idx):
        """kahlua → 칼루아 밀크 is a tier 2 match (en prefix)."""
        results = idx.search("kahlua", limit=10)
        row = next(r for r in results if r["name"] == "칼루아 밀크")
        assert row["tier"] == 2

    def test_kahlua_milk_ranks_above_brazilian_coffee_en(self, idx):
        """kahlua → 칼루아 밀크 ranks above 칼루아 브라질리언 커피."""
        results = idx.search("kahlua", limit=10)
        assert rank_of(results, "칼루아 밀크") < rank_of(
            results, "칼루아 브라질리언 커피"
        )

    def test_gin_prefix_finds_gin_tonic(self, idx):
        """gin (en prefix) finds 진토닉 (Gin Tonic)."""
        results = idx.search("gin", limit=10)
        assert "진토닉" in names(results)

    def test_gin_tonic_ranks_above_sloe_gin_fizz(self, idx):
        """gin → 진토닉 ranks above 슬로우 진 피즈 (prefix vs sub-string, spec §6-6 대응1)."""
        results = idx.search("gin", limit=10)
        # 진토닉 is a prefix match (tier 2); 슬로우 진 피즈 doesn't clear threshold
        # Either way 진토닉 must rank first or be the only gin result
        assert "진토닉" in names(results)
        if "슬로우 진 피즈" in names(results):
            assert rank_of(results, "진토닉") < rank_of(results, "슬로우 진 피즈")

    def test_martini_exact_match(self, idx):
        """martini (exact) finds 마티니 — tier 3."""
        results = idx.search("martini", limit=10)
        assert "마티니" in names(results)
        row = next(r for r in results if r["name"] == "마티니")
        assert row["tier"] == 3

    def test_cosmopolitan_exact_match(self, idx):
        """cosmopolitan (exact) finds 코스모폴리탄 — tier 3."""
        results = idx.search("cosmopolitan", limit=10)
        assert "코스모폴리탄" in names(results)

    def test_negroni_exact_match(self, idx):
        """negroni (exact) finds 네그로니 — tier 3."""
        results = idx.search("negroni", limit=10)
        assert "네그로니" in names(results)

    def test_tequila_sunrise_exact_match(self, idx):
        """tequila sunrise (exact) finds 데킬라 선라이즈 — tier 3."""
        results = idx.search("tequila sunrise", limit=10)
        assert "데킬라 선라이즈" in names(results)


class TestNullNameEn:
    """Null name_en safety (spec §6-3, §8-4 / §10-2 row 11)."""

    def test_korean_only_cocktail_found_by_korean_name(self, idx):
        """클라우드 나인 (name_en=None) is found by its Korean name."""
        results = idx.search("클라우드", limit=10)
        assert "클라우드 나인" in names(results)

    def test_korean_only_cocktail_matched_field_is_ko(self, idx):
        """클라우드 나인 match uses ko field (en_norm is None so skipped)."""
        results = idx.search("클라우드", limit=10)
        row = next(r for r in results if r["name"] == "클라우드 나인")
        assert row["matched_field"] == "ko"

    def test_korean_only_cocktail_name_en_is_none(self, idx):
        """클라우드 나인 result has name_en == None."""
        results = idx.search("클라우드", limit=10)
        row = next(r for r in results if r["name"] == "클라우드 나인")
        assert row["name_en"] is None

    def test_haejang_cocktail_found(self, idx):
        """해장 칵테일 (name_en=None) is found by Korean name prefix."""
        results = idx.search("해장", limit=10)
        assert "해장 칵테일" in names(results)

    def test_haejang_cocktail_no_en_norm_interference(self, idx):
        """해장 칵테일 search doesn't fail even with name_en=None in other items."""
        # Just confirm search works cleanly with mixed null/non-null fixture
        results = idx.search("해장", limit=10)
        assert len(results) >= 1

    def test_choseong_search_null_name_en(self, idx):
        """ㅋㄹㅇㄷㄴㅇ (choseong of 클라우드 나인) finds the null-name_en item."""
        results = idx.search("ㅋㄹㅇㄷㄴㅇ", limit=10)
        assert "클라우드 나인" in names(results)


class TestKoreanExactAndPrefix:
    """Korean exact and prefix matches."""

    def test_korean_exact_mojito(self, idx):
        """모히토 (Korean, exact norm) finds 모히토 — tier 3."""
        results = idx.search("모히토", limit=10)
        assert "모히토" in names(results)
        row = next(r for r in results if r["name"] == "모히토")
        assert row["tier"] == 3

    def test_korean_exact_martini(self, idx):
        """마티니 (Korean, exact norm) finds 마티니 — tier 3."""
        results = idx.search("마티니", limit=10)
        assert "마티니" in names(results)
        row = next(r for r in results if r["name"] == "마티니")
        assert row["tier"] == 3

    def test_korean_exact_black_russian(self, idx):
        """블랙 러시안 (Korean exact) finds 블랙 러시안 — tier 3."""
        results = idx.search("블랙 러시안", limit=10)
        assert "블랙 러시안" in names(results)
        row = next(r for r in results if r["name"] == "블랙 러시안")
        assert row["tier"] == 3

    def test_korean_prefix_mojito(self, idx):
        """모히 (Korean prefix) finds 모히토 — tier 2."""
        results = idx.search("모히", limit=10)
        assert "모히토" in names(results)
        row = next(r for r in results if r["name"] == "모히토")
        assert row["tier"] == 2

    def test_korean_prefix_daiquiri(self, idx):
        """다이키 (Korean prefix) finds 다이키리 — tier 2."""
        results = idx.search("다이키", limit=10)
        assert "다이키리" in names(results)


class TestEdgeCases:
    """Edge cases: empty, blank, limit, no-match (spec §6-6, §8-2)."""

    def test_empty_keyword_returns_empty_list(self, idx):
        """Empty string keyword returns []."""
        assert idx.search("", limit=10) == []

    def test_blank_keyword_returns_empty_list(self, idx):
        """Whitespace-only keyword returns [] (sanitized to empty)."""
        assert idx.search("   ", limit=10) == []

    def test_limit_caps_results(self, idx):
        """limit=2 returns at most 2 results even when more match."""
        results = idx.search("칼루아", limit=2)
        assert len(results) <= 2

    def test_limit_one(self, idx):
        """limit=1 returns exactly 1 result (when match exists)."""
        results = idx.search("마가리타", limit=1)
        assert len(results) == 1

    def test_no_match_returns_empty(self, idx):
        """A query with no matches returns an empty list."""
        results = idx.search("zzzxxx999", limit=10)
        assert results == []

    def test_results_ordered_by_tier_desc(self, idx):
        """Results are sorted by tier descending — higher tier items come first."""
        results = idx.search("kahlua", limit=10)
        tiers = [r["tier"] for r in results]
        assert tiers == sorted(tiers, reverse=True)

    def test_result_dicts_have_required_keys(self, idx):
        """Each result dict has id, name, name_en, tier, score, matched_field."""
        results = idx.search("마가리타", limit=10)
        assert len(results) > 0
        for r in results:
            for key in ("id", "name", "name_en", "tier", "score", "matched_field"):
                assert key in r, f"Missing key {key!r} in result {r}"


# ---------------------------------------------------------------------------
# Layer B — API tests via TestClient
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """TestClient with a pre-built fixture index; no DB calls."""
    from app.main import create_app
    from app.core.database import get_db

    # Build and install fixture index so ensure_index never queries DB
    fixture_idx = CocktailSearchIndex.build(FIXTURE)
    set_index(fixture_idx)

    application = create_app()

    # Override get_db so no DB connection is needed
    def _no_db():
        # Return a sentinel; ensure_index won't be called because _index is set
        yield None

    application.dependency_overrides[get_db] = _no_db

    with TestClient(application, raise_server_exceptions=True) as tc:
        yield tc


class TestAutocompleteAPI:
    """HTTP-level assertions for GET /search/autocomplete."""

    def test_basic_request_returns_200(self, client):
        """GET /search/autocomplete?keyword=마가리타 returns HTTP 200."""
        resp = client.get("/search/autocomplete", params={"keyword": "마가리타"})
        assert resp.status_code == 200

    def test_response_has_keyword_and_items(self, client):
        """Response body has 'keyword' and 'items' fields."""
        resp = client.get("/search/autocomplete", params={"keyword": "마가리타"})
        body = resp.json()
        assert "keyword" in body
        assert "items" in body

    def test_kahlua_milk_found_in_api(self, client):
        """깔루아밀크 query returns 칼루아 밀크 via API."""
        resp = client.get("/search/autocomplete", params={"keyword": "깔루아밀크"})
        assert resp.status_code == 200
        body = resp.json()
        item_names = [item["name"] for item in body["items"]]
        assert "칼루아 밀크" in item_names

    def test_keyword_echoed_sanitized(self, client):
        """API echoes the sanitized keyword back in the response."""
        resp = client.get("/search/autocomplete", params={"keyword": "  깔루아밀크  "})
        body = resp.json()
        # sanitize_keyword trims leading/trailing whitespace
        assert body["keyword"] == "깔루아밀크"

    def test_blank_keyword_returns_empty_items(self, client):
        """Empty keyword returns items: [] with 200."""
        resp = client.get("/search/autocomplete", params={"keyword": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body["keyword"] == ""
        assert body["items"] == []

    def test_missing_keyword_returns_empty_items(self, client):
        """Missing keyword param (defaults to '') returns items: [] with 200."""
        resp = client.get("/search/autocomplete")
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []

    def test_null_name_en_present_in_response(self, client):
        """Item with name_en=None has nameEn: null (key present) in API response."""
        resp = client.get("/search/autocomplete", params={"keyword": "클라우드"})
        assert resp.status_code == 200
        body = resp.json()
        items = {item["name"]: item for item in body["items"]}
        assert "클라우드 나인" in items
        cloud_item = items["클라우드 나인"]
        assert "nameEn" in cloud_item
        assert cloud_item["nameEn"] is None

    def test_non_null_name_en_present_in_response(self, client):
        """Item with name_en set has nameEn: str in API response."""
        resp = client.get("/search/autocomplete", params={"keyword": "마가리타"})
        body = resp.json()
        items = {item["name"]: item for item in body["items"]}
        assert "마가리타" in items
        item = items["마가리타"]
        assert "nameEn" in item
        assert item["nameEn"] == "Margarita"

    def test_limit_caps_api_results(self, client):
        """limit=2 query param returns at most 2 items."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "칼루아", "limit": 2}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) <= 2

    def test_debug_false_omits_debug_fields(self, client):
        """debug=0 (default) items do NOT include score/tier/matchedField."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "마가리타", "debug": "0"}
        )
        body = resp.json()
        assert len(body["items"]) > 0
        for item in body["items"]:
            assert "score" not in item
            assert "tier" not in item
            assert "matchedField" not in item

    def test_debug_true_adds_debug_fields(self, client):
        """debug=1 items include score, tier, matchedField."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "마가리타", "debug": "1"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) > 0
        for item in body["items"]:
            assert "score" in item, f"'score' missing from debug item: {item}"
            assert "tier" in item, f"'tier' missing from debug item: {item}"
            assert "matchedField" in item, (
                f"'matchedField' missing from debug item: {item}"
            )

    def test_debug_score_is_numeric(self, client):
        """debug=1 score is a finite float."""
        import math

        resp = client.get(
            "/search/autocomplete", params={"keyword": "마가리타", "debug": "1"}
        )
        body = resp.json()
        for item in body["items"]:
            assert isinstance(item["score"], float)
            assert math.isfinite(item["score"])

    def test_debug_matched_field_values(self, client):
        """debug=1 matchedField is one of 'ko', 'en', 'cho'."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "margharita", "debug": "1"}
        )
        body = resp.json()
        for item in body["items"]:
            assert item["matchedField"] in ("ko", "en", "cho")

    def test_api_items_have_id_name_nameEn(self, client):
        """Non-debug items have exactly id, name, nameEn fields."""
        resp = client.get("/search/autocomplete", params={"keyword": "mojito"})
        body = resp.json()
        assert len(body["items"]) > 0
        for item in body["items"]:
            assert "id" in item
            assert "name" in item
            assert "nameEn" in item

    def test_limit_boundary_ge1(self, client):
        """limit=1 is accepted and returns ≤1 items."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "마가리타", "limit": 1}
        )
        assert resp.status_code == 200
        assert len(resp.json()["items"]) <= 1

    def test_limit_out_of_range_422(self, client):
        """limit=0 violates ge=1 constraint → 422 Unprocessable Entity."""
        resp = client.get(
            "/search/autocomplete", params={"keyword": "마가리타", "limit": 0}
        )
        assert resp.status_code == 422
