"""Unit tests for normalization + similarity primitives (spec §10-1).

Covers:
  - normalize.norm()          — Hangul decomposition, NFC, case, separators
  - normalize.extract_choseong()
  - normalize.sanitize_keyword()
  - matching.trigrams()
  - matching.dice()
"""

import math
import unicodedata

import pytest

from app.cocktail.autocomplete.matching import dice, trigrams
from app.cocktail.autocomplete.normalize import (
    MAX_LEN,
    extract_choseong,
    norm,
    sanitize_keyword,
)


# ---------------------------------------------------------------------------
# norm() — core acceptance checks (spec §10-1)
# ---------------------------------------------------------------------------


class TestNorm:
    def test_kahlua_variants_differ(self):
        """칼루아 밀크 and 깔루아밀크 normalize to different strings (different choseong)."""
        assert norm("칼루아 밀크") != norm("깔루아밀크")

    def test_kahlua_variants_dice_above_threshold(self):
        """Dice similarity of the two 칼루아 variants is ≥ 0.6 (actual ≈ 0.77)."""
        result = dice(norm("칼루아 밀크"), norm("깔루아밀크"))
        assert result >= 0.6, f"dice was {result}"

    def test_mid_composition_input_is_prefix(self):
        """norm(밀크).startswith(norm(밀ㅋ)) is True — IME mid-composition works."""
        assert norm("밀크").startswith(norm("밀ㅋ"))

    def test_english_case_and_spaces_normalized(self):
        """norm(Kahlua Milk) == norm(kahluamilk) — lowercase + separator strip."""
        assert norm("Kahlua Milk") == norm("kahluamilk")

    def test_nfd_input_recomposes(self):
        """norm on NFD input equals norm on NFC input — NFC is applied first."""
        nfd = unicodedata.normalize("NFD", "칼")
        assert norm(nfd) == norm("칼")

    def test_empty_string_returns_empty(self):
        """norm('') returns ''."""
        assert norm("") == ""

    def test_none_returns_empty(self):
        """norm(None) returns ''."""
        assert norm(None) == ""

    def test_hangul_separators_stripped(self):
        """Separators (space, ·, -, /, etc.) are removed before decomposition."""
        # Space between syllables is stripped
        assert norm("칼루아밀크") == norm("칼루아 밀크")
        # Middle-dot separator
        assert norm("칼루아·밀크") == norm("칼루아밀크")

    def test_ascii_lowercased(self):
        """ASCII letters are lowercased."""
        assert norm("GIN") == norm("gin")

    def test_hangul_decomposed_to_jamo(self):
        """A single syllable decomposes into compatibility jamo (U+3130~)."""
        result = norm("칼")
        # 칼 = ㅋ(U+314B) + ㅏ(U+3131 area) + ㄹ
        assert result == "ㅋㅏㄹ"
        # Must NOT contain NFD/conjoining jamo (U+1100~U+11FF)
        for ch in result:
            assert ord(ch) < 0x1100 or ord(ch) > 0x11FF, (
                f"conjoining jamo found: U+{ord(ch):04X}"
            )


# ---------------------------------------------------------------------------
# extract_choseong() — spec §6-5
# ---------------------------------------------------------------------------


class TestExtractChoseong:
    def test_standard_example(self):
        """extract_choseong('칼루아 밀크') == 'ㅋㄹㅇㅁㅋ'."""
        assert extract_choseong("칼루아 밀크") == "ㅋㄹㅇㅁㅋ"

    def test_standalone_choseong_passthrough(self):
        """A bare choseong jamo char is included in the result unchanged."""
        assert extract_choseong("ㅋ") == "ㅋ"

    def test_vowels_and_ascii_skipped(self):
        """Vowels and ASCII characters are skipped."""
        result = extract_choseong("gin")
        assert result == ""

    def test_empty_string(self):
        """Empty input returns empty string."""
        assert extract_choseong("") == ""


# ---------------------------------------------------------------------------
# sanitize_keyword() — spec §8-1
# ---------------------------------------------------------------------------


class TestSanitizeKeyword:
    def test_non_string_returns_empty(self):
        """Non-str input (None, int, list) returns ''."""
        assert sanitize_keyword(None) == ""
        assert sanitize_keyword(42) == ""
        assert sanitize_keyword([]) == ""

    def test_empty_string_returns_empty(self):
        """Empty string returns ''."""
        assert sanitize_keyword("") == ""

    def test_whitespace_collapsed(self):
        """Multiple spaces/tabs/newlines are collapsed to a single space."""
        assert sanitize_keyword("hello   world") == "hello world"
        assert sanitize_keyword("  hello\t world  ") == "hello world"

    def test_control_chars_stripped(self):
        """Control characters (U+0001-U+001F, U+007F) are removed."""
        assert sanitize_keyword("hel\x01lo\x1fworld\x7f") == "helloworld"

    def test_zero_width_chars_stripped(self):
        """Zero-width chars (ZWSP U+200B, ZWJ U+200D, ZWNJ U+200C, BOM U+FEFF) removed."""
        zwsp = "hel​lo‍‌﻿world"
        assert sanitize_keyword(zwsp) == "helloworld"

    def test_max_len_truncated(self):
        """Strings longer than MAX_LEN are truncated to MAX_LEN chars."""
        long_input = "a" * (MAX_LEN + 10)
        result = sanitize_keyword(long_input)
        assert len(result) == MAX_LEN

    def test_nfc_normalization_applied(self):
        """NFC normalization is applied to the input."""
        nfd = unicodedata.normalize("NFD", "칼루아")
        result = sanitize_keyword(nfd)
        assert result == "칼루아"


# ---------------------------------------------------------------------------
# trigrams() — spec §6-6
# ---------------------------------------------------------------------------


class TestTrigrams:
    def test_short_string_non_empty(self):
        """trigrams('gi') is non-empty (padding ensures ≥1 trigram)."""
        result = trigrams("gi")
        assert len(result) > 0

    def test_empty_string_returns_empty_set(self):
        """trigrams('') returns empty set."""
        assert trigrams("") == set()

    def test_none_returns_empty_set(self):
        """trigrams(None) returns empty set."""
        assert trigrams(None) == set()

    def test_trigrams_are_length_three(self):
        """All returned trigrams have length 3."""
        for tg in trigrams("gin"):
            assert len(tg) == 3

    def test_padding_present_in_trigrams(self):
        """Padding (two leading spaces + one trailing) appears in trigram set."""
        result = trigrams("a")
        assert "  a" in result  # two leading spaces + char


# ---------------------------------------------------------------------------
# dice() — spec §6-6
# ---------------------------------------------------------------------------


class TestDice:
    def test_gi_gin_finite(self):
        """dice('gi', 'gin') does not raise and is finite (no NaN/ZeroDivision)."""
        result = dice("gi", "gin")
        assert math.isfinite(result)

    def test_empty_empty_returns_zero(self):
        """dice('', '') returns 0.0 — denom == 0 guard."""
        assert dice("", "") == 0.0

    def test_identical_strings_return_one(self):
        """dice(s, s) == 1.0 for any non-empty s."""
        assert dice("gin", "gin") == pytest.approx(1.0)

    def test_completely_different_strings_low_score(self):
        """dice of completely different short strings is < 1.0."""
        result = dice("aaa", "zzz")
        assert result < 1.0

    def test_kahlua_dice_value(self):
        """Exact dice for 칼루아 밀크 vs 깔루아밀크 pair — records the actual float.

        Spec §10-1 notes ≈0.77 with mandatory padding; assert >= 0.6.
        """
        result = dice(norm("칼루아 밀크"), norm("깔루아밀크"))
        # Record for the report
        assert result >= 0.6, f"Expected >= 0.6, got {result}"
        # Must be strictly less than 1.0 (they differ)
        assert result < 1.0
