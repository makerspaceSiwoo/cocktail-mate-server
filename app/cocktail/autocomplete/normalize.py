"""Text normalization for cocktail search autocomplete.

Compatibility-jamo decomposition (U+3130~), NOT NFD/conjoining (U+1100~).
This is the crux that makes IME mid-composition input and choseong search work.
"""

import re
import unicodedata

# fmt: off
# 19 leading consonants (choseong), compatibility jamo forms
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]

# 21 vowels (jungseong), compatibility jamo forms
JUNGSEONG = [
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ", "ㅙ",
    "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
]

# 28 trailing consonants (jongseong), index 0 = "" (no final consonant)
JONGSEONG = [
    "",       "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ",
    "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ",
    "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
# fmt: on

_CHOSEONG_SET = set(CHOSEONG)

# Separators to strip during norm()
SEPARATORS = set(" ·・-_/()[]")

MAX_LEN = 100

# Regex patterns for sanitize_keyword
# Control characters U+0001-U+001F and U+007F (avoid literal null in source)
_CTRL_RE = re.compile("[-]")
# Zero-width chars: ZWSP U+200B, ZWJ U+200D, ZWNJ U+200C, BOM U+FEFF
_ZWSP_RE = re.compile("[​-‍﻿]")
_WS_RE = re.compile(r"\s+")


def sanitize_keyword(raw) -> str:
    """Sanitize a raw keyword for display and API response (spec §8-1).

    Does NOT strip separators or decompose jamo -- that is norm()'s job.
    """
    if not isinstance(raw, str):
        return ""
    s = unicodedata.normalize("NFC", raw)
    s = _CTRL_RE.sub("", s)
    s = _ZWSP_RE.sub("", s)
    s = _WS_RE.sub(" ", s)
    s = s.strip()
    s = s[:MAX_LEN]
    s = s.strip()
    return s


def _decompose_syllable(ch: str) -> str:
    """Decompose a precomposed Hangul syllable into compatibility jamo.

    For any other character (ASCII, already-compatibility jamo U+3130~,
    digits, symbols), return the character unchanged (spec §5-1).
    """
    code = ord(ch)
    if 0xAC00 <= code <= 0xD7A3:
        s = code - 0xAC00
        cho = s // 588
        jung = (s % 588) // 28
        jong = s % 28
        return CHOSEONG[cho] + JUNGSEONG[jung] + JONGSEONG[jong]
    return ch


def norm(s) -> str:
    """Normalize a string for matching (spec §5-1).

    Steps: NFC -> lower -> strip separators -> decompose Hangul syllables.
    Pure function; no FastAPI/DB imports.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in SEPARATORS)
    s = "".join(_decompose_syllable(ch) for ch in s)
    return s


def extract_choseong(s) -> str:
    """Extract only the leading consonants (choseong) from Hangul syllables (spec §6-5).

    E.g. extract_choseong("칼루아 밀크") == "ㅋㄹㅇㅁㅋ"
    """
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in SEPARATORS)
    result = []
    for ch in s:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            cho = (code - 0xAC00) // 588
            result.append(CHOSEONG[cho])
        elif ch in _CHOSEONG_SET:
            result.append(ch)
        # else: skip (vowels, ASCII, digits, etc.)
    return "".join(result)
