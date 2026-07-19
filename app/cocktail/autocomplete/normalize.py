"""Text normalization for cocktail search autocomplete.

Compatibility-jamo decomposition (U+3130~), NOT NFD/conjoining (U+1100~).
This is the crux that makes IME mid-composition input and choseong search work.

Uses the `jamo` library (j2hcj(h2j(...))) for syllable decomposition. Since
sanitize_keyword applies NFC first, inputs reaching norm() are precomposed, so
this is functionally equivalent to the previous hand-rolled decomposition for all
practical inputs (precomposed syllables, standalone compat jamo, ASCII).
"""

import re
import unicodedata

from jamo import h2j, j2hcj

# fmt: off
# 19 leading consonants (choseong), compatibility jamo forms
CHOSEONG = [
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ",
    "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
]
# fmt: on

_CHOSEONG_SET = set(CHOSEONG)

# Separators to strip during norm()
SEPARATORS = set(" ·・-_/()[]")

MAX_LEN = 100

# Regex patterns for sanitize_keyword
# Control characters U+0000-U+001F (C0 controls) and U+007F (DEL)
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f]")
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


def norm(s) -> str:
    """Normalize a string for matching (spec §5-1).

    Steps: NFC -> lower -> strip separators -> decompose Hangul syllables to
    compatibility jamo (U+3130~) via the jamo library.
    Pure function; no FastAPI/DB imports.
    """
    if not s:
        return ""
    s = unicodedata.normalize("NFC", s)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in SEPARATORS)
    s = j2hcj(h2j(s))
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
