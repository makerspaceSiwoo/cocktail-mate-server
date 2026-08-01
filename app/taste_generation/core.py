"""Pure types, prompt construction, and CSV helpers for taste generation."""

from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CSV_COLUMNS = (
    "cocktail_id",
    "cocktail_name_ko",
    "cocktail_name_en",
    "recipe",
    "embedding_text",
)
TASTE_PROMPT_VERSION = 2


class TasteGenerationSettings(BaseSettings):
    """Settings used only by the offline Gemini taste worker."""

    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    gemini_text_model: str = "gemini-3.5-flash-lite"
    gemini_min_request_interval_seconds: float = 32.0
    gemini_max_transient_retries: int = 3
    gemini_max_validation_retries: int = 4


@dataclass(frozen=True, slots=True)
class IngredientSource:
    name: str
    name_en: str | None
    category: str | None
    amount: float | None
    unit: str | None
    description: str | None
    abv: float | None


@dataclass(frozen=True, slots=True)
class CocktailTasteSource:
    id: int
    name: str
    name_en: str | None
    recipe: tuple[str, ...]
    abv: float | None
    base_tag: str | None
    ingredients: tuple[IngredientSource, ...]

    @property
    def validation_error(self) -> str | None:
        if not self.name.strip():
            return "Korean cocktail name is required"
        if not self.recipe and not self.ingredients:
            return "recipe or ingredients are required"
        return None


class Intensity(str, Enum):
    NONE = "NONE"
    VERY_LOW = "VERY_LOW"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class Body(str, Enum):
    VERY_LIGHT = "VERY_LIGHT"
    LIGHT = "LIGHT"
    MEDIUM = "MEDIUM"
    FULL = "FULL"
    VERY_FULL = "VERY_FULL"


class ServingTemperature(str, Enum):
    VERY_COLD = "VERY_COLD"
    COLD = "COLD"
    COOL = "COOL"
    ROOM = "ROOM"
    WARM = "WARM"


class FinishLength(str, Enum):
    VERY_SHORT = "VERY_SHORT"
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"
    VERY_LONG = "VERY_LONG"


class AlcoholCharacter(str, Enum):
    NONE = "NONE"
    CLEAN = "CLEAN"
    SOFT = "SOFT"
    FERMENTED = "FERMENTED"
    WARM = "WARM"
    DISTINCT = "DISTINCT"
    PUNGENT = "PUNGENT"


class FruitNote(str, Enum):
    LEMON = "레몬"
    LIME = "라임"
    ORANGE = "오렌지"
    GRAPEFRUIT = "자몽"
    YUZU = "유자"
    TANGERINE = "귤"
    APPLE = "사과"
    PEAR = "배"
    PEACH = "복숭아"
    APRICOT = "살구"
    PLUM = "자두"
    CHERRY = "체리"
    STRAWBERRY = "딸기"
    RASPBERRY = "라즈베리"
    BLACKBERRY = "블랙베리"
    BLUEBERRY = "블루베리"
    CRANBERRY = "크랜베리"
    GRAPE = "포도"
    PINEAPPLE = "파인애플"
    MANGO = "망고"
    PASSION_FRUIT = "패션프루트"
    BANANA = "바나나"
    COCONUT = "코코넛"
    MELON = "멜론"
    WATERMELON = "수박"
    POMEGRANATE = "석류"
    LYCHEE = "리치"
    KIWI = "키위"
    FIG = "무화과"
    RAISIN = "건포도"


class OtherNote(str, Enum):
    JUNIPER = "주니퍼"
    MINT = "민트"
    BASIL = "바질"
    ROSEMARY = "로즈마리"
    THYME = "타임"
    HERB = "허브"
    GRASS = "풀"
    CUCUMBER = "오이"
    GINGER = "생강"
    CINNAMON = "계피"
    CLOVE = "정향"
    PEPPER = "후추"
    ANISE = "아니스"
    CARDAMOM = "카다멈"
    NUTMEG = "육두구"
    VANILLA = "바닐라"
    CARAMEL = "캐러멜"
    BROWN_SUGAR = "흑설탕"
    MOLASSES = "당밀"
    SUGARCANE = "사탕수수"
    MAPLE = "메이플"
    HONEY = "꿀"
    ALMOND = "아몬드"
    NUT = "견과"
    COFFEE = "커피"
    CACAO = "카카오"
    CHOCOLATE = "초콜릿"
    BLACK_TEA = "홍차"
    GREEN_TEA = "녹차"
    ROSE = "장미"
    ELDERFLOWER = "엘더플라워"
    FLORAL = "꽃"
    OAK = "오크"
    SMOKE = "연기"
    TOAST = "토스트"
    LEATHER = "가죽"
    EARTH = "흙"
    SALT = "소금"
    CREAM = "크림"
    BUTTER = "버터"
    YOGURT = "요거트"
    COLA = "콜라"
    LICORICE = "감초"
    AGAVE = "아가베"
    MALT = "몰트"
    GRAIN = "곡물"
    YEAST = "효모"
    BREAD = "빵"
    PEAT = "피트"
    MINERAL = "미네랄"
    WORMWOOD = "쑥"
    GENTIAN = "용담"


class Mouthfeel(str, Enum):
    REFRESHING = "청량함"
    CRISP = "산뜻함"
    CLEAN = "깔끔함"
    SMOOTH = "매끄러움"
    SILKY = "실키함"
    CREAMY = "크리미함"
    VELVETY = "벨벳감"
    JUICY = "과즙감"
    VISCOUS = "점성감"
    OILY = "오일리함"
    DRY = "드라이함"
    ASTRINGENT = "수렴감"
    FIZZY = "탄산 자극"
    WARMING = "알코올 열감"
    WEIGHTY = "묵직함"


class FinishCharacter(str, Enum):
    CLEAN = "깔끔함"
    DRY = "드라이함"
    SWEET = "달콤함"
    TART = "새콤함"
    BITTER = "쌉쌀함"
    FRUITY = "과일향"
    HERBAL = "허브향"
    SPICY = "향신료"
    OAKY = "오크"
    SMOKY = "연기"
    WARMING = "알코올 열감"
    CREAMY = "크리미함"
    NUTTY = "고소함"


_NEGATIVE_EMBEDDING_PATTERNS = (
    r"없",
    r"않",
    r"아니(?:다|며|고|지만|라|어서|므로)",
    r"낮(?:음|다|은|게)",
    r"약(?:함|한|하게)",
    r"미포함",
    r"무탄산",
    r"비탄산",
    r"무알코올",
    r"논알코올",
    r"제로",
)
_NON_SENSORY_EMBEDDING_TERMS = (
    "색상",
    "색깔",
    "색채",
    "투명",
    "불투명",
    "탁한",
    "레이어",
    "글라스",
    "가니시",
    "장식",
    "외관",
    "비주얼",
    "역사",
    "유래",
    "기원",
    "인기",
    "분위기",
)
_SWEET_TERMS = ("단맛", "달콤", "당도", "스위트", "디저트")
_ACIDITY_TERMS = ("산미", "새콤", "시큼")
_LIGHT_BODY_TERMS = ("가볍", "라이트", "산뜻", "청량", "경쾌")
_FULL_BODY_TERMS = ("밀도", "묵직", "리치", "실키", "벨벳", "점성")
_ALCOHOL_TERMS = ("알코올", "스피릿", "에탄올", "도수", "열감", "온기")
_CARBONATION_TERMS = ("탄산", "기포", "버블")


class FlavorProfile(BaseModel):
    """Machine facets plus the positive natural language used for embeddings."""

    model_config = ConfigDict(extra="forbid")

    sweetness: Intensity
    acidity: Intensity
    bitterness: Intensity
    salinity: Intensity
    umami: Intensity
    aroma_intensity: Intensity
    fruit_aromas: list[FruitNote] = Field(min_length=0, max_length=5)
    other_aromas: list[OtherNote] = Field(min_length=0, max_length=6)
    palate_fruit_notes: list[FruitNote] = Field(min_length=0, max_length=5)
    palate_other_notes: list[OtherNote] = Field(min_length=0, max_length=6)
    body: Body
    carbonation: Intensity
    creaminess: Intensity
    mouthfeel: list[Mouthfeel] = Field(min_length=1, max_length=5)
    serving_temperature: ServingTemperature
    alcohol_presence: Intensity
    alcohol_character: AlcoholCharacter
    finish_length: FinishLength
    finish_character: list[FinishCharacter] = Field(min_length=1, max_length=4)
    embedding_text: str = Field(
        min_length=80,
        max_length=500,
        description=(
            "Two to four Korean sensory sentences using only positively stated, "
            "actually perceived flavors and textures. This field alone is embedded."
        ),
    )

    @model_validator(mode="after")
    def validate_profile(self) -> FlavorProfile:
        for field_name in (
            "fruit_aromas",
            "other_aromas",
            "palate_fruit_notes",
            "palate_other_notes",
            "mouthfeel",
            "finish_character",
        ):
            values = getattr(self, field_name)
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
        if (self.alcohol_presence == Intensity.NONE) != (
            self.alcohol_character == AlcoholCharacter.NONE
        ):
            raise ValueError(
                "alcohol_presence and alcohol_character disagree about alcohol"
            )
        has_aroma_notes = bool(self.fruit_aromas or self.other_aromas)
        if (self.aroma_intensity == Intensity.NONE) == has_aroma_notes:
            raise ValueError(
                "aroma_intensity must agree with the presence of aroma notes"
            )
        if self.creaminess == Intensity.NONE and Mouthfeel.CREAMY in self.mouthfeel:
            raise ValueError("creaminess and creamy mouthfeel disagree")
        if self.carbonation == Intensity.NONE and Mouthfeel.FIZZY in self.mouthfeel:
            raise ValueError("carbonation and fizzy mouthfeel disagree")
        if self.alcohol_presence == Intensity.NONE and (
            Mouthfeel.WARMING in self.mouthfeel
            or FinishCharacter.WARMING in self.finish_character
        ):
            raise ValueError("non-alcoholic profile contains alcohol heat")
        self.embedding_text = _normalize_and_validate_embedding_text(
            self.embedding_text,
            profile=self,
        )
        return self


def _contains_any(value: str, terms: Iterable[str]) -> bool:
    return any(term in value for term in terms)


def _normalize_and_validate_embedding_text(
    value: str,
    *,
    profile: FlavorProfile,
) -> str:
    normalized = " ".join(value.split())
    if not 80 <= len(normalized) <= 500:
        raise ValueError("embedding_text must contain 80-500 normalized characters")
    if normalized[-1:] not in {".", "!", "?"}:
        raise ValueError("embedding_text must end with sentence punctuation")
    sentence_count = len(re.findall(r"[.!?](?=\s|$)", normalized))
    if not 2 <= sentence_count <= 4:
        raise ValueError("embedding_text must contain two to four sentences")
    if re.search(r"[|=]|\d|%", normalized):
        raise ValueError(
            "embedding_text must be natural prose without labels or numbers"
        )
    for pattern in _NEGATIVE_EMBEDDING_PATTERNS:
        if re.search(pattern, normalized):
            raise ValueError(
                f"embedding_text contains negative or level wording: {pattern}"
            )
    for term in _NON_SENSORY_EMBEDDING_TERMS:
        if term in normalized:
            raise ValueError(f"embedding_text contains non-sensory term: {term}")
    if re.search(r"잔(?:에|을|이|의|모양|속)", normalized):
        raise ValueError("embedding_text contains glass/presentation wording")

    low_intensities = {Intensity.NONE, Intensity.VERY_LOW, Intensity.LOW}
    if profile.sweetness in low_intensities:
        if _contains_any(normalized, _SWEET_TERMS):
            raise ValueError("low sweetness text must not mention sweetness tokens")
    if profile.acidity in low_intensities and _contains_any(
        normalized,
        _ACIDITY_TERMS,
    ):
        raise ValueError("low acidity text must not mention acidity tokens")
    if profile.body in {Body.VERY_LIGHT, Body.LIGHT} and not _contains_any(
        normalized,
        _LIGHT_BODY_TERMS,
    ):
        raise ValueError("light body requires a positive light/crisp descriptor")
    if profile.body in {Body.FULL, Body.VERY_FULL} and not _contains_any(
        normalized,
        _FULL_BODY_TERMS,
    ):
        raise ValueError("full body requires a positive rich/dense descriptor")
    if profile.carbonation in low_intensities and _contains_any(
        normalized,
        _CARBONATION_TERMS,
    ):
        raise ValueError("non-dominant carbonation must be omitted from embedding_text")
    if profile.creaminess in low_intensities and "크리미" in normalized:
        raise ValueError("non-dominant creaminess must be omitted from embedding_text")
    if profile.alcohol_presence == Intensity.NONE and _contains_any(
        normalized,
        _ALCOHOL_TERMS,
    ):
        raise ValueError("non-alcoholic embedding_text must omit alcohol tokens")
    if profile.alcohol_presence in {Intensity.HIGH, Intensity.VERY_HIGH} and not (
        _contains_any(normalized, _ALCOHOL_TERMS)
    ):
        raise ValueError("dominant alcohol perception must be positively described")

    selected_notes = {
        note.value
        for note in (
            *profile.fruit_aromas,
            *profile.other_aromas,
            *profile.palate_fruit_notes,
            *profile.palate_other_notes,
        )
    }
    if selected_notes and not _contains_any(normalized, selected_notes):
        raise ValueError(
            "embedding_text must include at least one selected flavor note"
        )
    return normalized


def render_embedding_text(profile: FlavorProfile) -> str:
    """Return only the positive natural-language field intended for embedding."""

    return profile.embedding_text


def _format_number(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:g}"


def build_taste_prompt(source: CocktailTasteSource) -> str:
    """Build a facts-first prompt for a controlled cocktail taste profile."""

    recipe_lines = "\n".join(
        f"{index}. {step.strip()}"
        for index, step in enumerate(source.recipe, start=1)
        if step.strip()
    )
    if not recipe_lines:
        recipe_lines = "(not provided)"

    ingredient_lines = "\n".join(
        (
            f"{index}. {ingredient.name}"
            f" / {ingredient.name_en or 'English name unknown'}"
            f" / amount={_format_number(ingredient.amount)}"
            f" {ingredient.unit or 'unit unknown'}"
            f" / category={ingredient.category or 'unknown'}"
            f" / ingredient_abv={_format_number(ingredient.abv)}%"
            f" / notes={ingredient.description or 'not provided'}"
        )
        for index, ingredient in enumerate(source.ingredients, start=1)
    )
    if not ingredient_lines:
        ingredient_lines = "(not provided)"

    return f"""You are a professional sensory analyst creating normalized Korean
cocktail taste data for semantic text embeddings and cosine-similarity search.

Infer the finished drink from the supplied facts. The recipe and measured
ingredients override general knowledge associated with the cocktail name.
Amounts, dilution, chilling, carbonation, and the finished ABV all matter.
Use the classic recipe only to fill a small gap when the supplied facts do not
answer it. Do not invent an ingredient or sensory note.

Hard constraints:
- Describe taste, retronasal aroma, mouthfeel, serving temperature, alcohol
  perception, and finish only.
- Never describe color, clarity, layers, appearance, garnish, glassware,
  presentation, history, origin, popularity, occasion, quality, or preference.
- Rate the finished drink, not each ingredient separately.
- Shaking does not by itself imply creaminess. Use creaminess only when dairy,
  egg, coconut cream, or another texture-producing ingredient supports it.
- Ignore a garnish unless the recipe explicitly expresses its oils or mixes an
  edible part into the drink.
- Keep aroma notes and palate notes distinct. Select only schema enum values,
  in descending dominance, with no duplicates. fruit_aromas and other_aromas
  mean aromas perceived by the nose; palate fields mean flavors perceived in
  the mouth and retronasally. Use an empty list when no note is supported.
- The facet fields are machine-only metadata and are never embedded. Fill every
  facet objectively, including NONE/VERY_LOW values. For a zero-ABV drink, set
  both alcohol fields to NONE.
- embedding_text is the only text that will be vectorized. Write two to four
  natural Korean sentences, 80-500 characters total. Do not repeat either
  cocktail name. Do not use headings, key=value labels, lists, scores, enum
  names, numbers, percentages, or ABV figures.
- In embedding_text, never state absence or weakness with words such as 없음,
  없다, 없는, 없이, 않다, 아니다, 낮음, 낮은, 약함, 약한, 미포함, 무탄산,
  비탄산, 무알코올, 논알코올, or 제로. Omit a non-perceptible axis instead of
  naming it. Mention only flavors and textures that are actually perceived.
- embedding_text must not mention color, clarity, layers, appearance, garnish,
  glassware, presentation, history, origin, popularity, or occasion.

Positive descriptor mapping for embedding_text:
- sweetness NONE/VERY_LOW/LOW: never use 단맛, 달콤함, 당도, or 스위트. Omit
  the axis when it is not material. Only when the low-sweetness character is
  itself important, use an evidence-based positive descriptor such as 드라이함,
  깔끔한 여운, 산뜻함, 담백함, or the ingredients' natural 씁쓸함.
- sweetness MEDIUM: use 은은한 단맛 or 균형 잡힌 당도 when it is material.
- sweetness HIGH/VERY_HIGH: use 달콤함, 진한 당도, 리치한 단맛, or
  디저트 스타일.
- acidity NONE/VERY_LOW/LOW: omit 산미, 새콤함, and 시큼함. Omit the axis when
  it is not material; otherwise describe the resulting experience positively
  with 부드러운 목넘김, 온화하고 둥근 풍미, 완만함, or 매끄러움.
- acidity MEDIUM/HIGH/VERY_HIGH: use lively or bright fruit-specific acidity,
  such as 선명한 레몬 산미, 새콤함, or 시트러스의 톡 쏘는 맛.
- body VERY_LIGHT/LIGHT: use 라이트하고 산뜻한 목넘김, 가볍고 청량한 질감,
  or 경쾌한 여운.
- body FULL/VERY_FULL: use 밀도감, 리치함, 묵직함, 실키함, 벨벳감, or
  점성감 only when supported by the ingredients.
- carbonation NONE/VERY_LOW/LOW: omit every carbonation-related word. Use
  매끄러운 질감 or 깔끔한 목넘김 only when accurate.
- carbonation HIGH/VERY_HIGH: describe lively bubbles, 탄산 자극, or a
  톡 쏘는 청량감.
- creaminess NONE/VERY_LOW/LOW: omit 크리미함. creaminess MEDIUM or above may
  use 크리미함, 실키함, or 벨벳감 when supported.
- alcohol NONE: omit all alcohol, ABV, spirit, heat, and proof words. For
  perceptible alcohol, describe only its sensory character with 부드러운 온기,
  깨끗한 스피릿감, 선명한 열감, or a similarly positive descriptor.
- Short finishes become 깔끔하고 경쾌한 여운; long finishes become 풍미가
  길게 이어지는 여운. Never write a negative duration statement.

Calibration rubric:
- All Intensity facets use one common finished-drink scale: NONE means not
  perceptible; VERY_LOW a trace; LOW secondary; MEDIUM clearly present but
  balanced; HIGH dominant; VERY_HIGH overwhelming.
- Body is relative to mixed drinks: VERY_LIGHT is watery/thin, LIGHT is crisp,
  MEDIUM has moderate weight, FULL is rich/viscous, and VERY_FULL is dense or
  dessert-like.
- Temperature follows the recipe: frozen or ice-blended is VERY_COLD; shaken,
  stirred, or explicitly chilled is normally COLD; cold ingredients without
  active chilling are COOL; no chilling is ROOM; heated is WARM.
- Finish length means the time a clear sensory impression remains after
  swallowing: under 3 seconds is VERY_SHORT, 3-6 SHORT, 7-15 MEDIUM, 16-30
  LONG, and over 30 VERY_LONG.
- Alcohol presence rates perceived ethanol heat/aroma, not ABV alone. Use the
  supplied finished ABV as supporting evidence.

Cocktail facts:
- Korean name: {source.name.strip()}
- English name: {(source.name_en or "(not provided)").strip()}
- Finished ABV: {_format_number(source.abv)}%
- Base category: {source.base_tag or "(not provided)"}
- Recipe:
{recipe_lines}
- Ingredients:
{ingredient_lines}

Return the machine facets and positive embedding_text as JSON matching the
response schema. Check that embedding_text remains truthful to the facets while
containing none of the prohibited negative or non-sensory wording.
Prompt version: {TASTE_PROMPT_VERSION}.
"""


def recipe_to_csv_value(recipe: Iterable[str]) -> str:
    return json.dumps(
        [step for step in recipe],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def csv_row_for(source: CocktailTasteSource, embedding_text: str) -> dict[str, str]:
    return {
        "cocktail_id": str(source.id),
        "cocktail_name_ko": source.name,
        "cocktail_name_en": source.name_en or "",
        "recipe": recipe_to_csv_value(source.recipe),
        "embedding_text": embedding_text,
    }


def row_matches_source(row: dict[str, str], source: CocktailTasteSource) -> bool:
    return (
        row.get("cocktail_name_ko") == source.name
        and row.get("cocktail_name_en") == (source.name_en or "")
        and row.get("recipe") == recipe_to_csv_value(source.recipe)
        and bool(row.get("embedding_text", "").strip())
    )


def read_existing_rows(path: Path) -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"CSV columns must be exactly {CSV_COLUMNS}, got "
                f"{tuple(reader.fieldnames or ())}: {path}"
            )
        rows: dict[int, dict[str, str]] = {}
        for line_number, row in enumerate(reader, start=2):
            try:
                cocktail_id = int(row["cocktail_id"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid cocktail_id at CSV line {line_number}: {path}"
                ) from error
            if cocktail_id in rows:
                raise ValueError(f"Duplicate cocktail_id={cocktail_id} in CSV: {path}")
            rows[cocktail_id] = dict(row)
        return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(data)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def write_csv_rows(path: Path, rows: dict[int, dict[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=CSV_COLUMNS,
        lineterminator="\n",
    )
    writer.writeheader()
    for cocktail_id in sorted(rows):
        writer.writerow(rows[cocktail_id])
    _atomic_write(path, output.getvalue().encode("utf-8-sig"))


class ModelRateLimiter:
    """Minimum-start-interval limiter keyed by model name."""

    def __init__(
        self,
        interval_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if interval_seconds < 0:
            raise ValueError("interval_seconds must be non-negative")
        self._interval_seconds = interval_seconds
        self._monotonic = monotonic
        self._sleep = sleep
        self._last_started_at: dict[str, float] = {}

    def wait(self, model: str) -> float:
        now = self._monotonic()
        last_started_at = self._last_started_at.get(model)
        waited = 0.0
        if last_started_at is not None:
            waited = max(0.0, self._interval_seconds - (now - last_started_at))
            if waited:
                self._sleep(waited)
        self._last_started_at[model] = self._monotonic()
        return waited
