"""Offline contracts for the sensory-48 Vertex AI Batch teacher run.

This module deliberately has no Google SDK import and performs no network or
credential-file I/O.  It prepares immutable local artifacts, validates recorded
Vertex responses, and exposes small protocols for a separately reviewed cloud
adapter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

from app.sensory_embedding.contracts import canonical_sha256, validate_sha256
from app.sensory_embedding.registry import SENSORY_V2_REGISTRY, SensoryAxis
from app.sensory_embedding.teacher_projection import (
    TEACHER_LABELS,
    AxisSoftLabels,
    teacher_source_sha256,
)

MODEL = "gemini-2.5-flash"
DEFAULT_PROJECT = "gen-lang-client-0477982146"
DEFAULT_LOCATION = "global"
CORPUS_ROWS = 602
AXIS_COUNT = 48
REQUEST_COUNT = CORPUS_ROWS * AXIS_COUNT
SHARD_COUNT = 8
SHARD_SIZE = REQUEST_COUNT // SHARD_COUNT
COHORT_SOURCE_FILE_SHA256 = (
    "8755a91cfd2709b87fad3a05e5daef158d7ea589cb08e6c3f09ab4ecabd4ab6f"
)
COHORT_ID_SET_SHA256 = (
    "56e77646b60ad9b45cbdcd43f4807dde994ef40b1d5e4461dbfa41ca2d59c05f"
)
INPUT_TOKENS_PER_REQUEST = 845
OUTPUT_TOKENS_PER_REQUEST = 32
PLANNING_INPUT_TOKEN_ENVELOPE = INPUT_TOKENS_PER_REQUEST
BATCH_INPUT_USD_PER_MILLION = Decimal("0.15")
BATCH_OUTPUT_USD_PER_MILLION = Decimal("1.25")
HISTORICAL_RESERVE_USD = Decimal("0.50")
SOFT_STOP_USD = Decimal("7.50")
HARD_CREATION_BLOCK_USD = Decimal("10.00")
GCS_LIFECYCLE_DAYS = 1
MANIFEST_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
PILOT_TOKEN_STATUS = "pilot_passed"
FULL_PRODUCTION_TOKEN_STATUS = "full_production_token_review_passed"
FULL_PRODUCTION_TOKEN_REVIEW_SCOPE = "full-production-token-envelope"
MIN_FULL_PRODUCTION_TOKEN_MEASUREMENTS = SHARD_COUNT

LABELS = tuple(TEACHER_LABELS)
SOURCE_PRIMARY_COLUMN = "normalized_recipe_json"
SOURCE_ALIAS_COLUMN = "recipe_facts"
SOURCE_PROVENANCE_COLUMN = "recipe_source_column"
AXIS_REGISTRY_PATH = (
    Path(__file__).with_name("data") / "sensory_axis_registry_48_ae_v2.csv"
)
AXIS_REGISTRY_FILE_SHA256 = (
    "f365ae66f7707c1c6ca15df380548f3682e488e0600a2a1a5d382fffe86c3fcf"
)

REQUEST_CONFIG: dict[str, object] = {
    "responseMimeType": "text/x.enum",
    "responseSchema": {"type": "STRING", "enum": list(LABELS)},
    "responseLogprobs": True,
    "logprobs": 20,
    "temperature": 1.0,
    "topP": 1.0,
    "maxOutputTokens": OUTPUT_TOKENS_PER_REQUEST,
    "thinkingConfig": {"thinkingBudget": 0},
}
REQUEST_CONFIG_SHA256 = canonical_sha256(REQUEST_CONFIG)

_INTENSITY_RUBRIC_KO = (
    "A=연상되거나 감지되지 않음, B=약함, C=보통, D=강함, E=매우 강함."
)
_AXIS_REGISTRY_COLUMNS = (
    "registry_version",
    "axis_order",
    "block_order",
    "category",
    "axis_id",
    "label_ko",
    "label_en",
    "definition_ko",
    "source_descriptors",
    "merge_action",
    "scale_id",
    "utility_profile_id",
    "status",
)


@dataclass(frozen=True, slots=True)
class PromptAxis:
    axis_order: int
    axis_id: str
    category: str
    label_ko: str
    label_en: str
    definition_ko: str


def load_prompt_axis_registry(
    path: Path = AXIS_REGISTRY_PATH,
) -> tuple[PromptAxis, ...]:
    """Load and hash-pin the local-experiment prompt registry."""

    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != AXIS_REGISTRY_FILE_SHA256:
            raise VertexSensoryBatchError("prompt axis registry SHA-256 mismatch")
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != _AXIS_REGISTRY_COLUMNS:
                raise VertexSensoryBatchError(
                    "prompt axis registry columns do not match the contract"
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VertexSensoryBatchError(
            f"cannot read prompt axis registry {path}: {error}"
        ) from error
    if len(raw_rows) != AXIS_COUNT:
        raise VertexSensoryBatchError("prompt axis registry must contain 48 rows")
    result: list[PromptAxis] = []
    for order, (raw, expected) in enumerate(
        zip(raw_rows, SENSORY_V2_REGISTRY.axes, strict=True)
    ):
        if (
            raw["registry_version"] != "sensory-48-ae-v2"
            or raw["status"] != "APPROVED_LOCAL_EXPERIMENT"
            or raw["axis_order"] != str(order)
            or raw["axis_id"] != expected.axis_id
            or raw["category"] != expected.category
        ):
            raise VertexSensoryBatchError(
                f"prompt axis registry row {order} violates order/identity/approval"
            )
        if (
            not raw["label_ko"].strip()
            or not raw["label_en"].strip()
            or not raw["definition_ko"].strip()
        ):
            raise VertexSensoryBatchError(
                f"prompt axis registry row {order} has an empty label/definition"
            )
        result.append(
            PromptAxis(
                axis_order=order,
                axis_id=raw["axis_id"],
                category=raw["category"],
                label_ko=raw["label_ko"].strip(),
                label_en=raw["label_en"].strip(),
                definition_ko=raw["definition_ko"].strip(),
            )
        )
    return tuple(result)


PROMPT_AXES = load_prompt_axis_registry()
_PROMPT_AXES_BY_ID = {axis.axis_id: axis for axis in PROMPT_AXES}

PROMPT_CONTRACT = {
    "language": "ko",
    "recipe_instruction": "이름이나 외부 지식이 아닌 제공된 레시피 사실만 사용",
    "axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
    "intensity_rubric": _INTENSITY_RUBRIC_KO,
    "output": "A-E 중 코드 하나만 출력",
}
PROMPT_SHA256 = canonical_sha256(PROMPT_CONTRACT)

TERMINAL_JOB_STATES = frozenset(
    {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }
)
ACTIVE_JOB_STATES = frozenset(
    {
        "JOB_STATE_PENDING",
        "JOB_STATE_RUNNING",
        "JOB_STATE_QUEUED",
        "JOB_STATE_UPDATING",
        "JOB_STATE_CANCELLING",
    }
)
ADC_ENV_ALLOWLIST = frozenset({"GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_QUOTA_PROJECT"})


class VertexSensoryBatchError(ValueError):
    """A local source, artifact, response, or safety contract is invalid."""


@dataclass(frozen=True, slots=True)
class FrozenCocktail:
    cocktail_id: int
    recipe_facts: object
    source_column: str

    def __post_init__(self) -> None:
        if type(self.cocktail_id) is not int or self.cocktail_id <= 0:
            raise VertexSensoryBatchError("cocktail_id must be a positive integer")
        if self.source_column not in {SOURCE_PRIMARY_COLUMN, SOURCE_ALIAS_COLUMN}:
            raise VertexSensoryBatchError("invalid recipe source column")
        _canonical_recipe(self.recipe_facts)

    @property
    def recipe_json(self) -> str:
        return _canonical_json(self.recipe_facts)


@dataclass(frozen=True, slots=True)
class SensoryBatchRequest:
    key: str
    cocktail_id: int
    row_index: int
    axis_order: int
    axis_id: str
    shard_index: int
    prompt: str

    def __post_init__(self) -> None:
        if self.shard_index != self.axis_order % SHARD_COUNT:
            raise VertexSensoryBatchError("request shard must be axis_order modulo 8")
        expected = f"r{self.row_index:04d}-a{self.axis_order:02d}"
        if self.key != expected:
            raise VertexSensoryBatchError("request key does not match row/axis order")

    def vertex_record(self) -> dict[str, object]:
        return {
            "request": {
                "contents": [{"role": "user", "parts": [{"text": self.prompt}]}],
                "generationConfig": REQUEST_CONFIG,
            }
        }

    def manifest_record(self) -> dict[str, object]:
        return {
            "key": self.key,
            "cocktail_id": self.cocktail_id,
            "row_index": self.row_index,
            "axis_order": self.axis_order,
            "axis_id": self.axis_id,
            "shard_index": self.shard_index,
            "prompt_sha256": hashlib.sha256(self.prompt.encode("utf-8")).hexdigest(),
        }


@dataclass(frozen=True, slots=True)
class ParsedDistribution:
    key: str
    cocktail_id: int
    axis_order: int
    axis_id: str
    selected_label: str
    probabilities: tuple[float, ...]
    response_sha256: str
    raw_response_sha256: str

    def __post_init__(self) -> None:
        AxisSoftLabels(self.axis_id, self.probabilities)
        if self.selected_label not in LABELS:
            raise VertexSensoryBatchError("selected label must be A-E")
        validate_sha256(self.response_sha256, field="response_sha256")
        validate_sha256(self.raw_response_sha256, field="raw_response_sha256")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "key": self.key,
            "cocktail_id": self.cocktail_id,
            "axis_order": self.axis_order,
            "axis_id": self.axis_id,
            "selected_label": self.selected_label,
            "labels": list(LABELS),
            "probabilities": list(self.probabilities),
            "response_sha256": self.response_sha256,
            "raw_response_sha256": self.raw_response_sha256,
        }


@dataclass(frozen=True, slots=True)
class QuarantineRecord:
    key: str
    shard_index: int
    line_number: int
    reason: str
    raw_response_sha256: str
    response_sha256: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class CostEstimate:
    request_count: int
    input_tokens: int
    output_tokens: int
    input_cost_usd: Decimal
    output_cost_usd: Decimal
    estimated_cost_usd: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, Decimal) else value
            for key, value in asdict(self).items()
        }


@dataclass(frozen=True, slots=True)
class PromptEnvelopeDiagnostics:
    request_count: int
    min_utf8_bytes: int
    max_utf8_bytes: int
    mean_utf8_bytes: float
    planning_input_tokens_per_request: int = PLANNING_INPUT_TOKEN_ENVELOPE

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GcsLifecycleContract:
    action: str = "Delete"
    age_days: int = GCS_LIFECYCLE_DAYS

    def __post_init__(self) -> None:
        if self.action != "Delete" or self.age_days != GCS_LIFECYCLE_DAYS:
            raise VertexSensoryBatchError(
                "GCS batch bucket must delete run objects after one day"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "rule": [
                {
                    "action": {"type": self.action},
                    "condition": {"age": self.age_days},
                }
            ]
        }


@dataclass(frozen=True, slots=True)
class JobLedgerEntry:
    run_id: str
    job_name: str
    state: str
    estimated_cost_usd: str
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class RunCostLedger:
    jobs: tuple[JobLedgerEntry, ...] = ()
    historical_cost_usd: str = "0.00"
    schema_version: int = LEDGER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LEDGER_SCHEMA_VERSION:
            raise VertexSensoryBatchError("unsupported run ledger schema")
        if Decimal(self.historical_cost_usd) < 0:
            raise VertexSensoryBatchError("historical cost cannot be negative")
        if len({entry.job_name for entry in self.jobs}) != len(self.jobs):
            raise VertexSensoryBatchError("run ledger job names must be unique")

    @property
    def most_recent_job(self) -> JobLedgerEntry | None:
        return self.jobs[-1] if self.jobs else None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "historical_cost_usd": self.historical_cost_usd,
            "jobs": [asdict(job) for job in self.jobs],
        }


class VertexBatchJobGateway(Protocol):
    """Cloud boundary; implementations must use ADC and preserve run metadata."""

    def create_job(
        self,
        *,
        manifest: Mapping[str, object],
        input_uri: str,
        output_uri: str,
        labels: Mapping[str, str],
    ) -> object: ...

    def get_job(self, job_name: str) -> object: ...


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise VertexSensoryBatchError("recipe facts must be valid JSON") from error


def _canonical_recipe(value: object) -> object:
    encoded = _canonical_json(value)
    if value is None or value == "" or value == [] or value == {}:
        raise VertexSensoryBatchError("recipe facts must be non-empty")
    return json.loads(encoded)


def _finite_number(value: object, *, field: str) -> int | float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise VertexSensoryBatchError(f"{field} must be a finite number")
    return value


def _without_nulls(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _without_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_nulls(item) for item in value if item is not None]
    return value


def minimal_recipe_facts(payload: object) -> dict[str, object]:
    """Remove names, IDs, nulls, and normalization boilerplate from lab JSON."""

    if not isinstance(payload, Mapping):
        raise VertexSensoryBatchError("normalized_recipe_json must be an object")
    raw_ingredients = payload.get("ingredients")
    if not isinstance(raw_ingredients, list) or not raw_ingredients:
        raise VertexSensoryBatchError(
            "normalized_recipe_json.ingredients must be a non-empty list"
        )
    ingredients: list[dict[str, object]] = []
    for index, raw in enumerate(raw_ingredients):
        if not isinstance(raw, Mapping):
            raise VertexSensoryBatchError(f"ingredient {index} must be an object")
        name = raw.get("canonical_name")
        category = raw.get("category")
        if not isinstance(name, str) or not name.strip():
            raise VertexSensoryBatchError(f"ingredient {index} requires canonical_name")
        if not isinstance(category, str) or not category.strip():
            raise VertexSensoryBatchError(f"ingredient {index} requires category")
        ingredient: dict[str, object] = {
            "canonical_name": name.strip(),
            "category": category.strip(),
        }
        ratio = raw.get("normalized_amount_ratio")
        amount_ml = raw.get("normalized_amount_ml")
        if ratio is not None:
            ingredient["normalized_amount_ratio"] = _finite_number(
                ratio,
                field=f"ingredient {index} normalized_amount_ratio",
            )
        elif amount_ml is not None:
            ingredient["normalized_amount_ml"] = _finite_number(
                amount_ml,
                field=f"ingredient {index} normalized_amount_ml",
            )
        ingredient["presence_only"] = ratio is None and amount_ml is None
        ingredients.append(ingredient)

    result: dict[str, object] = {"ingredients": ingredients}
    for source_key, target_key in (
        ("method", "method"),
        ("mixing_ice", "mixing_ice"),
        ("serving_ice", "serving_ice"),
        ("garnish", "garnish"),
    ):
        value = payload.get(source_key)
        if value is not None and value != "" and value != [] and value != {}:
            result[target_key] = _without_nulls(value)
    abv_status = payload.get(
        "abv_estimate_status",
        payload.get("estimated_pre_dilution_abv_status"),
    )
    if abv_status is not None and abv_status != "":
        result["estimated_pre_dilution_abv_status"] = _without_nulls(abv_status)
    carbonation = payload.get("carbonation", payload.get("carbonated"))
    if carbonation is not None:
        if not isinstance(carbonation, (bool, str)):
            raise VertexSensoryBatchError("carbonation must be boolean or string")
        result["carbonation"] = carbonation
    abv = payload.get("estimated_pre_dilution_abv_on_normalized_volume")
    if abv is None:
        abv = payload.get("estimated_pre_dilution_abv")
    if abv is not None:
        result["estimated_pre_dilution_abv"] = _finite_number(
            abv,
            field="estimated_pre_dilution_abv",
        )
    return cast(dict[str, object], json.loads(_canonical_json(result)))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def id_set_sha256(cocktail_ids: Iterable[int]) -> str:
    ids = sorted(cocktail_ids)
    if len(ids) != len(set(ids)):
        raise VertexSensoryBatchError("cocktail IDs must be unique")
    return canonical_sha256({"cocktail_ids": ids})


def load_cohort_ids_csv(
    path: Path,
    *,
    expected_rows: int = CORPUS_ROWS,
) -> tuple[int, ...]:
    """Load the explicit current-catalog ID allowlist without any DB access."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if "cocktail_id" not in set(reader.fieldnames or ()):
                raise VertexSensoryBatchError(
                    "cohort CSV requires a cocktail_id column"
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VertexSensoryBatchError(
            f"cannot read cohort CSV {path}: {error}"
        ) from error

    ids: list[int] = []
    for line_number, row in enumerate(raw_rows, start=2):
        raw_id = row.get("cocktail_id", "")
        try:
            cocktail_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: invalid cohort cocktail_id"
            ) from error
        if str(cocktail_id) != raw_id or cocktail_id <= 0:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: cohort cocktail_id must be canonical"
            )
        ids.append(cocktail_id)
    if len(ids) != expected_rows or len(set(ids)) != expected_rows:
        raise VertexSensoryBatchError(
            f"cohort CSV must contain exactly {expected_rows} unique IDs"
        )
    result = tuple(sorted(ids))
    if expected_rows == CORPUS_ROWS and id_set_sha256(result) != COHORT_ID_SET_SHA256:
        raise VertexSensoryBatchError(
            "cohort CSV ID set does not match the pinned current 602-ID cohort"
        )
    return result


def load_source_csv(
    path: Path,
    *,
    expected_rows: int | None = CORPUS_ROWS,
    included_cocktail_ids: Sequence[int] | None = None,
) -> tuple[FrozenCocktail, ...]:
    """Load recipe facts while intentionally ignoring names and other columns."""

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            fields = set(reader.fieldnames or ())
            if "cocktail_id" not in fields:
                raise VertexSensoryBatchError("source CSV requires cocktail_id")
            if not fields.intersection({SOURCE_PRIMARY_COLUMN, SOURCE_ALIAS_COLUMN}):
                raise VertexSensoryBatchError(
                    "source CSV requires normalized_recipe_json"
                )
            raw_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise VertexSensoryBatchError(
            f"cannot read source CSV {path}: {error}"
        ) from error

    if included_cocktail_ids is None:
        included: frozenset[int] | None = None
    else:
        included = frozenset(included_cocktail_ids)
        if len(included) != len(included_cocktail_ids):
            raise VertexSensoryBatchError("cohort allowlist contains duplicate IDs")
    result: list[FrozenCocktail] = []
    seen_source: set[int] = set()
    for line_number, raw in enumerate(raw_rows, start=2):
        try:
            cocktail_id = int(raw.get("cocktail_id", ""))
        except (TypeError, ValueError) as error:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: invalid cocktail_id"
            ) from error
        if cocktail_id <= 0 or cocktail_id in seen_source:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: non-positive or duplicate cocktail_id"
            )
        seen_source.add(cocktail_id)
        if included is not None and cocktail_id not in included:
            continue
        primary = (raw.get(SOURCE_PRIMARY_COLUMN) or "").strip()
        alias = (raw.get(SOURCE_ALIAS_COLUMN) or "").strip()
        if not primary and not alias:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: missing normalized recipe JSON"
            )
        try:
            primary_value = json.loads(primary) if primary else None
            alias_value = json.loads(alias) if alias else None
        except json.JSONDecodeError as error:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: invalid normalized recipe JSON"
            ) from error
        if (
            primary
            and alias
            and _canonical_json(primary_value) != _canonical_json(alias_value)
        ):
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: recipe source columns disagree"
            )
        explicit_source = (raw.get(SOURCE_PROVENANCE_COLUMN) or "").strip()
        if explicit_source and explicit_source not in {
            SOURCE_PRIMARY_COLUMN,
            SOURCE_ALIAS_COLUMN,
        }:
            raise VertexSensoryBatchError(
                f"{path}:{line_number}: invalid recipe_source_column"
            )
        source_column = (
            SOURCE_PRIMARY_COLUMN if primary else explicit_source or SOURCE_ALIAS_COLUMN
        )
        recipe = minimal_recipe_facts(primary_value if primary else alias_value)
        result.append(
            FrozenCocktail(
                cocktail_id=cocktail_id,
                recipe_facts=_canonical_recipe(recipe),
                source_column=source_column,
            )
        )
    result.sort(key=lambda row: row.cocktail_id)
    if included is not None and {row.cocktail_id for row in result} != included:
        missing = sorted(included - {row.cocktail_id for row in result})
        raise VertexSensoryBatchError(
            f"normalized source is missing cohort IDs: {missing}"
        )
    if expected_rows is not None and len(result) != expected_rows:
        raise VertexSensoryBatchError(
            f"expected {expected_rows} cocktail rows, got {len(result)}"
        )
    return tuple(result)


def frozen_csv_bytes(rows: Sequence[FrozenCocktail]) -> bytes:
    """Return the canonical, name-free source snapshot."""

    if not rows:
        raise VertexSensoryBatchError("cannot freeze an empty source")
    lines = ["cocktail_id,recipe_facts,recipe_source_column"]
    for row in sorted(rows, key=lambda item: item.cocktail_id):
        recipe = row.recipe_json.replace('"', '""')
        lines.append(f'{row.cocktail_id},"{recipe}",{row.source_column}')
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_prompt(recipe_facts: object, axis: SensoryAxis) -> str:
    """Render one name-free, ID-free Korean enum classification prompt."""

    prompt_axis = _PROMPT_AXES_BY_ID[axis.axis_id]
    recipe_json = _canonical_json(recipe_facts)
    return (
        "다음 정규화된 레시피 사실만 근거로 완성된 칵테일의 감각을 평가하세요. "
        "칵테일 이름, ID, 외부 지식은 추측하지 마세요.\n"
        f"레시피 사실: {recipe_json}\n"
        f"평가 축: {prompt_axis.label_ko} ({prompt_axis.label_en}). "
        f"정의: {prompt_axis.definition_ko}.\n"
        f"등급: {_INTENSITY_RUBRIC_KO}\n"
        "설명이나 문장부호 없이 A, B, C, D, E 중 코드 하나만 출력하세요."
    )


def prompt_envelope_diagnostics(
    shards: Sequence[Sequence[SensoryBatchRequest]],
) -> PromptEnvelopeDiagnostics:
    sizes = [
        len(request.prompt.encode("utf-8")) for shard in shards for request in shard
    ]
    if not sizes:
        raise VertexSensoryBatchError("prompt diagnostics require requests")
    return PromptEnvelopeDiagnostics(
        request_count=len(sizes),
        min_utf8_bytes=min(sizes),
        max_utf8_bytes=max(sizes),
        mean_utf8_bytes=math.fsum(sizes) / len(sizes),
    )


def validate_pilot_token_counts(
    token_counts: Mapping[str, int],
    requests: Sequence[SensoryBatchRequest],
) -> dict[str, object]:
    """Require measured counts for every pilot request and enforce 845 tokens."""

    expected = {request.key for request in requests}
    if set(token_counts) != expected:
        raise VertexSensoryBatchError(
            "pilot token-count keys must exactly match pilot request keys"
        )
    invalid = {
        key: count
        for key, count in token_counts.items()
        if type(count) is not int or count <= 0
    }
    if invalid:
        raise VertexSensoryBatchError("pilot token counts must be positive integers")
    maximum = max(token_counts.values())
    if maximum > PLANNING_INPUT_TOKEN_ENVELOPE:
        raise VertexSensoryBatchError(
            f"pilot token envelope exceeded: {maximum} > "
            f"{PLANNING_INPUT_TOKEN_ENVELOPE}"
        )
    return {
        "status": PILOT_TOKEN_STATUS,
        "planning_input_tokens_per_request": PLANNING_INPUT_TOKEN_ENVELOPE,
        "measured_request_count": len(token_counts),
        "measured_min_tokens": min(token_counts.values()),
        "measured_max_tokens": maximum,
        "measured_mean_tokens": math.fsum(token_counts.values()) / len(token_counts),
        "token_counts_sha256": canonical_sha256(dict(sorted(token_counts.items()))),
    }


def validate_full_production_token_counts(
    token_counts: Mapping[str, int],
    requests: Sequence[SensoryBatchRequest],
) -> dict[str, object]:
    """Create explicit reviewed evidence for a full 602-cocktail manifest.

    Ordinary pilot evidence deliberately retains ``pilot_passed`` and cannot
    authorize production. A full manifest builder must call this separate
    boundary with at least one measured request from every axis shard.
    """

    evidence = validate_pilot_token_counts(token_counts, requests)
    if len(requests) < MIN_FULL_PRODUCTION_TOKEN_MEASUREMENTS:
        raise VertexSensoryBatchError(
            "full production token review requires at least eight measurements"
        )
    shard_indexes = {request.shard_index for request in requests}
    if shard_indexes != set(range(SHARD_COUNT)):
        raise VertexSensoryBatchError(
            "full production token review must cover all eight shards"
        )
    return {
        **evidence,
        "status": FULL_PRODUCTION_TOKEN_STATUS,
        "review_scope": FULL_PRODUCTION_TOKEN_REVIEW_SCOPE,
        "full_production_authorized": True,
    }


def build_requests(
    rows: Sequence[FrozenCocktail],
) -> tuple[tuple[SensoryBatchRequest, ...], ...]:
    if not rows:
        raise VertexSensoryBatchError("cannot build requests from an empty source")
    shards: list[list[SensoryBatchRequest]] = [[] for _ in range(SHARD_COUNT)]
    for row_index, row in enumerate(sorted(rows, key=lambda item: item.cocktail_id)):
        for axis in SENSORY_V2_REGISTRY.axes:
            request = SensoryBatchRequest(
                key=f"r{row_index:04d}-a{axis.axis_order:02d}",
                cocktail_id=row.cocktail_id,
                row_index=row_index,
                axis_order=axis.axis_order,
                axis_id=axis.axis_id,
                shard_index=axis.axis_order % SHARD_COUNT,
                prompt=render_prompt(row.recipe_facts, axis),
            )
            shards[request.shard_index].append(request)
    return tuple(tuple(shard) for shard in shards)


def jsonl_bytes(requests: Sequence[SensoryBatchRequest]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                request.vertex_record(),
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            for request in requests
        )
        + "\n"
    ).encode("utf-8")


def estimate_cost(request_count: int) -> CostEstimate:
    if type(request_count) is not int or request_count < 0:
        raise VertexSensoryBatchError("request_count must be non-negative")
    input_tokens = request_count * INPUT_TOKENS_PER_REQUEST
    output_tokens = request_count * OUTPUT_TOKENS_PER_REQUEST
    input_cost = (
        Decimal(input_tokens) * BATCH_INPUT_USD_PER_MILLION / Decimal(1_000_000)
    )
    output_cost = (
        Decimal(output_tokens) * BATCH_OUTPUT_USD_PER_MILLION / Decimal(1_000_000)
    )
    return CostEstimate(
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        estimated_cost_usd=input_cost + output_cost,
    )


def build_manifest(
    rows: Sequence[FrozenCocktail],
    shards: Sequence[Sequence[SensoryBatchRequest]],
    *,
    input_sha256: str,
    run_id: str,
    created_at: str,
    sdk_version: str,
    project: str = DEFAULT_PROJECT,
    location: str = DEFAULT_LOCATION,
    pilot_token_envelope: Mapping[str, object] | None = None,
    cohort_source_sha256: str | None = None,
    cohort_id_set_sha256: str | None = None,
) -> dict[str, object]:
    validate_sha256(input_sha256, field="input_sha256")
    if cohort_source_sha256 is not None:
        validate_sha256(
            cohort_source_sha256,
            field="cohort_source_sha256",
        )
    if cohort_id_set_sha256 is not None:
        validate_sha256(
            cohort_id_set_sha256,
            field="cohort_id_set_sha256",
        )
    if len(shards) != SHARD_COUNT:
        raise VertexSensoryBatchError("manifest requires exactly 8 shards")
    request_count = sum(len(shard) for shard in shards)
    estimate = estimate_cost(request_count)
    diagnostics = prompt_envelope_diagnostics(shards)
    source_columns = sorted({row.source_column for row in rows})
    shard_records: list[dict[str, object]] = []
    request_records: list[dict[str, object]] = []
    for shard_index, shard in enumerate(shards):
        if any(request.shard_index != shard_index for request in shard):
            raise VertexSensoryBatchError("shard contains a misplaced request")
        payload = jsonl_bytes(shard)
        shard_records.append(
            {
                "shard_index": shard_index,
                "filename": f"requests-{shard_index:02d}.jsonl",
                "axis_orders": list(range(shard_index, AXIS_COUNT, SHARD_COUNT)),
                "record_count": len(shard),
                "sha256": sha256_bytes(payload),
            }
        )
        request_records.extend(request.manifest_record() for request in shard)
    ids_hash = id_set_sha256(row.cocktail_id for row in rows)
    if cohort_id_set_sha256 is not None and cohort_id_set_sha256 != ids_hash:
        raise VertexSensoryBatchError(
            "cohort ID-set SHA-256 does not match the frozen input rows"
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "input_sha256": input_sha256,
        "row_count": len(rows),
        "id_set_sha256": ids_hash,
        "input_row_count": len(rows),
        "input_id_set_sha256": ids_hash,
        "cohort_source_sha256": cohort_source_sha256,
        "cohort_id_set_sha256": cohort_id_set_sha256,
        "cohort_row_count": len(rows) if cohort_id_set_sha256 is not None else None,
        "input_recipe_source_columns": source_columns,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "model": MODEL,
        "project": project,
        "location": location,
        "sdk_version": sdk_version,
        "request_count": request_count,
        "shard_count": SHARD_COUNT,
        "shard_policy": "axis_order_mod_8",
        "shard_size": SHARD_SIZE if len(rows) == CORPUS_ROWS else None,
        "shards": shard_records,
        "prompt_envelope_diagnostics": diagnostics.to_dict(),
        "pilot_token_envelope": dict(pilot_token_envelope)
        if pilot_token_envelope is not None
        else {
            "status": "pilot_required_before_job_creation",
            "planning_input_tokens_per_request": PLANNING_INPUT_TOKEN_ENVELOPE,
        },
        "estimated_cost": estimate.to_dict(),
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
        "gcs_lifecycle": GcsLifecycleContract().to_dict(),
        "created_at": created_at,
        "requests": request_records,
    }


def validate_full_manifest(manifest: Mapping[str, object]) -> None:
    expected = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model": MODEL,
        "project": DEFAULT_PROJECT,
        "location": DEFAULT_LOCATION,
        "input_row_count": CORPUS_ROWS,
        "row_count": CORPUS_ROWS,
        "request_count": REQUEST_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_size": SHARD_SIZE,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "input_id_set_sha256": COHORT_ID_SET_SHA256,
        "id_set_sha256": COHORT_ID_SET_SHA256,
        "cohort_source_sha256": COHORT_SOURCE_FILE_SHA256,
        "cohort_id_set_sha256": COHORT_ID_SET_SHA256,
        "cohort_row_count": CORPUS_ROWS,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise VertexSensoryBatchError(
                f"manifest {field} must be {value!r}, got {manifest.get(field)!r}"
            )
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != SHARD_COUNT
        or any(
            not isinstance(shard, dict) or shard.get("record_count") != SHARD_SIZE
            for shard in shards
        )
    ):
        raise VertexSensoryBatchError(
            f"every production shard must have {SHARD_SIZE} records"
        )
    for shard_index, raw_shard in enumerate(shards):
        if not isinstance(raw_shard, dict):
            raise VertexSensoryBatchError("production manifest shard is invalid")
        shard = raw_shard
        expected_axis_orders = list(range(shard_index, AXIS_COUNT, SHARD_COUNT))
        if (
            shard.get("shard_index") != shard_index
            or shard.get("filename") != f"requests-{shard_index:02d}.jsonl"
            or shard.get("axis_orders") != expected_axis_orders
        ):
            raise VertexSensoryBatchError(
                "production manifest shard identity/order is invalid"
            )
        shard_hash = shard.get("sha256")
        if not isinstance(shard_hash, str):
            raise VertexSensoryBatchError("production shard SHA-256 is missing")
        validate_sha256(shard_hash, field="production shard sha256")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != REQUEST_COUNT:
        raise VertexSensoryBatchError(
            f"production manifest must contain {REQUEST_COUNT} request identities"
        )
    seen_keys: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()
    cocktail_ids: set[int] = set()
    cocktail_ids_by_row: dict[int, set[int]] = {}
    row_indexes_by_cocktail: dict[int, set[int]] = {}
    for raw in requests:
        if not isinstance(raw, Mapping):
            raise VertexSensoryBatchError("manifest request identity must be an object")
        try:
            key = raw["key"]
            cocktail_id = raw["cocktail_id"]
            row_index = raw["row_index"]
            axis_order = raw["axis_order"]
            axis_id = raw["axis_id"]
            shard_index = raw["shard_index"]
            prompt_hash = raw["prompt_sha256"]
        except KeyError as error:
            raise VertexSensoryBatchError(
                f"manifest request identity is missing {error.args[0]}"
            ) from error
        if (
            not isinstance(key, str)
            or type(cocktail_id) is not int
            or type(row_index) is not int
            or type(axis_order) is not int
            or not isinstance(axis_id, str)
            or type(shard_index) is not int
            or not isinstance(prompt_hash, str)
        ):
            raise VertexSensoryBatchError("manifest request identity has invalid types")
        expected_key = f"r{row_index:04d}-a{axis_order:02d}"
        if (
            key != expected_key
            or row_index not in range(CORPUS_ROWS)
            or axis_order not in range(AXIS_COUNT)
            or axis_id != SENSORY_V2_REGISTRY.axes[axis_order].axis_id
            or shard_index != axis_order % SHARD_COUNT
        ):
            raise VertexSensoryBatchError(
                "manifest request identity/order/shard contract is invalid"
            )
        validate_sha256(prompt_hash, field="prompt_sha256")
        pair = (row_index, axis_order)
        if key in seen_keys or pair in seen_pairs:
            raise VertexSensoryBatchError("manifest request identities are duplicated")
        seen_keys.add(key)
        seen_pairs.add(pair)
        cocktail_ids.add(cocktail_id)
        cocktail_ids_by_row.setdefault(row_index, set()).add(cocktail_id)
        row_indexes_by_cocktail.setdefault(cocktail_id, set()).add(row_index)
    if (
        len(cocktail_ids) != CORPUS_ROWS
        or id_set_sha256(cocktail_ids) != COHORT_ID_SET_SHA256
    ):
        raise VertexSensoryBatchError(
            "manifest request cocktail identities do not match the current cohort"
        )
    if (
        set(cocktail_ids_by_row) != set(range(CORPUS_ROWS))
        or any(len(ids) != 1 for ids in cocktail_ids_by_row.values())
        or any(len(indexes) != 1 for indexes in row_indexes_by_cocktail.values())
    ):
        raise VertexSensoryBatchError(
            "manifest row-to-cocktail identity mapping is inconsistent"
        )


def gcs_run_metadata(
    *,
    run_id: str,
    manifest_sha256: str,
    shard_index: int,
    object_sha256: str,
) -> dict[str, str]:
    validate_sha256(manifest_sha256, field="manifest_sha256")
    validate_sha256(object_sha256, field="object_sha256")
    if shard_index not in range(SHARD_COUNT):
        raise VertexSensoryBatchError("shard_index must be in [0, 7]")
    return {
        "schema-version": str(MANIFEST_SCHEMA_VERSION),
        "run-id": run_id,
        "manifest-sha256": manifest_sha256,
        "shard-index": str(shard_index),
        "object-sha256": object_sha256,
    }


def adc_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return non-secret ADC hints; API keys and credential JSON paths are ignored."""

    return {
        key: value
        for key, value in environ.items()
        if key in ADC_ENV_ALLOWLIST and value
    }


def guard_job_creation(
    ledger: RunCostLedger,
    estimate: CostEstimate,
    *,
    allow_soft_stop_override: bool = False,
) -> Decimal:
    """Enforce one known-terminal job and cumulative conservative spend bounds."""

    latest = ledger.most_recent_job
    if latest is not None and latest.state not in TERMINAL_JOB_STATES:
        state = latest.state if latest.state in ACTIVE_JOB_STATES else "UNKNOWN"
        raise VertexSensoryBatchError(
            f"new job blocked while most recent job state is {state}"
        )
    projected = (
        Decimal(ledger.historical_cost_usd)
        + HISTORICAL_RESERVE_USD
        + estimate.estimated_cost_usd
    )
    if projected >= HARD_CREATION_BLOCK_USD:
        raise VertexSensoryBatchError(
            f"hard creation block: projected ${projected} reaches "
            f"${HARD_CREATION_BLOCK_USD}"
        )
    if projected >= SOFT_STOP_USD and not allow_soft_stop_override:
        raise VertexSensoryBatchError(
            f"soft stop: projected ${projected} reaches ${SOFT_STOP_USD}"
        )
    return projected


def guard_production_job_creation(
    manifest: Mapping[str, object],
    ledger: RunCostLedger,
    *,
    allow_soft_stop_override: bool = False,
) -> Decimal:
    """Apply immutable production, pilot-token, one-job, and budget gates."""

    validate_full_manifest(manifest)
    pilot = manifest.get("pilot_token_envelope")
    if (
        not isinstance(pilot, Mapping)
        or pilot.get("status") != FULL_PRODUCTION_TOKEN_STATUS
        or pilot.get("review_scope") != FULL_PRODUCTION_TOKEN_REVIEW_SCOPE
        or pilot.get("full_production_authorized") is not True
    ):
        raise VertexSensoryBatchError(
            "production job creation requires explicit full-production token review"
        )
    planned = pilot.get("planning_input_tokens_per_request")
    measured_count = pilot.get("measured_request_count")
    minimum = pilot.get("measured_min_tokens")
    maximum = pilot.get("measured_max_tokens")
    mean = pilot.get("measured_mean_tokens")
    token_counts_hash = pilot.get("token_counts_sha256")
    if (
        planned != PLANNING_INPUT_TOKEN_ENVELOPE
        or type(measured_count) is not int
        or measured_count < MIN_FULL_PRODUCTION_TOKEN_MEASUREMENTS
        or measured_count > REQUEST_COUNT
        or type(minimum) is not int
        or minimum <= 0
        or type(maximum) is not int
        or maximum < minimum
        or maximum > PLANNING_INPUT_TOKEN_ENVELOPE
        or not isinstance(mean, (int, float))
        or isinstance(mean, bool)
        or not math.isfinite(float(mean))
        or not minimum <= float(mean) <= maximum
        or not isinstance(token_counts_hash, str)
    ):
        raise VertexSensoryBatchError("production pilot token envelope is invalid")
    validate_sha256(token_counts_hash, field="pilot token_counts_sha256")
    lifecycle = manifest.get("gcs_lifecycle")
    if lifecycle != GcsLifecycleContract().to_dict():
        raise VertexSensoryBatchError("production GCS lifecycle contract is invalid")
    return guard_job_creation(
        ledger,
        estimate_cost(REQUEST_COUNT),
        allow_soft_stop_override=allow_soft_stop_override,
    )


def load_ledger(path: Path) -> RunCostLedger:
    if not path.exists():
        return RunCostLedger()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        jobs = tuple(JobLedgerEntry(**item) for item in raw.get("jobs", []))
        return RunCostLedger(
            jobs=jobs,
            historical_cost_usd=str(raw.get("historical_cost_usd", "0.00")),
            schema_version=int(raw.get("schema_version", 0)),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise VertexSensoryBatchError(f"invalid run ledger {path}: {error}") from error


def update_job_state(
    ledger: RunCostLedger,
    *,
    job_name: str,
    state: str | None,
    updated_at: str,
) -> RunCostLedger:
    """Unknown/missing status is persisted as UNKNOWN and therefore blocks creation."""

    jobs = list(ledger.jobs)
    matches = [index for index, job in enumerate(jobs) if job.job_name == job_name]
    if len(matches) != 1:
        raise VertexSensoryBatchError("job ledger entry must exist exactly once")
    index = matches[0]
    old = jobs[index]
    jobs[index] = JobLedgerEntry(
        run_id=old.run_id,
        job_name=old.job_name,
        state=state if state in TERMINAL_JOB_STATES | ACTIVE_JOB_STATES else "UNKNOWN",
        estimated_cost_usd=old.estimated_cost_usd,
        created_at=old.created_at,
        updated_at=updated_at,
    )
    return RunCostLedger(
        jobs=tuple(jobs),
        historical_cost_usd=ledger.historical_cost_usd,
    )


def record_created_job(
    ledger: RunCostLedger,
    *,
    run_id: str,
    job_name: str,
    estimate: CostEstimate,
    created_at: str,
) -> RunCostLedger:
    guard_job_creation(ledger, estimate)
    entry = JobLedgerEntry(
        run_id=run_id,
        job_name=job_name,
        state="JOB_STATE_PENDING",
        estimated_cost_usd=str(estimate.estimated_cost_usd),
        created_at=created_at,
        updated_at=created_at,
    )
    return RunCostLedger(
        jobs=(*ledger.jobs, entry),
        historical_cost_usd=ledger.historical_cost_usd,
    )


def _response_object(record: Mapping[str, object]) -> Mapping[str, object]:
    response = record.get("response")
    if isinstance(response, Mapping):
        return response
    # Recorded fixtures may already contain the generateContent response.
    if "candidates" in record:
        return record
    raise VertexSensoryBatchError("response object is missing")


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(math.fsum(math.exp(value - maximum) for value in values))


def _candidate_logprobs(response: Mapping[str, object]) -> dict[str, float]:
    candidates = response.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(candidates[0], Mapping)
    ):
        raise VertexSensoryBatchError("response candidates are missing")
    log_result = candidates[0].get("logprobsResult")
    if not isinstance(log_result, Mapping):
        log_result = candidates[0].get("logprobs_result")
    if not isinstance(log_result, Mapping):
        raise VertexSensoryBatchError("logprobs result is missing")
    top = log_result.get("topCandidates")
    if top is None:
        top = log_result.get("top_candidates")
    if not isinstance(top, list):
        raise VertexSensoryBatchError("topCandidates is missing")

    by_label: dict[str, list[float]] = {label: [] for label in LABELS}
    for position in top:
        if not isinstance(position, Mapping):
            continue
        values = position.get("candidates")
        if not isinstance(values, list):
            continue
        for candidate in values:
            if not isinstance(candidate, Mapping):
                continue
            token = candidate.get("token")
            raw_logprob = candidate.get("logProbability")
            if raw_logprob is None:
                raw_logprob = candidate.get("log_probability")
            if not isinstance(token, str) or token.strip() not in LABELS:
                continue
            if isinstance(raw_logprob, bool) or not isinstance(
                raw_logprob, (int, float)
            ):
                continue
            logprob = float(raw_logprob)
            if math.isfinite(logprob):
                by_label[token.strip()].append(logprob)
    missing = [label for label, values in by_label.items() if not values]
    if missing:
        raise VertexSensoryBatchError(
            f"incomplete A-E logprobs; missing {','.join(missing)}"
        )
    return {label: _logsumexp(by_label[label]) for label in LABELS}


def _selected_label(response: Mapping[str, object]) -> str:
    candidates = response.get("candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or not isinstance(candidates[0], Mapping)
    ):
        raise VertexSensoryBatchError("response candidates are missing")
    content = candidates[0].get("content")
    if not isinstance(content, Mapping):
        raise VertexSensoryBatchError("candidate content is missing")
    parts = content.get("parts")
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], Mapping)
    ):
        raise VertexSensoryBatchError("candidate must contain exactly one text part")
    text = parts[0].get("text")
    if not isinstance(text, str) or text.strip() not in LABELS:
        raise VertexSensoryBatchError("enum response must be exactly one A-E code")
    return text.strip()


def parse_response_line(
    raw_line: bytes,
    request: Mapping[str, object],
) -> ParsedDistribution:
    raw_hash = sha256_bytes(raw_line)
    try:
        decoded = json.loads(raw_line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VertexSensoryBatchError("response line is invalid JSON") from error
    if not isinstance(decoded, Mapping):
        raise VertexSensoryBatchError("response line must be a JSON object")
    if decoded.get("error") or decoded.get("status") not in (None, "", {}):
        raise VertexSensoryBatchError("Vertex response contains an error status")
    response = _response_object(decoded)
    selected = _selected_label(response)
    logprobs = _candidate_logprobs(response)
    normalizer = _logsumexp(tuple(logprobs.values()))
    probabilities = tuple(math.exp(logprobs[label] - normalizer) for label in LABELS)
    response_hash = canonical_sha256(response)
    cocktail_id = request.get("cocktail_id")
    axis_order = request.get("axis_order")
    if type(cocktail_id) is not int or type(axis_order) is not int:
        raise VertexSensoryBatchError(
            "manifest request cocktail_id/axis_order must be integers"
        )
    return ParsedDistribution(
        key=str(request["key"]),
        cocktail_id=cocktail_id,
        axis_order=axis_order,
        axis_id=str(request["axis_id"]),
        selected_label=selected,
        probabilities=probabilities,
        response_sha256=response_hash,
        raw_response_sha256=raw_hash,
    )


def manifest_requests_by_shard(
    manifest: Mapping[str, object],
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    raw = manifest.get("requests")
    if not isinstance(raw, list):
        raise VertexSensoryBatchError("manifest requests are missing")
    shards: list[list[Mapping[str, object]]] = [[] for _ in range(SHARD_COUNT)]
    keys: set[str] = set()
    for request in raw:
        if not isinstance(request, Mapping):
            raise VertexSensoryBatchError("manifest request is invalid")
        key = request.get("key")
        shard_index = request.get("shard_index")
        if not isinstance(key, str) or key in keys:
            raise VertexSensoryBatchError("manifest request keys must be unique")
        if type(shard_index) is not int or shard_index not in range(SHARD_COUNT):
            raise VertexSensoryBatchError("manifest request shard is invalid")
        shards[shard_index].append(request)
        keys.add(key)
    return tuple(tuple(shard) for shard in shards)


def echoed_request_prompt_sha256(raw_line: bytes) -> str:
    """Validate one echoed Vertex request and return its prompt identity."""

    try:
        decoded = json.loads(raw_line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise VertexSensoryBatchError("response line is invalid JSON") from error
    if not isinstance(decoded, Mapping):
        raise VertexSensoryBatchError("response line must be a JSON object")
    echoed = decoded.get("request")
    if not isinstance(echoed, Mapping):
        raise VertexSensoryBatchError("response line has no echoed request")
    if set(echoed) != {"contents", "generationConfig"}:
        raise VertexSensoryBatchError("echoed request fields do not match the contract")
    if echoed.get("generationConfig") != REQUEST_CONFIG:
        raise VertexSensoryBatchError(
            "echoed request generationConfig does not match the contract"
        )
    contents = echoed.get("contents")
    if (
        not isinstance(contents, list)
        or len(contents) != 1
        or not isinstance(contents[0], Mapping)
        or contents[0].get("role") != "user"
    ):
        raise VertexSensoryBatchError("echoed request contents are invalid")
    parts = contents[0].get("parts")
    if (
        not isinstance(parts, list)
        or len(parts) != 1
        or not isinstance(parts[0], Mapping)
        or set(parts[0]) != {"text"}
        or not isinstance(parts[0].get("text"), str)
    ):
        raise VertexSensoryBatchError("echoed request prompt is invalid")
    prompt = str(parts[0]["text"])
    return sha256_bytes(prompt.encode("utf-8"))


def _requests_by_prompt_sha256(
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    if manifest.get("request_config_sha256") != REQUEST_CONFIG_SHA256:
        raise VertexSensoryBatchError(
            "manifest request config SHA-256 does not match the canonical contract"
        )
    shards = manifest_requests_by_shard(manifest)
    by_prompt: dict[str, Mapping[str, object]] = {}
    for request in (request for shard in shards for request in shard):
        prompt_sha256 = request.get("prompt_sha256")
        if not isinstance(prompt_sha256, str):
            raise VertexSensoryBatchError("manifest request prompt SHA-256 is missing")
        validate_sha256(prompt_sha256, field="manifest request prompt_sha256")
        if prompt_sha256 in by_prompt:
            raise VertexSensoryBatchError(
                "manifest request prompt SHA-256 values must be unique"
            )
        by_prompt[prompt_sha256] = request
    return by_prompt


def parse_recorded_outputs(
    manifest: Mapping[str, object],
    output_paths: Sequence[Path],
) -> tuple[tuple[ParsedDistribution, ...], tuple[QuarantineRecord, ...]]:
    """Parse echoed identities safely despite line or output-shard reordering."""

    if len(output_paths) != SHARD_COUNT:
        raise VertexSensoryBatchError("exactly 8 output shard paths are required")
    by_prompt = _requests_by_prompt_sha256(manifest)
    expected_keys = {str(request["key"]) for request in by_prompt.values()}
    parsed: list[ParsedDistribution] = []
    quarantined: list[QuarantineRecord] = []
    seen_keys: set[str] = set()
    all_lines: list[tuple[int, int, bytes]] = []
    for shard_index, path in enumerate(output_paths):
        try:
            lines = path.read_bytes().splitlines()
        except OSError as error:
            raise VertexSensoryBatchError(
                f"cannot read response shard {path}"
            ) from error
        all_lines.extend(
            (shard_index, line_number, line)
            for line_number, line in enumerate(lines, start=1)
        )
    if len(all_lines) != len(by_prompt):
        raise VertexSensoryBatchError(
            f"received {len(all_lines)} responses for {len(by_prompt)} requests"
        )
    for shard_index, line_number, line in all_lines:
        request: Mapping[str, object] | None = None
        key = ""
        try:
            prompt_sha256 = echoed_request_prompt_sha256(line)
            request = by_prompt.get(prompt_sha256)
            if request is None:
                raise VertexSensoryBatchError(
                    "echoed request prompt is absent from the manifest"
                )
            key = str(request.get("key", ""))
            if key in seen_keys:
                raise VertexSensoryBatchError("duplicate echoed request identity")
            seen_keys.add(key)
            parsed.append(parse_response_line(line, request))
        except (KeyError, TypeError, ValueError) as error:
            response_hash: str | None = None
            try:
                decoded = json.loads(line)
                if isinstance(decoded, Mapping):
                    response_hash = canonical_sha256(_response_object(decoded))
            except (ValueError, TypeError):
                pass
            quarantined.append(
                QuarantineRecord(
                    key=key,
                    shard_index=shard_index,
                    line_number=line_number,
                    reason=str(error),
                    raw_response_sha256=sha256_bytes(line),
                    response_sha256=response_hash,
                )
            )
    missing = sorted(expected_keys - seen_keys)
    if missing:
        raise VertexSensoryBatchError(
            f"missing {len(missing)} echoed request identities"
        )
    parsed.sort(key=lambda item: (item.cocktail_id, item.axis_order))
    quarantined.sort(key=lambda item: (item.shard_index, item.line_number))
    return tuple(parsed), tuple(quarantined)


def project_ready_records(
    distributions: Sequence[ParsedDistribution],
    *,
    expected_cocktails: int | None = CORPUS_ROWS,
) -> tuple[dict[str, object], ...]:
    """Aggregate only complete, registry-ordered 48-axis distributions."""

    grouped: dict[int, list[ParsedDistribution]] = {}
    for item in distributions:
        grouped.setdefault(item.cocktail_id, []).append(item)
    if expected_cocktails is not None and len(grouped) != expected_cocktails:
        raise VertexSensoryBatchError(
            f"expected {expected_cocktails} complete cocktails, got {len(grouped)}"
        )
    result: list[dict[str, object]] = []
    for cocktail_id in sorted(grouped):
        items = sorted(grouped[cocktail_id], key=lambda item: item.axis_order)
        expected_axes = tuple(axis.axis_id for axis in SENSORY_V2_REGISTRY.axes)
        if (
            tuple(item.axis_order for item in items) != tuple(range(AXIS_COUNT))
            or tuple(item.axis_id for item in items) != expected_axes
        ):
            raise VertexSensoryBatchError(
                f"cocktail {cocktail_id} does not have exactly 48 ordered axes"
            )
        axes = tuple(AxisSoftLabels(item.axis_id, item.probabilities) for item in items)
        raw = tuple(value for axis in axes for value in axis.probabilities)
        result.append(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "cocktail_id": cocktail_id,
                "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
                "labels": list(LABELS),
                "axes": [
                    {
                        "axis_order": item.axis_order,
                        "axis_id": item.axis_id,
                        "probabilities": list(item.probabilities),
                        "response_sha256": item.response_sha256,
                        "raw_response_sha256": item.raw_response_sha256,
                    }
                    for item in items
                ],
                "raw_probabilities": list(raw),
                "source_sha256": teacher_source_sha256(
                    SENSORY_V2_REGISTRY.registry_sha256,
                    raw,
                ),
            }
        )
    return tuple(result)


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def records_jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    lines = [
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for record in records
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def atomic_create(path: Path, data: bytes) -> None:
    """Create an artifact atomically and refuse to replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise VertexSensoryBatchError(
                f"refusing to replace existing artifact: {path}"
            ) from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
