from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from app.sensory_embedding import vertex_live
from app.sensory_embedding.registry import SENSORY_V2_REGISTRY
from app.sensory_embedding.vertex_batch import (
    AXIS_REGISTRY_FILE_SHA256,
    DEFAULT_PROJECT,
    COHORT_SOURCE_FILE_SHA256,
    GcsLifecycleContract,
    HARD_CREATION_BLOCK_USD,
    HISTORICAL_RESERVE_USD,
    FULL_PRODUCTION_TOKEN_REVIEW_SCOPE,
    FULL_PRODUCTION_TOKEN_STATUS,
    PROMPT_SHA256,
    REQUEST_CONFIG,
    REQUEST_CONFIG_SHA256,
    SHARD_COUNT,
    SOFT_STOP_USD,
    estimate_cost,
)
from app.sensory_embedding.vertex_live import (
    API_KEY_ENV_VARS,
    BUCKET_LOCATION,
    LIVE_CORPUS_ROWS,
    LIVE_ID_ALLOWLIST_SHA256,
    LIVE_PILOT_APPROVAL_MARKER,
    LIVE_PILOT_APPROVAL_MARKER_SHA256,
    LIVE_PILOT_APPROVED_MANIFEST_SHA256,
    LIVE_PILOT_FROZEN_SOURCE_SHA256,
    LIVE_PILOT_MANIFEST_TYPE,
    LIVE_PILOT_REQUEST_COUNT,
    LIVE_PILOT_ROWS,
    LIVE_PILOT_SELECTED_IDS,
    LIVE_PILOT_SELECTED_ID_SET_SHA256,
    LIVE_PILOT_SELECTION_POLICY,
    LIVE_PILOT_SHARD_SIZE,
    LIVE_PILOT_STATUS,
    LIVE_REQUEST_COUNT,
    LIVE_SHARD_SIZE,
    LOCATION as VERTEX_LOCATION,
    MODEL_ID,
    PILOT_RUN_SCOPE,
    REQUIRED_GCS_PERMISSIONS,
    UNKNOWN_REMOTE_STATE,
    AdcIdentity,
    AmbiguousRemoteCreateError,
    DedicatedBucketContract,
    GoogleBatchGateway,
    GoogleStorageGateway,
    RemoteJob,
    StorageObject,
    VertexLiveError,
    cleanup_run,
    download_outputs,
    refresh_job_status,
    submit_pilot_shard_once,
    submit_shard_once,
)

_TEST_DEDICATED_BUCKET = DedicatedBucketContract(
    name=(
        f"cm-sensory-{hashlib.sha256(DEFAULT_PROJECT.encode()).hexdigest()[:10]}-"
        f"{'0' * 32}"
    ),
    location=BUCKET_LOCATION,
)


def _identity() -> AdcIdentity:
    return AdcIdentity(
        credentials=object(),
        detected_project_id=DEFAULT_PROJECT,
        credential_project_id=DEFAULT_PROJECT,
        quota_project_id=DEFAULT_PROJECT,
        is_service_account=True,
    )


def _manifest_fixture(
    tmp_path: Path,
    *,
    pilot_status: str = FULL_PRODUCTION_TOKEN_STATUS,
) -> tuple[Path, dict[str, Any]]:
    missing_ids = {
        57,
        68,
        71,
        112,
        115,
        116,
        243,
        260,
        273,
        304,
        326,
        355,
        361,
        367,
        368,
        369,
        370,
        371,
        392,
        446,
        554,
        555,
        556,
    }
    cocktail_ids = [
        cocktail_id for cocktail_id in range(1, 626) if cocktail_id not in missing_ids
    ]
    assert len(cocktail_ids) == LIVE_CORPUS_ROWS
    requests = [
        {
            "key": f"r{row_index:04d}-a{axis.axis_order:02d}",
            "cocktail_id": cocktail_id,
            "row_index": row_index,
            "axis_order": axis.axis_order,
            "axis_id": axis.axis_id,
            "shard_index": axis.axis_order % SHARD_COUNT,
            "prompt_sha256": hashlib.sha256(
                f"fixture:{row_index}:{axis.axis_order}".encode()
            ).hexdigest(),
        }
        for row_index, cocktail_id in enumerate(cocktail_ids)
        for axis in SENSORY_V2_REGISTRY.axes
    ]
    requests_by_shard = [
        [request for request in requests if request["shard_index"] == shard_index]
        for shard_index in range(SHARD_COUNT)
    ]
    shards: list[dict[str, Any]] = []
    for shard_index, shard_requests in enumerate(requests_by_shard):
        payload = (
            "\n".join(
                json.dumps(
                    {
                        "request": {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "text": (
                                                f"fixture:{request['row_index']}:"
                                                f"{request['axis_order']}"
                                            )
                                        }
                                    ],
                                }
                            ],
                            "generationConfig": REQUEST_CONFIG,
                        }
                    },
                    separators=(",", ":"),
                )
                for request in shard_requests
            )
            + "\n"
        ).encode()
        filename = f"requests-{shard_index:02d}.jsonl"
        (tmp_path / filename).write_bytes(payload)
        shards.append(
            {
                "shard_index": shard_index,
                "filename": filename,
                "record_count": LIVE_SHARD_SIZE,
                "axis_orders": list(range(shard_index, 48, SHARD_COUNT)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": "sensory-live-unit-run",
        "model": MODEL_ID,
        "project": DEFAULT_PROJECT,
        "location": VERTEX_LOCATION,
        "row_count": LIVE_CORPUS_ROWS,
        "input_row_count": LIVE_CORPUS_ROWS,
        "id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "input_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "cohort_source_sha256": COHORT_SOURCE_FILE_SHA256,
        "cohort_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "cohort_row_count": LIVE_CORPUS_ROWS,
        "request_count": LIVE_REQUEST_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_size": LIVE_SHARD_SIZE,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "pilot_token_envelope": {
            "status": pilot_status,
            "review_scope": FULL_PRODUCTION_TOKEN_REVIEW_SCOPE,
            "full_production_authorized": True,
            "planning_input_tokens_per_request": 845,
            "measured_request_count": 8,
            "measured_min_tokens": 400,
            "measured_max_tokens": 844,
            "measured_mean_tokens": 622.0,
            "token_counts_sha256": "1" * 64,
        },
        "gcs_lifecycle": GcsLifecycleContract().to_dict(),
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
        "shards": shards,
        "requests": requests,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return path, manifest


def _pilot_manifest_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, dict[str, Any]]:
    requests = [
        {
            "key": f"r{row_index:04d}-a{axis.axis_order:02d}",
            "cocktail_id": cocktail_id,
            "row_index": row_index,
            "axis_order": axis.axis_order,
            "axis_id": axis.axis_id,
            "shard_index": axis.axis_order % SHARD_COUNT,
            "prompt_sha256": hashlib.sha256(
                f"pilot:{row_index}:{axis.axis_order}".encode()
            ).hexdigest(),
        }
        for row_index, cocktail_id in enumerate(LIVE_PILOT_SELECTED_IDS)
        for axis in SENSORY_V2_REGISTRY.axes
    ]
    shards: list[dict[str, Any]] = []
    for shard_index in range(SHARD_COUNT):
        shard_requests = [
            request for request in requests if request["shard_index"] == shard_index
        ]
        payload = (
            "\n".join(
                json.dumps(
                    {
                        "request": {
                            "contents": [
                                {
                                    "role": "user",
                                    "parts": [
                                        {
                                            "text": (
                                                f"pilot:{request['row_index']}:"
                                                f"{request['axis_order']}"
                                            )
                                        }
                                    ],
                                }
                            ],
                            "generationConfig": REQUEST_CONFIG,
                        }
                    },
                    separators=(",", ":"),
                )
                for request in shard_requests
            )
            + "\n"
        ).encode()
        filename = f"requests-{shard_index:02d}.jsonl"
        (tmp_path / filename).write_bytes(payload)
        shards.append(
            {
                "shard_index": shard_index,
                "filename": filename,
                "record_count": LIVE_PILOT_SHARD_SIZE,
                "axis_orders": list(range(shard_index, 48, SHARD_COUNT)),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_type": LIVE_PILOT_MANIFEST_TYPE,
        "run_id": "sensory-live-pilot-unit",
        "status": LIVE_PILOT_STATUS,
        "run_scope": PILOT_RUN_SCOPE,
        "full_production_authorized": False,
        "model": MODEL_ID,
        "project": DEFAULT_PROJECT,
        "location": VERTEX_LOCATION,
        "row_count": LIVE_PILOT_ROWS,
        "request_count": LIVE_PILOT_REQUEST_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_size": LIVE_PILOT_SHARD_SIZE,
        "parent_cohort_row_count": LIVE_CORPUS_ROWS,
        "parent_cohort_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "parent_cohort_source_sha256": COHORT_SOURCE_FILE_SHA256,
        "parent_frozen_source_sha256": LIVE_PILOT_FROZEN_SOURCE_SHA256,
        "selected_id_set_sha256": LIVE_PILOT_SELECTED_ID_SET_SHA256,
        "selected_cocktail_ids": list(LIVE_PILOT_SELECTED_IDS),
        "selection_policy": LIVE_PILOT_SELECTION_POLICY,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
        "user_approval_marker_sha256": LIVE_PILOT_APPROVAL_MARKER_SHA256,
        "estimated_cost_usd": str(
            estimate_cost(LIVE_PILOT_REQUEST_COUNT).estimated_cost_usd
        ),
        "gcs_lifecycle": GcsLifecycleContract().to_dict(),
        "shards": shards,
        "requests": requests,
    }
    path = tmp_path / "pilot-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        vertex_live,
        "LIVE_PILOT_APPROVED_MANIFEST_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    return path, manifest


class FakeStorage:
    def __init__(self) -> None:
        self.permission_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.upload_calls: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.download_calls: list[dict[str, Any]] = []
        self.delete_calls: list[dict[str, Any]] = []
        self.delete_bucket_calls: list[dict[str, Any]] = []
        self.list_results: list[tuple[StorageObject, ...]] = []
        self.payloads: dict[str, bytes] = {}
        self.closed = 0
        self.assert_api_keys_absent = False

    def _check_environment(self) -> None:
        if self.assert_api_keys_absent:
            assert all(variable not in os.environ for variable in API_KEY_ENV_VARS)

    def find_compatible_bucket(self, **kwargs: Any) -> str:
        self._check_environment()
        self.permission_calls.append(kwargs)
        return f"{kwargs['bucket_prefix']}{'0' * 32}"

    def upload_jsonl_create_only(self, **kwargs: Any) -> None:
        self._check_environment()
        self.upload_calls.append(kwargs)

    def list_objects(self, **kwargs: Any) -> tuple[StorageObject, ...]:
        self._check_environment()
        self.list_calls.append(kwargs)
        return self.list_results.pop(0)

    def download_object(self, **kwargs: Any) -> bytes:
        self._check_environment()
        self.download_calls.append(kwargs)
        return self.payloads[kwargs["object_name"]]

    def delete_object(self, **kwargs: Any) -> None:
        self._check_environment()
        self.delete_calls.append(kwargs)

    def delete_bucket(self, **kwargs: Any) -> None:
        self._check_environment()
        self.delete_bucket_calls.append(kwargs)

    def close(self) -> None:
        self.closed += 1


class FakeBatch:
    def __init__(self) -> None:
        self.create_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.create_error: Exception | None = None
        self.create_state = "JOB_STATE_PENDING"
        self.get_state = "JOB_STATE_SUCCEEDED"
        self.closed = 0
        self.assert_api_keys_absent = False

    def _check_environment(self) -> None:
        if self.assert_api_keys_absent:
            assert all(variable not in os.environ for variable in API_KEY_ENV_VARS)

    def create_job(self, **kwargs: Any) -> RemoteJob:
        self._check_environment()
        self.create_calls.append(kwargs)
        if self.create_error is not None:
            raise self.create_error
        return RemoteJob(
            name=(
                f"projects/{DEFAULT_PROJECT}/locations/{VERTEX_LOCATION}/"
                "batchPredictionJobs/123456"
            ),
            state=self.create_state,
        )

    def get_job(self, **kwargs: Any) -> RemoteJob:
        self._check_environment()
        self.get_calls.append(kwargs)
        return RemoteJob(name=kwargs["job_name"], state=self.get_state)

    def close(self) -> None:
        self.closed += 1


def _submit(
    manifest_path: Path,
    ledger_path: Path,
    storage: FakeStorage,
    batch: FakeBatch,
    *,
    execute_live: bool = True,
) -> dict[str, Any]:
    return submit_shard_once(
        execute_live=execute_live,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        shard_index=0,
        credential_loader=_identity,
        storage_factory=lambda credentials: storage,
        batch_factory=lambda credentials: batch,
        clock=lambda: "2026-08-06T01:02:03+00:00",
    )


def _read_ledger(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_live_flag_and_offline_pilot_gate_block_before_adc_or_network(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal calls
        calls += 1
        raise AssertionError("ADC must not be loaded")

    with pytest.raises(VertexLiveError, match="disabled"):
        submit_shard_once(
            execute_live=False,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert calls == 0
    assert not ledger_path.exists()

    manifest_path, _ = _manifest_fixture(
        tmp_path,
        pilot_status="pilot_required_before_job_creation",
    )
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "offline_gate"
    assert calls == 0
    assert not ledger_path.exists()


def test_live_pilot_scope_requires_marker_and_never_enters_full_create(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_path, _ = _pilot_manifest_fixture(tmp_path, monkeypatch)
    ledger_path = tmp_path / "pilot-ledger.json"
    adc_calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("ADC must not be loaded")

    with pytest.raises(VertexLiveError) as missing:
        submit_pilot_shard_once(
            execute_live_pilot=False,
            user_approval_marker=LIVE_PILOT_APPROVAL_MARKER,
            manifest_path=pilot_path,
            ledger_path=ledger_path,
            shard_index=0,
            dedicated_bucket=_TEST_DEDICATED_BUCKET,
            credential_loader=forbidden_adc,
        )
    assert missing.value.phase == "authorization"

    with pytest.raises(VertexLiveError) as wrong_marker:
        submit_pilot_shard_once(
            execute_live_pilot=True,
            user_approval_marker="wrong",
            manifest_path=pilot_path,
            ledger_path=ledger_path,
            shard_index=0,
            dedicated_bucket=_TEST_DEDICATED_BUCKET,
            credential_loader=forbidden_adc,
        )
    assert wrong_marker.value.phase == "offline_gate"
    assert adc_calls == 0
    assert not ledger_path.exists()

    with pytest.raises(VertexLiveError) as full_path:
        submit_shard_once(
            execute_live=True,
            manifest_path=pilot_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert full_path.value.phase == "offline_gate"
    assert adc_calls == 0


def test_live_pilot_creates_only_one_60_record_shard_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_path, _ = _pilot_manifest_fixture(tmp_path, monkeypatch)
    ledger_path = tmp_path / "pilot-ledger.json"
    storage = FakeStorage()
    batch = FakeBatch()

    result = submit_pilot_shard_once(
        execute_live_pilot=True,
        user_approval_marker=LIVE_PILOT_APPROVAL_MARKER,
        manifest_path=pilot_path,
        ledger_path=ledger_path,
        shard_index=0,
        dedicated_bucket=_TEST_DEDICATED_BUCKET,
        credential_loader=_identity,
        storage_factory=lambda credentials: storage,
        batch_factory=lambda credentials: batch,
        clock=lambda: "2026-08-06T01:02:03+00:00",
    )

    assert result["status"] == "BATCH_CREATED"
    assert result["run_scope"] == PILOT_RUN_SCOPE
    assert len(storage.upload_calls) == len(batch.create_calls) == 1
    assert storage.upload_calls[0]["data"].count(b"\n") == LIVE_PILOT_SHARD_SIZE
    ledger = _read_ledger(ledger_path)
    assert ledger["run_scope"] == PILOT_RUN_SCOPE
    assert ledger["shard_record_count"] == LIVE_PILOT_SHARD_SIZE
    assert ledger["jobs"][0]["estimated_cost_usd"] == str(
        estimate_cost(LIVE_PILOT_SHARD_SIZE).estimated_cost_usd
    )


def test_pilot_content_cannot_be_rebound_with_all_self_consistent_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot_path, manifest = _pilot_manifest_fixture(tmp_path, monkeypatch)
    shard_path = tmp_path / "requests-00.jsonl"
    lines = shard_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["request"]["contents"][0]["parts"][0]["text"] = "tampered prompt"
    lines[0] = json.dumps(first, separators=(",", ":"))
    payload = ("\n".join(lines) + "\n").encode()
    shard_path.write_bytes(payload)
    manifest["shards"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    first_request = next(
        request
        for request in manifest["requests"]
        if request["row_index"] == 0 and request["axis_order"] == 0
    )
    first_request["prompt_sha256"] = hashlib.sha256(b"tampered prompt").hexdigest()
    pilot_path.write_text(json.dumps(manifest), encoding="utf-8")
    adc_calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("ADC must not be loaded")

    with pytest.raises(VertexLiveError) as captured:
        submit_pilot_shard_once(
            execute_live_pilot=True,
            user_approval_marker=LIVE_PILOT_APPROVAL_MARKER,
            manifest_path=pilot_path,
            ledger_path=tmp_path / "ledger.json",
            shard_index=0,
            dedicated_bucket=_TEST_DEDICATED_BUCKET,
            credential_loader=forbidden_adc,
        )

    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0


def test_reviewed_prep_v2_manifest_digest_is_exactly_pinned() -> None:
    assert LIVE_PILOT_APPROVED_MANIFEST_SHA256 == (
        "06f7a1398537812bf5e31daecba9be7dfaa495ad54149003b6008034d059f396"
    )


def test_vertex_job_name_accepts_reviewed_numeric_project_name_only() -> None:
    numeric_name = (
        "projects/504835101849/locations/global/"
        "batchPredictionJobs/123456789"
    )
    assert vertex_live._validate_job_name(numeric_name) == numeric_name
    with pytest.raises(VertexLiveError, match="outside the reviewed"):
        vertex_live._validate_job_name(
            "projects/999999999999/locations/global/"
            "batchPredictionJobs/123456789"
        )


def test_one_measured_request_cannot_unlock_full_production(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_fixture(tmp_path)
    manifest["pilot_token_envelope"]["measured_request_count"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    adc_calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("ADC must not be loaded")

    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=tmp_path / "ledger.json",
            shard_index=0,
            credential_loader=forbidden_adc,
        )

    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0


def test_live_gate_rejects_622_shape_and_wrong_frozen_id_allowlist(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    adc_calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("ADC must not be loaded")

    manifest["location"] = "asia-northeast3"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0

    manifest["location"] = VERTEX_LOCATION
    manifest["row_count"] = 622
    manifest["input_row_count"] = 622
    manifest["request_count"] = 29_856
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0

    manifest["row_count"] = LIVE_CORPUS_ROWS
    manifest["input_row_count"] = LIVE_CORPUS_ROWS
    manifest["request_count"] = LIVE_REQUEST_COUNT
    manifest["id_set_sha256"] = "0" * 64
    manifest["input_id_set_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0

    manifest["id_set_sha256"] = LIVE_ID_ALLOWLIST_SHA256
    manifest["input_id_set_sha256"] = LIVE_ID_ALLOWLIST_SHA256
    manifest.pop("cohort_source_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "offline_gate"
    assert adc_calls == 0


def test_submit_uses_exact_boundary_once_and_isolates_api_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    storage = FakeStorage()
    batch = FakeBatch()
    storage.assert_api_keys_absent = True
    batch.assert_api_keys_absent = True
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-visible")
    monkeypatch.setenv("GOOGLE_API_KEY", "must-not-be-visible")

    result = _submit(manifest_path, ledger_path, storage, batch)

    assert result["status"] == "BATCH_CREATED"
    assert result["create_attempts"] == 1
    assert result["retry_attempted"] is False
    assert result["fallback_attempted"] is False
    assert storage.permission_calls == [
        {
            "project": DEFAULT_PROJECT,
            "bucket_prefix": storage.upload_calls[0]["bucket"][:-32],
            "expected_bucket": None,
            "location": BUCKET_LOCATION,
            "lifecycle_days": 1,
            "permissions": tuple(sorted(REQUIRED_GCS_PERMISSIONS)),
        }
    ]
    assert storage.create_calls == []
    assert len(storage.upload_calls) == 1
    upload = storage.upload_calls[0]
    assert upload["data"] == (tmp_path / "requests-00.jsonl").read_bytes()
    assert upload["metadata"]["shard-index"] == "0"
    assert (
        upload["metadata"]["object-sha256"]
        == hashlib.sha256(upload["data"]).hexdigest()
    )
    assert len(batch.create_calls) == 1
    create = batch.create_calls[0]
    assert create["input_uri"].startswith("gs://cm-sensory-")
    assert create["output_uri"].endswith("/output/shard-00")
    assert os.environ["GEMINI_API_KEY"] == "must-not-be-visible"
    assert os.environ["GOOGLE_API_KEY"] == "must-not-be-visible"
    assert storage.closed == batch.closed == 1

    ledger = _read_ledger(ledger_path)
    assert ledger["jobs"][0]["create_attempts"] == 1
    assert ledger["jobs"][0]["state"] == "JOB_STATE_PENDING"
    event_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "live-ledger.events").iterdir()
    )
    assert DEFAULT_PROJECT not in event_text
    assert "must-not-be-visible" not in ledger_path.read_text(encoding="utf-8")

    with pytest.raises(VertexLiveError) as captured:
        _submit(manifest_path, ledger_path, FakeStorage(), FakeBatch())
    assert captured.value.phase == "no_resubmit_gate"


def test_ambiguous_batch_failure_is_recorded_once_and_blocks_resubmit(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    storage = FakeStorage()
    batch = FakeBatch()
    batch.create_error = TimeoutError("ambiguous transport timeout")

    with pytest.raises(AmbiguousRemoteCreateError):
        _submit(manifest_path, ledger_path, storage, batch)

    assert len(batch.create_calls) == 1
    ledger = _read_ledger(ledger_path)
    assert ledger["jobs"][0]["create_attempts"] == 1
    assert ledger["jobs"][0]["state"] == UNKNOWN_REMOTE_STATE
    assert ledger["jobs"][0]["job_name"] is None

    forbidden_storage = FakeStorage()
    forbidden_batch = FakeBatch()
    with pytest.raises(VertexLiveError) as captured:
        _submit(
            manifest_path,
            ledger_path,
            forbidden_storage,
            forbidden_batch,
        )
    assert captured.value.phase == "no_resubmit_gate"
    assert not forbidden_storage.create_calls
    assert not forbidden_batch.create_calls


def test_failed_gcs_permission_preflight_blocks_mutations_and_is_not_repeated(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    adc_calls = 0

    def counted_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        return _identity()

    class BlockedStorage(FakeStorage):
        def find_compatible_bucket(self, **kwargs: Any) -> str:
            super().find_compatible_bucket(**kwargs)
            raise VertexLiveError(
                "required existing-bucket GCS permissions are missing",
                phase="gcs_permission_preflight",
            )

    storage = BlockedStorage()
    batch = FakeBatch()
    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=counted_adc,
            storage_factory=lambda credentials: storage,
            batch_factory=lambda credentials: batch,
            clock=lambda: "2026-08-06T01:30:00+00:00",
        )
    assert captured.value.phase == "gcs_permission_preflight"
    assert len(storage.permission_calls) == 1
    assert storage.create_calls == []
    assert storage.upload_calls == []
    assert batch.create_calls == []
    assert _read_ledger(ledger_path)["gcs_permission_preflight"]["state"] == "BLOCKED"

    with pytest.raises(VertexLiveError) as repeated:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=counted_adc,
            storage_factory=lambda credentials: storage,
            batch_factory=lambda credentials: batch,
        )
    assert repeated.value.phase == "gcs_permission_gate"
    assert len(storage.permission_calls) == 1
    assert adc_calls == 1


@pytest.mark.parametrize(
    ("historical_cost", "expected_message"),
    [
        ("6.50", "soft stop"),
        ("8.50", "hard cumulative"),
    ],
)
def test_cumulative_budget_gates_run_before_adc(
    tmp_path: Path,
    historical_cost: str,
    expected_message: str,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    _submit(manifest_path, ledger_path, FakeStorage(), FakeBatch())
    ledger = _read_ledger(ledger_path)
    ledger["jobs"][0]["state"] = "JOB_STATE_SUCCEEDED"
    ledger["historical_cost_usd"] = historical_cost
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    adc_calls = 0

    def forbidden_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        raise AssertionError("ADC must not be loaded")

    with pytest.raises(VertexLiveError, match=expected_message) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=1,
            credential_loader=forbidden_adc,
        )
    assert captured.value.phase == "budget_gate"
    assert adc_calls == 0


def test_service_account_project_mismatch_blocks_before_factories(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    factory_calls = 0

    def wrong_identity() -> AdcIdentity:
        return AdcIdentity(
            credentials=object(),
            detected_project_id="wrong",
            credential_project_id="wrong",
            quota_project_id="wrong",
            is_service_account=True,
        )

    def forbidden_factory(credentials: Any) -> FakeStorage:
        nonlocal factory_calls
        factory_calls += 1
        return FakeStorage()

    with pytest.raises(VertexLiveError) as captured:
        submit_shard_once(
            execute_live=True,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=wrong_identity,
            storage_factory=forbidden_factory,
        )
    assert captured.value.phase == "adc_validation"
    assert factory_calls == 0
    assert not ledger_path.exists()


def test_status_is_one_read_only_call_and_failed_read_becomes_unknown(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    _submit(manifest_path, ledger_path, FakeStorage(), FakeBatch())
    status_batch = FakeBatch()
    status_batch.get_state = "JOB_STATE_SUCCEEDED"

    result = refresh_job_status(
        execute_live=True,
        ledger_path=ledger_path,
        shard_index=0,
        credential_loader=_identity,
        batch_factory=lambda credentials: status_batch,
        clock=lambda: "2026-08-06T02:00:00+00:00",
    )

    assert result["job_state"] == "JOB_STATE_SUCCEEDED"
    assert result["remote_mutations"] == 0
    assert len(status_batch.get_calls) == 1
    assert status_batch.create_calls == []

    class FailingStatusBatch(FakeBatch):
        def get_job(self, **kwargs: Any) -> RemoteJob:
            self.get_calls.append(kwargs)
            raise TimeoutError("read failed")

    failing = FailingStatusBatch()
    with pytest.raises(TimeoutError):
        refresh_job_status(
            execute_live=True,
            ledger_path=ledger_path,
            shard_index=0,
            credential_loader=_identity,
            batch_factory=lambda credentials: failing,
            clock=lambda: "2026-08-06T02:01:00+00:00",
        )
    assert len(failing.get_calls) == 1
    assert _read_ledger(ledger_path)["jobs"][0]["state"] == UNKNOWN_REMOTE_STATE


def test_downloads_every_object_once_to_create_only_hashed_files(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    _submit(manifest_path, ledger_path, FakeStorage(), FakeBatch())
    ledger = _read_ledger(ledger_path)
    ledger["jobs"][0]["state"] = "JOB_STATE_SUCCEEDED"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    prefix = ledger["jobs"][0]["output_prefix"]
    first_name = f"{prefix}/predictions.jsonl"
    second_name = f"{prefix}/errors.jsonl"
    storage = FakeStorage()
    storage.payloads = {
        first_name: b'{"response":"A"}\n',
        second_name: b"",
    }
    storage.list_results = [
        (
            StorageObject(
                name=first_name,
                generation="11",
                size=len(storage.payloads[first_name]),
            ),
            StorageObject(name=second_name, generation="12", size=0),
        )
    ]
    output_dir = tmp_path / "downloads"

    result = download_outputs(
        execute_live=True,
        ledger_path=ledger_path,
        shard_index=0,
        output_dir=output_dir,
        credential_loader=_identity,
        storage_factory=lambda credentials: storage,
        clock=lambda: "2026-08-06T03:00:00+00:00",
    )

    assert result["status"] == "DOWNLOAD_VERIFIED"
    assert result["object_count"] == 2
    assert len(storage.list_calls) == 1
    assert len(storage.download_calls) == 2
    ledger = _read_ledger(ledger_path)
    assert ledger["jobs"][0]["download_state"] == "VERIFIED"
    for download in ledger["jobs"][0]["downloads"]:
        local_path = Path(download["local_path"])
        assert local_path.exists()
        assert hashlib.sha256(local_path.read_bytes()).hexdigest() == download["sha256"]

    with pytest.raises(VertexLiveError) as captured:
        download_outputs(
            execute_live=True,
            ledger_path=ledger_path,
            shard_index=0,
            output_dir=output_dir,
            credential_loader=_identity,
            storage_factory=lambda credentials: FakeStorage(),
        )
    assert captured.value.phase == "download_gate"


def _make_all_jobs_downloaded(ledger_path: Path) -> dict[str, Any]:
    ledger = _read_ledger(ledger_path)
    template = ledger["jobs"][0]
    jobs: list[dict[str, Any]] = []
    for shard_index in range(SHARD_COUNT):
        job = dict(template)
        job["shard_index"] = shard_index
        job["input_object"] = (
            f"runs/{ledger['manifest_sha256']}/input/requests-{shard_index:02d}.jsonl"
        )
        job["output_prefix"] = (
            f"runs/{ledger['manifest_sha256']}/output/shard-{shard_index:02d}"
        )
        job["job_name"] = (
            f"projects/{DEFAULT_PROJECT}/locations/{VERTEX_LOCATION}/"
            f"batchPredictionJobs/{1000 + shard_index}"
        )
        job["state"] = "JOB_STATE_SUCCEEDED"
        job["download_state"] = "VERIFIED"
        job["downloads"] = [
            {
                "object_name": f"output/{shard_index}",
                "sha256": "a" * 64,
            }
        ]
        jobs.append(job)
    ledger["jobs"] = jobs
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    return ledger


def test_cleanup_gate_and_verified_object_then_bucket_deletion(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _manifest_fixture(tmp_path)
    ledger_path = tmp_path / "live-ledger.json"
    _submit(manifest_path, ledger_path, FakeStorage(), FakeBatch())
    adc_calls = 0

    def counted_adc() -> AdcIdentity:
        nonlocal adc_calls
        adc_calls += 1
        return _identity()

    with pytest.raises(VertexLiveError) as captured:
        cleanup_run(
            execute_live=True,
            ledger_path=ledger_path,
            credential_loader=counted_adc,
        )
    assert captured.value.phase == "cleanup_gate"
    assert adc_calls == 0

    _make_all_jobs_downloaded(ledger_path)
    storage = FakeStorage()

    with pytest.raises(VertexLiveError, match="shared existing bucket"):
        cleanup_run(
            execute_live=True,
            ledger_path=ledger_path,
            credential_loader=counted_adc,
            storage_factory=lambda credentials: storage,
            clock=lambda: "2026-08-06T04:00:00+00:00",
        )
    assert adc_calls == 0
    assert storage.list_calls == storage.delete_calls == []
    assert storage.delete_bucket_calls == []


class _Response:
    def __init__(
        self,
        *,
        value: Mapping[str, Any] | None = None,
        content: bytes = b"",
        status_code: int = 200,
    ):
        self._value = dict(value or {})
        self.content = content
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._value


def test_concrete_storage_upload_uses_generation_zero_precondition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Session:
        def __init__(self, credentials: Any):
            self.post_calls: list[tuple[str, dict[str, Any]]] = []
            self.get_calls: list[tuple[str, dict[str, Any]]] = []

        def post(self, url: str, **kwargs: Any) -> _Response:
            self.post_calls.append((url, kwargs))
            return _Response(value={"bucket": "bucket", "name": "input.jsonl"})

        def get(self, url: str, **kwargs: Any) -> _Response:
            self.get_calls.append((url, kwargs))
            if url.endswith("/iam/testPermissions"):
                permissions = [
                    value for key, value in kwargs["params"] if key == "permissions"
                ]
                return _Response(value={"permissions": permissions})
            return _Response(
                value={
                    "items": [
                        {
                            "name": f"{kwargs['params']['prefix']}{'0' * 32}",
                            "location": BUCKET_LOCATION,
                            "lifecycle": {
                                "rule": [
                                    {
                                        "action": {"type": "Delete"},
                                        "condition": {"age": 1},
                                    }
                                ]
                            },
                        }
                    ]
                }
            )

        def close(self) -> None:
            pass

    session = Session(object())
    monkeypatch.setattr(
        "google.auth.transport.requests.AuthorizedSession",
        lambda credentials: session,
    )
    gateway = GoogleStorageGateway(object())

    bucket = gateway.find_compatible_bucket(
        project=DEFAULT_PROJECT,
        bucket_prefix="cm-sensory-test-",
        expected_bucket=None,
        location=BUCKET_LOCATION,
        lifecycle_days=1,
        permissions=tuple(sorted(REQUIRED_GCS_PERMISSIONS)),
    )
    gateway.upload_jsonl_create_only(
        bucket="bucket",
        object_name="input.jsonl",
        data=b"{}\n",
        metadata={"object-sha256": "a" * 64},
    )

    assert bucket == f"cm-sensory-test-{'0' * 32}"
    assert len(session.get_calls) == 2
    list_url, bucket_list = session.get_calls[0]
    assert list_url.endswith("/b")
    assert bucket_list["params"]["project"] == DEFAULT_PROJECT
    assert bucket_list["params"]["userProject"] == DEFAULT_PROJECT
    permission_url, permission_test = session.get_calls[1]
    assert permission_url.endswith("/iam/testPermissions")
    assert ("userProject", DEFAULT_PROJECT) in permission_test["params"]
    assert (
        "permissions",
        "storage.buckets.list",
    ) not in permission_test["params"]
    assert {
        value
        for key, value in permission_test["params"]
        if key == "permissions"
    } == set(REQUIRED_GCS_PERMISSIONS) - {"storage.buckets.list"}
    assert "cloudresourcemanager" not in " ".join(
        url for url, _ in session.get_calls + session.post_calls
    )
    assert len(session.post_calls) == 1
    _, request = session.post_calls[0]
    assert request["params"]["uploadType"] == "multipart"
    assert request["params"]["ifGenerationMatch"] == "0"
    assert b'"object-sha256"' in request["data"]


def test_concrete_batch_sdk_disables_retries_and_calls_create_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_options: list[Any] = []
    captured_client_args: list[dict[str, Any]] = []

    class Batches:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise TimeoutError("ambiguous")

    class Client:
        def __init__(self, **kwargs: Any):
            captured_client_args.append(kwargs)
            captured_options.append(kwargs["http_options"])
            self.batches = Batches()

        def close(self) -> None:
            pass

    clients: list[Client] = []

    def factory(**kwargs: Any) -> Client:
        client = Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr("google.genai.Client", factory)
    gateway = GoogleBatchGateway(object())

    with pytest.raises(AmbiguousRemoteCreateError):
        gateway.create_job(
            input_uri="gs://bucket/input.jsonl",
            output_uri="gs://bucket/output",
            display_name="unit",
        )

    assert len(captured_options) == 1
    assert captured_client_args[0]["project"] == DEFAULT_PROJECT
    assert captured_client_args[0]["location"] == "global"
    assert captured_options[0].retry_options.attempts == 1
    assert len(clients[0].batches.calls) == 1
    assert clients[0].batches.calls[0]["model"] == "gemini-2.5-flash"
