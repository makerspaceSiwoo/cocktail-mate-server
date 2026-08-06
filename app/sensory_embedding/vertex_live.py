"""Reviewed live cloud boundary for sensory-48 Vertex Batch jobs.

The offline :mod:`app.sensory_embedding.vertex_batch` module owns request and
manifest construction.  This module only moves those immutable artifacts across
the reviewed Google Cloud boundary.  Every mutating remote operation requires
an explicit live flag, uses service-account ADC, and is made at most once by
the calling operation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

from app.sensory_embedding.vertex_batch import (
    ACTIVE_JOB_STATES,
    AXIS_REGISTRY_FILE_SHA256,
    COHORT_ID_SET_SHA256,
    COHORT_SOURCE_FILE_SHA256,
    CORPUS_ROWS,
    DEFAULT_PROJECT,
    GCS_LIFECYCLE_DAYS,
    HARD_CREATION_BLOCK_USD,
    HISTORICAL_RESERVE_USD,
    MODEL,
    PROMPT_SHA256,
    REQUEST_CONFIG_SHA256,
    REQUEST_COUNT,
    SHARD_COUNT,
    SHARD_SIZE,
    SOFT_STOP_USD,
    TERMINAL_JOB_STATES,
    GcsLifecycleContract,
    RunCostLedger,
    VertexSensoryBatchError,
    atomic_create,
    estimate_cost,
    echoed_request_prompt_sha256,
    gcs_run_metadata,
    guard_production_job_creation,
    id_set_sha256,
    manifest_requests_by_shard,
    sha256_bytes,
    utc_now,
)
from app.sensory_embedding.registry import SENSORY_V2_REGISTRY

PROJECT = DEFAULT_PROJECT
LOCATION = "global"
BUCKET_LOCATION = "ASIA-NORTHEAST3"
MODEL_ID = MODEL
LIVE_CORPUS_ROWS = CORPUS_ROWS
LIVE_REQUEST_COUNT = REQUEST_COUNT
LIVE_SHARD_SIZE = SHARD_SIZE
LIVE_ID_ALLOWLIST_SHA256 = COHORT_ID_SET_SHA256
LIVE_ID_ALLOWLIST_SOURCE_FILE_SHA256 = COHORT_SOURCE_FILE_SHA256
LIVE_FLAG = "--execute-live"
LIVE_PILOT_FLAG = "--execute-live-pilot"
API_KEY_ENV_VARS = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
UNKNOWN_REMOTE_STATE = "UNKNOWN_REMOTE_STATE"
CREATION_DISPATCHING = "CREATION_DISPATCHING"
UPLOAD_DISPATCHING = "UPLOAD_DISPATCHING"
EXISTING_BUCKET_REQUIRED = "EXISTING_BUCKET_REQUIRED"
DOWNLOAD_DISPATCHING = "DOWNLOAD_DISPATCHING"
STORAGE_API = "https://storage.googleapis.com/storage/v1"
STORAGE_UPLOAD_API = "https://storage.googleapis.com/upload/storage/v1"
DEFAULT_TIMEOUT_SECONDS = 30.0
LEDGER_SCHEMA_VERSION = 1
EVENT_SCHEMA_VERSION = 1
FULL_RUN_SCOPE = "full-production-602"
PILOT_RUN_SCOPE = "live-pilot-10x48"
LIVE_PILOT_MANIFEST_TYPE = "sensory-teacher-live-pilot-v1"
LIVE_PILOT_STATUS = "PREPARED_FOR_LIVE_PILOT_ONLY"
LIVE_PILOT_ROWS = 10
LIVE_PILOT_REQUEST_COUNT = 480
LIVE_PILOT_SHARD_SIZE = 60
LIVE_PILOT_SELECTION_POLICY = "recipe_feature_jaccard_k_center_v1"
LIVE_PILOT_SELECTED_IDS = (40, 49, 129, 157, 188, 515, 522, 539, 544, 561)
LIVE_PILOT_SELECTED_ID_SET_SHA256 = (
    "fa79501cc27a54850808efdb90d071c2da665f69a5d238c8154b3c2e3247498a"
)
LIVE_PILOT_FROZEN_SOURCE_SHA256 = (
    "4a51835460938ddebc11507d34da2835796e4f73179b15279ddd94253523560b"
)
LIVE_PILOT_APPROVAL_MARKER = "USER_APPROVED_VERTEX_LIVE_PILOT_10X48_V1"
LIVE_PILOT_APPROVAL_MARKER_SHA256 = (
    "e595acef7f3dba3f0ea87812cbab7f78e555bbafa69fad171a83a8270ff96297"
)
LIVE_PILOT_APPROVED_MANIFEST_SHA256 = (
    "06f7a1398537812bf5e31daecba9be7dfaa495ad54149003b6008034d059f396"
)
_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
REQUIRED_GCS_PERMISSIONS = frozenset(
    {
        "storage.buckets.get",
        "storage.buckets.list",
        "storage.objects.create",
        "storage.objects.get",
        "storage.objects.list",
    }
)
_JOB_NAME_PATTERN = re.compile(
    rf"^projects/{re.escape(PROJECT)}/locations/{re.escape(LOCATION)}"
    r"/batchPredictionJobs/[^/]+$"
)


class VertexLiveError(RuntimeError):
    """Safe operational error that never contains a provider response body."""

    def __init__(self, message: str, *, phase: str):
        super().__init__(message)
        self.phase = phase


class AmbiguousRemoteCreateError(VertexLiveError):
    """A Batch create call may have reached the provider."""


@dataclass(frozen=True, slots=True)
class AdcIdentity:
    credentials: Any
    detected_project_id: str | None
    credential_project_id: str | None
    quota_project_id: str | None
    is_service_account: bool


@dataclass(frozen=True, slots=True)
class RemoteJob:
    name: str
    state: str


@dataclass(frozen=True, slots=True)
class StorageObject:
    name: str
    generation: str
    size: int


@dataclass(frozen=True, slots=True)
class DedicatedBucketContract:
    name: str
    location: str


class StorageGateway(Protocol):
    def find_compatible_bucket(
        self,
        *,
        project: str,
        bucket_prefix: str,
        expected_bucket: str | None,
        location: str,
        lifecycle_days: int,
        permissions: Sequence[str],
    ) -> str: ...

    def upload_jsonl_create_only(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        metadata: Mapping[str, str],
    ) -> None: ...

    def list_objects(self, *, bucket: str, prefix: str) -> Sequence[StorageObject]: ...

    def download_object(self, *, bucket: str, object_name: str) -> bytes: ...

    def delete_object(
        self,
        *,
        bucket: str,
        object_name: str,
        generation: str,
    ) -> None: ...

    def delete_bucket(self, *, bucket: str) -> None: ...

    def close(self) -> None: ...


class BatchGateway(Protocol):
    def create_job(
        self,
        *,
        input_uri: str,
        output_uri: str,
        display_name: str,
    ) -> RemoteJob: ...

    def get_job(self, *, job_name: str) -> RemoteJob: ...

    def close(self) -> None: ...


CredentialLoader = Callable[[], AdcIdentity]
StorageFactory = Callable[[Any], StorageGateway]
BatchFactory = Callable[[Any], BatchGateway]
Clock = Callable[[], str]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_live(execute_live: bool) -> None:
    if not execute_live:
        raise VertexLiveError(
            f"network access is disabled without {LIVE_FLAG}",
            phase="authorization",
        )


@contextmanager
def isolated_adc_environment() -> Iterator[None]:
    """Hide API keys while retaining official ADC configuration."""

    saved = {
        variable: os.environ[variable]
        for variable in API_KEY_ENV_VARS
        if variable in os.environ
    }
    for variable in API_KEY_ENV_VARS:
        os.environ.pop(variable, None)
    try:
        yield
    finally:
        for variable in API_KEY_ENV_VARS:
            os.environ.pop(variable, None)
        os.environ.update(saved)


def load_service_account_adc() -> AdcIdentity:
    """Load ADC only through official Google Auth APIs.

    This function never opens, parses, logs, or returns the contents of a
    credential JSON file.
    """

    try:
        import google.auth
        from google.oauth2 import service_account
    except ImportError as error:
        raise VertexLiveError(
            "official google-auth packages are required",
            phase="adc_import",
        ) from error
    try:
        credentials, detected_project = google.auth.default(
            scopes=[_CLOUD_PLATFORM_SCOPE],
            quota_project_id=PROJECT,
        )
    except Exception as error:
        raise VertexLiveError(
            "official ADC loading failed",
            phase="adc_load",
        ) from error
    return AdcIdentity(
        credentials=credentials,
        detected_project_id=detected_project,
        credential_project_id=getattr(credentials, "project_id", None),
        quota_project_id=getattr(credentials, "quota_project_id", None),
        is_service_account=isinstance(credentials, service_account.Credentials),
    )


def _validate_adc(identity: AdcIdentity) -> None:
    if (
        not identity.is_service_account
        or identity.detected_project_id != PROJECT
        or identity.credential_project_id != PROJECT
        or identity.quota_project_id != PROJECT
    ):
        raise VertexLiveError(
            "service-account ADC must match the reviewed project and quota project",
            phase="adc_validation",
        )


def _normalize_job_state(value: object) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw:
        raw = getattr(value, "name", None)
    if not isinstance(raw, str) or not raw:
        return UNKNOWN_REMOTE_STATE
    if raw in ACTIVE_JOB_STATES | TERMINAL_JOB_STATES:
        return raw
    return UNKNOWN_REMOTE_STATE


def _validate_job_name(value: str) -> str:
    if _JOB_NAME_PATTERN.fullmatch(value) is None:
        raise VertexLiveError(
            "Vertex returned a job outside the reviewed project/location",
            phase="batch_response",
        )
    return value


class GoogleStorageGateway:
    """Small official-auth Storage JSON API adapter with no retry loop."""

    def __init__(
        self,
        credentials: Any,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        try:
            from google.auth.transport.requests import AuthorizedSession
        except ImportError as error:
            raise VertexLiveError(
                "official google-auth transport is required",
                phase="storage_import",
            ) from error
        try:
            self._session = AuthorizedSession(credentials)  # type: ignore[no-untyped-call]
        except Exception as error:
            raise VertexLiveError(
                "authorized Storage session setup failed",
                phase="storage_setup",
            ) from error
        self._timeout_seconds = timeout_seconds

    def _request(self, method: str, url: str, *, phase: str, **kwargs: Any) -> Any:
        request_method = getattr(self._session, method)
        try:
            response = request_method(
                url,
                timeout=self._timeout_seconds,
                **kwargs,
            )
        except Exception as error:
            raise VertexLiveError(
                "Google Storage request failed",
                phase=phase,
            ) from error
        status = getattr(response, "status_code", None)
        if type(status) is not int or not 200 <= status < 300:
            raise VertexLiveError(
                "Google Storage request was not successful",
                phase=phase,
            )
        return response

    @staticmethod
    def _object_json(response: Any, *, phase: str) -> dict[str, Any]:
        try:
            value = response.json()
        except Exception as error:
            raise VertexLiveError(
                "Google Storage returned invalid metadata",
                phase=phase,
            ) from error
        if not isinstance(value, dict):
            raise VertexLiveError(
                "Google Storage returned invalid metadata",
                phase=phase,
            )
        return value

    def find_compatible_bucket(
        self,
        *,
        project: str,
        bucket_prefix: str,
        expected_bucket: str | None,
        location: str,
        lifecycle_days: int,
        permissions: Sequence[str],
    ) -> str:
        requested = tuple(sorted(set(permissions)))
        # storage.buckets.list is a project-scoped permission. The successful
        # bucket-list request above is its authorization check; including it in
        # a bucket resource's iam/testPermissions call makes the Storage API
        # reject the entire request with HTTP 400.
        bucket_permissions = tuple(
            permission
            for permission in requested
            if permission != "storage.buckets.list"
        )
        response = self._request(
            "get",
            f"{STORAGE_API}/b",
            phase="gcs_permission_preflight",
            params={
                "project": project,
                "userProject": project,
                "prefix": bucket_prefix,
                "fields": "items(name,location,lifecycle)",
            },
        )
        payload = self._object_json(response, phase="gcs_permission_preflight")
        items = payload.get("items", [])
        lifecycle = {
            "rule": [
                {
                    "action": {"type": "Delete"},
                    "condition": {"age": lifecycle_days},
                }
            ]
        }
        if not isinstance(items, list):
            raise VertexLiveError(
                "GCS bucket list returned invalid metadata",
                phase="gcs_permission_preflight",
            )
        compatible = sorted(
            str(item["name"])
            for item in items
            if isinstance(item, Mapping)
            and isinstance(item.get("name"), str)
            and str(item["name"]).startswith(bucket_prefix)
            and (expected_bucket is None or item["name"] == expected_bucket)
            and str(item.get("location", "")).upper() == location.upper()
            and item.get("lifecycle") == lifecycle
        )
        if not compatible:
            raise VertexLiveError(
                "no compatible existing GCS bucket is available",
                phase="gcs_permission_preflight",
            )
        bucket = compatible[0]
        response = self._request(
            "get",
            f"{STORAGE_API}/b/{quote(bucket, safe='')}/iam/testPermissions",
            phase="gcs_permission_preflight",
            params=[
                *[
                    ("permissions", permission)
                    for permission in bucket_permissions
                ],
                ("userProject", project),
            ],
        )
        payload = self._object_json(response, phase="gcs_permission_preflight")
        granted = payload.get("permissions", [])
        if not isinstance(granted, list) or any(
            not isinstance(permission, str) for permission in granted
        ):
            raise VertexLiveError(
                "GCS permission preflight returned invalid metadata",
                phase="gcs_permission_preflight",
            )
        if set(granted) != set(bucket_permissions):
            raise VertexLiveError(
                "required existing-bucket GCS permissions are missing",
                phase="gcs_permission_preflight",
            )
        return bucket

    def upload_jsonl_create_only(
        self,
        *,
        bucket: str,
        object_name: str,
        data: bytes,
        metadata: Mapping[str, str],
    ) -> None:
        if not data.endswith(b"\n"):
            raise VertexLiveError(
                "input shard must be newline-terminated JSONL",
                phase="upload_validation",
            )
        boundary = f"sensory-{sha256_bytes(data)[:24]}"
        metadata_document = json.dumps(
            {"name": object_name, "metadata": dict(metadata)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
        ).encode("ascii")
        body += metadata_document
        body += (f"\r\n--{boundary}\r\nContent-Type: application/jsonl\r\n\r\n").encode(
            "ascii"
        )
        body += data
        body += f"\r\n--{boundary}--\r\n".encode("ascii")
        response = self._request(
            "post",
            f"{STORAGE_UPLOAD_API}/b/{quote(bucket, safe='')}/o",
            phase="object_upload",
            params={
                "uploadType": "multipart",
                "ifGenerationMatch": "0",
            },
            headers={"Content-Type": f"multipart/related; boundary={boundary}"},
            data=body,
        )
        uploaded = self._object_json(response, phase="object_upload")
        if uploaded.get("bucket") != bucket or uploaded.get("name") != object_name:
            raise VertexLiveError(
                "uploaded object identity does not match the request",
                phase="object_upload",
            )

    def list_objects(self, *, bucket: str, prefix: str) -> Sequence[StorageObject]:
        objects: list[StorageObject] = []
        page_token: str | None = None
        while True:
            params = {
                "prefix": prefix,
                "fields": "items(name,generation,size),nextPageToken",
            }
            if page_token is not None:
                params["pageToken"] = page_token
            response = self._request(
                "get",
                f"{STORAGE_API}/b/{quote(bucket, safe='')}/o",
                phase="object_list",
                params=params,
            )
            payload = self._object_json(response, phase="object_list")
            items = payload.get("items", [])
            if not isinstance(items, list):
                raise VertexLiveError(
                    "Google Storage object listing is invalid",
                    phase="object_list",
                )
            for item in items:
                if not isinstance(item, Mapping):
                    raise VertexLiveError(
                        "Google Storage object listing is invalid",
                        phase="object_list",
                    )
                name = item.get("name")
                generation = item.get("generation")
                raw_size = item.get("size")
                if isinstance(raw_size, bool) or not isinstance(raw_size, (int, str)):
                    raise VertexLiveError(
                        "Google Storage object metadata is invalid",
                        phase="object_list",
                    )
                try:
                    size = int(raw_size)
                except (TypeError, ValueError) as error:
                    raise VertexLiveError(
                        "Google Storage object metadata is invalid",
                        phase="object_list",
                    ) from error
                if (
                    not isinstance(name, str)
                    or not isinstance(generation, str)
                    or size < 0
                ):
                    raise VertexLiveError(
                        "Google Storage object metadata is invalid",
                        phase="object_list",
                    )
                objects.append(
                    StorageObject(name=name, generation=generation, size=size)
                )
            token = payload.get("nextPageToken")
            if token in (None, ""):
                break
            if not isinstance(token, str):
                raise VertexLiveError(
                    "Google Storage pagination token is invalid",
                    phase="object_list",
                )
            page_token = token
        return tuple(objects)

    def download_object(self, *, bucket: str, object_name: str) -> bytes:
        response = self._request(
            "get",
            f"{STORAGE_API}/b/{quote(bucket, safe='')}/o/{quote(object_name, safe='')}",
            phase="object_download",
            params={"alt": "media"},
        )
        content = getattr(response, "content", None)
        if not isinstance(content, bytes):
            raise VertexLiveError(
                "Google Storage download did not return bytes",
                phase="object_download",
            )
        return content

    def delete_object(
        self,
        *,
        bucket: str,
        object_name: str,
        generation: str,
    ) -> None:
        self._request(
            "delete",
            f"{STORAGE_API}/b/{quote(bucket, safe='')}/o/{quote(object_name, safe='')}",
            phase="object_delete",
            params={"ifGenerationMatch": generation},
        )

    def delete_bucket(self, *, bucket: str) -> None:
        self._request(
            "delete",
            f"{STORAGE_API}/b/{quote(bucket, safe='')}",
            phase="bucket_delete",
        )

    def close(self) -> None:
        self._session.close()  # type: ignore[no-untyped-call]


class GoogleBatchGateway:
    """google-genai Batch adapter configured for one attempt and no fallback."""

    def __init__(
        self,
        credentials: Any,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        try:
            from google import genai
            from google.genai import types
        except ImportError as error:
            raise VertexLiveError(
                "official google-genai SDK is required",
                phase="batch_import",
            ) from error
        self._types = types
        http_options = types.HttpOptions(
            api_version="v1",
            timeout=int(timeout_seconds * 1000),
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        try:
            self._client = genai.Client(
                vertexai=True,
                project=PROJECT,
                location=LOCATION,
                credentials=credentials,
                http_options=http_options,
            )
        except Exception as error:
            raise VertexLiveError(
                "Vertex Batch client setup failed",
                phase="batch_setup",
            ) from error

    def create_job(
        self,
        *,
        input_uri: str,
        output_uri: str,
        display_name: str,
    ) -> RemoteJob:
        try:
            response = self._client.batches.create(
                model=MODEL_ID,
                src=input_uri,
                config=self._types.CreateBatchJobConfig(
                    display_name=display_name,
                    dest=output_uri,
                ),
            )
        except Exception as error:
            raise AmbiguousRemoteCreateError(
                "Vertex Batch create failed; remote state is ambiguous",
                phase="batch_create",
            ) from error
        name = getattr(response, "name", None)
        if not isinstance(name, str):
            raise AmbiguousRemoteCreateError(
                "Vertex Batch create returned no usable job name",
                phase="batch_response",
            )
        return RemoteJob(
            name=_validate_job_name(name),
            state=_normalize_job_state(getattr(response, "state", None)),
        )

    def get_job(self, *, job_name: str) -> RemoteJob:
        try:
            response = self._client.batches.get(name=job_name)
        except Exception as error:
            raise VertexLiveError(
                "Vertex Batch status read failed",
                phase="batch_status",
            ) from error
        name = getattr(response, "name", None)
        if name != job_name:
            raise VertexLiveError(
                "Vertex Batch status returned the wrong job",
                phase="batch_status",
            )
        return RemoteJob(
            name=_validate_job_name(job_name),
            state=_normalize_job_state(getattr(response, "state", None)),
        )

    def close(self) -> None:
        self._client.close()


def _default_storage_factory(credentials: Any) -> StorageGateway:
    return GoogleStorageGateway(credentials)


def _default_batch_factory(credentials: Any) -> BatchGateway:
    return GoogleBatchGateway(credentials)


def _load_json_object(path: Path, *, phase: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VertexLiveError(
            f"cannot load local JSON artifact {path.name}",
            phase=phase,
        ) from error
    if not isinstance(value, dict):
        raise VertexLiveError(
            f"local JSON artifact {path.name} must be an object",
            phase=phase,
        )
    return value


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        manifest = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VertexLiveError(
            "cannot load the offline manifest",
            phase="manifest_load",
        ) from error
    if not isinstance(manifest, dict):
        raise VertexLiveError(
            "offline manifest must be a JSON object",
            phase="manifest_load",
        )
    return manifest, sha256_bytes(payload)


def load_dedicated_bucket_contract(
    path: Path,
    *,
    manifest_sha256: str,
) -> DedicatedBucketContract:
    """Load a local reviewed destination contract without exposing its name."""

    try:
        if path.stat().st_mode & 0o777 != 0o600:
            raise VertexLiveError(
                "dedicated bucket contract must have mode 0600",
                phase="bucket_contract",
            )
        raw = json.loads(path.read_text(encoding="utf-8"))
    except VertexLiveError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VertexLiveError(
            "cannot load dedicated bucket contract",
            phase="bucket_contract",
        ) from error
    if not isinstance(raw, Mapping):
        raise VertexLiveError(
            "dedicated bucket contract is invalid",
            phase="bucket_contract",
        )
    name = raw.get("bucket_name")
    location = raw.get("location")
    valid = (
        _is_compatible_bucket_name(name)
        and isinstance(location, str)
        and bool(location)
        and raw.get("bucket_name_sha256") == _sha256_text(str(name))
        and raw.get("project_id") == PROJECT
        and raw.get("manifest_sha256") == manifest_sha256
        and raw.get("lifecycle_delete_age_days") == GCS_LIFECYCLE_DAYS
        and raw.get("uniform_bucket_level_access") is True
        and raw.get("public_access_prevention") == "enforced"
        and raw.get("object_count") == 0
        and raw.get("outcome") == "created"
    )
    if not valid:
        raise VertexLiveError(
            "dedicated bucket contract does not match the approved pilot",
            phase="bucket_contract",
        )
    return DedicatedBucketContract(name=str(name), location=str(location))


def _bucket_name(manifest_sha256: str) -> str:
    return f"cm-sensory-{_sha256_text(PROJECT)[:10]}-{manifest_sha256[:32]}"


def _bucket_prefix() -> str:
    return f"cm-sensory-{_sha256_text(PROJECT)[:10]}-"


def _is_compatible_bucket_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(
            rf"{re.escape(_bucket_prefix())}[0-9a-f]{{32}}",
            value,
        )
        is not None
    )


def _run_prefix(manifest_sha256: str) -> str:
    return f"runs/{manifest_sha256}"


def _new_ledger(
    *,
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    now: str,
    run_scope: str,
    shard_record_count: int,
    dedicated_bucket_name: str | None,
) -> dict[str, Any]:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise VertexLiveError(
            "offline manifest has no run ID",
            phase="manifest_validation",
        )
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "manifest_sha256": manifest_sha256,
        "run_id_sha256": _sha256_text(run_id),
        "run_scope": run_scope,
        "shard_record_count": shard_record_count,
        "historical_cost_usd": "0.00",
        "bucket": {
            "name": dedicated_bucket_name,
            "state": EXISTING_BUCKET_REQUIRED,
            "ownership": (
                "DEDICATED_PROJECT"
                if dedicated_bucket_name is not None
                else "EXISTING_SHARED"
            ),
            "created_at": now,
            "updated_at": now,
        },
        "gcs_permission_preflight": {
            "state": "NOT_STARTED",
            "updated_at": now,
        },
        "jobs": [],
        "cleanup": {"state": "NOT_STARTED", "updated_at": now},
        "event_count": 0,
    }


def _load_ledger(path: Path, *, manifest_sha256: str | None = None) -> dict[str, Any]:
    ledger = _load_json_object(path, phase="ledger_load")
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise VertexLiveError(
            "unsupported live ledger schema",
            phase="ledger_validation",
        )
    if manifest_sha256 is not None and ledger.get("manifest_sha256") != (
        manifest_sha256
    ):
        raise VertexLiveError(
            "live ledger does not match the offline manifest",
            phase="ledger_validation",
        )
    if not isinstance(ledger.get("jobs"), list):
        raise VertexLiveError(
            "live ledger jobs are invalid",
            phase="ledger_validation",
        )
    _validate_ledger_structure(ledger)
    return ledger


def _validate_ledger_structure(ledger: Mapping[str, Any]) -> None:
    manifest_sha = ledger.get("manifest_sha256")
    bucket = ledger.get("bucket")
    preflight = ledger.get("gcs_permission_preflight")
    cleanup = ledger.get("cleanup")
    if (
        not isinstance(manifest_sha, str)
        or re.fullmatch(r"[0-9a-f]{64}", manifest_sha) is None
        or not isinstance(bucket, Mapping)
        or (
            bucket.get("state") == EXISTING_BUCKET_REQUIRED
            and bucket.get("name") is not None
            and not _is_compatible_bucket_name(bucket.get("name"))
        )
        or (
            bucket.get("state") != EXISTING_BUCKET_REQUIRED
            and not _is_compatible_bucket_name(bucket.get("name"))
        )
        or bucket.get("ownership") not in {"EXISTING_SHARED", "DEDICATED_PROJECT"}
        or bucket.get("state")
        not in {
            EXISTING_BUCKET_REQUIRED,
            "READY",
            UNKNOWN_REMOTE_STATE,
        }
        or not isinstance(preflight, Mapping)
        or preflight.get("state")
        not in {
            "NOT_STARTED",
            "DISPATCHING",
            "PASSED",
            "BLOCKED",
            UNKNOWN_REMOTE_STATE,
        }
        or not isinstance(cleanup, Mapping)
        or cleanup.get("state")
        not in {
            "NOT_STARTED",
            "DISPATCHING",
            "VERIFIED_DELETED",
            UNKNOWN_REMOTE_STATE,
        }
    ):
        raise VertexLiveError(
            "live ledger control state is invalid",
            phase="ledger_validation",
        )
    try:
        historical = Decimal(str(ledger.get("historical_cost_usd")))
    except InvalidOperation as error:
        raise VertexLiveError(
            "live ledger historical cost is invalid",
            phase="ledger_validation",
        ) from error
    if not historical.is_finite() or historical < 0:
        raise VertexLiveError(
            "live ledger historical cost is invalid",
            phase="ledger_validation",
        )
    run_scope = ledger.get("run_scope", FULL_RUN_SCOPE)
    shard_record_count = ledger.get("shard_record_count", LIVE_SHARD_SIZE)
    if (
        run_scope not in {FULL_RUN_SCOPE, PILOT_RUN_SCOPE}
        or type(shard_record_count) is not int
        or shard_record_count
        != (LIVE_SHARD_SIZE if run_scope == FULL_RUN_SCOPE else LIVE_PILOT_SHARD_SIZE)
    ):
        raise VertexLiveError(
            "live ledger run scope is invalid",
            phase="ledger_validation",
        )
    expected_cost = estimate_cost(shard_record_count).estimated_cost_usd
    indexes: set[int] = set()
    names: set[str] = set()
    allowed_states = (
        ACTIVE_JOB_STATES
        | TERMINAL_JOB_STATES
        | {
            UPLOAD_DISPATCHING,
            CREATION_DISPATCHING,
            UNKNOWN_REMOTE_STATE,
        }
    )
    for job in ledger["jobs"]:
        if not isinstance(job, Mapping):
            raise VertexLiveError(
                "live ledger job entry is invalid",
                phase="ledger_validation",
            )
        shard_index = job.get("shard_index")
        name = job.get("job_name")
        if (
            type(shard_index) is not int
            or shard_index not in range(SHARD_COUNT)
            or shard_index in indexes
            or job.get("state") not in allowed_states
            or job.get("input_object")
            != f"{_run_prefix(manifest_sha)}/input/requests-{shard_index:02d}.jsonl"
            or job.get("output_prefix")
            != f"{_run_prefix(manifest_sha)}/output/shard-{shard_index:02d}"
            or job.get("estimated_cost_usd") != str(expected_cost)
            or job.get("create_attempts") not in {0, 1}
        ):
            raise VertexLiveError(
                "live ledger job entry is invalid",
                phase="ledger_validation",
            )
        if name is not None:
            if not isinstance(name, str) or name in names:
                raise VertexLiveError(
                    "live ledger job name is invalid",
                    phase="ledger_validation",
                )
            _validate_job_name(name)
            names.add(name)
        indexes.add(shard_index)


def _atomic_replace(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _save_ledger(
    path: Path,
    ledger: dict[str, Any],
    *,
    event: str,
    now: str,
    shard_index: int | None = None,
) -> None:
    ledger["event_count"] = int(ledger.get("event_count", 0)) + 1
    _atomic_replace(path, _json_bytes(ledger))
    bucket = ledger.get("bucket")
    bucket_name = bucket.get("name") if isinstance(bucket, Mapping) else None
    event_record: dict[str, Any] = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event": event,
        "at": now,
        "manifest_sha256": ledger["manifest_sha256"],
        "event_sequence": ledger["event_count"],
    }
    if shard_index is not None:
        event_record["shard_index"] = shard_index
        job = _find_job(ledger, shard_index)
        name = job.get("job_name")
        if isinstance(name, str):
            event_record["job_name_sha256"] = _sha256_text(name)
        event_record["job_state"] = job.get("state")
    if isinstance(bucket_name, str):
        event_record["bucket_name_sha256"] = _sha256_text(bucket_name)
    event_dir = path.parent / f"{path.stem}.events"
    event_name = f"{int(ledger['event_count']):04d}-{event}-{uuid.uuid4().hex}.json"
    atomic_create(event_dir / event_name, _json_bytes(event_record))


@contextmanager
def _operation_lock(ledger_path: Path) -> Iterator[None]:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(f".{ledger_path.name}.lock")
    try:
        descriptor = os.open(
            lock_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise VertexLiveError(
            "another live ledger operation is in progress",
            phase="ledger_lock",
        ) from error
    try:
        os.close(descriptor)
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def _find_job(ledger: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    matches = [
        job
        for job in ledger["jobs"]
        if isinstance(job, dict) and job.get("shard_index") == shard_index
    ]
    if len(matches) != 1:
        raise VertexLiveError(
            "live ledger must contain exactly one entry for the shard",
            phase="ledger_validation",
        )
    return matches[0]


def _validate_live_manifest(manifest: Mapping[str, Any]) -> None:
    """Apply the reviewed offline gates to the frozen 602-ID live cohort."""

    guard_production_job_creation(manifest, RunCostLedger())
    expected = {
        "schema_version": 1,
        "model": MODEL_ID,
        "project": PROJECT,
        "location": LOCATION,
        "row_count": LIVE_CORPUS_ROWS,
        "input_row_count": LIVE_CORPUS_ROWS,
        "request_count": LIVE_REQUEST_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_size": LIVE_SHARD_SIZE,
        "id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "input_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "cohort_source_sha256": LIVE_ID_ALLOWLIST_SOURCE_FILE_SHA256,
        "cohort_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "cohort_row_count": LIVE_CORPUS_ROWS,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
    }
    for field, required in expected.items():
        if manifest.get(field) != required:
            raise VertexSensoryBatchError(
                f"live manifest {field} does not match the reviewed 602 cohort"
            )
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != SHARD_COUNT
        or any(
            not isinstance(shard, Mapping)
            or shard.get("shard_index") != index
            or shard.get("filename") != f"requests-{index:02d}.jsonl"
            or shard.get("record_count") != LIVE_SHARD_SIZE
            for index, shard in enumerate(shards)
        )
    ):
        raise VertexSensoryBatchError(
            "live manifest requires eight ordered 3,612-record shards"
        )


def _validate_live_pilot_manifest(
    manifest: Mapping[str, Any],
    *,
    user_approval_marker: str,
) -> None:
    """Validate the isolated 10×48 pilot without authorizing production."""

    expected = {
        "schema_version": 1,
        "manifest_type": LIVE_PILOT_MANIFEST_TYPE,
        "status": LIVE_PILOT_STATUS,
        "run_scope": PILOT_RUN_SCOPE,
        "full_production_authorized": False,
        "model": MODEL_ID,
        "project": PROJECT,
        "location": LOCATION,
        "row_count": LIVE_PILOT_ROWS,
        "request_count": LIVE_PILOT_REQUEST_COUNT,
        "shard_count": SHARD_COUNT,
        "shard_size": LIVE_PILOT_SHARD_SIZE,
        "parent_cohort_row_count": LIVE_CORPUS_ROWS,
        "parent_cohort_id_set_sha256": LIVE_ID_ALLOWLIST_SHA256,
        "parent_cohort_source_sha256": LIVE_ID_ALLOWLIST_SOURCE_FILE_SHA256,
        "parent_frozen_source_sha256": LIVE_PILOT_FROZEN_SOURCE_SHA256,
        "selected_id_set_sha256": LIVE_PILOT_SELECTED_ID_SET_SHA256,
        "selection_policy": LIVE_PILOT_SELECTION_POLICY,
        "registry_sha256": SENSORY_V2_REGISTRY.registry_sha256,
        "prompt_axis_registry_file_sha256": AXIS_REGISTRY_FILE_SHA256,
        "prompt_sha256": PROMPT_SHA256,
        "request_config_sha256": REQUEST_CONFIG_SHA256,
        "historical_reserve_usd": str(HISTORICAL_RESERVE_USD),
        "soft_stop_usd": str(SOFT_STOP_USD),
        "hard_creation_block_usd": str(HARD_CREATION_BLOCK_USD),
        "user_approval_marker_sha256": LIVE_PILOT_APPROVAL_MARKER_SHA256,
        "gcs_lifecycle": GcsLifecycleContract().to_dict(),
    }
    for field, required in expected.items():
        if manifest.get(field) != required:
            raise VertexSensoryBatchError(
                f"live pilot manifest {field} does not match the reviewed contract"
            )
    if user_approval_marker != LIVE_PILOT_APPROVAL_MARKER or _sha256_text(
        user_approval_marker
    ) != manifest.get("user_approval_marker_sha256"):
        raise VertexSensoryBatchError(
            "live pilot requires the exact explicit user approval marker"
        )
    selected_ids = manifest.get("selected_cocktail_ids")
    if (
        not isinstance(selected_ids, list)
        or tuple(selected_ids) != LIVE_PILOT_SELECTED_IDS
        or id_set_sha256(selected_ids) != LIVE_PILOT_SELECTED_ID_SET_SHA256
    ):
        raise VertexSensoryBatchError(
            "live pilot selected IDs do not match the deterministic cohort sample"
        )
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != LIVE_PILOT_REQUEST_COUNT:
        raise VertexSensoryBatchError(
            "live pilot manifest must contain exactly 480 request identities"
        )
    seen: set[tuple[int, int]] = set()
    prompt_hashes: set[str] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            raise VertexSensoryBatchError("live pilot request identity is invalid")
        row_index = request.get("row_index")
        cocktail_id = request.get("cocktail_id")
        axis_order = request.get("axis_order")
        shard_index = request.get("shard_index")
        prompt_hash = request.get("prompt_sha256")
        if (
            type(row_index) is not int
            or row_index not in range(LIVE_PILOT_ROWS)
            or cocktail_id != LIVE_PILOT_SELECTED_IDS[row_index]
            or type(axis_order) is not int
            or axis_order not in range(48)
            or request.get("axis_id") != SENSORY_V2_REGISTRY.axes[axis_order].axis_id
            or request.get("key") != f"r{row_index:04d}-a{axis_order:02d}"
            or shard_index != axis_order % SHARD_COUNT
            or not isinstance(prompt_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", prompt_hash) is None
            or (row_index, axis_order) in seen
            or prompt_hash in prompt_hashes
        ):
            raise VertexSensoryBatchError(
                "live pilot request identities are incomplete or inconsistent"
            )
        seen.add((row_index, axis_order))
        prompt_hashes.add(prompt_hash)
    if len(seen) != LIVE_PILOT_REQUEST_COUNT:
        raise VertexSensoryBatchError(
            "live pilot request identities do not cover 10×48"
        )
    shards = manifest.get("shards")
    if (
        not isinstance(shards, list)
        or len(shards) != SHARD_COUNT
        or any(
            not isinstance(shard, Mapping)
            or shard.get("shard_index") != index
            or shard.get("filename") != f"requests-{index:02d}.jsonl"
            or shard.get("record_count") != LIVE_PILOT_SHARD_SIZE
            or shard.get("axis_orders") != list(range(index, 48, SHARD_COUNT))
            for index, shard in enumerate(shards)
        )
    ):
        raise VertexSensoryBatchError(
            "live pilot requires eight ordered 60-record shards"
        )
    estimate = estimate_cost(LIVE_PILOT_REQUEST_COUNT)
    projected = HISTORICAL_RESERVE_USD + estimate.estimated_cost_usd
    if (
        manifest.get("estimated_cost_usd") != str(estimate.estimated_cost_usd)
        or projected >= SOFT_STOP_USD
        or projected >= HARD_CREATION_BLOCK_USD
    ):
        raise VertexSensoryBatchError("live pilot budget contract is invalid")


def _validate_manifest_prompt_uniqueness(manifest: Mapping[str, Any]) -> None:
    request_shards = manifest_requests_by_shard(manifest)
    prompt_hashes = [
        request.get("prompt_sha256") for shard in request_shards for request in shard
    ]
    if any(
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in prompt_hashes
    ) or len(prompt_hashes) != len(set(prompt_hashes)):
        raise VertexLiveError(
            "manifest request prompt identities must be unique SHA-256 values",
            phase="manifest_validation",
        )


def _assert_no_active_job(ledger: Mapping[str, Any]) -> None:
    nonterminal = [
        job
        for job in ledger["jobs"]
        if isinstance(job, Mapping) and job.get("state") not in TERMINAL_JOB_STATES
    ]
    if nonterminal:
        raise VertexLiveError(
            "new job blocked while a prior job is active or ambiguous",
            phase="one_active_job_gate",
        )


def _guard_cumulative_budget(
    ledger: Mapping[str, Any],
    *,
    new_job_cost: Decimal,
    allow_soft_stop_override: bool,
) -> Decimal:
    try:
        costs = [
            Decimal(str(job["estimated_cost_usd"]))
            for job in ledger["jobs"]
            if isinstance(job, Mapping)
        ]
        historical = Decimal(str(ledger.get("historical_cost_usd", "0.00")))
    except (InvalidOperation, KeyError) as error:
        raise VertexLiveError(
            "live ledger contains an invalid cost",
            phase="budget_gate",
        ) from error
    if (
        not historical.is_finite()
        or historical < 0
        or any(not cost.is_finite() or cost < 0 for cost in costs)
    ):
        raise VertexLiveError(
            "live ledger contains an invalid cost",
            phase="budget_gate",
        )
    committed = sum(costs, Decimal("0"))
    projected = historical + HISTORICAL_RESERVE_USD + committed + new_job_cost
    if projected >= HARD_CREATION_BLOCK_USD:
        raise VertexLiveError(
            "hard cumulative budget block reached",
            phase="budget_gate",
        )
    if projected >= SOFT_STOP_USD and not allow_soft_stop_override:
        raise VertexLiveError(
            "cumulative soft stop reached; explicit override required",
            phase="budget_gate",
        )
    return projected


def _validate_shard(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    shard_index: int,
) -> tuple[bytes, Mapping[str, Any]]:
    if shard_index not in range(SHARD_COUNT):
        raise VertexLiveError(
            "shard index must be between 0 and 7",
            phase="shard_validation",
        )
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != SHARD_COUNT:
        raise VertexLiveError(
            "offline manifest shard list is invalid",
            phase="shard_validation",
        )
    raw = raw_shards[shard_index]
    if not isinstance(raw, Mapping) or raw.get("shard_index") != shard_index:
        raise VertexLiveError(
            "offline manifest shard identity is invalid",
            phase="shard_validation",
        )
    filename = raw.get("filename")
    if filename != f"requests-{shard_index:02d}.jsonl":
        raise VertexLiveError(
            "offline manifest shard filename is invalid",
            phase="shard_validation",
        )
    path = manifest_path.parent / filename
    try:
        data = path.read_bytes()
    except OSError as error:
        raise VertexLiveError(
            "cannot read the immutable input shard",
            phase="shard_validation",
        ) from error
    if sha256_bytes(data) != raw.get("sha256"):
        raise VertexLiveError(
            "input shard SHA-256 does not match the manifest",
            phase="shard_validation",
        )
    record_count = raw.get("record_count")
    if (
        type(record_count) is not int
        or data.count(b"\n") != record_count
        or not data.endswith(b"\n")
    ):
        raise VertexLiveError(
            "input shard record count does not match the manifest",
            phase="shard_validation",
        )
    try:
        request_shards = manifest_requests_by_shard(manifest)
    except VertexSensoryBatchError as error:
        raise VertexLiveError(
            "manifest request identities are invalid",
            phase="shard_validation",
        ) from error
    expected_requests = request_shards[shard_index]
    lines = data.splitlines()
    if len(expected_requests) != record_count:
        raise VertexLiveError(
            "manifest request identities do not match shard count",
            phase="shard_validation",
        )
    for line, request in zip(lines, expected_requests, strict=True):
        try:
            prompt_sha256 = echoed_request_prompt_sha256(line)
        except VertexSensoryBatchError as error:
            raise VertexLiveError(
                "input shard request line violates the canonical request contract",
                phase="shard_validation",
            ) from error
        if prompt_sha256 != request.get("prompt_sha256"):
            raise VertexLiveError(
                "input shard prompt does not match its manifest request identity",
                phase="shard_validation",
            )
    return data, raw


def _close(resource: object | None) -> None:
    close = getattr(resource, "close", None)
    if callable(close):
        close()


def submit_shard_once(
    *,
    execute_live: bool,
    manifest_path: Path,
    ledger_path: Path,
    shard_index: int,
    allow_soft_stop_override: bool = False,
    credential_loader: CredentialLoader = load_service_account_adc,
    storage_factory: StorageFactory = _default_storage_factory,
    batch_factory: BatchFactory = _default_batch_factory,
    clock: Clock = utc_now,
    _run_scope: str = FULL_RUN_SCOPE,
    _user_approval_marker: str = "",
    _dedicated_bucket: DedicatedBucketContract | None = None,
) -> dict[str, Any]:
    """Upload one immutable shard and create its Batch job exactly once."""

    _require_live(execute_live)
    with _operation_lock(ledger_path):
        manifest, manifest_sha = _load_manifest(manifest_path)
        try:
            if _run_scope == FULL_RUN_SCOPE:
                _validate_live_manifest(manifest)
            elif _run_scope == PILOT_RUN_SCOPE:
                if manifest_sha != LIVE_PILOT_APPROVED_MANIFEST_SHA256:
                    raise VertexSensoryBatchError(
                        "live pilot manifest SHA-256 is not the approved prep-v2 digest"
                    )
                _validate_live_pilot_manifest(
                    manifest,
                    user_approval_marker=_user_approval_marker,
                )
            else:
                raise VertexSensoryBatchError("unknown live run scope")
            _validate_manifest_prompt_uniqueness(manifest)
        except VertexSensoryBatchError as error:
            raise VertexLiveError(
                "offline manifest scope, approval, or lifecycle gate failed",
                phase="offline_gate",
            ) from error
        shard_data, shard = _validate_shard(
            manifest_path,
            manifest,
            shard_index,
        )
        now = clock()
        if ledger_path.exists():
            ledger = _load_ledger(
                ledger_path,
                manifest_sha256=manifest_sha,
            )
            if any(
                isinstance(job, Mapping) and job.get("shard_index") == shard_index
                for job in ledger["jobs"]
            ):
                raise VertexLiveError(
                    "this shard already has a live attempt; resubmission is forbidden",
                    phase="no_resubmit_gate",
                )
        else:
            ledger = _new_ledger(
                manifest=manifest,
                manifest_sha256=manifest_sha,
                now=now,
                run_scope=_run_scope,
                shard_record_count=(
                    LIVE_SHARD_SIZE
                    if _run_scope == FULL_RUN_SCOPE
                    else LIVE_PILOT_SHARD_SIZE
                ),
                dedicated_bucket_name=(
                    _dedicated_bucket.name if _dedicated_bucket is not None else None
                ),
            )

        if ledger.get("run_scope", FULL_RUN_SCOPE) != _run_scope:
            raise VertexLiveError(
                "live ledger scope does not match the create path",
                phase="ledger_validation",
            )
        _assert_no_active_job(ledger)
        job_estimate = estimate_cost(int(shard["record_count"]))
        projected = _guard_cumulative_budget(
            ledger,
            new_job_cost=job_estimate.estimated_cost_usd,
            allow_soft_stop_override=allow_soft_stop_override,
        )
        preflight = ledger.get("gcs_permission_preflight")
        if not isinstance(preflight, dict):
            raise VertexLiveError(
                "live ledger GCS permission preflight is invalid",
                phase="ledger_validation",
            )
        if preflight.get("state") not in {"NOT_STARTED", "PASSED"}:
            raise VertexLiveError(
                "prior GCS permission preflight is unresolved; it will not be repeated",
                phase="gcs_permission_gate",
            )

        identity: AdcIdentity
        storage: StorageGateway | None = None
        batch: BatchGateway | None = None
        with isolated_adc_environment():
            identity = credential_loader()
            _validate_adc(identity)
            storage = storage_factory(identity.credentials)
            try:
                if preflight.get("state") == "NOT_STARTED":
                    preflight["state"] = "DISPATCHING"
                    preflight["updated_at"] = clock()
                    _save_ledger(
                        ledger_path,
                        ledger,
                        event="gcs-permission-preflight-dispatching",
                        now=preflight["updated_at"],
                    )
                    try:
                        selected_bucket_name = storage.find_compatible_bucket(
                            project=PROJECT,
                            bucket_prefix=_bucket_prefix(),
                            expected_bucket=(
                                _dedicated_bucket.name
                                if _dedicated_bucket is not None
                                else None
                            ),
                            location=(
                                _dedicated_bucket.location
                                if _dedicated_bucket is not None
                                else BUCKET_LOCATION
                            ),
                            lifecycle_days=GCS_LIFECYCLE_DAYS,
                            permissions=tuple(sorted(REQUIRED_GCS_PERMISSIONS)),
                        )
                        if not _is_compatible_bucket_name(selected_bucket_name):
                            raise VertexLiveError(
                                "GCS returned an incompatible existing bucket",
                                phase="gcs_permission_preflight",
                            )
                    except Exception as error:
                        preflight["state"] = (
                            "BLOCKED"
                            if isinstance(error, VertexLiveError)
                            and error.phase == "gcs_permission_preflight"
                            else UNKNOWN_REMOTE_STATE
                        )
                        preflight["updated_at"] = clock()
                        _save_ledger(
                            ledger_path,
                            ledger,
                            event="gcs-permission-preflight-blocked",
                            now=preflight["updated_at"],
                        )
                        raise
                    bucket = ledger.get("bucket")
                    if (
                        not isinstance(bucket, dict)
                        or bucket.get("state") != EXISTING_BUCKET_REQUIRED
                    ):
                        raise VertexLiveError(
                            "live ledger bucket selection state is invalid",
                            phase="ledger_validation",
                        )
                    bucket["name"] = selected_bucket_name
                    bucket["state"] = "READY"
                    bucket["updated_at"] = clock()
                    preflight["state"] = "PASSED"
                    preflight["updated_at"] = clock()
                    _save_ledger(
                        ledger_path,
                        ledger,
                        event="gcs-permission-preflight-passed",
                        now=preflight["updated_at"],
                    )
                bucket = ledger["bucket"]
                if not isinstance(bucket, dict):
                    raise VertexLiveError(
                        "live ledger bucket entry is invalid",
                        phase="ledger_validation",
                    )
                bucket_name = bucket.get("name")
                if not isinstance(bucket_name, str) or not _is_compatible_bucket_name(
                    bucket_name
                ):
                    raise VertexLiveError(
                        "live ledger bucket identity is invalid",
                        phase="ledger_validation",
                    )
                if bucket.get("state") != "READY":
                    raise VertexLiveError(
                        "run bucket remote state is not known ready",
                        phase="bucket_gate",
                    )

                prefix = _run_prefix(manifest_sha)
                object_name = f"{prefix}/input/requests-{shard_index:02d}.jsonl"
                output_prefix = f"{prefix}/output/shard-{shard_index:02d}"
                input_uri = f"gs://{bucket_name}/{object_name}"
                output_uri = f"gs://{bucket_name}/{output_prefix}"
                job_entry: dict[str, Any] = {
                    "shard_index": shard_index,
                    "state": UPLOAD_DISPATCHING,
                    "job_name": None,
                    "input_object": object_name,
                    "input_sha256": sha256_bytes(shard_data),
                    "output_prefix": output_prefix,
                    "estimated_cost_usd": str(job_estimate.estimated_cost_usd),
                    "projected_cumulative_usd": str(projected),
                    "create_attempts": 0,
                    "download_state": "NOT_STARTED",
                    "downloads": [],
                    "created_at": clock(),
                    "updated_at": clock(),
                }
                ledger["jobs"].append(job_entry)
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="upload-dispatching",
                    now=job_entry["updated_at"],
                    shard_index=shard_index,
                )
                try:
                    storage.upload_jsonl_create_only(
                        bucket=bucket_name,
                        object_name=object_name,
                        data=shard_data,
                        metadata=gcs_run_metadata(
                            run_id=str(manifest["run_id"]),
                            manifest_sha256=manifest_sha,
                            shard_index=shard_index,
                            object_sha256=sha256_bytes(shard_data),
                        ),
                    )
                except Exception:
                    job_entry["state"] = UNKNOWN_REMOTE_STATE
                    job_entry["updated_at"] = clock()
                    _save_ledger(
                        ledger_path,
                        ledger,
                        event="upload-unknown",
                        now=job_entry["updated_at"],
                        shard_index=shard_index,
                    )
                    raise

                job_entry["state"] = CREATION_DISPATCHING
                job_entry["create_attempts"] = 1
                job_entry["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="batch-create-dispatching",
                    now=job_entry["updated_at"],
                    shard_index=shard_index,
                )
                batch = batch_factory(identity.credentials)
                try:
                    remote = batch.create_job(
                        input_uri=input_uri,
                        output_uri=output_uri,
                        display_name=(
                            f"sensory48-{manifest_sha[:12]}-shard-{shard_index:02d}"
                        ),
                    )
                    job_name = _validate_job_name(remote.name)
                except Exception as error:
                    job_entry["state"] = UNKNOWN_REMOTE_STATE
                    job_entry["updated_at"] = clock()
                    _save_ledger(
                        ledger_path,
                        ledger,
                        event="batch-create-unknown",
                        now=job_entry["updated_at"],
                        shard_index=shard_index,
                    )
                    if isinstance(error, VertexLiveError):
                        raise
                    raise AmbiguousRemoteCreateError(
                        "Vertex Batch create failed; remote state is ambiguous",
                        phase="batch_create",
                    ) from error
                job_entry["job_name"] = job_name
                job_entry["state"] = _normalize_job_state(remote.state)
                job_entry["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="batch-created",
                    now=job_entry["updated_at"],
                    shard_index=shard_index,
                )
            finally:
                _close(batch)
                _close(storage)

        return {
            "status": "BATCH_CREATED",
            "shard_index": shard_index,
            "job_state": job_entry["state"],
            "manifest_sha256": manifest_sha,
            "job_name_sha256": _sha256_text(str(job_entry["job_name"])),
            "create_attempts": 1,
            "retry_attempted": False,
            "fallback_attempted": False,
            "run_scope": _run_scope,
        }


def submit_pilot_shard_once(
    *,
    execute_live_pilot: bool,
    user_approval_marker: str,
    manifest_path: Path,
    ledger_path: Path,
    shard_index: int,
    dedicated_bucket: DedicatedBucketContract | None = None,
    credential_loader: CredentialLoader = load_service_account_adc,
    storage_factory: StorageFactory = _default_storage_factory,
    batch_factory: BatchFactory = _default_batch_factory,
    clock: Clock = utc_now,
) -> dict[str, Any]:
    """Use only the explicit 10×48 live-pilot manifest and approval scope."""

    if not execute_live_pilot:
        raise VertexLiveError(
            f"pilot network access is disabled without {LIVE_PILOT_FLAG}",
            phase="authorization",
        )
    if dedicated_bucket is None:
        raise VertexLiveError(
            "pilot requires an explicit dedicated bucket contract",
            phase="bucket_contract",
        )
    return submit_shard_once(
        execute_live=True,
        manifest_path=manifest_path,
        ledger_path=ledger_path,
        shard_index=shard_index,
        allow_soft_stop_override=False,
        credential_loader=credential_loader,
        storage_factory=storage_factory,
        batch_factory=batch_factory,
        clock=clock,
        _run_scope=PILOT_RUN_SCOPE,
        _user_approval_marker=user_approval_marker,
        _dedicated_bucket=dedicated_bucket,
    )


def refresh_job_status(
    *,
    execute_live: bool,
    ledger_path: Path,
    shard_index: int,
    credential_loader: CredentialLoader = load_service_account_adc,
    batch_factory: BatchFactory = _default_batch_factory,
    clock: Clock = utc_now,
) -> dict[str, Any]:
    """Read one job state once.  This never mutates remote job state."""

    _require_live(execute_live)
    with _operation_lock(ledger_path):
        ledger = _load_ledger(ledger_path)
        job = _find_job(ledger, shard_index)
        job_name = job.get("job_name")
        if not isinstance(job_name, str):
            raise VertexLiveError(
                "ambiguous create has no job name to poll",
                phase="status_gate",
            )
        batch: BatchGateway | None = None
        with isolated_adc_environment():
            identity = credential_loader()
            _validate_adc(identity)
            batch = batch_factory(identity.credentials)
            try:
                try:
                    remote = batch.get_job(job_name=job_name)
                    if remote.name != job_name:
                        raise VertexLiveError(
                            "status response job identity mismatch",
                            phase="batch_status",
                        )
                    job["state"] = _normalize_job_state(remote.state)
                except Exception:
                    job["state"] = UNKNOWN_REMOTE_STATE
                    job["updated_at"] = clock()
                    _save_ledger(
                        ledger_path,
                        ledger,
                        event="status-unknown",
                        now=job["updated_at"],
                        shard_index=shard_index,
                    )
                    raise
                job["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="status-read",
                    now=job["updated_at"],
                    shard_index=shard_index,
                )
            finally:
                _close(batch)
        return {
            "status": "STATUS_READ",
            "shard_index": shard_index,
            "job_state": job["state"],
            "job_name_sha256": _sha256_text(job_name),
            "remote_mutations": 0,
        }


def _download_path(
    output_dir: Path,
    *,
    shard_index: int,
    object_name: str,
) -> Path:
    basename = PurePosixPath(object_name).name
    safe_basename = re.sub(r"[^A-Za-z0-9._-]", "_", basename) or "output.bin"
    return (
        output_dir
        / f"shard-{shard_index:02d}"
        / f"{_sha256_text(object_name)[:16]}-{safe_basename}"
    )


def download_outputs(
    *,
    execute_live: bool,
    ledger_path: Path,
    shard_index: int,
    output_dir: Path,
    credential_loader: CredentialLoader = load_service_account_adc,
    storage_factory: StorageFactory = _default_storage_factory,
    clock: Clock = utc_now,
) -> dict[str, Any]:
    """Download every output object into create-only hash-verified files."""

    _require_live(execute_live)
    with _operation_lock(ledger_path):
        ledger = _load_ledger(ledger_path)
        job = _find_job(ledger, shard_index)
        if job.get("state") != "JOB_STATE_SUCCEEDED":
            raise VertexLiveError(
                "downloads require a recorded succeeded job state",
                phase="download_gate",
            )
        if job.get("download_state") != "NOT_STARTED":
            raise VertexLiveError(
                "download already started; replacing local files is forbidden",
                phase="download_gate",
            )
        bucket = ledger.get("bucket")
        if not isinstance(bucket, Mapping) or bucket.get("state") != "READY":
            raise VertexLiveError(
                "run bucket is not known ready",
                phase="download_gate",
            )
        bucket_name = bucket.get("name")
        output_prefix = job.get("output_prefix")
        if not isinstance(bucket_name, str) or not isinstance(output_prefix, str):
            raise VertexLiveError(
                "live ledger output identity is invalid",
                phase="download_gate",
            )
        storage: StorageGateway | None = None
        with isolated_adc_environment():
            identity = credential_loader()
            _validate_adc(identity)
            storage = storage_factory(identity.credentials)
            try:
                objects = tuple(
                    sorted(
                        storage.list_objects(
                            bucket=bucket_name,
                            prefix=f"{output_prefix}/",
                        ),
                        key=lambda item: item.name,
                    )
                )
                if not objects or len({item.name for item in objects}) != len(objects):
                    raise VertexLiveError(
                        "output object listing is empty or contains duplicates",
                        phase="download_validation",
                    )
                if any(
                    not item.name.startswith(f"{output_prefix}/") for item in objects
                ):
                    raise VertexLiveError(
                        "output listing escaped the job prefix",
                        phase="download_validation",
                    )
                paths = [
                    _download_path(
                        output_dir,
                        shard_index=shard_index,
                        object_name=item.name,
                    )
                    for item in objects
                ]
                if any(path.exists() for path in paths):
                    raise VertexLiveError(
                        "a local output file already exists",
                        phase="download_gate",
                    )
                job["download_state"] = DOWNLOAD_DISPATCHING
                job["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="download-dispatching",
                    now=job["updated_at"],
                    shard_index=shard_index,
                )
                downloads: list[dict[str, Any]] = []
                for item, path in zip(objects, paths, strict=True):
                    data = storage.download_object(
                        bucket=bucket_name,
                        object_name=item.name,
                    )
                    if len(data) != item.size:
                        raise VertexLiveError(
                            "downloaded object size does not match metadata",
                            phase="download_validation",
                        )
                    digest = sha256_bytes(data)
                    atomic_create(path, data)
                    if sha256_bytes(path.read_bytes()) != digest:
                        raise VertexLiveError(
                            "local output hash verification failed",
                            phase="download_validation",
                        )
                    downloads.append(
                        {
                            "object_name": item.name,
                            "object_name_sha256": _sha256_text(item.name),
                            "generation": item.generation,
                            "size": item.size,
                            "sha256": digest,
                            "local_path": str(path.resolve()),
                        }
                    )
                job["downloads"] = downloads
                job["download_state"] = "VERIFIED"
                job["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="download-verified",
                    now=job["updated_at"],
                    shard_index=shard_index,
                )
            finally:
                _close(storage)
        return {
            "status": "DOWNLOAD_VERIFIED",
            "shard_index": shard_index,
            "object_count": len(job["downloads"]),
            "sha256": [entry["sha256"] for entry in job["downloads"]],
            "remote_mutations": 0,
        }


def cleanup_run(
    *,
    execute_live: bool,
    ledger_path: Path,
    credential_loader: CredentialLoader = load_service_account_adc,
    storage_factory: StorageFactory = _default_storage_factory,
    clock: Clock = utc_now,
) -> dict[str, Any]:
    """Delete the dedicated run bucket only after all outputs are verified."""

    _require_live(execute_live)
    with _operation_lock(ledger_path):
        ledger = _load_ledger(ledger_path)
        jobs = ledger["jobs"]
        indexes = {job.get("shard_index") for job in jobs if isinstance(job, Mapping)}
        if (
            len(jobs) != SHARD_COUNT
            or indexes != set(range(SHARD_COUNT))
            or any(
                not isinstance(job, Mapping)
                or job.get("state") != "JOB_STATE_SUCCEEDED"
                or job.get("download_state") != "VERIFIED"
                or not job.get("downloads")
                for job in jobs
            )
        ):
            raise VertexLiveError(
                "cleanup requires verified downloads for all eight succeeded jobs",
                phase="cleanup_gate",
            )
        bucket = ledger.get("bucket")
        if (
            not isinstance(bucket, dict)
            or bucket.get("state") != "READY"
            or bucket.get("ownership") != "DEDICATED_RUN"
        ):
            raise VertexLiveError(
                "cleanup of a shared existing bucket is disabled",
                phase="cleanup_gate",
            )
        cleanup = ledger.get("cleanup")
        if not isinstance(cleanup, dict) or cleanup.get("state") != "NOT_STARTED":
            raise VertexLiveError(
                "cleanup has already started",
                phase="cleanup_gate",
            )
        bucket_name = bucket.get("name")
        if not isinstance(bucket_name, str):
            raise VertexLiveError(
                "run bucket identity is invalid",
                phase="cleanup_gate",
            )
        cleanup["state"] = "DISPATCHING"
        cleanup["updated_at"] = clock()
        _save_ledger(
            ledger_path,
            ledger,
            event="cleanup-dispatching",
            now=cleanup["updated_at"],
        )
        storage: StorageGateway | None = None
        with isolated_adc_environment():
            identity = credential_loader()
            _validate_adc(identity)
            storage = storage_factory(identity.credentials)
            try:
                objects = tuple(storage.list_objects(bucket=bucket_name, prefix=""))
                for item in objects:
                    storage.delete_object(
                        bucket=bucket_name,
                        object_name=item.name,
                        generation=item.generation,
                    )
                remaining = storage.list_objects(bucket=bucket_name, prefix="")
                if remaining:
                    raise VertexLiveError(
                        "run bucket is not empty after object cleanup",
                        phase="cleanup_verification",
                    )
                storage.delete_bucket(bucket=bucket_name)
            except Exception:
                cleanup["state"] = UNKNOWN_REMOTE_STATE
                cleanup["updated_at"] = clock()
                _save_ledger(
                    ledger_path,
                    ledger,
                    event="cleanup-unknown",
                    now=cleanup["updated_at"],
                )
                raise
            finally:
                _close(storage)
        cleanup["state"] = "VERIFIED_DELETED"
        cleanup["deleted_object_count"] = len(objects)
        cleanup["updated_at"] = clock()
        bucket["state"] = "DELETED"
        bucket["updated_at"] = cleanup["updated_at"]
        _save_ledger(
            ledger_path,
            ledger,
            event="cleanup-verified",
            now=cleanup["updated_at"],
        )
        return {
            "status": "RUN_BUCKET_DELETED",
            "deleted_object_count": len(objects),
            "bucket_name_sha256": _sha256_text(bucket_name),
        }
