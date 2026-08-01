"""Prepare, submit, monitor, and download paid Gemini image batch jobs."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from app.image_generation.core import atomic_write
from app.image_generation.gemini_batch import (
    GEMINI_BATCH_ASPECT_RATIO,
    GEMINI_BATCH_IMAGE_MODEL,
    GEMINI_BATCH_IMAGE_PRICE_USD,
    GEMINI_BATCH_IMAGE_SIZE,
    GeminiBatchError,
    build_batch_jsonl,
    build_batch_manifest,
    download_batch_results,
    estimated_image_output_cost,
    load_prompt_rows,
    manifest_request_map,
    select_batch_rows,
    sha256_bytes,
)

DEFAULT_CSV_PATH = Path("image-upload/cocktail-image-prompts.csv")
DEFAULT_IMAGE_DIR = Path("image-upload/images")
DEFAULT_BATCH_DIR = Path("image-upload/batch")
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def _add_batch_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cocktail images with the paid Gemini Batch API."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Create a no-charge JSONL request file from the prompt CSV.",
    )
    prepare.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    prepare.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    prepare.add_argument("--min-id", type=int, default=35)
    prepare.add_argument("--cocktail-id", type=int, action="append", default=[])
    prepare.add_argument(
        "--include-existing",
        action="store_true",
        help="Include IDs that already have a valid local image.",
    )
    prepare.add_argument(
        "--allow-reprepare",
        action="store_true",
        help="Replace request/manifest files even when a saved job exists.",
    )
    _add_batch_dir(prepare)

    submit = commands.add_parser(
        "submit",
        help="Upload the prepared JSONL and create a paid batch job.",
    )
    _add_batch_dir(submit)
    submit.add_argument("--max-requests", type=int, required=True)
    submit.add_argument("--max-image-output-cost-usd", required=True)
    submit.add_argument("--display-name")
    submit.add_argument(
        "--confirm-paid-batch",
        action="store_true",
        help="Required acknowledgement that this command creates paid usage.",
    )
    submit.add_argument(
        "--allow-new-job",
        action="store_true",
        help="Allow replacing a saved terminal job with a new submission.",
    )

    status = commands.add_parser("status", help="Fetch and save the batch job state.")
    _add_batch_dir(status)
    status.add_argument("--job-name")

    download = commands.add_parser(
        "download",
        help="Download a successful job and save cocktail-<id>.png files.",
    )
    _add_batch_dir(download)
    download.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    download.add_argument("--job-name")
    download.add_argument("--overwrite", action="store_true")

    wait_download = commands.add_parser(
        "wait-download",
        help="Poll until completion and then download generated images.",
    )
    _add_batch_dir(wait_download)
    wait_download.add_argument("--images", type=Path, default=DEFAULT_IMAGE_DIR)
    wait_download.add_argument("--job-name")
    wait_download.add_argument("--poll-seconds", type=int, default=60)
    wait_download.add_argument("--overwrite", action="store_true")
    return parser


def _load_environment() -> Path:
    env_file = Path(".env.local") if Path(".env.local").is_file() else Path(".env")
    load_dotenv(env_file, override=True)
    return env_file


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n").encode(
        "utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeminiBatchError(f"Cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise GeminiBatchError(f"Expected a JSON object in {path}")
    return value


def _job_state(job: object) -> str:
    state = getattr(job, "state", None)
    name = getattr(state, "name", None)
    return str(name or state or "JOB_STATE_UNSPECIFIED")


def _job_error(job: object) -> str | None:
    error = getattr(job, "error", None)
    return str(error) if error else None


def _job_result_file(job: object) -> str | None:
    destination = getattr(job, "dest", None)
    file_name = getattr(destination, "file_name", None)
    return str(file_name) if file_name else None


def _job_path(batch_dir: Path) -> Path:
    return batch_dir / "job.json"


def _manifest_path(batch_dir: Path) -> Path:
    return batch_dir / "manifest.json"


def _request_path(batch_dir: Path) -> Path:
    return batch_dir / "requests.jsonl"


def _resolve_job_name(batch_dir: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    state = _read_json(_job_path(batch_dir))
    job_name = state.get("job_name")
    if not isinstance(job_name, str) or not job_name:
        raise GeminiBatchError("Saved job state contains no job_name")
    return job_name


def _client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise GeminiBatchError("GEMINI_API_KEY is required")
    from google import genai  # noqa: PLC0415

    return genai.Client(api_key=api_key)


def _prepare(arguments: argparse.Namespace) -> dict[str, object]:
    if _job_path(arguments.batch_dir).exists() and not arguments.allow_reprepare:
        raise GeminiBatchError(
            "A saved batch job exists; download it before preparing another batch, "
            "or explicitly add --allow-reprepare"
        )
    csv_bytes = arguments.csv.read_bytes()
    rows = load_prompt_rows(arguments.csv)
    excluded_below_min = (
        0
        if arguments.cocktail_id
        else sum(row.cocktail_id < arguments.min_id for row in rows)
    )
    selected, skipped_existing = select_batch_rows(
        rows,
        arguments.images,
        min_id=arguments.min_id,
        cocktail_ids=tuple(arguments.cocktail_id),
        include_existing=arguments.include_existing,
    )
    jsonl = build_batch_jsonl(selected)
    manifest = build_batch_manifest(
        selected,
        csv_sha256=sha256_bytes(csv_bytes),
        jsonl_sha256=sha256_bytes(jsonl),
    )
    atomic_write(_request_path(arguments.batch_dir), jsonl)
    atomic_write(_manifest_path(arguments.batch_dir), _json_bytes(manifest))
    return {
        "charged": False,
        "requests": len(selected),
        "excluded_below_min_id": excluded_below_min,
        "skipped_existing": skipped_existing,
        "min_id": min(row.cocktail_id for row in selected),
        "max_id": max(row.cocktail_id for row in selected),
        "model": manifest["model"],
        "aspect_ratio": manifest["aspect_ratio"],
        "image_size": manifest["image_size"],
        "estimated_image_output_cost_usd": manifest["estimated_image_output_cost_usd"],
        "input_file": str(_request_path(arguments.batch_dir)),
        "manifest": str(_manifest_path(arguments.batch_dir)),
    }


def _decimal_argument(value: str, label: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as error:
        raise GeminiBatchError(f"{label} must be a decimal number") from error
    if result <= 0:
        raise GeminiBatchError(f"{label} must be greater than zero")
    return result


def _verify_submit_guards(
    arguments: argparse.Namespace,
    manifest: dict[str, Any],
    request_bytes: bytes,
) -> tuple[int, Decimal]:
    if not arguments.confirm_paid_batch:
        raise GeminiBatchError(
            "Paid submission blocked: add --confirm-paid-batch after reviewing "
            "the manifest and billing settings"
        )
    expected_settings = {
        "model": GEMINI_BATCH_IMAGE_MODEL,
        "aspect_ratio": GEMINI_BATCH_ASPECT_RATIO,
        "image_size": GEMINI_BATCH_IMAGE_SIZE,
        "batch_image_output_price_usd": str(GEMINI_BATCH_IMAGE_PRICE_USD),
    }
    for field, expected in expected_settings.items():
        if manifest.get(field) != expected:
            raise GeminiBatchError(
                f"Manifest {field}={manifest.get(field)!r}; expected {expected!r}"
            )
    request_count = int(manifest.get("request_count", 0))
    if request_count <= 0:
        raise GeminiBatchError("Manifest contains no requests")
    if request_count > arguments.max_requests:
        raise GeminiBatchError(
            f"Manifest has {request_count} requests, above --max-requests="
            f"{arguments.max_requests}"
        )
    expected_hash = manifest.get("jsonl_sha256")
    if expected_hash != sha256_bytes(request_bytes):
        raise GeminiBatchError("requests.jsonl does not match manifest.json")

    manifest_keys = set(manifest_request_map(manifest))
    jsonl_keys: list[str] = []
    for line_number, line in enumerate(request_bytes.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            request = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GeminiBatchError(
                f"requests.jsonl line {line_number} is invalid JSON"
            ) from error
        key = request.get("key") if isinstance(request, dict) else None
        if not isinstance(key, str):
            raise GeminiBatchError(
                f"requests.jsonl line {line_number} contains no request key"
            )
        jsonl_keys.append(key)
    if len(jsonl_keys) != request_count:
        raise GeminiBatchError(
            f"requests.jsonl contains {len(jsonl_keys)} requests; "
            f"manifest declares {request_count}"
        )
    if len(set(jsonl_keys)) != len(jsonl_keys) or set(jsonl_keys) != manifest_keys:
        raise GeminiBatchError("requests.jsonl keys do not match manifest.json")

    estimated = estimated_image_output_cost(request_count)
    try:
        configured_estimate = Decimal(
            str(manifest.get("estimated_image_output_cost_usd", "0"))
        )
    except InvalidOperation as error:
        raise GeminiBatchError("Manifest contains an invalid cost estimate") from error
    if configured_estimate != estimated:
        raise GeminiBatchError("Manifest cost estimate does not match request count")
    maximum = _decimal_argument(
        arguments.max_image_output_cost_usd,
        "--max-image-output-cost-usd",
    )
    if estimated > maximum:
        raise GeminiBatchError(
            f"Estimated image output cost ${estimated} exceeds approved ${maximum}"
        )
    return request_count, estimated


def _submit(arguments: argparse.Namespace) -> dict[str, object]:
    request_path = _request_path(arguments.batch_dir)
    manifest = _read_json(_manifest_path(arguments.batch_dir))
    request_bytes = request_path.read_bytes()
    request_count, estimated = _verify_submit_guards(
        arguments,
        manifest,
        request_bytes,
    )

    saved_job_path = _job_path(arguments.batch_dir)
    if saved_job_path.exists() and not arguments.allow_new_job:
        saved = _read_json(saved_job_path)
        raise GeminiBatchError(
            "A saved batch job already exists "
            f"({saved.get('job_name', 'unknown')}, "
            f"{saved.get('last_state', 'unknown')}); use status/download or "
            "explicitly add --allow-new-job"
        )

    from google.genai import types  # noqa: PLC0415

    client = _client()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    display_name = arguments.display_name or f"cocktail-images-{timestamp}"
    uploaded = client.files.upload(
        file=request_path,
        config=types.UploadFileConfig(
            display_name=display_name,
            mime_type="jsonl",
        ),
    )
    upload_name = getattr(uploaded, "name", None)
    if not upload_name:
        raise GeminiBatchError("Gemini File API returned no uploaded file name")
    try:
        job = client.batches.create(
            model=str(manifest.get("model") or GEMINI_BATCH_IMAGE_MODEL),
            src=str(upload_name),
            config={"display_name": display_name},
        )
    except Exception as error:  # noqa: BLE001
        atomic_write(
            saved_job_path,
            _json_bytes(
                {
                    "job_name": None,
                    "uploaded_file_name": str(upload_name),
                    "last_state": "SUBMIT_FAILED",
                    "error": str(error),
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                    "request_count": request_count,
                    "estimated_image_output_cost_usd": str(estimated),
                    "jsonl_sha256": manifest["jsonl_sha256"],
                }
            ),
        )
        raise

    job_name = getattr(job, "name", None)
    if not job_name:
        raise GeminiBatchError("Gemini Batch API returned no job name")
    state = {
        "job_name": str(job_name),
        "uploaded_file_name": str(upload_name),
        "display_name": display_name,
        "model": manifest["model"],
        "last_state": _job_state(job),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "request_count": request_count,
        "estimated_image_output_cost_usd": str(estimated),
        "jsonl_sha256": manifest["jsonl_sha256"],
    }
    atomic_write(saved_job_path, _json_bytes(state))
    return {**state, "charged": True}


def _status(arguments: argparse.Namespace) -> tuple[object, dict[str, object]]:
    job_name = _resolve_job_name(arguments.batch_dir, arguments.job_name)
    client = _client()
    job = client.batches.get(name=job_name)
    state = _job_state(job)
    result = {
        "job_name": job_name,
        "last_state": state,
        "error": _job_error(job),
        "result_file_name": _job_result_file(job),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    saved_path = _job_path(arguments.batch_dir)
    saved = _read_json(saved_path) if saved_path.exists() else {}
    atomic_write(saved_path, _json_bytes({**saved, **result}))
    return job, result


def _write_failures(batch_dir: Path, failures: list[object]) -> Path:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=("key", "cocktail_id", "cocktail_name", "error"),
    )
    writer.writeheader()
    for failure in failures:
        writer.writerow(asdict(failure))
    path = batch_dir / "failed.csv"
    atomic_write(path, output.getvalue().encode("utf-8-sig"))
    return path


def _download(
    arguments: argparse.Namespace, job: object | None = None
) -> dict[str, object]:
    if job is None:
        job, _ = _status(arguments)
    state = _job_state(job)
    if state != "JOB_STATE_SUCCEEDED":
        raise GeminiBatchError(
            f"Batch job is {state}; download is available only after success"
        )
    result_file = _job_result_file(job)
    if not result_file:
        raise GeminiBatchError("Successful batch job has no result file")

    manifest = _read_json(_manifest_path(arguments.batch_dir))
    client = _client()
    result_bytes = client.files.download(file=result_file)
    summary = download_batch_results(
        result_bytes,
        manifest,
        arguments.images,
        overwrite=arguments.overwrite,
    )
    failed_path = _write_failures(arguments.batch_dir, summary.failures)
    job_path = _job_path(arguments.batch_dir)
    saved = _read_json(job_path) if job_path.exists() else {}
    atomic_write(
        job_path,
        _json_bytes(
            {
                **saved,
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
                "download_saved": summary.saved,
                "download_skipped_existing": summary.skipped_existing,
                "download_failed": summary.failed,
            }
        ),
    )
    return {
        "job_name": getattr(job, "name", None),
        "saved": summary.saved,
        "skipped_existing": summary.skipped_existing,
        "failed": summary.failed,
        "images": str(arguments.images),
        "failed_csv": str(failed_path),
    }


def _wait_download(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.poll_seconds < 10:
        raise GeminiBatchError("--poll-seconds must be at least 10")
    while True:
        job, status = _status(arguments)
        state = str(status["last_state"])
        print(json.dumps(status, ensure_ascii=False, default=str), flush=True)
        if state in TERMINAL_STATES:
            if state != "JOB_STATE_SUCCEEDED":
                raise GeminiBatchError(
                    f"Batch ended as {state}: {status.get('error') or 'no detail'}"
                )
            return _download(arguments, job)
        time.sleep(arguments.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    env_file = _load_environment()
    try:
        if arguments.command == "prepare":
            result = _prepare(arguments)
        elif arguments.command == "submit":
            result = _submit(arguments)
        elif arguments.command == "status":
            _, result = _status(arguments)
        elif arguments.command == "download":
            result = _download(arguments)
        else:
            result = _wait_download(arguments)
    except KeyboardInterrupt:
        print("Interrupted; the remote batch job is not cancelled.")
        return 130
    except Exception as error:  # noqa: BLE001
        print(json.dumps({"error": str(error)}, ensure_ascii=False, indent=2))
        return 1

    print(
        json.dumps(
            {"env_file": str(env_file), **result},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )
    return 2 if int(result.get("failed", 0)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
