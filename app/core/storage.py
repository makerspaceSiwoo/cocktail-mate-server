"""S3 호환 오브젝트 스토리지 클라이언트.

로컬 개발에서는 MinIO, 프로덕션에서는 Oracle Object Storage(S3 호환 엔드포인트)를
같은 boto3 코드로 사용한다. 엔드포인트/키만 `.env`로 교체한다.
"""

from functools import lru_cache
from typing import BinaryIO

import boto3
from botocore.client import BaseClient
from botocore.config import Config

from app.core.config import get_settings


@lru_cache
def get_s3_client() -> BaseClient:
    settings = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=settings.storage_endpoint,
        region_name=settings.storage_region,
        aws_access_key_id=settings.storage_access_key,
        aws_secret_access_key=settings.storage_secret_key,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def upload_fileobj(fileobj: BinaryIO, key: str, content_type: str | None = None) -> str:
    """파일 객체를 버킷에 업로드하고 공개 URL을 반환한다."""
    settings = get_settings()
    extra = {"ContentType": content_type} if content_type else {}
    get_s3_client().upload_fileobj(
        fileobj, settings.storage_bucket, key, ExtraArgs=extra
    )
    return f"{settings.storage_public_base_url.rstrip('/')}/{key}"


def generate_presigned_url(key: str, expires_in: int = 3600) -> str:
    """비공개 객체 임시 접근용 presigned URL."""
    settings = get_settings()
    return get_s3_client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.storage_bucket, "Key": key},
        ExpiresIn=expires_in,
    )


def check_bucket() -> bool:
    """헬스체크용: 설정된 버킷에 접근 가능한지 확인한다."""
    settings = get_settings()
    get_s3_client().head_bucket(Bucket=settings.storage_bucket)
    return True
