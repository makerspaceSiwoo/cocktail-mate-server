"""Private API accepting at most ten manually generated images per request."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.image_generation.core import ImageGenerationSettings
from app.image_upload.auth import require_image_upload_key
from app.image_upload.schemas import BatchImageUploadResponse
from app.image_upload.service import (
    MAX_BATCH_FILES,
    MAX_UPLOAD_BYTES,
    ImageUploadError,
    IncomingImage,
    persist_batch,
)

router = APIRouter(prefix="/admin/cocktail-images", tags=["admin"])


@router.post("/batch", response_model=BatchImageUploadResponse)
def upload_cocktail_image_batch(
    files: list[UploadFile] = File(...),
    _authorized: None = Depends(require_image_upload_key),
    db: Session = Depends(get_db),
) -> BatchImageUploadResponse:
    if not 1 <= len(files) <= MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail="Each batch must contain between 1 and 10 images",
        )

    incoming: list[IncomingImage] = []
    for upload in files:
        filename = upload.filename or ""
        data = upload.file.read(MAX_UPLOAD_BYTES + 1)
        upload.file.close()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{filename}: file exceeds 15 MiB",
            )
        incoming.append(IncomingImage(filename=filename, data=data))

    try:
        items = persist_batch(db, incoming, ImageGenerationSettings())
    except ImageUploadError as error:
        raise HTTPException(
            status_code=error.status_code,
            detail=error.detail,
        ) from error

    return BatchImageUploadResponse(uploaded=len(items), items=items)
