"""POST /api/detect — one-shot detection from an uploaded image or video."""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import verify_api_key
from app.config import settings
from app.cv_pipeline import process_frame
from app.database import get_db
from app.schemas import DetectionItem, DetectResponse
from app.services.detection_pipeline import process_detections
from app.utils import (
    cap_frame_long_side,
    extract_video_frames,
    file_to_cv_image,
    FrameDedupBuffer,
    is_video_upload,
    run_inference,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["detect"])


@router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(...),
    camera_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """Detect, auto-register, and recognise visitors in an uploaded image/video."""
    camera_id = camera_id or "upload"

    if is_video_upload(file.filename, file.content_type):
        try:
            frames = await extract_video_frames(file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        if not frames:
            raise HTTPException(status_code=422, detail="Could not extract frames from video.")
    else:
        try:
            image = await file_to_cv_image(file)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        frames = [cap_frame_long_side(image)]

    all_detections = []
    frames_processed = 0
    dedup = FrameDedupBuffer(enabled=settings.FRAME_DEDUP_ENABLED and len(frames) > 1)

    for frame in frames:
        if dedup.is_duplicate(frame):
            continue

        detections = await run_inference(process_frame, frame, settings.ALLOW_BODY_FALLBACK)
        frames_processed += 1
        if not detections:
            continue
        processed = await process_detections(
            db, detections, frame=frame, camera_id=camera_id
        )
        all_detections.extend(processed)

    items = [
        DetectionItem(
            visitor_id=pd.visitor_id,
            is_new=pd.is_new,
            is_ambiguous=pd.is_ambiguous,
            visit_id=pd.visit_id,
            face_confidence=pd.face_confidence,
            body_confidence=pd.body_confidence,
            match_source=pd.match_source,
            bbox=pd.bbox,
        )
        for pd in all_detections
    ]
    new_count = sum(1 for pd in all_detections if pd.is_new)
    returning_count = sum(
        1 for pd in all_detections if pd.visitor_id is not None and not pd.is_new
    )

    return DetectResponse(
        detections=items,
        new_visitors_count=new_count,
        returning_visitors_count=returning_count,
        frames_processed=frames_processed,
    )
