"""Camera control endpoints."""

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from app.api import verify_api_key
from app.config import settings
from app.schemas import CameraStartRequest, CameraStatusResponse, RoiRequest, RoiResponse, BoundingBox
from app.services.camera_service import CameraService
from app.utils import is_video_upload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/camera", tags=["camera"])

# Non-erroring header check for the MJPEG stream, which also accepts the key as a
# query param (an <img> tag cannot send custom headers).
_stream_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# Directory for uploaded videos that the camera service streams from.
_VIDEO_UPLOAD_DIR = os.path.join("storage", "uploaded_videos")


@router.post("/start")
async def start_camera(
    request: CameraStartRequest,
    _key: str = Security(verify_api_key),
):
    cam = CameraService.get_instance()
    try:
        await cam.start(source=request.source, camera_id=request.camera_id, fps=request.fps)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "started", "source": cam.source}


@router.post("/upload-video")
async def upload_video_stream(
    file: UploadFile = File(...),
    fps: Optional[float] = Form(None),
    loop: bool = Form(False),
    _key: str = Security(verify_api_key),
):
    """
    Upload a video file and start streaming it through the detection pipeline.

    The file is persisted to disk and the camera service is pointed at it, so the
    annotated live feed (with bounding boxes + recognition labels) and the live
    stats apply exactly as they do for a webcam — viewable on the Video Studio /
    Live Monitor pages via the snapshot poller.
    """
    if not is_video_upload(file.filename, file.content_type):
        raise HTTPException(status_code=400, detail="File does not look like a video.")

    contents = await file.read()
    max_bytes = settings.VIDEO_MAX_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Video exceeds the {settings.VIDEO_MAX_SIZE_MB} MB limit.",
        )

    os.makedirs(_VIDEO_UPLOAD_DIR, exist_ok=True)
    suffix = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    fd, path = tempfile.mkstemp(suffix=suffix, dir=_VIDEO_UPLOAD_DIR)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}")

    cam = CameraService.get_instance()
    if cam.is_running:
        await cam.stop()

    try:
        await cam.start(
            source=path,
            camera_id=f"video:{os.path.basename(file.filename or 'upload')}",
            fps=fps or settings.CAMERA_FPS,
            loop=loop,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {
        "status": "streaming",
        "filename": file.filename,
        "source": path,
        "size_mb": round(len(contents) / (1024 * 1024), 2),
        "looping": loop,
    }


@router.post("/stop")
async def stop_camera(_key: str = Security(verify_api_key)):
    cam = CameraService.get_instance()
    await cam.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=CameraStatusResponse)
async def camera_status(_key: str = Security(verify_api_key)):
    return CameraStatusResponse(**CameraService.get_instance().status())


@router.get("/snapshot")
async def camera_snapshot(
    annotated: bool = True,
    _key: str = Security(verify_api_key),
):
    jpeg = CameraService.get_instance().snapshot_jpeg(annotated=annotated)
    if jpeg is None:
        raise HTTPException(status_code=404, detail="No frame available yet.")
    return Response(content=jpeg, media_type="image/jpeg")


@router.get("/stream")
async def camera_stream(
    api_key: Optional[str] = Query(None, description="API key (for <img> tags that can't set headers)"),
    header_key: Optional[str] = Security(_stream_api_key_header),
):
    """
    Live MJPEG push stream (multipart/x-mixed-replace). Frames are pushed to the
    client the moment the pipeline produces them — usable directly as an
    `<img src="/api/camera/stream?api_key=...">` source.

    Auth accepts either the x-api-key header or an `api_key` query param, since a
    browser <img> tag cannot send custom headers.
    """
    provided = header_key or api_key
    if not provided or provided != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")

    cam = CameraService.get_instance()
    if not cam.is_running:
        raise HTTPException(status_code=409, detail="Camera is not running.")

    return StreamingResponse(
        cam.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store", "Connection": "close"},
    )


@router.post("/roi", response_model=RoiResponse)
async def set_roi(
    body: RoiRequest,
    _key: str = Security(verify_api_key),
):
    cam = CameraService.get_instance()
    cam.roi = body.roi.model_dump() if body.roi else None
    return RoiResponse(roi=body.roi)


@router.get("/roi", response_model=RoiResponse)
async def get_roi(_key: str = Security(verify_api_key)):
    cam = CameraService.get_instance()
    roi = cam.roi
    return RoiResponse(
        roi=BoundingBox(**roi) if roi else None
    )
