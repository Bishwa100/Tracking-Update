"""
Image and video utility functions.
"""

import asyncio
import base64
import logging
import os
import tempfile
from typing import List, Optional

import cv2
import numpy as np
from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import settings

logger = logging.getLogger(__name__)

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ── Inference concurrency gate ───────────────────────────────
# Each heavy inference call (YOLO/ArcFace/OSNet) uses every CPU core, so letting
# several run concurrently from the threadpool makes them all slower. This
# semaphore serializes (or bounds) inference across all requests/tasks.
_inference_semaphore: Optional[asyncio.Semaphore] = None


def _get_inference_semaphore() -> asyncio.Semaphore:
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(max(1, settings.INFERENCE_MAX_CONCURRENCY))
    return _inference_semaphore


async def run_inference(func, /, *args, **kwargs):
    """Run a CPU-heavy inference function off the event loop, bounded by the
    global inference semaphore so concurrent work doesn't thrash the CPU."""
    async with _get_inference_semaphore():
        return await run_in_threadpool(func, *args, **kwargs)


def cap_frame_long_side(image: np.ndarray, max_side: Optional[int] = None) -> np.ndarray:
    """Downscale a frame so its longest side is at most `max_side` pixels."""
    if max_side is None:
        max_side = settings.MAX_FRAME_LONG_SIDE
    if max_side <= 0:
        return image
    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= max_side:
        return image
    scale = max_side / long_side
    return cv2.resize(
        image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


async def file_to_cv_image(upload_file: UploadFile) -> np.ndarray:
    """Convert a FastAPI UploadFile to an OpenCV BGR image."""
    contents = await upload_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image from file: {upload_file.filename}")
    return image


def _extract_video_frames_from_path(video_path: str, fps: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from a video file at the configured sampling FPS."""
    if fps is None:
        fps = settings.FRAMES_PER_SECOND

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    frames: List[np.ndarray] = []
    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 0:
            duration_seconds = total_frames / video_fps
            if duration_seconds > settings.VIDEO_MAX_DURATION_SECONDS:
                raise ValueError(
                    f"Video duration {duration_seconds:.1f}s exceeds the "
                    f"{settings.VIDEO_MAX_DURATION_SECONDS}s limit."
                )

        frame_interval = max(1, int(video_fps / fps))
        frame_count = 0
        # grab() advances without decoding; retrieve via read() only kept frames.
        while True:
            if frame_count % frame_interval == 0:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cap_frame_long_side(frame))
            else:
                if not cap.grab():
                    break
            frame_count += 1
    finally:
        cap.release()
    return frames


async def extract_video_frames(video_file: UploadFile, fps: Optional[int] = None) -> List[np.ndarray]:
    """Extract frames from an uploaded video file at the configured FPS rate."""
    contents = await video_file.read()
    tmp_path = None
    try:
        suffix = os.path.splitext(video_file.filename or "")[1].lower()
        if suffix not in VIDEO_SUFFIXES:
            suffix = ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        return await asyncio.to_thread(_extract_video_frames_from_path, tmp_path, fps)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def is_video_upload(filename: Optional[str], content_type: Optional[str]) -> bool:
    """True when an uploaded file looks like a video (by content-type or extension)."""
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if ctype.startswith("video/"):
        return True
    suffix = os.path.splitext(filename or "")[1].lower()
    return suffix in VIDEO_SUFFIXES


def compute_dhash(image: np.ndarray, hash_size: int = 8) -> int:
    """
    Difference hash (dHash): a perceptual fingerprint robust to compression
    noise and tiny lighting shifts. hash_size=8 → 64-bit, hash_size=16 → 256-bit.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int.from_bytes(np.packbits(diff).tobytes(), byteorder="big")


def frame_signature(frame: np.ndarray, size: int = 32) -> np.ndarray:
    """Cheap perceptual fingerprint of a frame (small grayscale thumbnail)."""
    small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)


def frames_are_similar(
    sig_a: Optional[np.ndarray],
    sig_b: Optional[np.ndarray],
    threshold: float,
) -> bool:
    """True if two frame signatures differ by less than `threshold` (mean abs diff)."""
    if sig_a is None or sig_b is None:
        return False
    return float(np.mean(np.abs(sig_a - sig_b))) < threshold


class FrameDedupBuffer:
    """
    Stateful frame de-duplication. Owns the previous frame signature and the
    threshold so the live-camera and upload paths share one implementation
    instead of each tracking their own `prev_sig`.

    Usage:
        dedup = FrameDedupBuffer()
        if dedup.is_duplicate(frame):   # near-identical to the previous frame
            continue                    # skip the heavy detection pass
    """

    def __init__(self, threshold: Optional[float] = None, enabled: Optional[bool] = None):
        self._threshold = (
            settings.FRAME_DEDUP_MAD_THRESHOLD if threshold is None else threshold
        )
        self._enabled = settings.FRAME_DEDUP_ENABLED if enabled is None else enabled
        self._prev_sig: Optional[np.ndarray] = None
        self.skipped = 0
        self.processed = 0

    def is_duplicate(self, frame: np.ndarray) -> bool:
        """Update state and return True when `frame` ~ the previous frame."""
        if not self._enabled:
            self.processed += 1
            return False
        sig = frame_signature(frame)
        dup = frames_are_similar(self._prev_sig, sig, self._threshold)
        self._prev_sig = sig
        if dup:
            self.skipped += 1
        else:
            self.processed += 1
        return dup


def cv_image_to_base64(image: np.ndarray, fmt: str = ".jpg", quality: int = 85) -> str:
    """Encode an OpenCV BGR image to a base64 string."""
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)] if fmt in (".jpg", ".jpeg") else []
    success, buffer = cv2.imencode(fmt, image, params)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.b64encode(buffer).decode("utf-8")


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    """Encode an OpenCV BGR image to raw JPEG bytes (for snapshot endpoints)."""
    success, buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not success:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


# Re-exported from app.similarity to keep one normalization implementation.
from app.similarity import normalize_embedding  # noqa: E402,F401


# Status → BGR colour for annotated detection boxes.
_DETECTION_COLORS = {
    "new": (0, 200, 0),        # green  — newly registered
    "returning": (255, 160, 0),  # blue-ish — recognised returning visitor
    "ambiguous": (0, 200, 200),  # yellow — skipped
    "none": (128, 128, 128),     # grey
}


def draw_detections(image: np.ndarray, annotations: list) -> np.ndarray:
    """
    Draw labelled bounding boxes for the live feed.

    annotations: list of {bbox: {x1,y1,x2,y2}, label: str, status: str}.
    """
    annotated = image.copy()
    img_h, img_w = annotated.shape[:2]
    font_scale = max(0.5, img_w / 1400.0)
    thickness = max(1, int(font_scale * 2))

    for ann in annotations:
        bbox = ann.get("bbox") or {}
        x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
        x2, y2 = int(bbox.get("x2", 0)), int(bbox.get("y2", 0))
        status = ann.get("status", "none")
        label = ann.get("label", "")
        color = _DETECTION_COLORS.get(status, (200, 200, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, max(2, thickness))
        if not label:
            continue

        size, base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = int(6 * font_scale)
        y_text = max(y1, size[1] + pad + 5)
        cv2.rectangle(
            annotated,
            (x1, y_text - size[1] - pad),
            (x1 + size[0] + pad * 2, y_text + base + pad),
            color, -1,
        )
        b, g, r = color
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if lum > 128 else (255, 255, 255)
        cv2.putText(
            annotated, label, (x1 + pad, y_text),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA,
        )
    return annotated


# ── Face preprocessing (CLAHE + gamma) ──────────────────────────────────────

def apply_clahe(face_crop: np.ndarray,
                clip_limit: float = 2.0,
                grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    CLAHE on the L channel of LAB.
    Improves recognition under uneven restaurant lighting (~2 ms per 112×112 crop).
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_eq = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def apply_gamma_correction(image: np.ndarray, gamma: float = None) -> np.ndarray:
    """
    Auto-gamma based on mean luminance.  Dark images are brightened, bright
    images slightly darkened — restores detail lost in restaurant back-lighting.
    """
    if image is None or image.size == 0:
        return image
    if gamma is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_lum = float(np.mean(gray)) / 255.0
        if mean_lum > 0:
            gamma = np.log(0.5) / np.log(max(mean_lum, 1e-6))
        else:
            gamma = 1.0
        gamma = float(np.clip(gamma, 0.5, 2.0))
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def preprocess_face_for_recognition(face_crop: np.ndarray) -> np.ndarray:
    """
    Gamma-then-CLAHE preprocessing before ArcFace embedding extraction.
    Controlled by FACE_PREPROCESSING_GAMMA / FACE_PREPROCESSING_CLAHE settings.
    The original crop is unchanged (thumbnail still looks natural).
    """
    if face_crop is None or face_crop.size == 0:
        return face_crop
    result = face_crop.copy()
    if settings.FACE_PREPROCESSING_GAMMA:
        result = apply_gamma_correction(result)
    if settings.FACE_PREPROCESSING_CLAHE:
        result = apply_clahe(result,
                             clip_limit=settings.CLAHE_CLIP_LIMIT,
                             grid_size=tuple(settings.CLAHE_GRID_SIZE))
    return result
