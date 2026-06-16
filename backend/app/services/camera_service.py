"""
Camera service — background webcam/RTSP/file processing loop (CPU, ~1 FPS).
"""

import asyncio
import logging
from datetime import datetime, timezone
from time import perf_counter
from typing import Optional, Union

import cv2
import numpy as np

from app.config import settings
from app.cv_pipeline import process_frame
from app.database import AsyncSessionLocal
from app.ml_models import FaceEmbeddingCache
from app.services.detection_pipeline import process_detections
from app.utils import (
    cap_frame_long_side,
    draw_detections,
    encode_jpeg,
    frame_signature,
    frames_are_similar,
    run_inference,
)

logger = logging.getLogger(__name__)


def _parse_source(source: str) -> Union[int, str]:
    """'0' → int 0 (webcam index); anything else stays a string (URL/path)."""
    s = (source or "").strip()
    return int(s) if s.isdigit() else s


def _bbox_center_in_roi(bbox: dict, roi: dict) -> bool:
    """Check if the center of a detection bbox falls within the ROI."""
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2
    return roi["x1"] <= cx <= roi["x2"] and roi["y1"] <= cy <= roi["y2"]


def _filter_by_roi(detections: list, roi: dict) -> list:
    """Filter detections to only those within the ROI."""
    return [d for d in detections if _bbox_center_in_roi(d.bbox, roi)]


class CameraService:
    """Singleton background camera processor."""

    _instance: Optional["CameraService"] = None

    def __init__(self):
        self.capture: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_annotated: Optional[np.ndarray] = None
        self.source: Optional[str] = None
        self.camera_id: str = settings.CAMERA_ID
        self.fps: float = settings.CAMERA_FPS
        self.started_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.loop_file: bool = False  # replay a finished video file from the start
        self.roi: Optional[dict] = None  # {"x1", "y1", "x2", "y2"} or None
        self.stats = {
            "frames_processed": 0,
            "frames_skipped": 0,
            "persons_detected": 0,
            "new_visitors": 0,
            "returning_visitors": 0,
        }

    @classmethod
    def get_instance(cls) -> "CameraService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(
        self,
        source: Optional[str] = None,
        camera_id: Optional[str] = None,
        fps: Optional[float] = None,
        loop: bool = False,
    ) -> None:
        if self.is_running:
            raise RuntimeError("Camera is already running.")

        self.source = source if source is not None else settings.CAMERA_SOURCE
        self.camera_id = camera_id or settings.CAMERA_ID
        self.fps = fps or settings.CAMERA_FPS
        self.loop_file = loop
        self.last_error = None

        cap_source = _parse_source(self.source)
        self.capture = await asyncio.to_thread(cv2.VideoCapture, cap_source)
        if not self.capture.isOpened():
            self.capture = None
            raise RuntimeError(f"Could not open camera source: {self.source}")

        self.is_running = True
        self.started_at = perf_counter()
        for k in self.stats:
            self.stats[k] = 0
        self._task = asyncio.create_task(self._processing_loop())
        logger.info("Camera started (source=%s, id=%s, fps=%.2f).", self.source, self.camera_id, self.fps)

    async def stop(self) -> None:
        self.is_running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        if self.capture is not None:
            await asyncio.to_thread(self.capture.release)
            self.capture = None
        logger.info("Camera stopped.")

    async def _processing_loop(self) -> None:
        prev_sig = None
        embedding_cache = FaceEmbeddingCache()
        interval = 1.0 / max(self.fps, 0.1)

        try:
            while self.is_running:
                loop_start = perf_counter()
                ret, frame = await asyncio.to_thread(self.capture.read)
                if not ret or frame is None:
                    # Files end; live cameras may hiccup — retry briefly.
                    if isinstance(_parse_source(self.source), int) or str(self.source).startswith("rtsp"):
                        await asyncio.sleep(interval)
                        continue
                    # Video file ended — replay from the start when looping.
                    if self.loop_file and self.capture is not None:
                        await asyncio.to_thread(
                            self.capture.set, cv2.CAP_PROP_POS_FRAMES, 0
                        )
                        prev_sig = None
                        continue
                    logger.info("Camera source ended.")
                    break

                frame = cap_frame_long_side(frame)
                self._last_frame = frame

                if settings.FRAME_DEDUP_ENABLED:
                    sig = frame_signature(frame)
                    if frames_are_similar(prev_sig, sig, settings.FRAME_DEDUP_MAD_THRESHOLD):
                        self.stats["frames_skipped"] += 1
                        prev_sig = sig
                        await self._sleep_remaining(loop_start, interval)
                        continue
                    prev_sig = sig

                try:
                    detections = await run_inference(
                        process_frame, frame, True, embedding_cache
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    logger.exception("Inference failed: %s", exc)
                    await self._sleep_remaining(loop_start, interval)
                    continue

                if self.roi and detections:
                    detections = _filter_by_roi(detections, self.roi)

                self.stats["frames_processed"] += 1
                self.stats["persons_detected"] += len(detections)

                processed = []
                if detections:
                    now = datetime.now(timezone.utc)
                    async with AsyncSessionLocal() as db:
                        try:
                            processed = await process_detections(
                                db, detections, frame=frame,
                                camera_id=self.camera_id, timestamp=now,
                            )
                        except Exception as exc:
                            self.last_error = str(exc)
                            logger.exception("Detection processing failed: %s", exc)
                            await db.rollback()

                    for pd in processed:
                        if pd.is_new:
                            self.stats["new_visitors"] += 1
                        elif pd.visitor_id is not None:
                            self.stats["returning_visitors"] += 1

                annotations = [
                    {"bbox": pd.bbox, "label": pd.label, "status": pd.status}
                    for pd in processed
                ]
                self._last_annotated = (
                    draw_detections(frame, annotations) if annotations else frame.copy()
                )

                if self.roi and self._last_annotated is not None:
                    r = self.roi
                    cv2.rectangle(
                        self._last_annotated,
                        (r["x1"], r["y1"]), (r["x2"], r["y2"]),
                        (0, 100, 255), 2
                    )
                    cv2.putText(
                        self._last_annotated, "Detection Zone",
                        (r["x1"] + 4, r["y1"] + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1
                    )

                await self._sleep_remaining(loop_start, interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Camera loop crashed: %s", exc)
        finally:
            self.is_running = False

    @staticmethod
    async def _sleep_remaining(loop_start: float, interval: float) -> None:
        elapsed = perf_counter() - loop_start
        if elapsed < interval:
            await asyncio.sleep(interval - elapsed)

    def snapshot_jpeg(self, annotated: bool = True) -> Optional[bytes]:
        frame = self._last_annotated if annotated else self._last_frame
        if frame is None:
            return None
        return encode_jpeg(frame, settings.LIVE_FEED_JPEG_QUALITY)

    def status(self) -> dict:
        src = self.source or ""
        is_file = bool(src) and not src.isdigit() and not src.startswith(("rtsp", "http"))
        return {
            "is_running": self.is_running,
            "source": self.source,
            "source_kind": "video" if is_file else "camera",
            "looping": self.loop_file,
            "camera_id": self.camera_id,
            "fps": self.fps,
            "frames_processed": self.stats["frames_processed"],
            "frames_skipped": self.stats["frames_skipped"],
            "persons_detected": self.stats["persons_detected"],
            "new_visitors": self.stats["new_visitors"],
            "returning_visitors": self.stats["returning_visitors"],
            "uptime_seconds": (perf_counter() - self.started_at) if self.started_at else 0.0,
            "last_error": self.last_error,
        }
