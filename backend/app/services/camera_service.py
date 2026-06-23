"""
Camera service — background webcam/RTSP/file processor.

Two execution modes (selected by settings.PIPELINE_PARALLEL):

* Parallel (default): a multi-stage pipeline whose stages run concurrently —
  a capture task that keeps only the newest frame (drops backlog for a
  low-latency live view), one or more inference workers, and a post-process
  task that does DB writes + annotation + JPEG encoding. The GPU stays busy on
  inference while the CPU captures, writes to the DB and encodes in parallel.
* Sequential: the original single read→infer→DB→annotate→sleep loop.
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
    """Keep only detections whose *drawn* box falls inside the ROI.

    The label is rendered on the face box when a face is present (otherwise the
    person box), so we gate on that same box — a tall person whose body center
    is in the zone but whose face is above it must not produce a label outside
    the zone.
    """
    return [
        d for d in detections
        if _bbox_center_in_roi(getattr(d, "face_bbox", None) or d.bbox, roi)
    ]


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
        self._last_jpeg: Optional[bytes] = None  # latest pre-encoded annotated frame
        # Parallel-pipeline state (created fresh on each start()).
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_frame_id: int = 0   # incremented by the capture stage
        self._claimed_id: int = 0        # highest frame id claimed by a worker
        self._display_id: int = 0        # highest frame id shown / encoded
        self._last_sig: Optional[np.ndarray] = None
        self._last_annotations: list = []  # most recent detection overlays
        self._annotations_id: int = 0      # frame id the overlays came from
        self._frame_cond: Optional[asyncio.Condition] = None
        self._display_cond: Optional[asyncio.Condition] = None
        self._results: Optional[asyncio.Queue] = None
        self._pipeline_tasks: list = []
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
        if not self.capture.isOpened() and str(self.source).startswith("rtsp"):
            # OpenCV's FFmpeg can't do SHA-256 RTSP digest (common on modern IP
            # cameras). Fall back to our native RTSP client, which authenticates
            # itself and decodes via FFmpeg. See services/rtsp_native.py.
            try:
                await asyncio.to_thread(self.capture.release)
            except Exception:  # noqa: BLE001
                pass
            logger.info("OpenCV could not open RTSP source; trying native RTSP client.")
            from app.services.rtsp_native import RtspNativeCapture
            self.capture = await asyncio.to_thread(RtspNativeCapture, self.source)
        if not self.capture.isOpened():
            self.capture = None
            raise RuntimeError(f"Could not open camera source: {self.source}")

        self.is_running = True
        self.started_at = perf_counter()
        for k in self.stats:
            self.stats[k] = 0
        self._reset_pipeline_state()
        if settings.PIPELINE_PARALLEL:
            self._task = asyncio.create_task(self._run_parallel())
            logger.info(
                "Camera started — parallel pipeline (source=%s, id=%s, workers=%d, max_fps=%s).",
                self.source, self.camera_id, max(1, settings.INFERENCE_WORKERS),
                settings.PIPELINE_MAX_FPS or "unlimited",
            )
        else:
            self._task = asyncio.create_task(self._processing_loop())
            logger.info(
                "Camera started — sequential loop (source=%s, id=%s, fps=%.2f).",
                self.source, self.camera_id, self.fps,
            )

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

    # ── Parallel pipeline ────────────────────────────────────

    def _reset_pipeline_state(self) -> None:
        """Fresh synchronization primitives + counters for a new run."""
        self._latest_frame = None
        self._latest_frame_id = 0
        self._claimed_id = 0
        self._display_id = 0
        self._last_sig = None
        self._last_jpeg = None
        self._last_annotations = []
        self._annotations_id = 0
        self._frame_cond = asyncio.Condition()
        self._display_cond = asyncio.Condition()
        self._results = None
        self._pipeline_tasks = []

    def _is_file_source(self) -> bool:
        src = self.source or ""
        return bool(src) and not src.isdigit() and not src.startswith(("rtsp", "http"))

    async def _run_parallel(self) -> None:
        """Spawn capture + inference workers + consumer; tear them all down together."""
        n_workers = max(1, settings.INFERENCE_WORKERS)
        self._results = asyncio.Queue(maxsize=n_workers + 1)
        tasks = [
            asyncio.create_task(self._capture_loop()),
            asyncio.create_task(self._display_loop()),
        ]
        for i in range(n_workers):
            tasks.append(asyncio.create_task(self._inference_worker(i)))
        tasks.append(asyncio.create_task(self._consumer_loop()))
        self._pipeline_tasks = tasks
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("Parallel pipeline crashed: %s", exc)
        finally:
            self.is_running = False
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Wake any client streams blocked waiting for a frame.
            async with self._display_cond:
                self._display_cond.notify_all()

    async def _capture_loop(self) -> None:
        """Continuously grab frames, keeping only the newest (drops backlog)."""
        # Optional pacing: explicit cap, or a video file's own frame rate so it
        # plays in real time rather than blasting through. Live cameras pace
        # themselves, so leave them unthrottled.
        throttle_fps = settings.PIPELINE_MAX_FPS
        if throttle_fps <= 0 and self._is_file_source():
            native = 0.0
            try:
                native = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                native = 0.0
            throttle_fps = native if native > 1.0 else self.fps
        interval = 1.0 / throttle_fps if throttle_fps > 0 else 0.0

        try:
            while self.is_running:
                loop_start = perf_counter()
                ret, frame = await asyncio.to_thread(self.capture.read)
                if not ret or frame is None:
                    if isinstance(_parse_source(self.source), int) or str(self.source).startswith("rtsp"):
                        await asyncio.sleep(0.05)
                        continue
                    if self.loop_file and self.capture is not None:
                        await asyncio.to_thread(self.capture.set, cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    logger.info("Camera source ended.")
                    break

                frame = cap_frame_long_side(frame)
                self._last_frame = frame
                async with self._frame_cond:
                    self._latest_frame = frame
                    self._latest_frame_id += 1
                    self._frame_cond.notify_all()

                if interval:
                    await self._sleep_remaining(loop_start, interval)
        except asyncio.CancelledError:
            raise
        finally:
            self.is_running = False
            async with self._frame_cond:
                self._frame_cond.notify_all()  # release blocked workers

    async def _claim_latest(self):
        """Block until a frame newer than the last claimed one exists, then take
        it — skipping every frame in between (newest-wins, low latency)."""
        async with self._frame_cond:
            await self._frame_cond.wait_for(
                lambda: not self.is_running or self._latest_frame_id > self._claimed_id
            )
            if not self.is_running:
                return None, 0
            self._claimed_id = self._latest_frame_id
            return self._latest_frame, self._claimed_id

    async def _display_loop(self) -> None:
        """Encode the latest captured frame at a steady preview rate, overlaying
        the most recent detection boxes. Runs independently of detection/dedup so
        the live feed never freezes when detection is skipped."""
        preview_fps = settings.LIVE_PREVIEW_FPS
        interval = 1.0 / preview_fps if preview_fps > 0 else 0.0
        last_seen_id = 0
        try:
            while self.is_running:
                loop_start = perf_counter()
                async with self._frame_cond:
                    await self._frame_cond.wait_for(
                        lambda: not self.is_running or self._latest_frame_id != last_seen_id
                    )
                    if not self.is_running:
                        break
                    frame = self._latest_frame
                    last_seen_id = self._latest_frame_id

                if frame is None:
                    continue

                annotations = self._last_annotations
                out = draw_detections(frame, annotations) if annotations else frame.copy()
                self._draw_roi_overlay(out)
                self._last_annotated = out

                jpeg = await asyncio.to_thread(
                    encode_jpeg, out, settings.LIVE_FEED_JPEG_QUALITY
                )
                async with self._display_cond:
                    self._last_jpeg = jpeg
                    self._display_id = last_seen_id
                    self._display_cond.notify_all()

                if interval:
                    await self._sleep_remaining(loop_start, interval)
        except asyncio.CancelledError:
            raise

    async def _inference_worker(self, worker_id: int) -> None:
        """Pull the newest frame and run the CV pipeline off the event loop.

        When an ROI is set, inference runs on the ROI crop only — so YOLO,
        ArcFace and OSNet never compute (and nothing ever registers) for anyone
        outside the zone. Boxes are offset back to full-frame coordinates.
        """
        embedding_cache = FaceEmbeddingCache()
        try:
            while self.is_running:
                frame, fid = await self._claim_latest()
                if frame is None:
                    break

                # Restrict all heavy compute to the zone.
                infer_frame, ox, oy = self._roi_crop(frame, self.roi)

                # Dedup on the region we actually process, so only motion inside
                # the zone triggers a (skippable) detection pass.
                if settings.FRAME_DEDUP_ENABLED:
                    sig = frame_signature(infer_frame)
                    if frames_are_similar(self._last_sig, sig, settings.FRAME_DEDUP_MAD_THRESHOLD):
                        self._last_sig = sig
                        self.stats["frames_skipped"] += 1
                        continue
                    self._last_sig = sig

                try:
                    detections = await run_inference(
                        process_frame, infer_frame,
                        settings.ALLOW_BODY_FALLBACK, embedding_cache
                    )
                except Exception as exc:
                    self.last_error = str(exc)
                    logger.exception("Inference failed (worker %d): %s", worker_id, exc)
                    continue

                if ox or oy:
                    self._offset_detections(detections, ox, oy)

                if self._results is not None:
                    await self._results.put((fid, frame, detections))
        except asyncio.CancelledError:
            raise

    @staticmethod
    def _roi_crop(frame: np.ndarray, roi: Optional[dict]):
        """Return (crop, offset_x, offset_y). No ROI → the full frame at (0, 0)."""
        if not roi or frame is None:
            return frame, 0, 0
        h, w = frame.shape[:2]
        x1 = max(0, min(int(roi["x1"]), w))
        y1 = max(0, min(int(roi["y1"]), h))
        x2 = max(0, min(int(roi["x2"]), w))
        y2 = max(0, min(int(roi["y2"]), h))
        if x2 - x1 < 10 or y2 - y1 < 10:
            return frame, 0, 0
        # Contiguous copy — a sliced view can break downstream cv2/ORT ops.
        return np.ascontiguousarray(frame[y1:y2, x1:x2]), x1, y1

    @staticmethod
    def _offset_detections(detections: list, ox: int, oy: int) -> None:
        """Shift crop-relative boxes/landmarks back into full-frame coords (in place)."""
        for d in detections:
            for box in (d.bbox, getattr(d, "face_bbox", None)):
                if box:
                    box["x1"] += ox
                    box["x2"] += ox
                    box["y1"] += oy
                    box["y2"] += oy
            lmk = getattr(d, "face_landmarks", None)
            if lmk is not None:
                d.face_landmarks = lmk + np.asarray([ox, oy], dtype=lmk.dtype)

    async def _consumer_loop(self) -> None:
        """Post-process inference results: ROI filter, DB write, and publish the
        detection overlays for the display loop to draw.

        Runs concurrently with capture+inference+display so neither the GPU nor
        the capture stage waits on the database, and the live feed is never
        blocked by detection.
        """
        try:
            while self.is_running:
                try:
                    fid, frame, detections = await asyncio.wait_for(
                        self._results.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # Out-of-order result from a slower worker — keep the newest
                # overlays so boxes never jump backwards.
                if fid <= self._annotations_id:
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

                # Publish overlays; the display loop draws them on the live frame.
                self._last_annotations = [
                    {"bbox": pd.bbox, "label": pd.label, "status": pd.status}
                    for pd in processed
                ]
                self._annotations_id = fid
        except asyncio.CancelledError:
            raise

    def _draw_roi_overlay(self, frame: Optional[np.ndarray]) -> None:
        """Visualize the detection zone on a frame (in place): dim everything
        outside the ROI so the active area stands out, then a bright border +
        label. No-op when no ROI is set."""
        if not (self.roi and frame is not None):
            return
        r = self.roi
        h, w = frame.shape[:2]
        x1 = max(0, min(int(r["x1"]), w))
        y1 = max(0, min(int(r["y1"]), h))
        x2 = max(0, min(int(r["x2"]), w))
        y2 = max(0, min(int(r["y2"]), h))
        if x2 <= x1 or y2 <= y1:
            return

        # Darken the whole frame, then restore the ROI region at full brightness.
        darkened = cv2.convertScaleAbs(frame, alpha=0.45, beta=0)
        darkened[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        np.copyto(frame, darkened)

        # Bright amber border + label tab.
        color = (0, 165, 255)  # BGR amber
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "Detection Zone"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 8 if y1 - th - 10 >= 0 else y1 + th + 8
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(
            frame, label, (x1 + 4, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    async def mjpeg_frames(self):
        """Async generator yielding multipart MJPEG parts as new frames arrive.

        Pushes the latest annotated frame to the client the instant the consumer
        produces it — no polling. Used by GET /api/camera/stream.
        """
        last_id = -1
        boundary = b"--frame\r\n"
        while self.is_running:
            async with self._display_cond:
                await self._display_cond.wait_for(
                    lambda: not self.is_running
                    or (self._display_id != last_id and self._last_jpeg is not None)
                )
                if not self.is_running:
                    break
                jpeg = self._last_jpeg
                last_id = self._display_id
            yield (
                boundary
                + b"Content-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )

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
                        process_frame, frame,
                        settings.ALLOW_BODY_FALLBACK, embedding_cache
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
                self._draw_roi_overlay(self._last_annotated)

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
        # Reuse the consumer's pre-encoded annotated frame when available.
        if annotated and self._last_jpeg is not None:
            return self._last_jpeg
        frame = self._last_annotated if annotated else self._last_frame
        if frame is None:
            return None
        return encode_jpeg(frame, settings.LIVE_FEED_JPEG_QUALITY)

    def status(self) -> dict:
        src = self.source or ""
        is_file = bool(src) and not src.isdigit() and not src.startswith(("rtsp", "http"))
        return {
            "pipeline": "parallel" if settings.PIPELINE_PARALLEL else "sequential",
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
