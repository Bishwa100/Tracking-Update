"""
Detection orchestration shared by /api/detect and the camera service.

For a list of DetectedPerson from one frame:
  resolve identity → register/update visitor → track visit → write audit event.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional
from uuid import UUID

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.cv_pipeline import DetectedPerson
from app.models import DetectionEvent, Visitor
from app.services import auto_enroller, identity_resolver
from app.services.visit_tracker import VisitTracker
from app.services.temporal_consistency import temporal_gate
from app.services.mask_detector import is_masked as _is_masked, masked_threshold_offset

logger = logging.getLogger(__name__)


@dataclass
class ProcessedDetection:
    bbox: dict
    visitor_id: Optional[UUID] = None
    is_new: bool = False
    is_ambiguous: bool = False
    visit_id: Optional[UUID] = None
    face_confidence: Optional[float] = None
    body_confidence: Optional[float] = None
    match_source: str = "none"

    @property
    def status(self) -> str:
        if self.is_ambiguous:
            return "ambiguous"
        if self.is_new:
            return "new"
        if self.visitor_id is not None:
            return "returning"
        return "none"

    @property
    def label(self) -> str:
        if self.is_ambiguous:
            return "AMBIGUOUS"
        if self.is_new:
            return "NEW visitor"
        if self.visitor_id is not None:
            sim = self.face_confidence or self.body_confidence or 0.0
            return f"Visitor {str(self.visitor_id)[:8]} ({sim:.2f})"
        return ""


def _crop(frame: np.ndarray, bbox: Optional[dict]) -> Optional[np.ndarray]:
    if frame is None or not bbox:
        return None
    h, w = frame.shape[:2]
    x1 = max(0, min(w, int(bbox.get("x1", 0))))
    y1 = max(0, min(h, int(bbox.get("y1", 0))))
    x2 = max(0, min(w, int(bbox.get("x2", 0))))
    y2 = max(0, min(h, int(bbox.get("y2", 0))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


async def process_detections(
    db: AsyncSession,
    detections: List[DetectedPerson],
    frame: Optional[np.ndarray] = None,
    camera_id: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    frame_path: Optional[str] = None,
) -> List[ProcessedDetection]:
    """Resolve, enroll, visit-track and audit every face-bearing detection."""
    timestamp = timestamp or datetime.now(timezone.utc)
    tracker = VisitTracker.get_instance()

    matchable = [d for d in detections if d.face_embedding]
    if not matchable:
        return []

    # Mark masked detections and build face dicts for the resolver
    threshold_offsets: list[float] = []
    for d in matchable:
        face_crop = _crop(frame, d.face_bbox or d.bbox)
        if settings.MASK_DETECTION_ENABLED and face_crop is not None and _is_masked(face_crop):
            d.is_masked = True
        threshold_offsets.append(masked_threshold_offset() if d.is_masked else 0.0)

    faces = [
        {
            "face_embedding": d.face_embedding,
            "body_embedding": d.body_embedding,
            "det_score": d.face_det_score or 0.0,
            "pose_bin": d.pose.bin.value if d.pose else "unknown",
            "threshold_offset": threshold_offsets[i],
        }
        for i, d in enumerate(matchable)
    ]
    resolutions = await identity_resolver.resolve_batch(faces, db)

    out: List[ProcessedDetection] = []
    for det, res in zip(matchable, resolutions):
        det_score = det.face_det_score or 0.0
        face_crop = _crop(frame, det.face_bbox or det.bbox)

        pd = ProcessedDetection(
            bbox=det.face_bbox or det.bbox,
            is_ambiguous=res.is_ambiguous,
            face_confidence=round(res.face_similarity, 4) if res.face_similarity else None,
            body_confidence=round(res.body_similarity, 4) if res.body_similarity else None,
            match_source=res.match_source,
        )

        if res.is_ambiguous:
            db.add(
                DetectionEvent(
                    detected_at=timestamp,
                    face_similarity=res.face_similarity or None,
                    is_new_visitor=False,
                    is_ambiguous=True,
                    match_source="none",
                    camera_id=camera_id,
                    frame_path=frame_path,
                    bbox=pd.bbox,
                )
            )
            out.append(pd)
            continue

        if res.is_new:
            # Temporal gate: maybe this is a known person who just turned away
            temporal_match = temporal_gate.check(
                new_embedding=det.face_embedding,
                new_bbox=det.face_bbox or det.bbox,
                timestamp=timestamp,
            )
            if temporal_match is not None:
                # Treat as returning rather than creating a new record
                pd.visitor_id = temporal_match
                pd.is_new = False
                res.visitor_id = temporal_match
                res.is_new = False
                res.match_source = "temporal"
                visitor = await db.get(Visitor, temporal_match)
            else:
                visitor = await auto_enroller.register_new_visitor(
                    db,
                    face_embedding=det.face_embedding,
                    det_score=det_score,
                    body_embedding=det.body_embedding,
                    face_crop=face_crop,
                    pose=det.pose,
                )
                pd.visitor_id = visitor.id
                pd.is_new = True
                # Flag low-quality or near-duplicate new registrations
                from app.services.review_queue import maybe_flag_new_visitor
                await maybe_flag_new_visitor(
                    db,
                    visitor_id=visitor.id,
                    det_score=det_score,
                    top_similarity=res.face_similarity,
                )
        elif res.visitor_id is not None:
            pd.visitor_id = res.visitor_id
            visitor = await db.get(Visitor, res.visitor_id)
            if visitor is not None and res.match_source == "face":
                await auto_enroller.update_after_match(
                    db,
                    visitor,
                    face_embedding=det.face_embedding,
                    det_score=det_score,
                    face_similarity=res.face_similarity,
                    body_embedding=det.body_embedding,
                    face_crop=face_crop,
                    pose=det.pose,
                )
        else:
            # Dropped (grey zone, low quality) — record nothing identity-bearing.
            out.append(pd)
            continue

        visit_id, is_new_visit = await tracker.process_detection(
            db,
            visitor_id=pd.visitor_id,
            timestamp=timestamp,
            confidence=det_score,
            camera_id=camera_id,
        )
        pd.visit_id = visit_id

        # Feed confirmed detections into the temporal gate
        if det.face_embedding:
            temporal_gate.add_detection(
                visitor_id=pd.visitor_id,
                embedding=det.face_embedding,
                bbox=det.face_bbox or det.bbox,
                timestamp=timestamp,
                confidence=det_score,
            )

        db.add(
            DetectionEvent(
                visitor_id=pd.visitor_id,
                visit_id=visit_id,
                detected_at=timestamp,
                face_similarity=res.face_similarity or None,
                body_similarity=res.body_similarity or None,
                is_new_visitor=pd.is_new,
                is_ambiguous=False,
                match_source=res.match_source,
                camera_id=camera_id,
                frame_path=frame_path,
                bbox=pd.bbox,
            )
        )
        out.append(pd)

    await db.commit()
    return out
