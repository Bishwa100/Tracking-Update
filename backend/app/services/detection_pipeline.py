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
from app.cv_pipeline import DetectedPerson, FacePose, PoseBin, _is_group_frame
from app.geometry import crop_from_frame
from app.ml_models import ModelManager
from app.models import DetectionEvent, Visitor
from app.services import auto_enroller, identity_resolver
from app.services.visit_tracker import VisitTracker
from app.services.temporal_consistency import temporal_gate
from app.services.tracklet import tracklet_buffer
from app.services import cross_camera
from app.services.mask_detector import (
    is_masked as _is_masked,
    masked_threshold_offset,
    extract_periocular_region,
)
from app.utils import normalize_embedding

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


def _better_resolution(
    full: "identity_resolver.ResolutionResult",
    perio: "identity_resolver.ResolutionResult",
) -> "identity_resolver.ResolutionResult":
    """
    Pick the stronger of a masked face's full-face vs periocular resolution.
    A confident returning ("face") match wins over a held/new one; if both (or
    neither) are returning, the higher similarity wins. Ambiguous full-face
    results are kept (don't let a periocular guess override a no-merge decision).
    """
    if full.is_ambiguous:
        return full
    full_face = full.match_source == "face"
    perio_face = perio.match_source == "face"
    if perio_face and not full_face:
        return perio
    if full_face and not perio_face:
        return full
    return perio if perio.face_similarity > full.face_similarity else full


def _crop(frame: np.ndarray, bbox: Optional[dict]) -> Optional[np.ndarray]:
    return crop_from_frame(frame, bbox, min_size=4)


def _pose_allows_registration(pose: Optional[FacePose]) -> bool:
    """
    Whether a face's head-pose is good enough to SEED a brand-new visitor.

    Only gates CREATION of new visitors — matching/recognising existing visitors
    is never restricted by pose. A non-frontal first embedding is the main cause
    of one person being registered as several records, so we wait for a frontal
    view (the tracklet observation safety-valve in process_detections registers a
    profile-only person eventually so nobody is lost).
    """
    policy = settings.REGISTRATION_POSE_POLICY
    if policy == "any":
        return True
    if pose is None:
        # No landmarks → pose unknown; can't confirm it's frontal, so hold and
        # wait for a frame we can judge (the safety-valve covers the rare case
        # where a conforming pose never arrives).
        return False
    if policy == "frontal_or_down":
        return pose.bin in (PoseBin.FRONTAL, PoseBin.DOWNWARD)
    # default / "frontal"
    return pose.bin == PoseBin.FRONTAL


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

    # Crowded, high-overlap scenes are where face↔person assignment is least
    # reliable — be extra conservative about creating NEW visitors there.
    group_frame = _is_group_frame(detections)

    # Mark masked detections and build face dicts for the resolver. For masked
    # faces, also extract a periocular (eye-region) embedding — the lower face is
    # occluded so the full-face embedding is unreliable; the eye region usually
    # matches better.
    threshold_offsets: list[float] = []
    periocular_embeddings: list[Optional[list]] = []
    for d in matchable:
        face_crop = _crop(frame, d.face_bbox or d.bbox)
        perio_emb: Optional[list] = None
        if settings.MASK_DETECTION_ENABLED and face_crop is not None and _is_masked(face_crop):
            d.is_masked = True
            perio_crop = extract_periocular_region(face_crop)
            if perio_crop is not None:
                fd = ModelManager.get_instance().extract_face_data(perio_crop)
                if fd is not None and fd.get("embedding") is not None:
                    perio_emb = normalize_embedding(fd["embedding"])
        threshold_offsets.append(masked_threshold_offset() if d.is_masked else 0.0)
        periocular_embeddings.append(perio_emb)

    # ── Tracklet fast-path ──────────────────────────────────────────────
    # Associate each detection with its per-camera tracklet up front. A tracklet
    # already pinned to a visitor (and not due for re-verify) is attributed
    # DIRECTLY to that visitor without the expensive HNSW gallery search — the big
    # CPU win for stationary/seated patrons on frame 2..N. Everything else still
    # goes through resolve_batch below. The tracklet for each detection is reused
    # later in the loop (so we associate exactly once per detection).
    tracklets: list = [None] * len(matchable)
    fast_visitor: list = [None] * len(matchable)
    if settings.TRACKLET_ENABLED and camera_id is not None:
        for i, d in enumerate(matchable):
            tr = tracklet_buffer.get_or_create(
                camera_id, d.face_bbox or d.bbox, timestamp
            )
            tracklets[i] = tr
            # Masked faces always take the full path: their full-face embedding is
            # unreliable and they get a periocular re-resolve below, so we must not
            # short-circuit them.
            if (
                settings.TRACKLET_FAST_PATH
                and not d.is_masked
                and not tracklet_buffer.needs_reverify(
                    tr, d.face_bbox or d.bbox, timestamp, d.face_embedding
                )
            ):
                fast_visitor[i] = tr.visitor_id

    # Only faces WITHOUT a trusted fast-path pin need the gallery search.
    resolve_idx = [i for i in range(len(matchable)) if fast_visitor[i] is None]
    faces = [
        {
            "face_embedding": matchable[i].face_embedding,
            "body_embedding": matchable[i].body_embedding,
            "det_score": matchable[i].face_det_score or 0.0,
            "pose_bin": matchable[i].pose.bin.value if matchable[i].pose else "unknown",
            "yaw": matchable[i].pose.yaw if matchable[i].pose else None,
            "threshold_offset": threshold_offsets[i],
        }
        for i in resolve_idx
    ]
    resolved = await identity_resolver.resolve_batch(faces, db) if faces else []

    # Re-expand to one resolution per matchable detection, in order. Fast-path
    # detections get a synthetic confident result attributed to the pinned visitor.
    resolutions: list = [None] * len(matchable)
    for j, i in enumerate(resolve_idx):
        resolutions[i] = resolved[j]
    for i in range(len(matchable)):
        if fast_visitor[i] is not None:
            resolutions[i] = identity_resolver.ResolutionResult(
                visitor_id=fast_visitor[i],
                is_new=False,
                face_similarity=0.0,
                match_source="tracklet_fast",
                top_match_id=fast_visitor[i],
            )

    # Resolve the periocular embeddings for masked faces and keep whichever of
    # {full-face, periocular} gives the stronger result.
    perio_idx = [i for i, e in enumerate(periocular_embeddings) if e is not None]
    if perio_idx:
        perio_faces = [
            {
                "face_embedding": periocular_embeddings[i],
                "body_embedding": matchable[i].body_embedding,
                "det_score": matchable[i].face_det_score or 0.0,
                "pose_bin": "unknown",
                "threshold_offset": threshold_offsets[i],
            }
            for i in perio_idx
        ]
        perio_res = await identity_resolver.resolve_batch(perio_faces, db)
        for i, pres in zip(perio_idx, perio_res):
            resolutions[i] = _better_resolution(resolutions[i], pres)

    out: List[ProcessedDetection] = []
    for i, (det, res) in enumerate(zip(matchable, resolutions)):
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

        # Reuse the tracklet associated up front (the fast-path block already
        # called get_or_create once per detection — don't double-count observations).
        tracklet = tracklets[i]

        if res.match_source == "tracklet_fast" and res.visitor_id is not None:
            # Fast-path: a pinned, not-yet-due-for-reverify tracklet. Attribute
            # directly to the visitor WITHOUT the gallery search or gallery growth
            # (we didn't search, and the verified frame already grew it). Still
            # heartbeats the visit + writes an audit event below.
            pd.visitor_id = res.visitor_id

        elif res.match_source == "face" and res.visitor_id is not None:
            # Confident returning match — attribute + grow gallery/centroid.
            pd.visitor_id = res.visitor_id
            visitor = await db.get(Visitor, res.visitor_id)
            if visitor is not None:
                await auto_enroller.update_after_match(
                    db, visitor,
                    face_embedding=det.face_embedding, det_score=det_score,
                    face_similarity=res.face_similarity,
                    body_embedding=det.body_embedding, face_crop=face_crop,
                    pose=det.pose, camera_id=camera_id,
                )
            if tracklet is not None:
                # Record this as the verified pin so later frames can fast-path.
                tracklet_buffer.mark_resolved(
                    tracklet, res.visitor_id,
                    verified_ts=timestamp, verified_embedding=det.face_embedding,
                )

        elif res.match_source == "body" and res.visitor_id is not None:
            # Same-session body fallback — attribute only (clothing-dependent).
            pd.visitor_id = res.visitor_id
            if tracklet is not None:
                tracklet_buffer.mark_resolved(tracklet, res.visitor_id)

        elif res.is_new or res.match_source == "grey_zone":
            # This tracklet already resolved to a visitor on an earlier frame →
            # attach instead of re-resolving (prevents a duplicate record).
            if tracklet is not None and tracklet.visitor_id is not None:
                visitor = await db.get(Visitor, tracklet.visitor_id)
                if visitor is not None:
                    pd.visitor_id = visitor.id
                    res.visitor_id = visitor.id
                    res.is_new = False
                    res.match_source = "tracklet"
                    await auto_enroller.update_after_match(
                        db, visitor,
                        face_embedding=det.face_embedding, det_score=det_score,
                        face_similarity=res.face_similarity,
                        body_embedding=det.body_embedding, face_crop=face_crop,
                        pose=det.pose, match_source="temporal", camera_id=camera_id,
                    )
                else:
                    tracklet_buffer.clear_visitor(tracklet.visitor_id)
                    tracklet = tracklet_buffer.get_or_create(
                        camera_id, det.face_bbox or det.bbox, timestamp
                    )

            if pd.visitor_id is None:
                # Temporal gate: a known person who just turned away/reappeared.
                temporal_match = temporal_gate.check(
                    new_embedding=det.face_embedding,
                    new_bbox=det.face_bbox or det.bbox,
                    timestamp=timestamp,
                    camera_id=camera_id,
                )
                if temporal_match is not None:
                    pd.visitor_id = temporal_match
                    res.visitor_id = temporal_match
                    res.is_new = False
                    res.match_source = "temporal"
                    visitor = await db.get(Visitor, temporal_match)
                    if visitor is not None:
                        await auto_enroller.update_after_match(
                            db, visitor,
                            face_embedding=det.face_embedding, det_score=det_score,
                            face_similarity=res.face_similarity,
                            body_embedding=det.body_embedding, face_crop=face_crop,
                            pose=det.pose, match_source="temporal", camera_id=camera_id,
                        )
                    if tracklet is not None:
                        tracklet_buffer.mark_resolved(tracklet, temporal_match)

            # Cross-camera: maybe this "new" face is someone seen seconds ago on
            # another camera (flag-gated; conservative thresholds, inert unless
            # CROSS_CAMERA_ENABLED).
            cross = None
            if pd.visitor_id is None and camera_id is not None and det.face_embedding:
                cross = await cross_camera.find_cross_camera_candidate(
                    db, det.face_embedding, camera_id, timestamp
                )
                if cross is not None and cross["decision"] == "auto":
                    visitor = await db.get(Visitor, cross["visitor_id"])
                    if visitor is not None:
                        pd.visitor_id = visitor.id
                        res.visitor_id = visitor.id
                        res.is_new = False
                        res.match_source = "cross_camera"
                        res.face_similarity = cross["similarity"]
                        await auto_enroller.update_after_match(
                            db, visitor,
                            face_embedding=det.face_embedding, det_score=det_score,
                            face_similarity=res.face_similarity,
                            body_embedding=det.body_embedding, face_crop=face_crop,
                            pose=det.pose, match_source="temporal", camera_id=camera_id,
                        )
                        if tracklet is not None:
                            tracklet_buffer.mark_resolved(tracklet, visitor.id)

            if pd.visitor_id is None:
                # Decide whether to register NOW or HOLD for more frames.
                confident_stranger = res.is_new and res.match_source == "new"
                enough_obs = (
                    tracklet is not None
                    and tracklet.observations >= settings.TRACKLET_MIN_OBSERVATIONS_NEW
                )
                if tracklet is None:
                    allow_register = True               # tracklet disabled → legacy
                elif confident_stranger and not group_frame:
                    allow_register = True               # clear stranger, uncrowded
                else:
                    allow_register = enough_obs          # grey-zone / crowded → persist first

                # Pose gate: don't SEED a new visitor from a profile / steep-angle
                # face — that weak first embedding is the main cause of one person
                # becoming several records. A tracklet seen enough times without
                # ever presenting a conforming pose registers anyway (safety valve)
                # so a profile-only person isn't lost. Matching existing visitors
                # above is never gated by pose.
                held_source = res.match_source or "grey_zone"
                if allow_register and not _pose_allows_registration(det.pose):
                    obs = tracklet.observations if tracklet is not None else 0
                    valve = settings.REGISTRATION_POSE_FALLBACK_OBSERVATIONS
                    if not (valve > 0 and obs >= valve):
                        allow_register = False
                        held_source = "pose_hold"

                if not allow_register:
                    # HELD — record an audit event (measurable hold rate) and do
                    # not register; later frames may confirm or turn frontal.
                    db.add(
                        DetectionEvent(
                            detected_at=timestamp,
                            face_similarity=res.face_similarity or None,
                            is_new_visitor=False,
                            is_ambiguous=False,
                            match_source=held_source,
                            camera_id=camera_id,
                            frame_path=frame_path,
                            bbox=pd.bbox,
                        )
                    )
                    out.append(pd)
                    continue

                visitor = await auto_enroller.register_new_visitor(
                    db,
                    face_embedding=det.face_embedding,
                    det_score=det_score,
                    body_embedding=det.body_embedding,
                    face_crop=face_crop,
                    pose=det.pose,
                    camera_id=camera_id,
                )
                pd.visitor_id = visitor.id
                pd.is_new = True
                res.is_new = True
                res.match_source = "new"
                if tracklet is not None:
                    tracklet_buffer.mark_resolved(tracklet, visitor.id)
                from app.services.review_queue import maybe_flag_new_visitor
                await maybe_flag_new_visitor(
                    db,
                    visitor_id=visitor.id,
                    det_score=det_score,
                    top_similarity=res.face_similarity,
                    top_match_id=res.top_match_id,
                )
                # Medium-confidence cross-camera candidate → flag the new record
                # as a probable duplicate of the other-camera visitor for the
                # dedup sweep / operator to merge.
                if cross is not None and cross["decision"] == "review":
                    from app.services.review_queue import _insert_flag
                    await _insert_flag(
                        db,
                        visitor_id=visitor.id,
                        flag_type="probable_duplicate",
                        detail=f"cross_camera_probable_duplicate: sim={cross['similarity']:.3f}",
                        matched_visitor_id=cross["visitor_id"],
                        similarity=cross["similarity"],
                    )
        else:
            # Nothing identity-bearing (e.g. low-quality drop).
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
                camera_id=camera_id,
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
