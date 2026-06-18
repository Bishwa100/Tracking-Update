"""
Visitor merge — fold one visitor's history into another and delete the source.

Shared by the manual admin merge endpoint and the review-queue "auto-merge
duplicates" sweep so both behave identically (re-point child rows, recompute the
target's aggregates, evict the source from the live tracker).
"""

import logging
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import (
    DetectionEvent, Visit, Visitor, VisitorFace, VisitorMergeAudit,
)
from app.services.visit_tracker import VisitTracker

logger = logging.getLogger(__name__)


class MergeError(Exception):
    """Raised when a merge cannot proceed (self-merge or missing visitor)."""


async def _trim_gallery(db: AsyncSession, visitor_id: UUID) -> int:
    """After a merge the pooled gallery can exceed the cap — drop the lowest
    det_score faces back down to MAX_FACES_PER_VISITOR. Returns count removed."""
    from app.services.auto_enroller import _delete_gallery_face

    faces = (
        await db.execute(
            select(VisitorFace).where(VisitorFace.visitor_id == visitor_id)
        )
    ).scalars().all()
    cap = settings.MAX_FACES_PER_VISITOR
    if len(faces) <= cap:
        return 0
    faces.sort(key=lambda f: (f.det_score or 0.0), reverse=True)
    removed = 0
    for f in faces[cap:]:
        await _delete_gallery_face(db, f)
        removed += 1
    return removed


async def merge_visitors(
    db: AsyncSession,
    source_id: UUID,
    target_id: UUID,
    reason: str = "manual",
    similarity: Optional[float] = None,
    merged_by: str = "manual",
) -> dict:
    """
    Merge `source_id` INTO `target_id`, then delete the source. Commits the tx.

    Raises MergeError on a self-merge or when either visitor is missing.
    """
    if source_id == target_id:
        raise MergeError("Cannot merge a visitor into itself.")

    source = await db.get(Visitor, source_id)
    target = await db.get(Visitor, target_id)
    if source is None or target is None:
        raise MergeError("Source or target visitor not found.")

    # Re-point all child rows to the target.
    for model, col in (
        (Visit, Visit.visitor_id),
        (VisitorFace, VisitorFace.visitor_id),
        (DetectionEvent, DetectionEvent.visitor_id),
    ):
        await db.execute(
            update(model).where(col == source_id).values(visitor_id=target_id)
        )

    merged_visits = await db.scalar(
        select(func.count(Visit.id)).where(Visit.visitor_id == target_id)
    )

    # Recompute target aggregates.
    target.visit_count = (target.visit_count or 0) + (source.visit_count or 0)
    if source.first_seen_at and (
        not target.first_seen_at or source.first_seen_at < target.first_seen_at
    ):
        target.first_seen_at = source.first_seen_at
    if source.last_seen_at and (
        not target.last_seen_at or source.last_seen_at > target.last_seen_at
    ):
        target.last_seen_at = source.last_seen_at
    target.total_faces_recorded = (
        (target.total_faces_recorded or 0) + (source.total_faces_recorded or 0)
    )

    # Pooled gallery may exceed the cap — trim before recomputing the centroid.
    try:
        trimmed = await _trim_gallery(db, target_id)
    except Exception as exc:
        trimmed = 0
        logger.warning("Gallery trim after merge %s→%s failed (%s).", source_id, target_id, exc)

    # Rebuild the centroid (+ adaptive thresholds) from the now-pooled gallery so
    # it reflects every face, not the target's stale adaptive average. Done
    # before the source delete; the re-pointed faces are already visible here.
    try:
        from app.services.auto_enroller import (
            recompute_centroid_from_gallery, recompute_adaptive_thresholds,
        )
        await recompute_centroid_from_gallery(db, target)
        await recompute_adaptive_thresholds(db, target)
    except Exception as exc:
        logger.warning(
            "Centroid recompute after merge %s→%s failed (%s) — keeping existing.",
            source_id, target_id, exc,
        )

    # Drop the merged visitor from the in-memory tracker + temporal gate +
    # tracklet pins so new detections never match a now-deleted visitor.
    tracker = VisitTracker.get_instance()
    tracker.active_visits.pop(source_id, None)
    try:
        from app.services.temporal_consistency import temporal_gate
        from app.services.tracklet import tracklet_buffer
        temporal_gate.clear_visitor(source_id)
        tracklet_buffer.clear_visitor(source_id)
    except Exception:
        pass

    # Append an audit row before the cascade delete removes the source.
    db.add(
        VisitorMergeAudit(
            id=uuid4(),
            source_visitor_id=source_id,
            target_visitor_id=target_id,
            reason=reason,
            similarity=similarity,
            merged_by=merged_by,
        )
    )

    await db.delete(source)
    await db.commit()

    logger.info("Merged visitor %s into %s (%s).", source_id, target_id, reason)
    return {
        "merged_into": str(target_id),
        "merged_visits": merged_visits or 0,
        "gallery_trimmed": trimmed,
    }
