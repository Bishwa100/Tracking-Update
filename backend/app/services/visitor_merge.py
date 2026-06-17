"""
Visitor merge — fold one visitor's history into another and delete the source.

Shared by the manual admin merge endpoint and the review-queue "auto-merge
duplicates" sweep so both behave identically (re-point child rows, recompute the
target's aggregates, evict the source from the live tracker).
"""

import logging
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DetectionEvent, Visit, Visitor, VisitorFace
from app.services.visit_tracker import VisitTracker

logger = logging.getLogger(__name__)


class MergeError(Exception):
    """Raised when a merge cannot proceed (self-merge or missing visitor)."""


async def merge_visitors(
    db: AsyncSession,
    source_id: UUID,
    target_id: UUID,
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

    # Drop the merged visitor from the in-memory tracker if present.
    tracker = VisitTracker.get_instance()
    tracker.active_visits.pop(source_id, None)

    await db.delete(source)
    await db.commit()

    logger.info("Merged visitor %s into %s.", source_id, target_id)
    return {"merged_into": str(target_id), "merged_visits": merged_visits or 0}
