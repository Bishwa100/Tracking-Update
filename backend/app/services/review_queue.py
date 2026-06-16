"""
Human review queue.

Auto-flags suspicious new registrations and probable gallery duplicates so an
operator can confirm or merge them rather than letting errors compound silently.

Flag triggers
─────────────
• new_low_quality   — new visitor registered with face det_score < threshold
• probable_duplicate — new visitor's best embedding similarity (against all
                       galleries) is suspiciously close (just below NEW_VISITOR_MAX_SIMILARITY)
• high_ambiguity    — ambiguous match rate for a visitor exceeds 20% of detections
• opted_out_match   — a detection matched a visitor who has since opted out

Queue entries land in the `review_queue` DB table (created by migration 006).
"""

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

_QUALITY_THRESHOLD = 0.50     # flag new visitors whose best face score < this
_NEAR_THRESHOLD_MARGIN = 0.05  # flag when top similarity is within this of rejection


async def maybe_flag_new_visitor(
    db: AsyncSession,
    visitor_id: UUID,
    det_score: float,
    top_similarity: Optional[float],
) -> None:
    """Call after registering a new visitor. Flags low-quality or near-duplicate."""
    reason: Optional[str] = None

    if det_score < _QUALITY_THRESHOLD:
        reason = f"new_low_quality: det_score={det_score:.3f} < {_QUALITY_THRESHOLD}"
    elif (
        top_similarity is not None
        and top_similarity >= settings.NEW_VISITOR_MAX_SIMILARITY - _NEAR_THRESHOLD_MARGIN
    ):
        reason = (
            f"probable_duplicate: top_similarity={top_similarity:.3f} near "
            f"threshold={settings.NEW_VISITOR_MAX_SIMILARITY}"
        )

    if reason:
        await _insert_flag(db, visitor_id=visitor_id, flag_type=reason.split(":")[0], detail=reason)


async def flag_ambiguous_visitor(
    db: AsyncSession,
    visitor_id: UUID,
) -> None:
    """Call when ambiguous match rate for a visitor is anomalously high."""
    reason = "high_ambiguity: ambiguous detection rate > 20%"
    await _insert_flag(db, visitor_id=visitor_id, flag_type="high_ambiguity", detail=reason)


async def flag_opted_out_match(
    db: AsyncSession,
    visitor_id: UUID,
    detected_at: datetime,
) -> None:
    """Call when a detection matched a visitor who has opted out."""
    reason = f"opted_out_match: detection at {detected_at.isoformat()} matched opted-out visitor"
    await _insert_flag(db, visitor_id=visitor_id, flag_type="opted_out_match", detail=reason)


async def _insert_flag(
    db: AsyncSession,
    visitor_id: UUID,
    flag_type: str,
    detail: str,
) -> None:
    try:
        await db.execute(
            text("""
                INSERT INTO review_queue (visitor_id, flag_type, detail, created_at, resolved)
                VALUES (:vid, :ftype, :detail, :now, FALSE)
                ON CONFLICT DO NOTHING
            """),
            {
                "vid": str(visitor_id),
                "ftype": flag_type,
                "detail": detail,
                "now": datetime.now(timezone.utc),
            },
        )
        logger.info("Review flag [%s] queued for visitor %s.", flag_type, visitor_id)
    except Exception as exc:
        # Table may not exist yet (pre-migration) — log but don't crash
        logger.debug("review_queue insert skipped: %s", exc)


async def get_pending_flags(db: AsyncSession, limit: int = 50) -> list[dict]:
    """Return unresolved flags for the admin UI."""
    try:
        rows = (await db.execute(text("""
            SELECT id, visitor_id, flag_type, detail, created_at
            FROM review_queue
            WHERE resolved = FALSE
            ORDER BY created_at DESC
            LIMIT :lim
        """), {"lim": limit})).all()
    except Exception:
        return []

    return [
        {
            "id": str(r.id),
            "visitor_id": str(r.visitor_id),
            "flag_type": r.flag_type,
            "detail": r.detail,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def resolve_flag(db: AsyncSession, flag_id: UUID) -> bool:
    """Mark a review flag as resolved."""
    try:
        await db.execute(
            text("""
                UPDATE review_queue
                SET resolved = TRUE, resolved_at = NOW()
                WHERE id = :fid
            """),
            {"fid": str(flag_id)},
        )
        await db.commit()
        return True
    except Exception as exc:
        logger.error("Could not resolve flag %s: %s", flag_id, exc)
        return False
