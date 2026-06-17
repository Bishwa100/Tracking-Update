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
    top_match_id: Optional[UUID] = None,
) -> None:
    """Call after registering a new visitor. Flags low-quality or near-duplicate."""
    reason: Optional[str] = None
    matched_visitor_id: Optional[UUID] = None
    similarity: Optional[float] = None

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
        # Record WHICH known visitor this duplicate resembled, and how closely.
        matched_visitor_id = top_match_id
        similarity = top_similarity

    if reason:
        await _insert_flag(
            db,
            visitor_id=visitor_id,
            flag_type=reason.split(":")[0],
            detail=reason,
            matched_visitor_id=matched_visitor_id,
            similarity=similarity,
        )


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
    matched_visitor_id: Optional[UUID] = None,
    similarity: Optional[float] = None,
) -> None:
    try:
        await db.execute(
            text("""
                INSERT INTO review_queue
                    (visitor_id, flag_type, detail, matched_visitor_id, similarity, created_at, resolved)
                VALUES (:vid, :ftype, :detail, :mvid, :sim, :now, FALSE)
                ON CONFLICT DO NOTHING
            """),
            {
                "vid": str(visitor_id),
                "ftype": flag_type,
                "detail": detail,
                "mvid": str(matched_visitor_id) if matched_visitor_id else None,
                "sim": similarity,
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
            SELECT rq.id, rq.visitor_id, rq.flag_type, rq.detail,
                   rq.matched_visitor_id, rq.similarity, rq.created_at,
                   mv.name AS matched_visitor_name
            FROM review_queue rq
            LEFT JOIN visitors mv ON mv.id = rq.matched_visitor_id
            WHERE rq.resolved = FALSE
            ORDER BY rq.created_at DESC
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
            "matched_visitor_id": str(r.matched_visitor_id) if r.matched_visitor_id else None,
            "matched_visitor_name": r.matched_visitor_name,
            "similarity": float(r.similarity) if r.similarity is not None else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def auto_merge_duplicates(db: AsyncSession, limit: int = 200) -> dict:
    """
    Global one-click dedup: merge every unresolved `probable_duplicate` into the
    existing visitor it resembles (highest-confidence first), collapsing each pair
    into a single user. Merging the source visitor cascade-deletes its own flag, so
    only un-mergeable flags (missing/already-merged target) are explicitly resolved.

    Returns {merged, skipped, total}.
    """
    from app.services.visitor_merge import MergeError, merge_visitors

    try:
        rows = (await db.execute(text("""
            SELECT id, visitor_id, matched_visitor_id, similarity
            FROM review_queue
            WHERE resolved = FALSE
              AND flag_type = 'probable_duplicate'
              AND matched_visitor_id IS NOT NULL
            ORDER BY similarity DESC NULLS LAST
            LIMIT :lim
        """), {"lim": limit})).all()
    except Exception as exc:
        logger.error("auto_merge_duplicates query failed: %s", exc)
        return {"merged": 0, "skipped": 0, "total": 0}

    merged = 0
    skipped = 0
    gone: set[str] = set()  # sources already merged away this sweep

    for r in rows:
        target = str(r.matched_visitor_id)
        # Target was itself merged away, or a self-reference — can't merge.
        if target in gone or r.visitor_id == r.matched_visitor_id:
            await resolve_flag(db, r.id)
            skipped += 1
            continue
        try:
            await merge_visitors(db, r.visitor_id, r.matched_visitor_id)
        except MergeError:
            await resolve_flag(db, r.id)
            skipped += 1
            continue
        gone.add(str(r.visitor_id))
        merged += 1

    return {"merged": merged, "skipped": skipped, "total": len(rows)}


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
