"""
Cross-camera identity resolution (Phase 4).

The same person walking from camera A to camera B is, today, registered twice:
each camera resolves identity independently and pixel-distance temporal gating
is camera-local. This module reconciles across cameras — conservatively.

Two entry points:
  • find_cross_camera_candidate() — called live before creating a NEW visitor:
    is this "new" face actually someone seen seconds ago on another camera?
  • reconcile_recent_duplicates() — periodic background sweep that merges/flags
    duplicate visitor records that slipped through live.

Everything is gated behind settings.CROSS_CAMERA_ENABLED and uses a high
confidence bar for any automatic action — a duplicate is annoying, a false merge
corrupts a gallery, so medium-confidence cases go to the review queue.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)


async def _topology(
    db: AsyncSession, cam_a: str, cam_b: str
) -> Optional[dict]:
    """Return the transition constraint between two cameras (either direction)."""
    try:
        row = (
            await db.execute(
                text("""
                    SELECT min_travel_seconds, max_expected_seconds, transition_enabled
                    FROM camera_topology
                    WHERE (camera_a = :a AND camera_b = :b)
                       OR (camera_a = :b AND camera_b = :a)
                    LIMIT 1
                """),
                {"a": cam_a, "b": cam_b},
            )
        ).first()
    except Exception:
        return None  # table may not exist yet (pre-migration)
    if row is None:
        return None
    return {
        "min_travel_seconds": row.min_travel_seconds,
        "max_expected_seconds": row.max_expected_seconds,
        "transition_enabled": row.transition_enabled,
    }


async def _last_camera(
    db: AsyncSession, visitor_id: UUID, cutoff: datetime
) -> Optional[tuple[str, datetime]]:
    """The visitor's most recent (camera_id, detected_at) since cutoff, if any."""
    row = (
        await db.execute(
            text("""
                SELECT camera_id, detected_at
                FROM detection_events
                WHERE visitor_id = :v AND camera_id IS NOT NULL
                  AND detected_at >= :cutoff
                ORDER BY detected_at DESC
                LIMIT 1
            """),
            {"v": str(visitor_id), "cutoff": cutoff},
        )
    ).first()
    if row is None:
        return None
    return row.camera_id, row.detected_at


async def find_cross_camera_candidate(
    db: AsyncSession,
    embedding: list,
    camera_id: str,
    timestamp: datetime,
) -> Optional[dict]:
    """
    Before registering a NEW visitor, check whether this face matches someone
    recently seen on a DIFFERENT camera.

    Returns {visitor_id, similarity, decision} where decision is:
      • "auto"   — high confidence + topology-plausible → treat as returning now
      • "review" — medium confidence → register but flag for operator/dedup
      • None when there is no credible cross-camera candidate.
    """
    if not settings.CROSS_CAMERA_ENABLED or not embedding:
        return None

    cutoff = timestamp - timedelta(seconds=settings.CROSS_CAMERA_LOOKBACK_SECONDS)
    try:
        rows = (
            await db.execute(
                text(r"""
                    SELECT vf.visitor_id, MAX(1 - (vf.embedding <=> :emb\:\:vector)) AS sim
                    FROM visitor_faces vf
                    JOIN visitors vis ON vis.id = vf.visitor_id
                    WHERE vis.is_active = TRUE
                      AND vis.consent_status != 'opted_out'
                      AND vis.last_seen_at >= :cutoff
                    GROUP BY vf.visitor_id
                    ORDER BY sim DESC
                    LIMIT 5
                """),
                {"emb": str(embedding), "cutoff": cutoff},
            )
        ).all()
    except Exception as exc:
        logger.debug("cross-camera candidate search skipped: %s", exc)
        return None

    for r in rows:
        sim = float(r.sim)
        if sim < settings.CROSS_CAMERA_REVIEW_THRESHOLD:
            break  # rows are sorted desc — nothing else qualifies

        last = await _last_camera(db, r.visitor_id, cutoff)
        if last is None:
            continue
        last_cam, last_ts = last
        if last_cam == camera_id:
            continue  # same camera → handled by the normal/temporal path

        topo = await _topology(db, last_cam, camera_id)
        if topo is not None:
            if not topo["transition_enabled"]:
                continue  # this transition is physically impossible
            delta = abs((timestamp - last_ts).total_seconds())
            min_travel = topo.get("min_travel_seconds")
            if min_travel is not None and delta < min_travel:
                continue  # too soon to have walked between these cameras

        decision = (
            "auto" if sim >= settings.CROSS_CAMERA_AUTO_THRESHOLD else "review"
        )
        return {"visitor_id": r.visitor_id, "similarity": sim, "decision": decision}

    return None


async def reconcile_recent_duplicates(db: AsyncSession) -> dict:
    """
    Background sweep: find pairs of recently-active visitors whose centroids are
    near-identical (likely the same person split across cameras) and either
    auto-merge (very high confidence) or flag for review. Conservative by design.

    Returns {merged, flagged, scanned}.
    """
    if not settings.CROSS_CAMERA_ENABLED:
        return {"merged": 0, "flagged": 0, "scanned": 0}

    from app.services.review_queue import _insert_flag
    from app.services.visitor_merge import MergeError, merge_visitors

    cutoff = datetime.now(timezone.utc) - timedelta(
        seconds=settings.CROSS_CAMERA_LOOKBACK_SECONDS
    )
    try:
        pairs = (
            await db.execute(
                text(r"""
                    SELECT a.id AS a_id, b.id AS b_id,
                           1 - (a.face_embedding <=> b.face_embedding) AS sim
                    FROM visitors a
                    JOIN visitors b
                      ON b.id > a.id
                    WHERE a.is_active AND b.is_active
                      AND a.consent_status != 'opted_out'
                      AND b.consent_status != 'opted_out'
                      AND a.face_embedding IS NOT NULL
                      AND b.face_embedding IS NOT NULL
                      AND a.last_seen_at >= :cutoff
                      AND b.last_seen_at >= :cutoff
                      AND a.is_staff = FALSE AND b.is_staff = FALSE
                      AND (1 - (a.face_embedding <=> b.face_embedding))
                          >= :review_thr
                    ORDER BY sim DESC
                    LIMIT 100
                """),
                {"cutoff": cutoff, "review_thr": settings.CROSS_CAMERA_REVIEW_THRESHOLD},
            )
        ).all()
    except Exception as exc:
        logger.debug("cross-camera reconcile query skipped: %s", exc)
        return {"merged": 0, "flagged": 0, "scanned": 0}

    merged = 0
    flagged = 0
    gone: set[str] = set()
    for p in pairs:
        a_id, b_id, sim = str(p.a_id), str(p.b_id), float(p.sim)
        if a_id in gone or b_id in gone:
            continue
        if sim >= settings.CROSS_CAMERA_AUTO_MERGE_THRESHOLD:
            try:
                # Merge the newer (b) into the older (a) — b.id > a.id, but ids are
                # random UUIDs; keep target = a for determinism.
                await merge_visitors(
                    db, source_id=p.b_id, target_id=p.a_id,
                    reason="cross_camera_auto", similarity=sim, merged_by="cross_camera",
                )
                gone.add(b_id)
                merged += 1
            except MergeError:
                continue
        else:
            await _insert_flag(
                db,
                visitor_id=p.b_id,
                flag_type="probable_duplicate",
                detail=f"cross_camera_probable_duplicate: sim={sim:.3f}",
                matched_visitor_id=p.a_id,
                similarity=sim,
            )
            flagged += 1

    if flagged:
        await db.commit()
    return {"merged": merged, "flagged": flagged, "scanned": len(pairs)}
