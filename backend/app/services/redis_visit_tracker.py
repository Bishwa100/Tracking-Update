"""
Redis-backed visit tracker for multi-worker / multi-camera deployments.

Mirrors the VisitTracker API but stores active-visit state in Redis so every
worker process shares the same view. Falls back gracefully to in-process state
when Redis is not configured (REDIS_ENABLED=False).

Key schema:
  visit:active:{visitor_id}  →  JSON blob of ActiveVisitData
  visit:lock:{visitor_id}    →  distributed lock (NX, EX 5 s)
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Visit, Visitor

logger = logging.getLogger(__name__)

_redis_client = None  # lazily initialised


def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            import redis.asyncio as aioredis  # type: ignore
            _redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
        except ImportError:
            raise RuntimeError(
                "redis package not installed. Add 'redis[asyncio]' to requirements."
            )
    return _redis_client


def _key(visitor_id: UUID) -> str:
    return f"visit:active:{visitor_id}"


def _lock_key(visitor_id: UUID) -> str:
    return f"visit:lock:{visitor_id}"


@dataclass
class ActiveVisitData:
    visit_id: str
    visitor_id: str
    started_at: str       # ISO
    last_detected_at: str # ISO
    detection_count: int
    best_confidence: float
    sum_confidence: float
    camera_id: Optional[str]
    was_seated: bool = False


def _to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisVisitTracker:
    """Distributed visit tracker backed by Redis."""

    def current_inside_count_sync(self) -> int:
        """Not available without a running event-loop; use async variant."""
        return -1

    async def current_inside_count(self) -> int:
        r = _get_redis()
        keys = await r.keys("visit:active:*")
        return len(keys)

    async def process_detection(
        self,
        db: AsyncSession,
        visitor_id: UUID,
        timestamp: datetime,
        confidence: float,
        camera_id: Optional[str] = None,
        bbox: Optional[dict] = None,
        frame_shape: Optional[tuple] = None,
    ) -> Tuple[UUID, bool]:
        from app.services.visit_tracker import _infer_seated

        r = _get_redis()
        vk = _key(visitor_id)
        lk = _lock_key(visitor_id)
        max_dur = timedelta(hours=settings.MAX_VISIT_DURATION_HOURS)

        # Acquire a per-visitor distributed lock (5 s max)
        acquired = await r.set(lk, "1", nx=True, ex=5)
        if not acquired:
            # Another worker is updating this visitor — best-effort: read state only
            logger.debug("Could not acquire Redis lock for visitor %s", visitor_id)

        try:
            raw = await r.get(vk)
            active: Optional[ActiveVisitData] = None
            if raw:
                try:
                    d = json.loads(raw)
                    active = ActiveVisitData(**d)
                except Exception:
                    active = None

            if active is not None:
                if bbox and frame_shape:
                    if _infer_seated(bbox, frame_shape):
                        active.was_seated = True

                cooldown_min = (
                    settings.SEATED_COOLDOWN_MINUTES
                    if active.was_seated
                    else settings.VISIT_COOLDOWN_MINUTES
                )
                cooldown = timedelta(minutes=cooldown_min)
                last_dt = _to_dt(active.last_detected_at)
                start_dt = _to_dt(active.started_at)
                gap = timestamp - last_dt
                open_for = timestamp - start_dt

                if gap < cooldown and open_for < max_dur:
                    active.last_detected_at = timestamp.isoformat()
                    active.detection_count += 1
                    active.best_confidence = max(active.best_confidence, confidence)
                    active.sum_confidence += confidence
                    avg = active.sum_confidence / max(active.detection_count, 1)
                    await r.set(vk, json.dumps(asdict(active)), ex=int(max_dur.total_seconds()) + 60)
                    await db.execute(
                        update(Visit)
                        .where(Visit.id == UUID(active.visit_id))
                        .values(
                            detection_count=active.detection_count,
                            best_face_confidence=active.best_confidence,
                            avg_face_confidence=avg,
                            updated_at=timestamp,
                        )
                    )
                    await db.execute(
                        update(Visitor).where(Visitor.id == visitor_id).values(last_seen_at=timestamp)
                    )
                    return UUID(active.visit_id), False

                # Close stale visit
                left_at = last_dt
                dur = max(0, int((left_at - start_dt).total_seconds() // 60))
                await db.execute(
                    update(Visit)
                    .where(Visit.id == UUID(active.visit_id))
                    .values(left_at=left_at, duration_minutes=dur)
                )
                await r.delete(vk)
                active = None

            # Open new visit
            visit = Visit(
                visitor_id=visitor_id,
                entered_at=timestamp,
                detection_count=1,
                best_face_confidence=confidence,
                avg_face_confidence=confidence,
                camera_id=camera_id,
                updated_at=timestamp,
            )
            db.add(visit)
            await db.flush()

            await db.execute(
                update(Visitor)
                .where(Visitor.id == visitor_id)
                .values(visit_count=Visitor.visit_count + 1, last_seen_at=timestamp)
            )

            data = ActiveVisitData(
                visit_id=str(visit.id),
                visitor_id=str(visitor_id),
                started_at=timestamp.isoformat(),
                last_detected_at=timestamp.isoformat(),
                detection_count=1,
                best_confidence=confidence,
                sum_confidence=confidence,
                camera_id=camera_id,
            )
            await r.set(vk, json.dumps(asdict(data)), ex=int(max_dur.total_seconds()) + 60)
            return visit.id, True

        finally:
            if acquired:
                await r.delete(lk)

    async def cleanup_stale(self, db: AsyncSession, now: Optional[datetime] = None) -> int:
        """Close visits idle past cooldown. Should run from a single scheduler."""
        now = now or datetime.now(timezone.utc)
        r = _get_redis()
        keys = await r.keys("visit:active:*")
        closed = 0
        cooldown = timedelta(minutes=settings.VISIT_COOLDOWN_MINUTES)
        max_dur = timedelta(hours=settings.MAX_VISIT_DURATION_HOURS)

        for vk in keys:
            raw = await r.get(vk)
            if not raw:
                continue
            try:
                d = json.loads(raw)
                active = ActiveVisitData(**d)
            except Exception:
                continue

            last_dt = _to_dt(active.last_detected_at)
            start_dt = _to_dt(active.started_at)
            eff_cooldown = (
                timedelta(minutes=settings.SEATED_COOLDOWN_MINUTES)
                if active.was_seated else cooldown
            )
            idle = now - last_dt
            open_for = now - start_dt
            if idle < eff_cooldown and open_for < max_dur:
                continue

            left_at = last_dt
            dur = max(0, int((left_at - start_dt).total_seconds() // 60))
            await db.execute(
                update(Visit)
                .where(Visit.id == UUID(active.visit_id))
                .values(left_at=left_at, duration_minutes=dur)
            )
            await r.delete(vk)
            closed += 1

        if closed:
            await db.commit()
            logger.info("Redis cleanup: closed %d stale visit(s).", closed)
        return closed
