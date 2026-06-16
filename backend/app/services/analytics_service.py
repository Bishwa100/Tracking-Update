"""
Analytics query builders. Staff and soft-deleted visitors are excluded.
Confidence-weighted metrics discount low-quality detections so false registrations
don't inflate unique-visitor counts.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings


def _range(since: Optional[datetime], until: Optional[datetime]) -> tuple[datetime, datetime]:
    until = until or datetime.now(timezone.utc)
    since = since or (until - timedelta(days=settings.ANALYTICS_DEFAULT_DAYS))
    return since, until


async def summary(db: AsyncSession, since: Optional[datetime], until: Optional[datetime]) -> dict:
    since, until = _range(since, until)
    params = {"since": since, "until": until}

    row = (await db.execute(text("""
        WITH period_visits AS (
            SELECT v.visitor_id, v.duration_minutes, vis.first_seen_at
            FROM visits v
            JOIN visitors vis ON vis.id = v.visitor_id
            WHERE v.entered_at >= :since AND v.entered_at < :until
              AND vis.is_staff = FALSE AND vis.is_active = TRUE
        )
        SELECT
            COUNT(*) AS total_visits,
            COUNT(DISTINCT visitor_id) AS unique_visitors,
            COUNT(DISTINCT visitor_id) FILTER (WHERE first_seen_at >= :since) AS new_visitors,
            COALESCE(AVG(duration_minutes) FILTER (WHERE duration_minutes IS NOT NULL), 0) AS avg_duration
        FROM period_visits
    """), params)).one()

    unique = row.unique_visitors or 0
    new = row.new_visitors or 0
    returning = max(0, unique - new)

    by_day = (await db.execute(text("""
        SELECT date_trunc('day', v.entered_at) AS day, COUNT(*) AS visits
        FROM visits v
        JOIN visitors vis ON vis.id = v.visitor_id
        WHERE v.entered_at >= :since AND v.entered_at < :until
          AND vis.is_staff = FALSE AND vis.is_active = TRUE
        GROUP BY day ORDER BY day
    """), params)).all()

    return {
        "total_unique_visitors": unique,
        "total_visits": row.total_visits or 0,
        "new_visitors": new,
        "returning_visitors": returning,
        "average_duration_minutes": round(float(row.avg_duration or 0), 1),
        "return_rate": round(returning / unique, 4) if unique else 0.0,
        "visits_by_day": [
            {"day": r.day.date().isoformat(), "visits": r.visits} for r in by_day
        ],
    }


async def frequency(db: AsyncSession) -> dict:
    rows = (await db.execute(text("""
        SELECT visit_count, COUNT(*) AS n
        FROM visitors
        WHERE is_staff = FALSE AND is_active = TRUE AND visit_count > 0
        GROUP BY visit_count
    """))).all()

    dist = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for r in rows:
        vc = r.visit_count
        key = str(vc) if vc <= 3 else "4+"
        dist[key] += r.n
    return {"distribution": dist}


async def hourly(db: AsyncSession, since: Optional[datetime], until: Optional[datetime]) -> dict:
    since, until = _range(since, until)
    rows = (await db.execute(text("""
        SELECT CAST(EXTRACT(HOUR FROM v.entered_at) AS INTEGER) AS hour,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE NOT EXISTS (
                   SELECT 1 FROM visits v2
                   WHERE v2.visitor_id = v.visitor_id AND v2.entered_at < v.entered_at
               )) AS new_count
        FROM visits v
        JOIN visitors vis ON vis.id = v.visitor_id
        WHERE v.entered_at >= :since AND v.entered_at < :until
          AND vis.is_staff = FALSE AND vis.is_active = TRUE
        GROUP BY hour ORDER BY hour
    """), {"since": since, "until": until})).all()

    by_hour = {r.hour: (r.total, r.new_count) for r in rows}
    return {
        "hourly": [
            {
                "hour": h,
                "new": by_hour.get(h, (0, 0))[1],
                "returning": by_hour.get(h, (0, 0))[0] - by_hour.get(h, (0, 0))[1],
            }
            for h in range(24)
        ]
    }


async def top_visitors(db: AsyncSession, limit: int = 10) -> list[dict]:
    rows = (await db.execute(text("""
        SELECT vis.id, vis.name, vis.visit_count, vis.first_seen_at, vis.last_seen_at,
               AVG(v.duration_minutes) AS avg_dur
        FROM visitors vis
        LEFT JOIN visits v ON v.visitor_id = vis.id
        WHERE vis.is_staff = FALSE AND vis.is_active = TRUE AND vis.visit_count > 0
        GROUP BY vis.id
        ORDER BY vis.visit_count DESC, vis.last_seen_at DESC
        LIMIT :limit
    """), {"limit": max(1, limit)})).all()

    return [
        {
            "visitor_id": r.id,
            "name": r.name,
            "visit_count": r.visit_count,
            "first_visit": r.first_seen_at,
            "last_visit": r.last_seen_at,
            "avg_duration_minutes": round(float(r.avg_dur), 1) if r.avg_dur is not None else None,
        }
        for r in rows
    ]


# ── Confidence-weighted analytics ────────────────────────────────────────────

async def confidence_weighted_summary(
    db: AsyncSession,
    since: Optional[datetime],
    until: Optional[datetime],
    min_confidence: float = 0.40,
) -> dict:
    """
    Like summary() but weights each detection by face_similarity so that
    low-confidence detections contribute fractionally rather than equally.
    The effective_unique count is a confidence-weighted head count and is
    always ≤ the raw unique count.
    """
    since, until = _range(since, until)
    params = {"since": since, "until": until, "min_conf": min_confidence}

    row = (await db.execute(text("""
        WITH weighted AS (
            SELECT
                de.visitor_id,
                COALESCE(de.face_similarity, 0.5) AS sim
            FROM detection_events de
            JOIN visitors vis ON vis.id = de.visitor_id
            WHERE de.detected_at >= :since AND de.detected_at < :until
              AND vis.is_staff = FALSE AND vis.is_active = TRUE
              AND COALESCE(de.face_similarity, 0) >= :min_conf
              AND de.is_ambiguous = FALSE
        ),
        per_visitor AS (
            SELECT visitor_id, MAX(sim) AS max_sim
            FROM weighted
            GROUP BY visitor_id
        )
        SELECT
            COUNT(*) AS unique_count,
            COALESCE(SUM(max_sim), 0) AS effective_unique,
            COALESCE(AVG(max_sim), 0) AS avg_confidence
        FROM per_visitor
    """), params)).one()

    # Also pull raw numbers for comparison
    raw = await summary(db, since, until)
    raw["confidence_weighted"] = {
        "unique_visitors": int(row.unique_count or 0),
        "effective_unique": round(float(row.effective_unique or 0), 1),
        "avg_confidence": round(float(row.avg_confidence or 0), 4),
        "min_confidence_filter": min_confidence,
    }
    return raw


async def detection_quality_report(
    db: AsyncSession,
    since: Optional[datetime],
    until: Optional[datetime],
) -> dict:
    """
    Breakdown of detection quality bands to surface systematic issues.
    Bands: high (≥0.65), medium (0.45–0.65), low (<0.45 or null).
    """
    since, until = _range(since, until)
    rows = (await db.execute(text("""
        SELECT
            CASE
                WHEN face_similarity >= 0.65 THEN 'high'
                WHEN face_similarity >= 0.45 THEN 'medium'
                ELSE 'low'
            END AS band,
            COUNT(*) AS n
        FROM detection_events
        WHERE detected_at >= :since AND detected_at < :until
          AND is_ambiguous = FALSE
        GROUP BY band
    """), {"since": since, "until": until})).all()

    bands = {"high": 0, "medium": 0, "low": 0}
    for r in rows:
        bands[r.band] = r.n
    total = sum(bands.values()) or 1
    return {
        "bands": bands,
        "pct_high": round(bands["high"] / total, 4),
        "pct_medium": round(bands["medium"] / total, 4),
        "pct_low": round(bands["low"] / total, 4),
        "total_detections": total,
    }
