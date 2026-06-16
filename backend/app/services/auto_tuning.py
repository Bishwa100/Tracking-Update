"""
Auto-tuning: weekly adjustment of RETURNING_FACE_THRESHOLD.

Logic:
  1. Compute false-new rate = (new_flags / total_detections) over the last
     AUTO_TUNING_INTERVAL_DAYS days.
  2. If false-new rate > 5% the threshold is too low → raise by 0.02.
  3. If false-new rate < 1% we're probably too strict → lower by 0.01.
  4. Clamp to [0.45, 0.75] to keep the system sane.
  5. Write the adjustment to the auto_tuning_log table and apply in-process.
"""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

_MIN_THRESHOLD = 0.45
_MAX_THRESHOLD = 0.75
_STEP_UP = 0.02
_STEP_DOWN = 0.01
_FALSE_NEW_HIGH = 0.05  # raise threshold if false-new rate exceeds this
_FALSE_NEW_LOW = 0.01   # lower threshold if false-new rate is below this


async def run_auto_tuning(db: AsyncSession) -> dict:
    """
    Analyse recent detection quality and adjust RETURNING_FACE_THRESHOLD if needed.
    Returns a summary dict describing the action taken (or skipped).
    """
    if not settings.AUTO_TUNING_ENABLED:
        return {"status": "disabled"}

    interval = timedelta(days=settings.AUTO_TUNING_INTERVAL_DAYS)
    since = datetime.now(timezone.utc) - interval

    try:
        row = (await db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE is_new_visitor = TRUE) AS new_count,
                COUNT(*) FILTER (
                    WHERE is_new_visitor = TRUE
                      AND (face_similarity IS NULL OR face_similarity < :thresh)
                ) AS low_conf_new
            FROM detection_events
            WHERE detected_at >= :since AND is_ambiguous = FALSE
        """), {"since": since, "thresh": settings.RETURNING_FACE_THRESHOLD})).one()
    except Exception as exc:
        logger.warning("Auto-tuning query failed: %s", exc)
        return {"status": "error", "detail": str(exc)}

    total = row.total or 0
    new_count = row.new_count or 0
    low_conf_new = row.low_conf_new or 0

    if total < 100:
        return {"status": "insufficient_data", "total_detections": total}

    false_new_rate = low_conf_new / total
    old_threshold = settings.RETURNING_FACE_THRESHOLD
    new_threshold = old_threshold
    action = "no_change"

    if false_new_rate > _FALSE_NEW_HIGH:
        new_threshold = min(_MAX_THRESHOLD, round(old_threshold + _STEP_UP, 4))
        action = "raised"
    elif false_new_rate < _FALSE_NEW_LOW:
        new_threshold = max(_MIN_THRESHOLD, round(old_threshold - _STEP_DOWN, 4))
        action = "lowered"

    summary = {
        "status": "ok",
        "action": action,
        "old_threshold": old_threshold,
        "new_threshold": new_threshold,
        "false_new_rate": round(false_new_rate, 4),
        "total_detections": total,
        "new_registrations": new_count,
        "low_conf_new": low_conf_new,
        "interval_days": settings.AUTO_TUNING_INTERVAL_DAYS,
    }

    if action != "no_change":
        object.__setattr__(settings, "RETURNING_FACE_THRESHOLD", new_threshold)
        logger.info(
            "Auto-tuning: threshold %s → %.4f (false_new_rate=%.3f)",
            action, new_threshold, false_new_rate,
        )
        await _log_adjustment(db, old_threshold, new_threshold, false_new_rate, summary)

    return summary


async def _log_adjustment(
    db: AsyncSession,
    old_val: float,
    new_val: float,
    false_new_rate: float,
    detail: dict,
) -> None:
    try:
        import json
        await db.execute(
            text("""
                INSERT INTO auto_tuning_log
                  (tuned_at, setting_key, old_value, new_value, false_new_rate, detail)
                VALUES (:at, 'RETURNING_FACE_THRESHOLD', :old, :new, :fnr, :det)
            """),
            {
                "at": datetime.now(timezone.utc),
                "old": old_val,
                "new": new_val,
                "fnr": false_new_rate,
                "det": json.dumps(detail),
            },
        )
        await db.commit()
    except Exception as exc:
        logger.debug("auto_tuning_log insert skipped: %s", exc)
