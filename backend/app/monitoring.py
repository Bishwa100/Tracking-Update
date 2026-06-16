"""
Health monitoring and alerting.

Runs a background task (every HEALTH_CHECK_INTERVAL_SECONDS) that:
  • checks DB connectivity
  • checks model availability
  • tracks frame processing latency
  • fires a webhook alert when a check fails (once per cool-down period)
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

_last_alert_sent: dict[str, datetime] = {}
_ALERT_COOLDOWN = timedelta(minutes=10)

# Simple sliding-window frame latency tracker
_latency_samples: list[float] = []
_MAX_SAMPLES = 100


def record_frame_latency(seconds: float) -> None:
    """Call after each frame is processed to track pipeline latency."""
    _latency_samples.append(seconds)
    if len(_latency_samples) > _MAX_SAMPLES:
        _latency_samples.pop(0)


def p95_latency() -> Optional[float]:
    if not _latency_samples:
        return None
    sorted_samples = sorted(_latency_samples)
    idx = int(len(sorted_samples) * 0.95)
    return sorted_samples[min(idx, len(sorted_samples) - 1)]


async def check_database(db: AsyncSession) -> dict:
    try:
        t0 = time.perf_counter()
        await db.execute(text("SELECT 1"))
        latency = time.perf_counter() - t0
        return {"ok": True, "latency_ms": round(latency * 1000, 1)}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def check_models() -> dict:
    try:
        from app.ml_models import ModelManager
        mgr = ModelManager.get_instance()
        return {
            "ok": True,
            "yolo_loaded": mgr.has_person_model,
            "face_loaded": mgr.has_face_model,
            "body_loaded": mgr.has_body_model,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def visit_tracker_status() -> dict:
    try:
        from app.services.visit_tracker import VisitTracker
        tracker = VisitTracker.get_instance()
        return {"ok": True, "active_visits": tracker.current_inside_count()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def full_health_check(db: AsyncSession) -> dict:
    db_check = await check_database(db)
    model_check = check_models()
    tracker_check = visit_tracker_status()
    p95 = p95_latency()

    overall_ok = db_check["ok"] and model_check["ok"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ok": overall_ok,
        "database": db_check,
        "models": model_check,
        "visit_tracker": tracker_check,
        "frame_p95_latency_s": round(p95, 3) if p95 is not None else None,
    }


async def _fire_alert(name: str, detail: str) -> None:
    now = datetime.now(timezone.utc)
    last = _last_alert_sent.get(name)
    if last and (now - last) < _ALERT_COOLDOWN:
        return

    _last_alert_sent[name] = now
    webhook_url = settings.ALERT_WEBHOOK_URL
    if not webhook_url:
        logger.warning("ALERT [%s]: %s (no webhook configured)", name, detail)
        return

    payload = {
        "alert": name,
        "detail": detail,
        "timestamp": now.isoformat(),
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        logger.info("Alert fired: %s", name)
    except Exception as exc:
        logger.error("Alert webhook failed: %s", exc)


async def monitoring_loop(get_db_func) -> None:
    """Background task — call once at startup."""
    interval = settings.HEALTH_CHECK_INTERVAL_SECONDS
    logger.info("Monitoring loop started (interval=%ds).", interval)
    while True:
        await asyncio.sleep(interval)
        try:
            async for db in get_db_func():
                result = await full_health_check(db)
                if not result["ok"]:
                    if not result["database"]["ok"]:
                        await _fire_alert("db_down", result["database"].get("error", ""))
                    if not result["models"]["ok"]:
                        await _fire_alert("models_unavailable", result["models"].get("error", ""))
                else:
                    p95 = result.get("frame_p95_latency_s")
                    if p95 is not None and p95 > 5.0:
                        await _fire_alert(
                            "high_latency",
                            f"p95 frame latency {p95:.1f}s exceeds 5 s",
                        )
                break
        except Exception as exc:
            logger.error("Monitoring check error: %s", exc)
