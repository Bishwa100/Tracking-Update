"""
Runtime configuration API — change thresholds without restarting the server.

PATCH /api/admin/settings  →  update one or more settings fields in-process.
GET  /api/admin/settings   →  read current runtime values.

Changes are ephemeral (lost on restart) unless your deployment loads them from
a DB-backed runtime_settings table (see migration 005). For now the endpoint
mutates the in-process `settings` object so every subsequent request sees the
new value immediately.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Security
from pydantic import BaseModel, model_validator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import verify_admin_api_key
from app.config import settings
from app.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-config"])

# Fields that may be changed at runtime (whitelist for safety)
_PATCHABLE = {
    "RETURNING_FACE_THRESHOLD",
    "REJECT_SIMILARITY",
    "AMBIGUITY_MARGIN",
    "STRONG_MATCH_THRESHOLD",
    "NEW_VISITOR_MAX_SIMILARITY",
    "VISIT_COOLDOWN_MINUTES",
    "SEATED_COOLDOWN_MINUTES",
    "MAX_VISIT_DURATION_HOURS",
    "TEMPORAL_WINDOW_SECONDS",
    "TEMPORAL_MAX_PIXEL_DISTANCE",
    "TEMPORAL_MIN_SIMILARITY",
    "FACE_CONF_SKIP_BODY",
    "FACE_PREPROCESSING_CLAHE",
    "FACE_PREPROCESSING_GAMMA",
    "CLAHE_CLIP_LIMIT",
    "POSE_AWARE_GALLERY",
    "MASK_DETECTION_ENABLED",
    "MASKED_FACE_THRESHOLD_OFFSET",
    "AUTO_TUNING_ENABLED",
    "YOLO_PERSON_CONFIDENCE",
    "MIN_FACE_DET_SCORE",
    "FACE_QUALITY_CUTOFF",
}


class SettingsPatch(BaseModel):
    updates: Dict[str, Any]

    @model_validator(mode="after")
    def check_keys(self) -> "SettingsPatch":
        bad = set(self.updates) - _PATCHABLE
        if bad:
            raise ValueError(f"Non-patchable or unknown keys: {sorted(bad)}")
        return self


@router.get("/settings")
async def get_settings(_key: str = Security(verify_admin_api_key)):
    """Return current runtime values for all patchable settings."""
    return {k: getattr(settings, k, None) for k in sorted(_PATCHABLE)}


@router.patch("/settings")
async def patch_settings(
    body: SettingsPatch,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_admin_api_key),
):
    """Update one or more runtime settings in-process (survives until next restart)."""
    applied: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    for key, value in body.updates.items():
        try:
            expected_type = type(getattr(settings, key))
            cast_value = expected_type(value)
            object.__setattr__(settings, key, cast_value)
            applied[key] = cast_value

            # Persist to runtime_settings table if it exists (migration 005)
            try:
                await db.execute(
                    text("""
                        INSERT INTO runtime_settings (key, value, updated_at)
                        VALUES (:key, :value, NOW())
                        ON CONFLICT (key) DO UPDATE
                          SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """),
                    {"key": key, "value": str(cast_value)},
                )
            except Exception:
                pass  # Table may not exist yet — not critical

        except Exception as exc:
            errors[key] = str(exc)
            logger.warning("Failed to apply setting %s=%r: %s", key, value, exc)

    if applied:
        logger.info("Runtime settings updated: %s", applied)
        await db.commit()

    return {"applied": applied, "errors": errors}


@router.get("/review-queue")
async def get_review_queue(
    limit: int = 50,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Return unresolved human-review flags."""
    from app.services.review_queue import get_pending_flags
    return await get_pending_flags(db, limit=limit)


@router.post("/review-queue/{flag_id}/resolve")
async def resolve_review_flag(
    flag_id: str,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Mark a review flag as resolved."""
    import uuid
    from app.services.review_queue import resolve_flag
    ok = await resolve_flag(db, uuid.UUID(flag_id))
    return {"success": ok, "flag_id": flag_id}


@router.post("/settings/reload")
async def reload_settings_from_db(
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_admin_api_key),
):
    """Reload persisted settings from the runtime_settings table (if present)."""
    try:
        rows = (await db.execute(text("SELECT key, value FROM runtime_settings"))).all()
    except Exception:
        return {"reloaded": 0, "message": "runtime_settings table not found"}

    reloaded = 0
    for row in rows:
        key, value = row.key, row.value
        if key not in _PATCHABLE:
            continue
        try:
            expected_type = type(getattr(settings, key))
            object.__setattr__(settings, key, expected_type(value))
            reloaded += 1
        except Exception as exc:
            logger.warning("Could not reload %s=%r: %s", key, value, exc)

    logger.info("Reloaded %d runtime setting(s) from DB.", reloaded)
    return {"reloaded": reloaded}
