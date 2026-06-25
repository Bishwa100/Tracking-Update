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

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator, model_validator
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
    # Max face embeddings kept per visitor. Higher = more multi-pose data per
    # person (better recall) + larger gallery; merges only trim faces ABOVE this.
    "MAX_FACES_PER_VISITOR",
    # ── Multi-angle identity (Phase 3) ──
    "POSE_CONTINUOUS_SEARCH",
    "ADAPTIVE_VISITOR_THRESHOLDS",
    "IDENTITY_TOP_K",
    "GREY_ZONE_POLICY",
    "TRACKLET_ENABLED",
    "TRACKLET_WINDOW_SECONDS",
    "TRACKLET_MAX_PIXEL_DISTANCE",
    "TRACKLET_MIN_OBSERVATIONS_NEW",
    # ── Cross-camera identity (Phase 4) ──
    # NOTE: toggling CROSS_CAMERA_ENABLED live enables the per-detection candidate
    # check immediately; the background reconciliation loop is only (re)started on
    # app startup. FACE_CACHE_* and CROSS_CAMERA_DEDUP_INTERVAL_MINUTES are
    # intentionally NOT patchable (they change cache/loop structure → need restart).
    "CROSS_CAMERA_ENABLED",
    "CROSS_CAMERA_LOOKBACK_SECONDS",
    "CROSS_CAMERA_REVIEW_THRESHOLD",
    "CROSS_CAMERA_AUTO_THRESHOLD",
    "CROSS_CAMERA_AUTO_MERGE_THRESHOLD",
    # ── Entry/Exit gate counting (two-camera directional visits) ──
    # ENTRY_CAMERA_ID / EXIT_CAMERA_ID are free-form camera ids (set from the
    # dashboard Gate card), so they're patchable strings with no fixed enum.
    "GATE_COUNTING_ENABLED",
    "ENTRY_CAMERA_ID",
    "EXIT_CAMERA_ID",
    "GATE_MIN_DWELL_SECONDS",
    "GATE_MAX_DWELL_SECONDS",
    "GATE_REQUIRE_ENTRY_FIRST",
}

# String-enum settings: restrict patch values to a known set.
_ENUM_CHOICES: Dict[str, set] = {
    "GREY_ZONE_POLICY": {"review", "tracklet", "register"},
}


class SettingsPatch(BaseModel):
    updates: Dict[str, Any]

    @model_validator(mode="after")
    def check_keys(self) -> "SettingsPatch":
        bad = set(self.updates) - _PATCHABLE
        if bad:
            raise ValueError(f"Non-patchable or unknown keys: {sorted(bad)}")
        for key, choices in _ENUM_CHOICES.items():
            if key in self.updates and self.updates[key] not in choices:
                raise ValueError(
                    f"{key} must be one of {sorted(choices)}; got {self.updates[key]!r}"
                )
        # Entry and exit cameras must differ when both are provided in the patch.
        entry = self.updates.get("ENTRY_CAMERA_ID")
        exit_ = self.updates.get("EXIT_CAMERA_ID")
        if entry and exit_ and str(entry).strip() == str(exit_).strip():
            raise ValueError("ENTRY_CAMERA_ID and EXIT_CAMERA_ID must differ.")
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


# ── Processing device (CPU / GPU) ────────────────────────────

_DEVICE_CHOICES = {"auto", "cpu", "cuda"}


class DevicePatch(BaseModel):
    device: str

    @field_validator("device")
    @classmethod
    def check_device(cls, v: str) -> str:
        norm = (v or "").strip().lower()
        if norm == "gpu":
            norm = "cuda"
        if norm not in _DEVICE_CHOICES:
            raise ValueError(f"device must be one of {sorted(_DEVICE_CHOICES)}")
        return norm


def _device_status() -> Dict[str, Any]:
    """Current device state + GPU capability for the dashboard."""
    from app.ml_models import ModelManager, cuda_available, gpu_info

    mgr = ModelManager.get_instance()
    info = gpu_info()
    return {
        "requested": getattr(settings, "DEVICE", "auto"),
        "current_device": mgr.device,
        "cuda_available": cuda_available(),
        "gpu_name": info.get("name"),
        "gpu_memory_mb": info.get("memory_mb"),
        "gpu_memory_used_mb": info.get("memory_used_mb"),
        "models_loaded": mgr.is_loaded,
    }


@router.get("/device")
async def get_device(_key: str = Security(verify_admin_api_key)):
    """Report the active processing device and whether a CUDA GPU is usable."""
    return _device_status()


@router.post("/device")
async def set_device(
    body: DevicePatch,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_admin_api_key),
):
    """
    Switch processing between CPU and GPU live — reloads all models onto the new
    device in-process. Inference is paused for the duration of the reload.
    """
    from app.ml_models import ModelManager, resolve_device
    from app.utils import _get_inference_semaphore

    requested = body.device
    resolved = resolve_device(requested)

    # Pause inference (camera loop / requests) while models are swapped. With
    # INFERENCE_MAX_CONCURRENCY=1 this fully serializes against in-flight work.
    semaphore = _get_inference_semaphore()
    async with semaphore:
        try:
            await run_in_threadpool(ModelManager.get_instance().reload, resolved)
        except Exception as exc:
            logger.exception("Device reload failed (%s): %s", resolved, exc)
            raise HTTPException(
                status_code=500, detail=f"Failed to switch device: {exc}"
            )

    # Remember the user's REQUESTED choice (e.g. "auto"), not the resolved value.
    object.__setattr__(settings, "DEVICE", requested)
    try:
        await db.execute(
            text("""
                INSERT INTO runtime_settings (key, value, updated_at)
                VALUES ('DEVICE', :value, NOW())
                ON CONFLICT (key) DO UPDATE
                  SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """),
            {"value": requested},
        )
        await db.commit()
    except Exception:
        pass  # runtime_settings table may not exist yet — not critical

    logger.info("Processing device switched to %s (requested=%s).", resolved, requested)
    return _device_status()


@router.get("/review-queue")
async def get_review_queue(
    limit: int = 50,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Return unresolved human-review flags."""
    from app.services.review_queue import get_pending_flags
    return await get_pending_flags(db, limit=limit)


@router.post("/review-queue/auto-merge-duplicates")
async def auto_merge_review_duplicates(
    min_similarity: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Only merge duplicate flags with similarity >= this "
                    "(defaults to AUTO_MERGE_MIN_SIMILARITY).",
    ),
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Merge probable-duplicate flags above a confidence floor into their matched
    visitor (global dedup). Weaker pairs are left for human review."""
    from app.services.review_queue import auto_merge_duplicates
    return await auto_merge_duplicates(db, min_similarity=min_similarity)


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


@router.post("/visitors/{visitor_id}/clean-faces")
async def clean_visitor_faces(
    visitor_id: str,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Auto-remove unclear faces from a visitor's gallery (clarity-based pruning)."""
    import uuid
    from app.services.auto_enroller import clean_visitor_gallery
    return await clean_visitor_gallery(db, uuid.UUID(visitor_id))


# ── Camera topology (Phase 4 cross-camera) ───────────────────

class CameraTopologyUpsert(BaseModel):
    camera_a: str
    camera_b: str
    min_travel_seconds: Optional[float] = None
    max_expected_seconds: Optional[float] = None
    transition_enabled: bool = True

    @model_validator(mode="after")
    def check(self) -> "CameraTopologyUpsert":
        if not self.camera_a.strip() or not self.camera_b.strip():
            raise ValueError("camera_a and camera_b are required.")
        if self.camera_a.strip() == self.camera_b.strip():
            raise ValueError("camera_a and camera_b must differ.")
        return self


@router.get("/cameras")
async def list_known_cameras(
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Distinct camera ids seen in detections/visits, for topology dropdowns."""
    ids: set[str] = {settings.CAMERA_ID}
    for tbl in ("detection_events", "visits"):
        try:
            rows = (await db.execute(
                text(f"SELECT DISTINCT camera_id FROM {tbl} WHERE camera_id IS NOT NULL")
            )).all()
            ids.update(r.camera_id for r in rows if r.camera_id)
        except Exception:
            pass
    return sorted(ids)


@router.get("/camera-topology")
async def list_camera_topology(
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """List configured camera-transition constraints."""
    from app.models import CameraTopology
    from sqlalchemy import select

    try:
        rows = (await db.execute(select(CameraTopology))).scalars().all()
    except Exception:
        return []
    return [
        {
            "id": str(r.id),
            "camera_a": r.camera_a,
            "camera_b": r.camera_b,
            "min_travel_seconds": r.min_travel_seconds,
            "max_expected_seconds": r.max_expected_seconds,
            "transition_enabled": r.transition_enabled,
        }
        for r in rows
    ]


@router.post("/camera-topology")
async def upsert_camera_topology(
    body: CameraTopologyUpsert,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Create or update a camera-pair transition constraint (unique by pair)."""
    import uuid
    from app.models import CameraTopology
    from sqlalchemy import select, or_, and_

    a, b = body.camera_a.strip(), body.camera_b.strip()
    existing = (
        await db.execute(
            select(CameraTopology).where(
                or_(
                    and_(CameraTopology.camera_a == a, CameraTopology.camera_b == b),
                    and_(CameraTopology.camera_a == b, CameraTopology.camera_b == a),
                )
            )
        )
    ).scalars().first()

    if existing is None:
        existing = CameraTopology(id=uuid.uuid4(), camera_a=a, camera_b=b)
        db.add(existing)
    existing.min_travel_seconds = body.min_travel_seconds
    existing.max_expected_seconds = body.max_expected_seconds
    existing.transition_enabled = body.transition_enabled
    await db.commit()
    return {"id": str(existing.id), "camera_a": a, "camera_b": b}


@router.delete("/camera-topology/{topology_id}")
async def delete_camera_topology(
    topology_id: str,
    _key: str = Security(verify_admin_api_key),
    db: AsyncSession = Depends(get_db),
):
    """Delete a camera-transition constraint."""
    import uuid
    from app.models import CameraTopology

    obj = await db.get(CameraTopology, uuid.UUID(topology_id))
    if obj is not None:
        await db.delete(obj)
        await db.commit()
    return {"success": obj is not None, "id": topology_id}


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
