"""
FastAPI application — Restaurant Visitor Tracker.

Detects persons via webcam/upload, auto-registers first-time visitors,
recognises returning visitors (ArcFace gallery), tracks visit sessions, and
serves visitor + analytics APIs for the dashboard.
"""

import os

# Configure CPU thread usage for all math backends BEFORE numpy/torch/onnxruntime
# are imported (they read the thread count once at import time). Default: every
# physical core. Everything runs on CPU — no GPU.
_cpu_threads = os.environ.get("CPU_THREADS", "").strip()
if not _cpu_threads or _cpu_threads == "0":
    _cpu_threads = str(os.cpu_count() or 1)
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = _cpu_threads

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from time import perf_counter

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import AsyncSessionLocal, engine, get_db, init_db
from app.ml_models import ModelManager
from app.models import Visit, Visitor
from app.schemas import HealthResponse
from app.services.camera_service import CameraService
from app.services.visit_tracker import VisitTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

_background_tasks: set = set()


async def _stale_visit_loop():
    """Periodically close visits idle past the cooldown / open past the max cap."""
    tracker = VisitTracker.get_instance()
    while True:
        await asyncio.sleep(settings.STALE_CHECK_INTERVAL_SECONDS)
        try:
            async with AsyncSessionLocal() as db:
                await tracker.cleanup_stale(db)
        except Exception as exc:
            logger.warning("Stale-visit cleanup failed: %s", exc)


async def _auto_tuning_loop():
    """Run weekly threshold auto-tuning."""
    from app.services.auto_tuning import run_auto_tuning
    interval = settings.AUTO_TUNING_INTERVAL_DAYS * 86400
    await asyncio.sleep(interval)  # first run after one full interval
    while True:
        try:
            async with AsyncSessionLocal() as db:
                result = await run_auto_tuning(db)
                logger.info("Auto-tuning result: %s", result)
        except Exception as exc:
            logger.warning("Auto-tuning run failed: %s", exc)
        await asyncio.sleep(interval)


async def _retention_loop():
    """Purge visitors whose last_seen_at is older than the retention window."""
    if settings.VISITOR_RETENTION_DAYS <= 0:
        return
    while True:
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=settings.VISITOR_RETENTION_DAYS)
            async with AsyncSessionLocal() as db:
                result = await db.execute(
                    text("DELETE FROM visitors WHERE last_seen_at IS NOT NULL AND last_seen_at < :cutoff"),
                    {"cutoff": cutoff},
                )
                await db.commit()
                if result.rowcount:
                    logger.info("Retention purge removed %d visitor(s).", result.rowcount)
        except Exception as exc:
            logger.warning("Retention purge failed: %s", exc)
        await asyncio.sleep(settings.RETENTION_PURGE_INTERVAL_HOURS * 3600)


def _spawn(coro):
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Restaurant Tracker — initializing database...")
    await init_db()

    import torch
    cpu_threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(cpu_threads)
    logger.info("PyTorch CPU thread count set to %d (CPU-only).", cpu_threads)

    logger.info("Loading ML models (CPU)...")
    ModelManager.get_instance().load_all(
        yolo_path=settings.YOLO_MODEL_PATH,
        insightface_name=settings.INSIGHTFACE_MODEL_NAME,
    )

    # Recover any visits left open by a previous run.
    async with AsyncSessionLocal() as db:
        await VisitTracker.get_instance().recover_active(db)

    _spawn(_stale_visit_loop())
    _spawn(_retention_loop())

    if settings.AUTO_TUNING_ENABLED:
        _spawn(_auto_tuning_loop())

    # Load persisted runtime settings from DB (migration 005) if table exists
    try:
        from app.api.admin_config import _PATCHABLE
        async with AsyncSessionLocal() as db:
            rows = (await db.execute(text("SELECT key, value FROM runtime_settings"))).all()
            for row in rows:
                if row.key in _PATCHABLE:
                    try:
                        expected_type = type(getattr(settings, row.key))
                        object.__setattr__(settings, row.key, expected_type(row.value))
                    except Exception:
                        pass
            if rows:
                logger.info("Loaded %d runtime setting(s) from DB.", len(rows))
    except Exception:
        pass  # runtime_settings table may not exist yet

    # Start monitoring background loop
    from app.monitoring import monitoring_loop
    _spawn(monitoring_loop(get_db))

    if settings.CAMERA_AUTOSTART:
        try:
            await CameraService.get_instance().start()
        except Exception as exc:
            logger.warning("Camera autostart failed: %s", exc)

    logger.info("Startup complete.")
    yield

    logger.info("Shutting down...")
    await CameraService.get_instance().stop()
    for task in list(_background_tasks):
        task.cancel()
    await engine.dispose()


app = FastAPI(
    title="Restaurant Visitor Tracker",
    description="Auto-registering visitor detection, recognition, and analytics.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware — temporarily disabled for WebSocket debugging
# WebSocket upgrade requests need special handling that FastAPI's CORS middleware doesn't support well
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


@app.middleware("http")
async def add_timing(request, call_next):
    start = perf_counter()
    response = await call_next(request)
    elapsed_ms = (perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ── Routers ──────────────────────────────────────────────────
from app.api import (  # noqa: E402
    activity,
    admin,
    admin_config,
    analytics,
    camera,
    detect,
    settings as settings_api,
    visitors,
    websocket,
)

app.include_router(detect.router)
app.include_router(visitors.router)
app.include_router(analytics.router)
app.include_router(activity.router)
app.include_router(camera.router)
app.include_router(admin.router)
app.include_router(admin_config.router)
app.include_router(settings_api.router)
app.include_router(websocket.router)


@app.get("/api/health", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_db)):
    model_mgr = ModelManager.get_instance()
    status = model_mgr.status()

    db_status = "connected"
    visitors_count = 0
    total_visits = 0
    try:
        await db.execute(text("SELECT 1"))
        visitors_count = await db.scalar(select(func.count(Visitor.id))) or 0
        total_visits = await db.scalar(select(func.count(Visit.id))) or 0
    except Exception:
        db_status = "error"

    healthy = db_status == "connected" and model_mgr.is_loaded
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=db_status,
        models_loaded=model_mgr.is_loaded,
        yolo_loaded=status["yolo_loaded"],
        arcface_loaded=status["arcface_loaded"],
        body_model=status["body_model"],
        camera_running=CameraService.get_instance().is_running,
        visitors_count=visitors_count,
        total_visits=total_visits,
    )
