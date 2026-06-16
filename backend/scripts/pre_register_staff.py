"""
Staff pre-registration script.

Usage:
    python -m scripts.pre_register_staff \
        --name "Alice" \
        --photos staff_photos/alice1.jpg staff_photos/alice2.jpg \
        [--camera-id cam-0]

Inserts a Visitor record with is_staff=True and populates their face gallery
from the supplied photo files, so staff are never counted in analytics.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Ensure the app package is importable when run from backend/
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.cv_pipeline import estimate_pose, PoseBin, FacePose
from app.ml_models import ModelManager
from app.models import Visitor, VisitorFace
from app.utils import normalize_embedding

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def _register(name: str, photo_paths: list[str], camera_id: str) -> None:
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    model_mgr = ModelManager.get_instance()

    async with AsyncSessionLocal() as db:
        visitor = Visitor(
            name=name,
            is_staff=True,
            is_active=True,
            consent_status="explicit",
            consent_method="staff_registration",
        )
        db.add(visitor)
        await db.flush()

        added = 0
        for path in photo_paths:
            img = cv2.imread(path)
            if img is None:
                logger.warning("Cannot read %s — skipping.", path)
                continue

            face_data = model_mgr.extract_face_data(img)
            if face_data is None:
                logger.warning("No face detected in %s — skipping.", path)
                continue

            if face_data["det_score"] < settings.FACE_QUALITY_CUTOFF:
                logger.warning(
                    "Face quality %.2f < cutoff %.2f in %s — skipping.",
                    face_data["det_score"], settings.FACE_QUALITY_CUTOFF, path,
                )
                continue

            emb = normalize_embedding(face_data["embedding"])
            kps = face_data.get("kps")
            kps_arr = np.asarray(kps, dtype=float) if kps is not None else None
            pose = estimate_pose(kps_arr) if kps_arr is not None else FacePose(0.0, 0.0, 0.0, PoseBin.UNKNOWN)

            face = VisitorFace(
                visitor_id=visitor.id,
                embedding=emb,
                det_score=face_data["det_score"],
                pose_bin=pose.bin.value,
            )
            db.add(face)
            added += 1
            logger.info("Added face from %s (score=%.3f, pose=%s).", path, face_data["det_score"], pose.bin.value)

        if added == 0:
            logger.error("No valid faces found — aborting without saving.")
            await db.rollback()
            return

        await db.commit()
        logger.info(
            "Registered staff member '%s' (id=%s) with %d face(s).",
            name, visitor.id, added,
        )

    await engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-register a staff member.")
    parser.add_argument("--name", required=True, help="Staff member's name")
    parser.add_argument("--photos", nargs="+", required=True, help="Photo file paths")
    parser.add_argument("--camera-id", default="cam-0", help="Camera ID tag")
    args = parser.parse_args()

    asyncio.run(_register(args.name, args.photos, args.camera_id))


if __name__ == "__main__":
    main()
