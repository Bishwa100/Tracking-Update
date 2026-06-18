"""
Auto-enroller + gallery manager.

  • New visitor    → create record (centroid = first face) + first gallery face.
  • Returning      → add face to gallery (top-N by quality, pose-aware) + adaptive centroid.
  • Best face crop → saved as the visitor thumbnail.
  • Pose-aware gallery → up to MAX_FACES_PER_POSE_BIN faces per pose bin;
    underrepresented bins can evict overrepresented ones.
"""

import logging
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import cv2
import numpy as np
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Visitor, VisitorFace
from app.similarity import cosine_similarity, pairwise_cosine
from app.utils import normalize_embedding
from app.cv_pipeline import PoseBin, FacePose

logger = logging.getLogger(__name__)


def _save_thumbnail(visitor_id: UUID, face_crop: Optional[np.ndarray]) -> Optional[str]:
    """Save a face crop as the visitor thumbnail. Returns the path or None."""
    if face_crop is None or face_crop.size == 0:
        return None
    out_dir = Path(settings.VISITOR_PHOTO_DIR) / str(visitor_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "thumbnail.jpg"
    try:
        cv2.imwrite(str(path), face_crop)
    except Exception as exc:
        logger.warning("Could not write thumbnail for %s: %s", visitor_id, exc)
        return None
    return str(path)


def _save_face_crop(
    visitor_id: UUID, face_id: UUID, face_crop: Optional[np.ndarray]
) -> Optional[str]:
    """Persist a gallery face's tight crop so its clarity can be re-scored later."""
    if face_crop is None or face_crop.size == 0:
        return None
    out_dir = Path(settings.VISITOR_PHOTO_DIR) / str(visitor_id) / "faces"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{face_id}.jpg"
    try:
        cv2.imwrite(str(path), face_crop)
    except Exception as exc:
        logger.warning("Could not write face crop for %s: %s", visitor_id, exc)
        return None
    return str(path)


def _add_gallery_face(
    db: AsyncSession,
    visitor_id: UUID,
    embedding: list,
    det_score: float,
    body_embedding: Optional[list],
    pose: Optional[FacePose],
    face_crop: Optional[np.ndarray],
    camera_id: Optional[str] = None,
) -> VisitorFace:
    """Build a VisitorFace (with explicit id), persist its crop, and stage it."""
    face = VisitorFace(
        id=uuid4(),
        visitor_id=visitor_id,
        embedding=embedding,
        det_score=det_score,
        body_embedding=body_embedding,
        pose_bin=pose.bin.value if pose else "unknown",
        yaw=pose.yaw if pose else None,
        pitch=pose.pitch if pose else None,
        roll=pose.roll if pose else None,
        source_camera_id=camera_id,
    )
    face.crop_path = _save_face_crop(visitor_id, face.id, face_crop)
    db.add(face)
    return face


async def _delete_gallery_face(db: AsyncSession, face: VisitorFace) -> None:
    """Delete a gallery face row and remove its crop file from disk."""
    if face.crop_path:
        try:
            Path(face.crop_path).unlink(missing_ok=True)
        except Exception as exc:
            logger.debug("Could not remove face crop %s: %s", face.crop_path, exc)
    await db.delete(face)


async def register_new_visitor(
    db: AsyncSession,
    face_embedding: list,
    det_score: float,
    body_embedding: Optional[list] = None,
    face_crop: Optional[np.ndarray] = None,
    pose: Optional[FacePose] = None,
    camera_id: Optional[str] = None,
) -> Visitor:
    """Create a new visitor with the first face seeding both centroid and gallery."""
    visitor = Visitor(
        id=uuid4(),
        face_embedding=face_embedding,
        body_embedding=body_embedding,
        visit_count=0,
        best_face_det_score=det_score,
        total_faces_recorded=1,
        consent_status="implicit",
        visit_confidence=0.3,
    )
    db.add(visitor)
    await db.flush()  # assign PK / make it queryable in this tx

    _add_gallery_face(
        db, visitor.id, face_embedding, det_score, body_embedding, pose,
        face_crop, camera_id=camera_id,
    )

    thumb = _save_thumbnail(visitor.id, face_crop)
    if thumb:
        visitor.thumbnail_path = thumb

    logger.info(
        "Registered new visitor %s (det_score=%.3f, pose=%s).",
        visitor.id, det_score, pose.bin.value if pose else "unknown",
    )
    return visitor


# A face this similar to one already in the same pose bin adds no information —
# reject it regardless of confidence so the bounded gallery stays diverse.
_GALLERY_NEAR_DUP_SIM = 0.97


async def add_face_to_gallery(
    db: AsyncSession,
    visitor_id: UUID,
    embedding: list,
    det_score: float,
    body_embedding: Optional[list] = None,
    pose: Optional[FacePose] = None,
    face_crop: Optional[np.ndarray] = None,
    camera_id: Optional[str] = None,
) -> None:
    """
    Insert a face using pose-aware eviction policy.
    Enforces diversity across pose bins (frontal / left / right / down) and
    rejects near-duplicates that would waste a bounded gallery slot.
    """
    bin_name = pose.bin.value if pose else "unknown"
    max_per_bin = settings.MAX_FACES_PER_POSE_BIN
    min_per_bin = settings.MIN_FACES_PER_POSE_BIN

    rows = (
        await db.execute(
            select(VisitorFace).where(VisitorFace.visitor_id == visitor_id)
        )
    ).scalars().all()

    total = len(rows)

    # Count faces already in each bin
    bin_counts: dict[str, list[VisitorFace]] = {}
    for f in rows:
        b = f.pose_bin or "unknown"
        bin_counts.setdefault(b, []).append(f)

    current_bin_faces = bin_counts.get(bin_name, [])
    current_bin_count = len(current_bin_faces)

    # Gallery quality gate: skip faces that are near-identical to one already in
    # the same pose bin (no new angular/appearance information).
    for f in current_bin_faces:
        if f.embedding is not None and cosine_similarity(
            embedding, f.embedding, assume_normalized=True
        ) >= _GALLERY_NEAR_DUP_SIM:
            return

    def _add():
        _add_gallery_face(
            db, visitor_id, embedding, det_score, body_embedding, pose,
            face_crop, camera_id=camera_id,
        )

    if total < settings.MAX_FACES_PER_VISITOR:
        # Gallery has room — add if this bin isn't over its cap
        if current_bin_count < max_per_bin:
            _add()
        return

    # Gallery is full — smart eviction
    if current_bin_count < min_per_bin:
        # This bin is underrepresented; evict worst from most overcrowded bin
        worst = _find_eviction_candidate(rows, bin_counts, bin_name)
        if worst is not None and det_score > worst.det_score * 0.9:
            await _delete_gallery_face(db, worst)
            _add()
    else:
        # Bin already has enough faces — only replace worst-quality in same bin
        worst_in_bin = (
            min(current_bin_faces, key=lambda f: (f.det_score or 0.0))
            if current_bin_faces else None
        )
        if worst_in_bin is not None and det_score > (worst_in_bin.det_score or 0.0):
            await _delete_gallery_face(db, worst_in_bin)
            _add()


def _find_eviction_candidate(
    all_faces: list,
    bin_counts: dict,
    target_bin: str,
) -> Optional[VisitorFace]:
    """Return the lowest-quality face from the most over-represented bin."""
    overcrowded = [
        (b, faces)
        for b, faces in bin_counts.items()
        if b != target_bin and len(faces) > settings.MAX_FACES_PER_POSE_BIN
    ]
    if overcrowded:
        _, candidates = max(overcrowded, key=lambda x: len(x[1]))
    else:
        # No bin is overcrowded — evict global worst
        candidates = all_faces
    if not candidates:
        return None
    return min(candidates, key=lambda f: (f.det_score or 0.0))


async def update_centroid(
    db: AsyncSession,
    visitor: Visitor,
    new_embedding: list,
    det_score: float,
) -> None:
    """
    Weighted moving average of the face centroid. The learning rate shrinks as a
    visitor accumulates visits (their centroid becomes more trusted) and scales
    with detection quality.
    """
    if det_score < settings.FACE_QUALITY_CUTOFF or visitor.face_embedding is None:
        return

    current = np.asarray(visitor.face_embedding, dtype=np.float32)
    incoming = np.asarray(new_embedding, dtype=np.float32)

    alpha = (
        settings.CENTROID_ALPHA_BASE
        * min(det_score * 2, 1.0)
        * max(0.05, 1.0 / (1 + visitor.visit_count * 0.1))
    )
    updated = (1 - alpha) * current + alpha * incoming
    visitor.face_embedding = normalize_embedding(updated)


async def recompute_centroid_from_gallery(db: AsyncSession, visitor: Visitor) -> bool:
    """
    Rebuild a visitor's face (and body) centroid from its current gallery faces,
    quality-weighted by det_score. Mutates `visitor` in place WITHOUT committing —
    the caller owns the transaction.

    Used after a merge (or gallery edit) so the centroid reflects the pooled set
    of faces rather than a stale adaptive average. Returns False (centroid left
    untouched) when the visitor has no usable gallery embeddings.
    """
    rows = await db.execute(
        select(
            VisitorFace.embedding,
            VisitorFace.det_score,
            VisitorFace.body_embedding,
        ).where(VisitorFace.visitor_id == visitor.id)
    )

    face_vecs: list[np.ndarray] = []
    weights: list[float] = []
    body_vecs: list[np.ndarray] = []
    best_det = 0.0

    for emb, det_score, body_emb in rows.all():
        if emb is None:
            continue
        face_vecs.append(np.asarray(emb, dtype=np.float32))
        # Weight by quality, with a floor so every gallery face still counts.
        weights.append(max(float(det_score or 0.0), 0.05))
        best_det = max(best_det, float(det_score or 0.0))
        if body_emb is not None:
            body_vecs.append(np.asarray(body_emb, dtype=np.float32))

    if not face_vecs:
        return False

    w = np.asarray(weights, dtype=np.float32)[:, None]
    face_centroid = (np.stack(face_vecs) * w).sum(axis=0) / w.sum()
    visitor.face_embedding = normalize_embedding(face_centroid)

    if best_det > 0:
        visitor.best_face_det_score = max(
            float(visitor.best_face_det_score or 0.0), best_det
        )
    if body_vecs:
        visitor.body_embedding = normalize_embedding(
            np.mean(np.stack(body_vecs), axis=0)
        )

    logger.info(
        "Recomputed centroid for visitor %s from %d gallery face(s).",
        visitor.id, len(face_vecs),
    )
    return True


async def recompute_adaptive_thresholds(db: AsyncSession, visitor: Visitor) -> bool:
    """
    Recompute a visitor's personal thresholds from the within-gallery pairwise
    similarity distribution. A visitor whose faces are mutually consistent
    (compact cluster) keeps a strict threshold; one with high variance (mixed
    angles/lighting/masks) gets a looser personal_returning_threshold so the same
    person isn't fragmented. Clamped to [0.40, RETURNING_FACE_THRESHOLD]; needs
    >= 3 gallery faces to have a meaningful distribution. Mutates `visitor`
    in place WITHOUT committing — the caller owns the transaction.
    """
    if not settings.ADAPTIVE_VISITOR_THRESHOLDS:
        return False

    rows = await db.execute(
        select(VisitorFace.embedding).where(VisitorFace.visitor_id == visitor.id)
    )
    gallery = [e for e in rows.scalars().all() if e is not None]
    if len(gallery) < 3:
        return False

    sims = pairwise_cosine(gallery)
    if sims.size == 0:
        return False

    mean = float(sims.mean())
    std = float(sims.std())
    base = settings.RETURNING_FACE_THRESHOLD
    # 2σ below the mean within-person similarity, clamped so we never exceed the
    # global returning bar (only ever loosen) and never drop below 0.40.
    personal_returning = float(np.clip(mean - 2.0 * std, 0.40, base))

    visitor.expected_match_similarity = mean
    visitor.match_similarity_std = std
    visitor.personal_returning_threshold = personal_returning
    visitor.personal_new_threshold = min(
        settings.NEW_VISITOR_MAX_SIMILARITY, max(0.0, personal_returning - 0.10)
    )
    return True


async def _is_diverse_embedding(
    db: AsyncSession,
    visitor_id: UUID,
    new_embedding: list,
    diversity_threshold: float = 0.85,
) -> bool:
    """
    Check if the new embedding is sufficiently different from existing gallery faces.
    Returns True if the embedding should be added (no near-duplicate in gallery).
    Threshold 0.85 means > 85% similar is considered a near-duplicate.
    """
    rows = await db.execute(
        select(VisitorFace.embedding).where(VisitorFace.visitor_id == visitor_id)
    )
    gallery = rows.scalars().all()
    if not gallery:
        return True

    for existing_emb in gallery:
        similarity = cosine_similarity(new_embedding, existing_emb, assume_normalized=True)
        if similarity >= diversity_threshold:
            return False
    return True


async def update_after_match(
    db: AsyncSession,
    visitor: Visitor,
    face_embedding: list,
    det_score: float,
    face_similarity: float,
    body_embedding: Optional[list] = None,
    face_crop: Optional[np.ndarray] = None,
    pose: Optional[FacePose] = None,
    match_source: str = "face",
    camera_id: Optional[str] = None,
) -> None:
    """
    Self-improvement on a confident returning match: grow the gallery, refresh
    the adaptive centroid, and update the thumbnail when a better face is seen.

    Gallery growth paths:
    - High confidence (>= STRONG_MATCH_THRESHOLD): always add + update centroid
    - Medium confidence (>= RETURNING_FACE_THRESHOLD): add if pose-diverse + update centroid
    - Temporal recovery (match_source="temporal"): the temporal gate already
      confirmed the same person via similarity + spatial + time proximity, so the
      raw gallery similarity is low (that's WHY the gate fired). Learn this hard
      angle anyway — add if pose-diverse and refresh the centroid — otherwise the
      system never absorbs the very angle that caused the near-miss.
    """
    if det_score < settings.FACE_QUALITY_CUTOFF:
        return

    added_to_gallery = False

    if match_source == "temporal":
        if await _is_diverse_embedding(db, visitor.id, face_embedding):
            await add_face_to_gallery(
                db, visitor.id, face_embedding, det_score, body_embedding,
                pose=pose, face_crop=face_crop, camera_id=camera_id,
            )
            added_to_gallery = True
        await update_centroid(db, visitor, face_embedding, det_score)
    elif face_similarity >= settings.STRONG_MATCH_THRESHOLD:
        await add_face_to_gallery(
            db, visitor.id, face_embedding, det_score, body_embedding,
            pose=pose, face_crop=face_crop, camera_id=camera_id,
        )
        await update_centroid(db, visitor, face_embedding, det_score)
        added_to_gallery = True
    elif face_similarity >= settings.RETURNING_FACE_THRESHOLD:
        if await _is_diverse_embedding(db, visitor.id, face_embedding):
            await add_face_to_gallery(
                db, visitor.id, face_embedding, det_score, body_embedding,
                pose=pose, face_crop=face_crop, camera_id=camera_id,
            )
            added_to_gallery = True
        # Medium-confidence matches now also nudge the centroid — a visitor who
        # only ever appears at medium confidence (glasses, angle) otherwise keeps
        # a stale centroid that drifts away and eventually fragments them.
        await update_centroid(db, visitor, face_embedding, det_score)

    if added_to_gallery:
        visitor.total_faces_recorded = (visitor.total_faces_recorded or 0) + 1
        # Gallery changed → refresh the per-visitor adaptive thresholds.
        await recompute_adaptive_thresholds(db, visitor)

    if det_score > (visitor.best_face_det_score or 0.0):
        visitor.best_face_det_score = det_score
        thumb = _save_thumbnail(visitor.id, face_crop)
        if thumb:
            visitor.thumbnail_path = thumb


async def clean_visitor_gallery(db: AsyncSession, visitor_id: UUID) -> dict:
    """
    Score every gallery face for clarity (landmark frontality + blur + det_score)
    and delete the unclear ones, always keeping the single clearest face. Promotes
    the clearest remaining crop to the visitor thumbnail.

    Returns a summary: {visitor_id, removed, kept, scores:[...]}.
    """
    from app.services.face_quality import compute_clarity

    visitor = await db.get(Visitor, visitor_id)
    if visitor is None:
        return {"visitor_id": str(visitor_id), "removed": 0, "kept": 0, "scores": []}

    faces = (
        await db.execute(
            select(VisitorFace).where(VisitorFace.visitor_id == visitor_id)
        )
    ).scalars().all()

    # Score each face; cache the clarity on the row.
    scored: list[tuple[VisitorFace, dict]] = []
    for f in faces:
        crop = cv2.imread(f.crop_path) if f.crop_path else None
        result = compute_clarity(crop, f.det_score, f.pose_bin)
        f.clarity_score = result["clarity"]
        scored.append((f, result))

    # Always keep the clearest face, even if it's below the cutoff.
    scored.sort(key=lambda sf: sf[1]["clarity"], reverse=True)
    keeper = scored[0][0] if scored else None

    removed = 0
    for f, result in scored:
        if f is keeper:
            continue
        if result["clarity"] < settings.FACE_CLARITY_CUTOFF:
            await _delete_gallery_face(db, f)
            removed += 1

    # Refresh thumbnail + best score from the clearest surviving face.
    if keeper is not None:
        if keeper.crop_path and Path(keeper.crop_path).exists():
            crop = cv2.imread(keeper.crop_path)
            thumb = _save_thumbnail(visitor_id, crop)
            if thumb:
                visitor.thumbnail_path = thumb
        survivors = [f for f, _ in scored if f is keeper or f.clarity_score is None
                     or f.clarity_score >= settings.FACE_CLARITY_CUTOFF]
        visitor.best_face_det_score = max(
            (f.det_score or 0.0 for f in survivors), default=0.0
        )

    # Removing faces changes the gallery — rebuild the centroid from survivors.
    if removed:
        await recompute_centroid_from_gallery(db, visitor)

    await db.commit()

    return {
        "visitor_id": str(visitor_id),
        "removed": removed,
        "kept": len(scored) - removed,
        "scores": [
            {
                "face_id": str(f.id),
                "pose_bin": f.pose_bin,
                "kept": (f is keeper) or (r["clarity"] >= settings.FACE_CLARITY_CUTOFF),
                **r,
            }
            for f, r in scored
        ],
    }
