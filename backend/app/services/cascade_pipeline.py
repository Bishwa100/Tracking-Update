"""
Cascade pipeline: face-first, body only when face confidence < FACE_CONF_SKIP_BODY.

Saves ~30-40% CPU by skipping OSNet when ArcFace already has a confident match.
Wraps process_frame() transparently.
"""

import logging
from typing import List, Optional

import numpy as np

from app.config import settings
from app.cv_pipeline import DetectedPerson, process_frame
from app.ml_models import FaceEmbeddingCache

logger = logging.getLogger(__name__)


def process_frame_cascade(
    image: np.ndarray,
    embedding_cache: Optional[FaceEmbeddingCache] = None,
) -> List[DetectedPerson]:
    """
    Run YOLOv8 + ArcFace first. For each detected person, run OSNet only when
    face_det_score < FACE_CONF_SKIP_BODY (or when there is no face at all).

    This is a two-pass approach:
      Pass 1 — face-only (extract_body=False).
      Pass 2 — body only for the subset that needs it.
    The re-ID model is called once per weak-face person rather than for every
    person on every frame.
    """
    threshold = settings.FACE_CONF_SKIP_BODY

    # Pass 1: face detection only (skips OSNet entirely)
    persons = process_frame(image, extract_body=False, embedding_cache=embedding_cache)

    # Identify persons that need body embeddings
    needs_body = [
        p for p in persons
        if p.body_embedding is None and (p.face_det_score or 0.0) < threshold
    ]

    if not needs_body:
        logger.debug(
            "Cascade: %d person(s), all confident faces — skipping body pass.",
            len(persons),
        )
        return persons

    # Pass 2: run OSNet only for the weak-face subset
    from app.ml_models import ModelManager
    model_mgr = ModelManager.get_instance()
    if not model_mgr.has_body_model:
        return persons

    h, w = image.shape[:2]
    crops = []
    for p in needs_body:
        b = p.bbox
        x1 = max(0, min(w, int(b["x1"])))
        y1 = max(0, min(h, int(b["y1"])))
        x2 = max(0, min(w, int(b["x2"])))
        y2 = max(0, min(h, int(b["y2"])))
        crops.append(image[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else image[0:1, 0:1])

    try:
        from app.utils import normalize_embedding
        embeddings = model_mgr.extract_body_embeddings(crops)
        for person, body_emb in zip(needs_body, embeddings):
            person.body_embedding = normalize_embedding(body_emb)
    except Exception as e:
        logger.error("Cascade body pass failed: %s", e)

    logger.debug(
        "Cascade: %d person(s), %d needed body pass.",
        len(persons), len(needs_body),
    )
    return persons
