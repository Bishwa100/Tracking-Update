"""
Temporal consistency gate.

Prevents same-person fragmentation: if a "new" detection appears within
TEMPORAL_WINDOW_SECONDS and TEMPORAL_MAX_PIXEL_DISTANCE pixels of a recently
seen known visitor, it is treated as that visitor re-appearing (e.g. after
turning their head) rather than a new registration.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

import numpy as np

from app.config import settings


def _bbox_center_distance(b1: dict, b2: dict) -> float:
    c1x = (b1["x1"] + b1["x2"]) / 2.0
    c1y = (b1["y1"] + b1["y2"]) / 2.0
    c2x = (b2["x1"] + b2["x2"]) / 2.0
    c2y = (b2["y1"] + b2["y2"]) / 2.0
    return float(np.sqrt((c1x - c2x) ** 2 + (c1y - c2y) ** 2))


def _cosine_sim(a: list, b: list) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(va, vb) / denom)


class TemporalConsistencyGate:
    """
    Sliding-window buffer of recent successful detections.
    Call add_detection() on every RETURNING match.
    Call check() before registering a NEW visitor — it may return an existing
    visitor_id that should be used instead.
    """

    def __init__(self):
        self._recent: List[dict] = []

    def _evict_old(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=settings.TEMPORAL_WINDOW_SECONDS * 2)
        self._recent = [r for r in self._recent if r["timestamp"] > cutoff]
        # Hard cap
        if len(self._recent) > 2000:
            self._recent = self._recent[-2000:]

    def add_detection(
        self,
        visitor_id: UUID,
        embedding: list,
        bbox: dict,
        timestamp: datetime,
        confidence: float,
    ) -> None:
        """Record a confirmed detection."""
        self._evict_old(timestamp)
        self._recent.append(
            {
                "visitor_id": visitor_id,
                "embedding": embedding,
                "bbox": bbox,
                "timestamp": timestamp,
                "confidence": confidence,
            }
        )

    def check(
        self,
        new_embedding: list,
        new_bbox: dict,
        timestamp: datetime,
    ) -> Optional[UUID]:
        """
        Return visitor_id if this 'new' detection looks like a recently seen
        visitor who temporarily disappeared; None otherwise.
        """
        if not new_embedding:
            return None

        cutoff = timestamp - timedelta(seconds=settings.TEMPORAL_WINDOW_SECONDS)
        max_dist = settings.TEMPORAL_MAX_PIXEL_DISTANCE
        min_sim = settings.TEMPORAL_MIN_SIMILARITY

        best_id: Optional[UUID] = None
        best_score = -1.0

        for entry in self._recent:
            if entry["timestamp"] < cutoff:
                continue
            if not entry.get("embedding"):
                continue

            px_dist = _bbox_center_distance(new_bbox, entry["bbox"])
            if px_dist > max_dist:
                continue

            sim = _cosine_sim(new_embedding, entry["embedding"])
            if sim < min_sim:
                continue

            spatial_score = max(0.0, 1.0 - px_dist / max_dist)
            score = sim * 0.7 + spatial_score * 0.3

            if score > best_score:
                best_score = score
                best_id = entry["visitor_id"]

        return best_id

    def clear_visitor(self, visitor_id: UUID) -> None:
        """Remove all entries for a visitor (e.g. after opt-out)."""
        self._recent = [r for r in self._recent if r["visitor_id"] != visitor_id]


# Module-level singleton shared within the process
temporal_gate = TemporalConsistencyGate()
