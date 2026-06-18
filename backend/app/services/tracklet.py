"""
Tracklet buffer — defers NEW-visitor creation until a sighting persists.

A single bad/ambiguous frame should never create a permanent duplicate visitor.
The buffer groups consecutive detections of the same body in one camera (by bbox
proximity within a short time window) into a *tracklet*. A grey-zone or
first-sighting face is HELD until its tracklet has accrued enough observations,
and once a tracklet resolves to a visitor, later frames of that same tracklet are
attached to that visitor instead of re-resolved (which would risk a duplicate).

This is process-local state (like the temporal gate). It does not touch the DB.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from app.config import settings
from app.geometry import bbox_center


@dataclass
class Tracklet:
    camera_id: Optional[str]
    last_bbox: dict
    last_ts: datetime
    created_ts: datetime
    observations: int = 1
    visitor_id: Optional[UUID] = None  # set once the tracklet resolves to a visitor


def _center_distance(a: dict, b: dict) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


class TrackletBuffer:
    """Sliding-window buffer of open tracklets, one set per process."""

    def __init__(self):
        self._tracklets: List[Tracklet] = []

    def _evict_old(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=settings.TRACKLET_WINDOW_SECONDS * 2)
        self._tracklets = [t for t in self._tracklets if t.last_ts > cutoff]
        if len(self._tracklets) > 2000:
            self._tracklets = self._tracklets[-2000:]

    def get_or_create(
        self, camera_id: Optional[str], bbox: dict, timestamp: datetime
    ) -> Tracklet:
        """
        Associate `bbox` with the nearest open tracklet on the same camera within
        the window + max pixel distance, or open a new one. Updates the matched
        tracklet's position/timestamp and bumps its observation count.
        """
        self._evict_old(timestamp)
        window = timedelta(seconds=settings.TRACKLET_WINDOW_SECONDS)
        max_dist = settings.TRACKLET_MAX_PIXEL_DISTANCE

        best: Optional[Tracklet] = None
        best_dist = max_dist
        for t in self._tracklets:
            if t.camera_id != camera_id:
                continue
            if timestamp - t.last_ts > window:
                continue
            d = _center_distance(bbox, t.last_bbox)
            if d <= best_dist:
                best_dist = d
                best = t

        if best is not None:
            best.last_bbox = bbox
            best.last_ts = timestamp
            best.observations += 1
            return best

        tr = Tracklet(
            camera_id=camera_id, last_bbox=bbox,
            last_ts=timestamp, created_ts=timestamp,
        )
        self._tracklets.append(tr)
        return tr

    def mark_resolved(self, tracklet: Tracklet, visitor_id: UUID) -> None:
        """Pin a tracklet to a visitor so later frames attach instead of re-resolving."""
        tracklet.visitor_id = visitor_id

    def clear_visitor(self, visitor_id: UUID) -> None:
        """Drop a visitor's pin (e.g. after a merge or opt-out)."""
        for t in self._tracklets:
            if t.visitor_id == visitor_id:
                t.visitor_id = None


# Module-level singleton shared within the process.
tracklet_buffer = TrackletBuffer()
