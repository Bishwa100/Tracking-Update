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
from app.geometry import bbox_center, bbox_iou


@dataclass
class Tracklet:
    camera_id: Optional[str]
    last_bbox: dict
    last_ts: datetime
    created_ts: datetime
    observations: int = 1
    visitor_id: Optional[UUID] = None  # set once the tracklet resolves to a visitor
    # When the pin was last confirmed by a full gallery search. Used by the
    # fast-path: a pinned tracklet may attribute directly to its visitor until a
    # re-verify is due (time elapsed / IoU drop / face change), then it must
    # re-resolve. None until first resolved.
    last_verified_ts: Optional[datetime] = None
    # Face embedding from the last full resolve that confirmed the pin. The
    # fast-path compares the incoming face against this: if it drifts too far the
    # tracklet may have swapped to a different nearby person, so force a re-verify.
    last_verified_embedding: Optional[List[float]] = None


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

    def mark_resolved(
        self,
        tracklet: Tracklet,
        visitor_id: UUID,
        verified_ts: Optional[datetime] = None,
        verified_embedding: Optional[List[float]] = None,
    ) -> None:
        """Pin a tracklet to a visitor so later frames attach instead of re-resolving.

        `verified_ts` records when this pin was confirmed by a full gallery search;
        `verified_embedding` is the face that confirmed it. The fast-path uses both
        to decide when a re-verify is due.

        Only a CONFIDENT FACE match passes `verified_embedding` — that's the only
        signal strong enough to license skipping the gallery search next frame.
        Weaker pins (body / temporal / cross-camera / tracklet-attach) set the
        visitor but deliberately leave verification state untouched, so the next
        frame still does a full resolve rather than fast-pathing off a weak match.
        """
        tracklet.visitor_id = visitor_id
        if verified_embedding is not None:
            tracklet.last_verified_ts = verified_ts or tracklet.last_ts
            tracklet.last_verified_embedding = verified_embedding

    def needs_reverify(
        self,
        tracklet: Tracklet,
        bbox: dict,
        now: datetime,
        embedding: Optional[List[float]] = None,
    ) -> bool:
        """
        Whether a pinned tracklet must be re-resolved by a full gallery search
        rather than fast-attributed to its pinned visitor.

        Returns True (force a full resolve) when the fast-path is unsafe:
          • the tracklet isn't pinned yet, or was never verified,
          • more than TRACKLET_REVERIFY_SECONDS elapsed since the last verify,
          • the body box moved enough that IoU with the last box dropped below
            TRACKLET_REVERIFY_IOU (possible tracking swap),
          • the incoming face drifted from the one that last confirmed the pin
            (cosine below the verified bar) — the tracklet may have swapped to a
            different nearby person.
        Returns False only when all guards pass → safe to skip the gallery search.
        """
        if tracklet.visitor_id is None or tracklet.last_verified_ts is None:
            return True

        reverify_secs = settings.TRACKLET_REVERIFY_SECONDS
        if reverify_secs > 0:
            if (now - tracklet.last_verified_ts) >= timedelta(seconds=reverify_secs):
                return True

        if bbox_iou(bbox, tracklet.last_bbox) < settings.TRACKLET_REVERIFY_IOU:
            return True

        # If the incoming face has drifted from the one that confirmed the pin, the
        # cheap association may have jumped to a different person — re-resolve. The
        # bar is deliberately high (near-identical face) since within a few seconds
        # the same person's face is very stable; anything looser is a swap risk.
        if embedding is not None and tracklet.last_verified_embedding is not None:
            from app.similarity import cosine_similarity

            sim = cosine_similarity(
                embedding, tracklet.last_verified_embedding, assume_normalized=True
            )
            if sim < settings.RETURNING_FACE_THRESHOLD:
                return True

        return False

    def clear_visitor(self, visitor_id: UUID) -> None:
        """Drop a visitor's pin (e.g. after a merge or opt-out)."""
        for t in self._tracklets:
            if t.visitor_id == visitor_id:
                t.visitor_id = None
                t.last_verified_ts = None


# Module-level singleton shared within the process.
tracklet_buffer = TrackletBuffer()
