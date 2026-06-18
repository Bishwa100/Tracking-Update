"""
Bounding-box geometry helpers.

Single source of truth for the bbox clamp / crop / IoU / centre / offset logic
that was previously copy-pasted across cv_pipeline, detection_pipeline,
cascade_pipeline and camera_service. All boxes are {"x1","y1","x2","y2"} dicts
in pixel coordinates.
"""

from typing import Optional

import numpy as np


def clamp_bbox(bbox: dict, width: int, height: int) -> dict:
    """Clamp a bbox to the frame bounds [0, width] × [0, height] (ints)."""
    return {
        "x1": max(0, min(width, int(bbox.get("x1", 0)))),
        "y1": max(0, min(height, int(bbox.get("y1", 0)))),
        "x2": max(0, min(width, int(bbox.get("x2", 0)))),
        "y2": max(0, min(height, int(bbox.get("y2", 0)))),
    }


def bbox_area(bbox: dict) -> float:
    """Area of a bbox (0 for degenerate boxes)."""
    return max(0.0, bbox["x2"] - bbox["x1"]) * max(0.0, bbox["y2"] - bbox["y1"])


def bbox_center(bbox: dict) -> tuple[float, float]:
    """(cx, cy) centre of a bbox."""
    return ((bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0)


def bbox_iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two bboxes."""
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def offset_bbox(bbox: dict, dx: int, dy: int) -> dict:
    """Translate a bbox by (dx, dy) — e.g. mapping a crop-local box to frame coords."""
    return {
        "x1": int(bbox["x1"]) + dx,
        "y1": int(bbox["y1"]) + dy,
        "x2": int(bbox["x2"]) + dx,
        "y2": int(bbox["y2"]) + dy,
    }


def crop_from_frame(
    frame: Optional[np.ndarray],
    bbox: Optional[dict],
    margin: float = 0.0,
    min_size: int = 4,
) -> Optional[np.ndarray]:
    """
    Crop `bbox` (optionally expanded by `margin` fraction of its size) from a
    frame, clamped to bounds. Returns None when the frame/bbox is missing or the
    resulting crop is smaller than `min_size` on either side.
    """
    if frame is None or not bbox:
        return None
    h, w = frame.shape[:2]
    box = bbox
    if margin:
        bw = bbox.get("x2", 0) - bbox.get("x1", 0)
        bh = bbox.get("y2", 0) - bbox.get("y1", 0)
        mx, my = int(bw * margin), int(bh * margin)
        box = {
            "x1": bbox["x1"] - mx, "y1": bbox["y1"] - my,
            "x2": bbox["x2"] + mx, "y2": bbox["y2"] + my,
        }
    c = clamp_bbox(box, w, h)
    if c["x2"] - c["x1"] < min_size or c["y2"] - c["y1"] < min_size:
        return None
    return frame[c["y1"]:c["y2"], c["x1"]:c["x2"]]
