"""
Face clarity scoring — "is this face clearly visible?".

Combines three cheap, model-free signals into one [0, 1] clarity score:

  • frontality — how head-on the face is, from the 5-point-landmark pose bin
    (frontal faces are clearest; profiles / downward / unknown are penalized).
  • sharpness  — Laplacian variance of the crop, normalized by FACE_BLUR_REF
    (blurry crops score low). Requires crop pixels.
  • det_score  — InsightFace detection confidence already stored per face.

Used by the review-queue "auto-clean faces" action to prune unclear gallery
faces. No extra model or dependency — only OpenCV + values already on the row.
"""

from typing import Optional

import cv2
import numpy as np

from app.config import settings

# Frontality weight per landmark-derived pose bin (see cv_pipeline.estimate_pose).
_FRONTALITY = {
    "frontal": 1.0,
    "down": 0.55,
    "left": 0.45,
    "right": 0.45,
    "unknown": 0.6,
}


def sharpness_score(crop_bgr: Optional[np.ndarray]) -> Optional[float]:
    """Laplacian-variance sharpness normalized to [0, 1]; None if no crop."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return max(0.0, min(var / settings.FACE_BLUR_REF, 1.0))


def compute_clarity(
    crop_bgr: Optional[np.ndarray],
    det_score: Optional[float],
    pose_bin: Optional[str],
) -> dict:
    """
    Return clarity sub-scores and a combined score in [0, 1].

    With a crop available we weight frontality/sharpness/det (0.40/0.35/0.25);
    without one (legacy faces) we drop the blur term and reweight to
    frontality/det (0.60/0.40).
    """
    frontality = _FRONTALITY.get((pose_bin or "unknown").lower(), 0.6)
    det = max(0.0, min(float(det_score or 0.0), 1.0))
    blur = sharpness_score(crop_bgr)

    if blur is not None:
        clarity = 0.40 * frontality + 0.35 * blur + 0.25 * det
    else:
        clarity = 0.60 * frontality + 0.40 * det

    return {
        "clarity": round(float(clarity), 4),
        "frontality": round(float(frontality), 4),
        "blur": round(float(blur), 4) if blur is not None else None,
        "det": round(float(det), 4),
    }
