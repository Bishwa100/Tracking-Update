"""
Mask detection via periocular heuristic.

When the lower face (nose+mouth region) is occluded we fall back to the
upper-face (periocular) region for embedding extraction, and loosen the
similarity threshold by MASKED_FACE_THRESHOLD_OFFSET.
"""

from typing import Optional

import cv2
import numpy as np

from app.config import settings


def _lower_face_brightness(face_crop: np.ndarray) -> float:
    """Mean pixel value of the lower 40% of the face crop (nose-mouth zone)."""
    h = face_crop.shape[0]
    lower = face_crop[int(h * 0.6):, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY) if lower.ndim == 3 else lower
    return float(np.mean(gray))


def _upper_face_uniformity(face_crop: np.ndarray) -> float:
    """Std-dev of the upper 40% of the face crop (forehead-eye zone)."""
    h = face_crop.shape[0]
    upper = face_crop[: int(h * 0.4), :]
    gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY) if upper.ndim == 3 else upper
    return float(np.std(gray))


def is_masked(face_crop: np.ndarray) -> bool:
    """
    Heuristic mask detector.

    Returns True when the lower-face region is unusually uniform (solid colour
    mask) AND the upper-face region has normal texture variance (real eyes).
    Tuned for surgical and cloth masks in restaurant lighting.
    """
    if face_crop is None or face_crop.size == 0:
        return False
    h, w = face_crop.shape[:2]
    if h < 32 or w < 32:
        return False

    lower_std = _lower_std(face_crop)
    upper_std = _upper_face_uniformity(face_crop)
    lower_bright = _lower_face_brightness(face_crop)

    # A mask makes the lower face FLAT (low texture) while the eye region keeps
    # normal texture (real eyes). But a flat lower face alone is not enough — a
    # clean-shaven chin or a plain shirt in shadow is also flat and would falsely
    # loosen the match threshold. A real surgical/cloth mask is also a distinctly
    # NON-skin tone: clearly bright (white surgical) or clearly dark (black cloth),
    # not a mid-tone like skin. Require all three so we don't mask-flag bare faces.
    lower_is_flat = lower_std < 18.0
    upper_has_texture = upper_std > 12.0
    non_skin_tone = lower_bright > 170.0 or lower_bright < 70.0
    return lower_is_flat and upper_has_texture and non_skin_tone


def _lower_std(face_crop: np.ndarray) -> float:
    h = face_crop.shape[0]
    lower = face_crop[int(h * 0.6):, :]
    gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY) if lower.ndim == 3 else lower
    return float(np.std(gray))


def extract_periocular_region(face_crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Return a crop of the eye-region (upper 55% of the face) upscaled to
    112x112 so ArcFace can still produce an embedding from a masked face.
    Returns None when the crop would be too small.
    """
    if face_crop is None or face_crop.size == 0:
        return None
    h, w = face_crop.shape[:2]
    if h < 32 or w < 32:
        return None
    upper = face_crop[: int(h * 0.55), :]
    if upper.size == 0:
        return None
    return cv2.resize(upper, (112, 112), interpolation=cv2.INTER_CUBIC)


def masked_threshold_offset() -> float:
    """Return threshold offset to apply when matching a masked face."""
    return settings.MASKED_FACE_THRESHOLD_OFFSET if settings.MASK_DETECTION_ENABLED else 0.0
