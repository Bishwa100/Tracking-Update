"""
Computer-vision pipeline.
Processes a frame through YOLOv8 (persons) → ArcFace (faces) → OSNet (bodies)
and returns one DetectedPerson per detected face/person.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import List, Optional

import cv2
import numpy as np

from app.ml_models import ModelManager, FaceEmbeddingCache
from app.config import settings
from app.geometry import bbox_iou as _compute_iou
from app.utils import normalize_embedding

logger = logging.getLogger(__name__)


def _is_group_frame(persons: List["DetectedPerson"], iou_threshold: float = 0.4) -> bool:
    """Return True when ≥3 person bboxes have significant pairwise overlap."""
    if len(persons) < 3:
        return False
    overlap = sum(
        1
        for i in range(len(persons))
        for j in range(i + 1, len(persons))
        if _compute_iou(persons[i].bbox, persons[j].bbox) > iou_threshold
    )
    total_pairs = len(persons) * (len(persons) - 1) / 2
    return (overlap / total_pairs) > 0.3


class PoseBin(str, Enum):
    FRONTAL = "frontal"       # yaw -15° to +15°
    LEFT_PROFILE = "left"     # yaw -90° to -15°
    RIGHT_PROFILE = "right"   # yaw +15° to +90°
    DOWNWARD = "down"         # pitch > 20° (looking at phone/menu)
    UNKNOWN = "unknown"


@dataclass
class FacePose:
    yaw: float
    pitch: float
    roll: float
    bin: PoseBin


def estimate_pose(face_landmarks: Optional[np.ndarray]) -> FacePose:
    """
    Geometric head-pose estimate from 5-point InsightFace landmarks.
    Landmarks: [left_eye, right_eye, nose, left_mouth, right_mouth]
    No extra model required — purely landmark geometry.
    """
    if face_landmarks is None or len(face_landmarks) < 5:
        return FacePose(yaw=0.0, pitch=0.0, roll=0.0, bin=PoseBin.UNKNOWN)

    left_eye, right_eye, nose, left_mouth, right_mouth = (
        np.asarray(face_landmarks[i], dtype=float) for i in range(5)
    )

    eye_center = (left_eye + right_eye) / 2.0
    mouth_center = (left_mouth + right_mouth) / 2.0
    iod = float(np.linalg.norm(right_eye - left_eye))
    if iod < 1e-6:
        return FacePose(yaw=0.0, pitch=0.0, roll=0.0, bin=PoseBin.UNKNOWN)

    # Yaw: nose offset from eye-center, normalized by IOD
    nose_offset_x = (nose[0] - eye_center[0]) / iod
    yaw = float(-np.degrees(np.arctan2(nose_offset_x, 1.0)) * 1.5)

    # Pitch: vertical nose position relative to eye-mouth midpoint
    face_mid_y = (eye_center[1] + mouth_center[1]) / 2.0
    nose_offset_y = (nose[1] - face_mid_y) / iod
    pitch = float(np.degrees(np.arctan2(nose_offset_y, 1.0)) * 2.0)

    # Roll from eye-line angle
    roll = float(np.degrees(np.arctan2(
        right_eye[1] - left_eye[1],
        right_eye[0] - left_eye[0],
    )))

    if abs(yaw) <= 15:
        bin_ = PoseBin.FRONTAL
    elif yaw < -15:
        bin_ = PoseBin.LEFT_PROFILE
    else:
        bin_ = PoseBin.RIGHT_PROFILE

    if pitch > 20 and bin_ == PoseBin.FRONTAL:
        bin_ = PoseBin.DOWNWARD

    return FacePose(yaw=yaw, pitch=pitch, roll=roll, bin=bin_)


@dataclass
class DetectedPerson:
    """A single detected person with extracted features."""
    bbox: dict  # {x1, y1, x2, y2}
    person_confidence: float
    face_embedding: Optional[List[float]] = None
    body_embedding: Optional[List[float]] = None
    face_bbox: Optional[dict] = None
    face_det_score: Optional[float] = None
    has_face: bool = False
    pose: Optional[FacePose] = None           # head-pose estimate
    face_landmarks: Optional[np.ndarray] = None  # 5-pt kps for downstream use
    is_masked: bool = False                   # set by mask detector


def face_passes_quality(face: dict) -> bool:
    """Reject faces too small or low-confidence to yield a reliable embedding."""
    if face["det_score"] < settings.MIN_FACE_DET_SCORE:
        return False
    fb = face["bbox"]
    return min(fb["x2"] - fb["x1"], fb["y2"] - fb["y1"]) >= settings.MIN_FACE_SIZE_PX


def refine_small_face(image: np.ndarray, face: dict) -> Optional[dict]:
    """
    Second chance for a face that failed the size/score gate: crop with margin,
    upscale so the face is ~160px, and re-run ArcFace. Returns a face dict with
    the bbox mapped back to frame coordinates, or None if it still fails.
    """
    model_mgr = ModelManager.get_instance()
    h, w = image.shape[:2]
    fb = face["bbox"]
    fw = fb["x2"] - fb["x1"]
    fh = fb["y2"] - fb["y1"]
    if fw <= 0 or fh <= 0:
        return None

    mx, my = int(fw * 0.5), int(fh * 0.5)
    x1, y1 = max(0, fb["x1"] - mx), max(0, fb["y1"] - my)
    x2, y2 = min(w, fb["x2"] + mx), min(h, fb["y2"] + my)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    scale = max(1.0, 160.0 / max(1, min(fw, fh)))
    if scale > 1.0:
        crop = cv2.resize(
            crop,
            (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    refined = model_mgr.extract_face_data(crop)
    if refined is None or refined["det_score"] < settings.MIN_FACE_DET_SCORE:
        return None

    refined["bbox"] = {
        "x1": int(refined["bbox"]["x1"] / scale) + x1,
        "y1": int(refined["bbox"]["y1"] / scale) + y1,
        "x2": int(refined["bbox"]["x2"] / scale) + x1,
        "y2": int(refined["bbox"]["y2"] / scale) + y1,
    }
    return refined


def process_frame(
    image: np.ndarray,
    extract_body: bool = True,
    embedding_cache: Optional[FaceEmbeddingCache] = None,
) -> List[DetectedPerson]:
    """
    Full CV pipeline for one frame:
      1. YOLOv8n → person boxes.
      2. One full-frame ArcFace pass → all faces (with small-face rescue).
      3. Assign each face to the person box containing it; per-crop fallback for
         persons with no full-frame face.
      4. One batched OSNet pass → body embeddings (when extract_body and a body
         model are loaded).

    Returns one DetectedPerson per person box (plus any face-only detections
    that fell outside every person box).
    """
    model_mgr = ModelManager.get_instance()
    detected_persons: List[DetectedPerson] = []

    _t_yolo = perf_counter()
    persons = model_mgr.detect_persons(image, confidence=settings.YOLO_PERSON_CONFIDENCE)
    yolo_secs = perf_counter() - _t_yolo
    logger.debug("Detected %d person(s) in frame.", len(persons))

    # All faces in one pass; rescue small faces rather than dropping them.
    _t_face = perf_counter()
    frame_faces = model_mgr.extract_all_faces(image, embedding_cache=embedding_cache)
    gated_faces: List[dict] = []
    for ff in frame_faces:
        if face_passes_quality(ff):
            gated_faces.append(ff)
        else:
            refined = refine_small_face(image, ff)
            if refined is not None:
                gated_faces.append(refined)
    frame_faces = gated_faces
    face_secs = perf_counter() - _t_face

    # Nothing to attribute identity to and no bodies to embed → skip the rest.
    if not persons and not frame_faces:
        logger.debug(
            "process_frame timing: yolo=%.3fs arcface=%.3fs (0 person(s), 0 face(s)).",
            yolo_secs, face_secs,
        )
        return detected_persons

    body_queue: List[tuple] = []  # (DetectedPerson, crop)
    h, w = image.shape[:2]
    used_faces: set = set()

    for person in persons:
        bbox = person["bbox"]
        x1, y1 = max(0, bbox["x1"]), max(0, bbox["y1"])
        x2, y2 = min(w, bbox["x2"]), min(h, bbox["y2"])
        if x2 - x1 < 20 or y2 - y1 < 20:
            continue

        person_crop = image[y1:y2, x1:x2]
        detected = DetectedPerson(bbox=bbox, person_confidence=person["confidence"])

        # Assign the highest-scoring full-frame face whose centre is in this box.
        face_data = None
        best_score = -1.0
        best_idx = -1
        for idx, ff in enumerate(frame_faces):
            if idx in used_faces:
                continue
            fb = ff["bbox"]
            cx = (fb["x1"] + fb["x2"]) * 0.5
            cy = (fb["y1"] + fb["y2"]) * 0.5
            if x1 <= cx <= x2 and y1 <= cy <= y2 and ff["det_score"] > best_score:
                best_score = ff["det_score"]
                face_data = ff
                best_idx = idx
        face_from_full_frame = face_data is not None
        if best_idx >= 0:
            used_faces.add(best_idx)

        # Fallback: no full-frame face landed here — retry on the upscaled crop.
        # Costs one extra ArcFace pass per faceless person box (the dominant
        # per-frame cost in crowds), so it's gated behind PER_PERSON_FACE_FALLBACK.
        if face_data is None and settings.PER_PERSON_FACE_FALLBACK:
            _t_fb = perf_counter()
            face_data = model_mgr.extract_face_data(person_crop)
            face_secs += perf_counter() - _t_fb
            face_from_full_frame = False
            if face_data is not None and face_data["det_score"] < settings.MIN_FACE_DET_SCORE:
                face_data = None

        if face_data is not None:
            detected.has_face = True
            detected.face_embedding = normalize_embedding(face_data["embedding"])
            detected.face_det_score = face_data["det_score"]
            if face_from_full_frame:
                detected.face_bbox = face_data["bbox"]
            else:
                detected.face_bbox = {
                    "x1": face_data["bbox"]["x1"] + x1,
                    "y1": face_data["bbox"]["y1"] + y1,
                    "x2": face_data["bbox"]["x2"] + x1,
                    "y2": face_data["bbox"]["y2"] + y1,
                }
            # Pose estimation from 5-point landmarks (kps key from InsightFace)
            kps = face_data.get("kps")
            if kps is not None:
                kps_arr = np.asarray(kps, dtype=float)
                detected.face_landmarks = kps_arr
                detected.pose = estimate_pose(kps_arr)
            else:
                detected.pose = FacePose(yaw=0.0, pitch=0.0, roll=0.0, bin=PoseBin.UNKNOWN)

        if extract_body and model_mgr.has_body_model:
            body_queue.append((detected, person_crop))

        detected_persons.append(detected)

    # Faces that fell outside every person box still count (e.g. seated patrons
    # whose body is occluded by a table). Emit them as face-only detections.
    for idx, ff in enumerate(frame_faces):
        if idx in used_faces:
            continue
        fb = ff["bbox"]
        kps = ff.get("kps")
        kps_arr = np.asarray(kps, dtype=float) if kps is not None else None
        pose = estimate_pose(kps_arr) if kps_arr is not None else FacePose(0.0, 0.0, 0.0, PoseBin.UNKNOWN)
        detected_persons.append(
            DetectedPerson(
                bbox=fb,
                person_confidence=float(ff["det_score"]),
                face_embedding=normalize_embedding(ff["embedding"]),
                face_bbox=fb,
                face_det_score=float(ff["det_score"]),
                has_face=True,
                pose=pose,
                face_landmarks=kps_arr,
            )
        )

    # Single batched body-model pass for all queued person crops.
    if body_queue:
        try:
            embeddings = model_mgr.extract_body_embeddings([c for _, c in body_queue])
            for (det, _), body_emb in zip(body_queue, embeddings):
                det.body_embedding = normalize_embedding(body_emb)
        except Exception as e:
            logger.error("Failed to extract body embeddings: %s", e)

    logger.debug(
        "process_frame timing: yolo=%.3fs arcface=%.3fs (%d person(s), %d face(s)).",
        yolo_secs, face_secs, len(persons), len(frame_faces),
    )
    return detected_persons
