"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ── Common ───────────────────────────────────────────────────


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


# ── Camera ROI ───────────────────────────────────────────────

class RoiRequest(BaseModel):
    roi: Optional[BoundingBox] = None


class RoiResponse(BaseModel):
    roi: Optional[BoundingBox] = None


# ── Detection (/api/detect) ──────────────────────────────────


class DetectionItem(BaseModel):
    visitor_id: Optional[UUID] = None
    is_new: bool
    is_ambiguous: bool = False
    visit_id: Optional[UUID] = None
    face_confidence: Optional[float] = None
    body_confidence: Optional[float] = None
    match_source: str = "none"  # "face" | "body" | "new" | "none"
    bbox: Optional[BoundingBox] = None


class DetectResponse(BaseModel):
    detections: List[DetectionItem]
    new_visitors_count: int
    returning_visitors_count: int
    frames_processed: int


# ── Visitors ─────────────────────────────────────────────────


class VisitorSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: Optional[str] = None
    visit_count: int
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    is_staff: bool = False
    is_active: bool = True
    best_face_det_score: Optional[float] = None
    thumbnail_url: Optional[str] = None


class VisitorListResponse(BaseModel):
    total: int
    visitors: List[VisitorSummary]


class VisitSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    entered_at: datetime
    left_at: Optional[datetime] = None
    duration_minutes: Optional[int] = None
    detection_count: int = 0
    best_face_confidence: Optional[float] = None
    camera_id: Optional[str] = None
    is_active: bool = False


class VisitorDetailResponse(VisitorSummary):
    notes: Optional[str] = None
    total_faces_recorded: int = 0
    latest_visit: Optional[VisitSummary] = None
    consent_status: Optional[str] = None
    consent_at: Optional[datetime] = None
    consent_method: Optional[str] = None
    opted_out_at: Optional[datetime] = None


class ConsentUpdateRequest(BaseModel):
    # "explicit" | "implicit" | "opted_out"
    consent_status: str
    method: Optional[str] = "staff"


class VisitListResponse(BaseModel):
    total: int
    visits: List[VisitSummary]


class VisitorUpdateRequest(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    is_staff: Optional[bool] = None


# ── Camera control ───────────────────────────────────────────


class CameraStartRequest(BaseModel):
    # "0" = USB webcam, "rtsp://..." = IP camera, "/path/to.mp4" = file.
    source: Optional[str] = None
    camera_id: Optional[str] = None
    fps: Optional[float] = None


class CameraStatusResponse(BaseModel):
    is_running: bool
    source: Optional[str] = None
    source_kind: Optional[str] = None  # "video" | "camera"
    looping: bool = False
    camera_id: Optional[str] = None
    fps: Optional[float] = None
    frames_processed: int = 0
    frames_skipped: int = 0
    persons_detected: int = 0
    new_visitors: int = 0
    returning_visitors: int = 0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None


# ── Admin ────────────────────────────────────────────────────


class MergeRequest(BaseModel):
    # Merge this visitor's history INTO target_visitor_id, then delete it.
    target_visitor_id: UUID


class MarkStaffRequest(BaseModel):
    is_staff: bool = True


# ── Analytics ────────────────────────────────────────────────


class AnalyticsSummary(BaseModel):
    total_unique_visitors: int
    total_visits: int
    new_visitors: int
    returning_visitors: int
    average_duration_minutes: float
    return_rate: float
    visits_by_day: List[dict]


class FrequencyDistribution(BaseModel):
    distribution: dict


class HourlyBreakdown(BaseModel):
    hourly: List[dict]


class TopVisitor(BaseModel):
    visitor_id: UUID
    name: Optional[str] = None
    visit_count: int
    first_visit: Optional[datetime] = None
    last_visit: Optional[datetime] = None
    avg_duration_minutes: Optional[float] = None


# ── Activity feed ────────────────────────────────────────────


class ActivityEvent(BaseModel):
    id: UUID
    detected_at: datetime
    visitor_id: Optional[UUID] = None
    visitor_name: Optional[str] = None
    thumbnail_url: Optional[str] = None
    visit_id: Optional[UUID] = None
    face_similarity: Optional[float] = None
    body_similarity: Optional[float] = None
    is_new_visitor: bool = False
    is_ambiguous: bool = False
    match_source: Optional[str] = None
    camera_id: Optional[str] = None


class ActivityResponse(BaseModel):
    total: int
    events: List[ActivityEvent]


# ── Settings (read-only) ─────────────────────────────────────


class SettingsResponse(BaseModel):
    # Recognition thresholds
    returning_face_threshold: float
    new_visitor_max_similarity: float
    reject_similarity: float
    ambiguity_margin: float
    strong_match_threshold: float
    allow_body_fallback: bool
    returning_body_threshold: float
    # Gallery
    max_faces_per_visitor: int
    face_quality_cutoff: float
    # Visit sessions
    visit_cooldown_minutes: int
    max_visit_duration_hours: int
    stale_check_interval_seconds: int
    # Camera
    camera_source: str
    camera_fps: float
    frame_dedup_enabled: bool
    # Privacy
    visitor_retention_days: int


# ── Health ───────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    database: str
    models_loaded: bool
    yolo_loaded: bool
    arcface_loaded: bool
    body_model: str
    camera_running: bool
    visitors_count: int
    total_visits: int
