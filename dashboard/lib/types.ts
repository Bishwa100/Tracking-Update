// Mirrors the backend Pydantic schemas (app/schemas.py).

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RegionOfInterest {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface RoiResponse {
  roi: RegionOfInterest | null;
}

export interface VisitorSummary {
  id: string;
  name: string | null;
  visit_count: number;
  first_seen_at: string | null;
  last_seen_at: string | null;
  is_staff: boolean;
  is_active: boolean;
  best_face_det_score: number | null;
  thumbnail_url: string | null;
}

export interface VisitorListResponse {
  total: number;
  visitors: VisitorSummary[];
}

export interface VisitSummary {
  id: string;
  entered_at: string;
  left_at: string | null;
  duration_minutes: number | null;
  detection_count: number;
  best_face_confidence: number | null;
  camera_id: string | null;
  is_active: boolean;
}

export interface VisitorDetail extends VisitorSummary {
  notes: string | null;
  total_faces_recorded: number;
  latest_visit: VisitSummary | null;
  consent_status?: string | null;
  consent_at?: string | null;
  consent_method?: string | null;
  opted_out_at?: string | null;
}

export interface VisitListResponse {
  total: number;
  visits: VisitSummary[];
}

export interface CameraStatus {
  is_running: boolean;
  source: string | null;
  source_kind?: "video" | "camera" | null;
  looping?: boolean;
  camera_id: string | null;
  fps: number | null;
  frames_processed: number;
  frames_skipped: number;
  persons_detected: number;
  new_visitors: number;
  returning_visitors: number;
  uptime_seconds: number;
  last_error: string | null;
}

export interface AnalyticsSummary {
  total_unique_visitors: number;
  total_visits: number;
  new_visitors: number;
  returning_visitors: number;
  average_duration_minutes: number;
  return_rate: number;
  visits_by_day: { day: string; visits: number }[];
}

export interface FrequencyDistribution {
  distribution: Record<string, number>;
}

export interface HourlyBreakdown {
  hourly: { hour: number; new: number; returning: number }[];
}

export interface TopVisitor {
  visitor_id: string;
  name: string | null;
  visit_count: number;
  first_visit: string | null;
  last_visit: string | null;
  avg_duration_minutes: number | null;
}

export interface ActivityEvent {
  id: string;
  detected_at: string;
  visitor_id: string | null;
  visitor_name: string | null;
  thumbnail_url: string | null;
  visit_id: string | null;
  face_similarity: number | null;
  body_similarity: number | null;
  is_new_visitor: boolean;
  is_ambiguous: boolean;
  match_source: string | null;
  camera_id: string | null;
}

export interface ActivityResponse {
  total: number;
  events: ActivityEvent[];
}

export interface SettingsResponse {
  returning_face_threshold: number;
  new_visitor_max_similarity: number;
  reject_similarity: number;
  ambiguity_margin: number;
  strong_match_threshold: number;
  allow_body_fallback: boolean;
  returning_body_threshold: number;
  max_faces_per_visitor: number;
  face_quality_cutoff: number;
  visit_cooldown_minutes: number;
  max_visit_duration_hours: number;
  stale_check_interval_seconds: number;
  camera_source: string;
  camera_fps: number;
  frame_dedup_enabled: boolean;
  visitor_retention_days: number;
}

export interface HealthResponse {
  status: string;
  database: string;
  models_loaded: boolean;
  yolo_loaded: boolean;
  arcface_loaded: boolean;
  body_model: string;
  camera_running: boolean;
  visitors_count: number;
  total_visits: number;
}

export interface DetectionItem {
  visitor_id: string | null;
  is_new: boolean;
  is_ambiguous: boolean;
  visit_id: string | null;
  face_confidence: number | null;
  body_confidence: number | null;
  match_source: string;
  bbox: BoundingBox | null;
}

export interface DetectResponse {
  detections: DetectionItem[];
  new_visitors_count: number;
  returning_visitors_count: number;
  frames_processed: number;
}

// ── New backend capabilities ────────────────────────────────

export interface ReviewFlag {
  id: string;
  visitor_id: string;
  flag_type: string;
  detail: string;
  created_at: string | null;
}

export interface ConfidenceWeighted {
  unique_visitors: number;
  effective_unique: number;
  avg_confidence: number;
  min_confidence_filter: number;
}

export interface ConfidenceWeightedSummary extends AnalyticsSummary {
  confidence_weighted: ConfidenceWeighted;
}

export interface DetectionQuality {
  bands: { high: number; medium: number; low: number };
  pct_high: number;
  pct_medium: number;
  pct_low: number;
  total_detections: number;
}

// Runtime-editable admin settings (from GET/PATCH /api/admin/settings).
export type AdminSettings = Record<string, number | boolean | string>;

export interface VideoStreamResponse {
  status: string;
  filename: string;
  source: string;
  size_mb: number;
  looping: boolean;
}

export interface LiveFeedMessage {
  type: string;
  is_running: boolean;
  currently_inside: number;
  stats: {
    frames_processed: number;
    frames_skipped: number;
    persons_detected: number;
    new_visitors: number;
    returning_visitors: number;
  };
  frame: string | null;
}
