"""
Application configuration via pydantic-settings.
All values read from environment variables / .env file.
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator
import json


class Settings(BaseSettings):
    # ── Database ─────────────────────────────────────────────
    DATABASE_URL: str = (
        "postgresql+asyncpg://tracker:tracker_pass@localhost:3018/restaurant_tracker"
    )

    # ── Authentication ───────────────────────────────────────
    API_KEY: str = "changeme-set-a-real-key"

    # ── Server / CORS ────────────────────────────────────────
    CORS_ORIGINS: List[str] = ["http://localhost:3003"]

    # ── Identity resolution thresholds ───────────────────────
    # Minimum face cosine similarity to recognise a detection as a RETURNING
    # visitor. buffalo_l ArcFace same-person similarity is typically 0.5-0.7 and
    # different-person 0.1-0.3; 0.55 is a sensible starting point but MUST be
    # re-calibrated on your own camera footage.
    RETURNING_FACE_THRESHOLD: float = 0.55
    # Below this, a detection is treated as definitely a NEW visitor (no body
    # check). Between this and RETURNING_FACE_THRESHOLD is the "grey zone".
    REJECT_SIMILARITY: float = 0.35
    # Top-2 gallery matches must differ by at least this much, otherwise the
    # detection is AMBIGUOUS (two known visitors explain the face equally well)
    # and is skipped rather than risk a false merge.
    AMBIGUITY_MARGIN: float = 0.05
    # Above this similarity the centroid is updated and the face is considered
    # for the gallery (a confident, high-quality match).
    STRONG_MATCH_THRESHOLD: float = 0.65
    # A new visitor is only auto-created when the best gallery similarity is at
    # or below this value AND face quality clears FACE_QUALITY_CUTOFF. Keeping
    # creation conservative (low) avoids fragmenting one person into many
    # visitor records in the grey zone.
    NEW_VISITOR_MAX_SIMILARITY: float = 0.45

    # ── Body (OSNet) re-ID fallback ──────────────────────────
    # OSNet body embeddings are CLOTHING/APPEARANCE dependent: they only
    # discriminate "same person, same outfit, minutes apart", NOT the same
    # regular customer on a different day. Body matching is therefore a
    # short-term, same-session re-acquisition signal only and is OFF by default
    # as a cross-detection identity signal. Enable only if you understand it
    # will NOT recognise returning visitors across visits.
    ALLOW_BODY_FALLBACK: bool = False
    RETURNING_BODY_THRESHOLD: float = 0.55

    # ── Gallery management ───────────────────────────────────
    # Keep the top-N highest-quality face embeddings per visitor (multi-pose
    # gallery). More poses = better recall over time at a bounded table size.
    MAX_FACES_PER_VISITOR: int = 10
    # Minimum ArcFace det_score for a face to be stored in the gallery or to
    # seed a new visitor. Low-quality faces produce unreliable embeddings.
    FACE_QUALITY_CUTOFF: float = 0.45
    # Operator-triggered "auto-clean faces": gallery faces whose combined clarity
    # (landmark frontality + Laplacian sharpness + det_score) falls below this are
    # deleted, except the single clearest face which is always kept.
    FACE_CLARITY_CUTOFF: float = 0.45
    # Laplacian-variance reference for blur normalization; a crop at/above this is
    # treated as fully sharp (blur sub-score = 1.0).
    FACE_BLUR_REF: float = 120.0
    # Weighted-moving-average learning rate base for the adaptive centroid.
    CENTROID_ALPHA_BASE: float = 0.15

    # ── Visit session tracking ───────────────────────────────
    # After this many minutes with no detection, an active visit is closed.
    # The next detection of that visitor starts a NEW visit (and increments
    # their visit_count). A brief absence under this window is treated as the
    # same visit (e.g. a bathroom break).
    VISIT_COOLDOWN_MINUTES: int = 20
    # Hard cap: auto-close visits that have been open this long (prevents
    # forever-open records if someone is mis-tracked).
    MAX_VISIT_DURATION_HOURS: int = 4
    # Background task cadence for closing stale visits.
    STALE_CHECK_INTERVAL_SECONDS: int = 60

    # ── Camera service ───────────────────────────────────────
    # "0" = first USB webcam, "rtsp://..." = IP camera, "/path/to.mp4" = file.
    CAMERA_SOURCE: str = "0"
    # Frames processed per second. Webcams run at 30 fps; 1 fps is ample for
    # visitor analytics and cuts CPU ~30x.
    CAMERA_FPS: float = 1.0
    # Camera id stored on visits/events (lets analytics distinguish cameras).
    CAMERA_ID: str = "cam-0"
    # Auto-start the camera loop on application startup.
    CAMERA_AUTOSTART: bool = False
    # JPEG quality (0-100) for snapshot / live-feed frames.
    LIVE_FEED_JPEG_QUALITY: int = 70
    # Live preview encode rate (frames/sec). The display stays at this rate
    # independent of how often detection runs, so the feed never freezes when
    # frame-dedup skips the (expensive) detection pass on near-identical frames.
    LIVE_PREVIEW_FPS: float = 15.0

    # ── Frame preprocessing ──────────────────────────────────
    # Downscale frames so the longest side is at most this many pixels before
    # inference. 0 disables the cap.
    MAX_FRAME_LONG_SIDE: int = 1280

    # ── Frame de-duplication (video / camera) ────────────────
    # Skip the heavy YOLO+ArcFace pass on a frame near-identical to the previous
    # one (static scene). ~1 ms comparison vs ~1 s of detection.
    FRAME_DEDUP_ENABLED: bool = True
    # Mean absolute pixel difference (0-255 on a 32x32 grayscale thumbnail)
    # below which two consecutive frames are treated as the same scene.
    FRAME_DEDUP_MAD_THRESHOLD: float = 4.0

    # ── Processing device ────────────────────────────────────
    # Where inference runs: "auto" (GPU if a CUDA device is available, else CPU),
    # "cpu" (force CPU), or "cuda" (force GPU; falls back to CPU with a warning if
    # no CUDA device is present). Switchable at runtime from the dashboard — see
    # /api/admin/device. GPU requires a CUDA torch build + onnxruntime-gpu.
    DEVICE: str = "auto"

    # ── Models ───────────────────────────────────────────────
    YOLO_MODEL_PATH: str = "yolov8n.pt"
    # Export YOLO to ONNX once at startup (2-3x faster than PyTorch on CPU).
    YOLO_USE_ONNX: bool = True
    # Person-detection confidence threshold.
    YOLO_PERSON_CONFIDENCE: float = 0.5

    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    # ArcFace detector input size. 640 keeps full recall for small/distant
    # faces; 480/320 are faster for close-up footage.
    INSIGHTFACE_DET_SIZE: int = 640

    # Body re-ID model: "osnet" (OSNet x0.25, 512-d, fast) or "none".
    BODY_MODEL_TYPE: str = "osnet"
    OSNET_WEIGHTS_PATH: str = "models/osnet_x0_25_msmt17.pth"
    OSNET_WEIGHTS_URL: str = (
        "https://huggingface.co/kaiyangzhou/osnet/resolve/main/"
        "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_"
        "lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
    )

    # ── CPU performance ──────────────────────────────────────
    # Math-library / torch thread count. 0 = use every physical core.
    CPU_THREADS: int = 0
    # Max frames running heavy inference at once. Each inference already uses
    # every core, so 1 = serialize for predictable latency. Keep equal to
    # INFERENCE_WORKERS; raising both shares one set of model objects across
    # threads (see INFERENCE_WORKERS caveat).
    INFERENCE_MAX_CONCURRENCY: int = 1

    # ── Parallel streaming pipeline ──────────────────────────
    # Run the camera as a multi-stage pipeline (capture / inference / post-
    # process+DB / encode), each stage concurrent, so the GPU stays busy on
    # inference while the CPU handles capture, DB writes and JPEG encoding in
    # parallel. This is the main throughput win and is safe with one worker.
    # False falls back to the original single sequential loop.
    PIPELINE_PARALLEL: bool = True
    # Number of concurrent inference workers. 1 is safe and already keeps the GPU
    # busy back-to-back. >1 overlaps CPU preprocessing with GPU compute for more
    # throughput, BUT the shared YOLO/InsightFace/OSNet objects are not
    # guaranteed thread-safe — only raise this once each worker has its own model
    # instances. Also raise INFERENCE_MAX_CONCURRENCY to match.
    INFERENCE_WORKERS: int = 1
    # Optional cap on processed frames/sec (0 = unlimited, process as fast as the
    # GPU allows). For video files this also paces playback; live cameras are
    # naturally bounded by their own frame rate.
    PIPELINE_MAX_FPS: float = 0.0

    # ── Face quality gates (detection) ───────────────────────
    # Faces smaller than this (px, min of width/height) are ignored — too small
    # for a reliable embedding.
    MIN_FACE_SIZE_PX: int = 40
    MIN_FACE_DET_SCORE: float = 0.40

    # ── Upload limits (/api/detect) ──────────────────────────
    VIDEO_MAX_SIZE_MB: int = 100
    VIDEO_MAX_DURATION_SECONDS: int = 60
    # Frames sampled per second of an uploaded video.
    FRAMES_PER_SECOND: int = 1

    # ── Storage ──────────────────────────────────────────────
    # Best face crops saved here as thumbnails (one per visitor, updated when a
    # higher-quality face is seen).
    VISITOR_PHOTO_DIR: str = "storage/visitor_photos"
    # Scratch dir for uploaded media in /api/detect (cleaned after processing).
    DETECT_TMP_DIR: str = "storage/tmp_detect"
    # Save annotated detection frames to disk for the audit trail.
    DETECT_SAVE_FRAMES: bool = False

    # ── Privacy / retention ──────────────────────────────────
    # Auto-purge visitors (and their cascade rows) whose last_seen_at is older
    # than this many days. 0 disables purging. Auto-enrolling biometric data of
    # patrons may be regulated (GDPR / BIPA / etc.) — set an appropriate value
    # and document your consent/notice posture before deploying.
    VISITOR_RETENTION_DAYS: int = 0
    RETENTION_PURGE_INTERVAL_HOURS: int = 24

    # ── Analytics ────────────────────────────────────────────
    ANALYTICS_DEFAULT_DAYS: int = 30

    # ── Face preprocessing (CLAHE + gamma) ──────────────────
    FACE_PREPROCESSING_CLAHE: bool = True
    FACE_PREPROCESSING_GAMMA: bool = True
    CLAHE_CLIP_LIMIT: float = 2.0
    CLAHE_GRID_SIZE: List[int] = [8, 8]

    # ── Pose-aware gallery ───────────────────────────────────
    POSE_AWARE_GALLERY: bool = True
    MIN_FACES_PER_POSE_BIN: int = 2
    MAX_FACES_PER_POSE_BIN: int = 4

    # ── Mask detection ───────────────────────────────────────
    MASK_DETECTION_ENABLED: bool = True
    MASKED_FACE_THRESHOLD_OFFSET: float = -0.05

    # ── Temporal consistency gate ────────────────────────────
    TEMPORAL_WINDOW_SECONDS: float = 30.0
    TEMPORAL_MAX_PIXEL_DISTANCE: float = 150.0
    TEMPORAL_MIN_SIMILARITY: float = 0.50

    # ── Smart / seated cooldown ──────────────────────────────
    SEATED_COOLDOWN_MINUTES: int = 45

    # ── Cascade (skip body when face is strong) ──────────────
    FACE_CONF_SKIP_BODY: float = 0.60

    # ── Redis (for multi-worker visit state) ─────────────────
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Consent / privacy ────────────────────────────────────
    DEFAULT_CONSENT_MODE: str = "implicit"
    PHYSICAL_NOTICE_REQUIRED: bool = True
    CONSENT_NOTICE_TEXT: str = (
        "This premises uses facial recognition for analytics. "
        "By entering you consent to biometric processing. "
        "Opt out: ask staff."
    )
    CONSENT_QR_URL: str = ""
    ACTIVE_VISITOR_RETENTION_DAYS: int = 30
    OPTED_OUT_EMBEDDING_TTL_DAYS: int = 7

    # ── Auto-tuning ──────────────────────────────────────────
    AUTO_TUNING_ENABLED: bool = True
    AUTO_TUNING_INTERVAL_DAYS: int = 7

    # ── Monitoring ───────────────────────────────────────────
    HEALTH_CHECK_INTERVAL_SECONDS: int = 30
    ALERT_WEBHOOK_URL: str = ""

    # ── Admin API ────────────────────────────────────────────
    ADMIN_API_KEY: str = ""

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [origin.strip() for origin in v.split(",")]
        return v

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
