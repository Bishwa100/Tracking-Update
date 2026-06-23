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
    # Number of gallery rows fetched per face from the HNSW search. Rows are
    # collapsed to the best score PER VISITOR before the ambiguity gate, so a
    # value > 2 lets us compare the top visitor against the closest DIFFERENT
    # visitor (top-2 rows can otherwise both belong to the same person, hiding
    # the real runner-up).
    IDENTITY_TOP_K: int = 10
    # pgvector HNSW search breadth (the size of the dynamic candidate list). The
    # default of 40 silently under-fetches once the gallery search also filters on
    # is_active / consent_status / pose_bin INSIDE the LIMIT, so true matches can
    # be missed → the same person re-registered as new. Set per resolve
    # transaction via `SET LOCAL hnsw.ef_search`. Must be >= IDENTITY_TOP_K; 100 is
    # a strong recall/latency trade-off for galleries up to ~1e6 rows. 0 leaves the
    # server default (no SET issued).
    #
    # Note: pgvector's `<=>` distance for a returned row is already EXACT (we use
    # L2-normalized embeddings, so `1 - cosine_distance` is the true cosine). HNSW
    # is approximate only in WHICH rows it returns — so the right lever for recall
    # is ef_search (this), not a NumPy re-rank of the rows it already returned.
    HNSW_EF_SEARCH: int = 100
    # What to do with a "grey zone" face (REJECT_SIMILARITY < top_sim <
    # RETURNING_FACE_THRESHOLD): it is neither a confident match nor a confident
    # stranger, so creating a NEW visitor from it is the #1 cause of duplicate
    # records for the same person seen at a new angle/lighting/camera.
    #   "review"   — hold (do not register); record a grey_zone audit event.
    #   "tracklet" — (Phase 3) buffer across frames, resolve once. Until the
    #                tracklet buffer ships this behaves like "review".
    #   "register" — legacy behaviour: create a NEW visitor if quality clears the
    #                cutoff (kept only as an explicit escape hatch).
    GREY_ZONE_POLICY: str = "review"
    # Default similarity floor for the review-queue "auto-merge duplicates" sweep.
    # Only probable_duplicate flags whose recorded similarity is >= this value are
    # mass-merged; weaker pairs stay for human review. Set conservatively (a
    # confident same-person match) so a bulk sweep never fuses two people.
    AUTO_MERGE_MIN_SIMILARITY: float = 0.65

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

    # ── Face embedding cache (dHash) ─────────────────────────
    # The per-stream ArcFace embedding cache is bounded with LRU eviction so a
    # 24/7 stream can't grow it without limit (OOM). ~10k entries ≈ 20 MB.
    FACE_CACHE_MAX_ENTRIES: int = 10_000
    # Drop cache entries older than this many seconds (0 = no TTL, LRU only).
    FACE_CACHE_TTL_SECONDS: int = 3_600

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
    # When the single full-frame ArcFace pass assigns no face to a person box,
    # retry face detection on that person's upscaled crop. This rescues small /
    # partially-occluded faces the full-frame detector missed, but costs one extra
    # ArcFace forward pass PER such person — the dominant per-frame cost in crowds.
    # Set False to trust the full-frame pass only (faster; slightly lower face
    # recall on small distant faces). The full-frame small-face rescue
    # (refine_small_face) still runs regardless.
    PER_PERSON_FACE_FALLBACK: bool = True

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

    # ── Multi-angle identity (Phase 3) ───────────────────────
    # Rank gallery candidates by continuous head-pose angular distance (yaw)
    # before vector distance, so a side-angle query prefers same-angle gallery
    # faces of the right person over frontal faces of the wrong person.
    POSE_CONTINUOUS_SEARCH: bool = True
    # Use per-visitor adaptive returning thresholds derived from each visitor's
    # gallery similarity distribution (loosen only for high-variance visitors).
    ADAPTIVE_VISITOR_THRESHOLDS: bool = True
    # Tracklet buffer: hold grey-zone / first-sighting faces across a short window
    # and resolve once, so a single bad frame can't create a duplicate visitor.
    TRACKLET_ENABLED: bool = True
    TRACKLET_WINDOW_SECONDS: float = 2.0
    # Associate a detection with an open tracklet when bbox centres are within
    # this many pixels (same camera) — a coarse same-body proxy.
    TRACKLET_MAX_PIXEL_DISTANCE: float = 160.0
    # Require at least this many observations before a tracklet that never matched
    # a known visitor is allowed to register a NEW visitor.
    TRACKLET_MIN_OBSERVATIONS_NEW: int = 2

    # ── Tracklet fast-path (skip gallery search for known tracklets) ──
    # Once a tracklet is confidently resolved to a visitor, later frames of that
    # SAME tracklet (same camera, overlapping body) skip the expensive HNSW gallery
    # search and attribute directly to the pinned visitor. This is the big CPU win
    # for stationary/seated patrons: frame 2..N of a known person become a cheap
    # IoU association + visit-tracker heartbeat instead of a full re-resolve.
    #
    # OFF by default so behaviour is unchanged until you opt in. Identity can't
    # silently drift because the pin is RE-VERIFIED (a full gallery search is
    # forced) whenever any of these fire:
    #   • more than TRACKLET_REVERIFY_SECONDS elapsed since the last verification,
    #   • the body bbox moved enough that IoU with the last box drops below
    #     TRACKLET_REVERIFY_IOU (a possible tracking swap in a crowd),
    #   • the aligned face crop changed materially (the per-stream face-embedding
    #     cache missed → not the same stable face we verified).
    TRACKLET_FAST_PATH: bool = False
    # Re-verify a pinned tracklet at least this often (seconds). Lower = safer
    # (more frequent full resolves), higher = faster. 0 disables the time trigger
    # (only the IoU / face-change triggers force a re-verify — most aggressive).
    # Default 2.0 is conservative: it keeps the per-visit speed win but bounds any
    # mis-attribution to a ~2 s window, so the fast-path stays close to the
    # every-frame slow path on accuracy. Raise toward 5–10 s for sparse/seated
    # scenes where more throughput is worth a slightly wider window.
    TRACKLET_REVERIFY_SECONDS: float = 2.0
    # Force a re-verify when the current body box's IoU with the tracklet's last
    # box falls below this — i.e. the person moved/swapped enough that the cheap
    # association is no longer trustworthy. 1.0 would re-verify every frame. 0.7 is
    # conservative (a fairly stationary body); lower it toward 0.5 to fast-path
    # through more movement at a slightly higher swap risk in crowds.
    TRACKLET_REVERIFY_IOU: float = 0.7

    # ── Registration pose gate (reduce duplicate registrations) ──
    # A brand-new visitor is only CREATED from a roughly front-facing view.
    # Profile / steep-angle first frames yield an embedding that later frontal
    # views of the SAME person match weakly (below RETURNING_FACE_THRESHOLD), so
    # they get registered again — the #1 cause of one person becoming several
    # records. This gate applies ONLY to creating a new visitor; recognising and
    # matching EXISTING visitors is never restricted by pose (you still want to
    # re-identify a returning patron seen in profile).
    #   "frontal"         — only a frontal face (|yaw| ≤ 15°, not steep-down) seeds
    #                        a new visitor (recommended; safe default).
    #   "frontal_or_down" — also allow downward-looking faces (menu / phone), but
    #                        not hard left/right profiles.
    #   "any"             — no pose gate (legacy behaviour).
    REGISTRATION_POSE_POLICY: str = "frontal"
    # Safety valve: if a tracklet is observed at least this many times but never
    # presents a pose satisfying REGISTRATION_POSE_POLICY, register it anyway from
    # the best available frame — so a person only ever seen in profile is not lost
    # forever. 0 disables the valve (strict: never register a non-conforming pose).
    REGISTRATION_POSE_FALLBACK_OBSERVATIONS: int = 5

    # ── Cross-camera identity (Phase 4) ──────────────────────
    # Master switch. Keep OFF until camera topology + review workflow are set up;
    # the camera-aware temporal gate and candidate check are inert when disabled.
    CROSS_CAMERA_ENABLED: bool = False
    # How far back to search other cameras' recent visitors before creating a new
    # visitor (a person walking between cameras reappears within this window).
    CROSS_CAMERA_LOOKBACK_SECONDS: float = 180.0
    # Face-similarity bands for the cross-camera candidate / reconciliation logic.
    CROSS_CAMERA_REVIEW_THRESHOLD: float = 0.52   # queue for human review
    CROSS_CAMERA_AUTO_THRESHOLD: float = 0.68     # accept as returning live
    CROSS_CAMERA_AUTO_MERGE_THRESHOLD: float = 0.72  # auto-merge duplicate records
    # Background reconciliation sweep cadence (0 disables the loop).
    CROSS_CAMERA_DEDUP_INTERVAL_MINUTES: int = 5

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
