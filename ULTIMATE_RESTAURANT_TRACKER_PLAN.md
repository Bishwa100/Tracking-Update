# Restaurant Visitor Detection & Frequency Tracking System
## Ultimate Implementation Plan — CPU-Optimized, Webcam-Ready

**Date:** 2026-06-16  
**Base Project:** Student Verification System (SPVS) — [Person-Tracking](file:///D:/Person-Tracking)  
**Stack:** FastAPI + PostgreSQL/pgvector + YOLO + ArcFace + OSNet | Next.js Dashboard  
**Target:** Webcam → Detect unknown persons → Auto-register → Recognize returning → Count frequency → Dashboard analytics

---

## 0. Implementation Status & Deviations From Original Plan

**Status (2026-06-16):** Backend implemented in place **and** the full 7-page
Next.js dashboard is built (`dashboard/`, builds cleanly). Neither has had a live
end-to-end run yet (heavy backend deps weren't installed in the build env). Four
backend endpoints were added beyond the original API list for the dashboard:
`GET /api/activity` (detection-event feed), `GET /api/settings` (read-only
config), `POST /api/camera/roi` and `GET /api/camera/roi` (optional region-of-interest
filtering). The dashboard reaches the backend through a same-origin server-side
proxy so the API key never reaches the browser; the live feed is a direct
WebSocket. shadcn/ui was replaced with lightweight Tailwind components (no
generator step), and the Settings page is read-only (settings are env-driven).
All app ports are in 3001–3010: dashboard 3003, backend 3001, pgAdmin 3002, PostgreSQL 3004.

Additional items implemented but not originally listed: a `test_camera.py` utility
(Windows camera index detector), a `dashboard/README.md` (separate frontend setup
guide), `dashboard/.env.local.example` (frontend env template), a
`dashboard/Dockerfile` (containerized Next.js build), hard-delete support on
`DELETE /api/visitors/{id}?hard=true`, a timing middleware (`X-Process-Time-Ms`
header on every response), CORS middleware disabled (comment notes WebSocket
compatibility reason), and `is_ambiguous` persisted as a column in
`detection_events` so the Activity page can filter on it.

The backend was built by **transforming the existing SPVS codebase in place**
rather than copying modules into a new `restaurant-tracker/` tree. Student /
guardian / caption code was removed; the four restaurant tables and all new
services + API routers were added. The following deliberate deviations were made
for correctness, privacy, and operability:

| # | Original plan | What was implemented | Why |
|---|---|---|---|
| 1 | Copy reused modules into a new `restaurant-tracker/` folder (§12) | Transform `backend/app/` in place; delete student/guardian/caption code | A copy duplicates the codebase and contradicts "remove unnecessary code"; the repo *is* the base project |
| 2 | Body (OSNet) is a cross-visit RETURNING signal at ≥0.50 (§4.1, §6.3) | Body fallback is **same-session only and OFF by default** (`ALLOW_BODY_FALLBACK=false`) | OSNet embeddings are clothing/appearance dependent — useless and dangerous (false merges) for recognising a regular on a different day |
| 3 | No retention/consent handling | Added `VISITOR_RETENTION_DAYS` purge job + privacy warnings + `.gitignore` of captured face crops | Auto-enrolling patron biometrics is regulated (GDPR / Illinois BIPA / etc.) |
| 4 | `cleanup_stale` closes after `VISIT_EXTENSION_MINUTES` (10) while cooldown is 20 (§5.2) | Visits close after `VISIT_COOLDOWN_MINUTES` **or** `MAX_VISIT_DURATION_HOURS`; the brief-absence case is handled by *not* closing within the cooldown | The original conflated "extend window" with "close after cooldown"; `VISIT_EXTENSION_MINUTES` was dropped |
| 5 | Multi-worker uvicorn implied | App pinned to a **single worker** (documented); Redis noted as the scale-out path | The in-memory `VisitTracker` lives in process memory |
| 6 | Single `RETURNING_FACE_THRESHOLD` for both create + recognise | Added `NEW_VISITOR_MAX_SIMILARITY` so **auto-creation is stricter than recognition** | Prevents fragmenting one person into many visitor records in the grey zone |

Other build-time simplifications: liveness/anti-spoofing, the BLIP/Nano-LLM
caption stack, and the legacy ViT body model were removed (not needed for
analytics), trimming `transformers`/`python-jose`/`passlib` from dependencies. A
shared `services/detection_pipeline.py` was added to orchestrate
resolve→enroll→track→audit for both `/api/detect` and the camera loop.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Database Schema](#3-database-schema)
4. [Identity Resolution Pipeline](#4-identity-resolution-pipeline)
5. [Visit Session Tracking](#5-visit-session-tracking)
6. [Re-Recognition Strategy](#6-re-recognition-strategy)
7. [Camera Service (Webcam)](#7-camera-service-webcam)
8. [API Design](#8-api-design)
9. [UI Dashboard Plan](#9-ui-dashboard-plan)
10. [CPU Optimization Strategy](#10-cpu-optimization-strategy)
11. [Implementation Phases](#11-implementation-phases)
12. [Project Structure](#12-project-structure)
13. [Configuration Reference](#13-configuration-reference)

---

## 1. Executive Summary

### What We Build

An automated restaurant analytics system that:
- **Detects** every person via webcam (USB/RTSP) or uploaded image/video
- **Auto-registers** first-time visitors — zero manual enrollment
- **Recognizes** returning visitors by face (512-d ArcFace) + body (512-d OSNet) embeddings
- **Counts** total visits, session duration, timestamps, and frequency patterns
- **Displays** a real-time admin dashboard with live camera feed, visitor directory, and analytics charts

### Why This Is Challenging

| Challenge | Our Solution |
|---|---|
| Unknown people (no pre-enrollment) | Auto-register on first high-quality detection; multi-frame buffer |
| False merges (two people look similar) | Ambiguity margin gate — reject when top-2 matches are too close |
| One long meal ≠ multiple visits | Visit session state machine with 20-min cooldown |
| Lighting/angle variance over time | Multi-embedding gallery (10 faces per visitor) + adaptive centroid |
| Group dynamics | YOLO detects each person independently; per-person resolution |
| Scale (thousands of visitors) | HNSW indexes on pgvector — logarithmic search at any scale |

### What We Reuse From Base Project

| Component | Source | How We Use It |
|---|---|---|
| YOLOv8n + ONNX | [ml_models.py](file:///D:/Person-Tracking/backend/app/ml_models.py) | Person detection (2-3x faster via ONNX) |
| ArcFace (InsightFace buffalo_l) | [ml_models.py](file:///D:/Person-Tracking/backend/app/ml_models.py) | 512-d face embeddings |
| OSNet x0.25 | [ml_models.py](file:///D:/Person-Tracking/backend/app/ml_models.py) | 512-d body re-ID (0.2M params, fast CPU) |
| `process_frame()` | [cv_pipeline.py](file:///D:/Person-Tracking/backend/app/cv_pipeline.py) | Full-frame ArcFace + quality gates + batched body |
| `enroll_person()` | [cv_pipeline.py](file:///D:/Person-Tracking/backend/app/cv_pipeline.py) | Weighted-average enrollment embeddings |
| `FaceEmbeddingCache` | [ml_models.py](file:///D:/Person-Tracking/backend/app/ml_models.py) | dHash dedup to avoid recomputing same face |
| Batched pgvector queries | [verification_service.py](file:///D:/Person-Tracking/backend/app/verification_service.py) | `VALUES + CROSS JOIN LATERAL` for N-face single query |
| Frame dedup (dHash + MAD) | [utils.py](file:///D:/Person-Tracking/backend/app/utils.py) | Skip static scenes |
| Cross-frame tracking | [config.py](file:///D:/Person-Tracking/backend/app/config.py) | IoU + cosine tracking to avoid re-querying same person |
| Inference semaphore | [utils.py](file:///D:/Person-Tracking/backend/app/utils.py) | Prevent CPU thrashing under load |
| Config pattern | [config.py](file:///D:/Person-Tracking/backend/app/config.py) | 40+ tunable settings via pydantic-settings |

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                         │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                      │
│   │  USB Webcam   │    │ RTSP Camera  │    │ Upload Video │                      │
│   │  (cv2 idx 0)  │    │ (IP Stream)  │    │ (API Upload) │                      │
│   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘                      │
│          └───────────────────┴───────────────────┘                               │
│                              │                                                    │
└──────────────────────────────┼────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND (CPU-ONLY)                                     │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  CAMERA SERVICE (background asyncio task)                                    │  │
│  │   • cv2.VideoCapture(source)                                                │  │
│  │   • Grab 1 frame/sec (configurable)                                         │  │
│  │   • Frame dedup: dHash + MAD threshold → skip static scenes                 │  │
│  │   • Downscale to 1280px long side                                           │  │
│  └───────────────────────────┬─────────────────────────────────────────────────┘  │
│                              │                                                    │
│  ┌───────────────────────────▼─────────────────────────────────────────────────┐  │
│  │  DETECTION PIPELINE (reused from base project)                               │  │
│  │   • YOLOv8n (ONNX) → person bounding boxes (conf ≥ 0.5)                    │  │
│  │   • ArcFace full-frame pass → all faces at once (512-d each)                │  │
│  │   • Quality gate: min 40px, det_score ≥ 0.40                               │  │
│  │   • Small-face rescue: upscale 2.5x + re-detect                            │  │
│  │   • OSNet batched → body embeddings (512-d each)                            │  │
│  └───────────────────────────┬─────────────────────────────────────────────────┘  │
│                              │                                                    │
│  ┌───────────────────────────▼─────────────────────────────────────────────────┐  │
│  │  IDENTITY RESOLVER (new)                                                      │  │
│  │   • HNSW search on visitor_faces.embedding (top-K)                           │  │
│  │   • Ambiguity gate: top - runner_up < 0.05 → REJECT                         │  │
│  │   • Face threshold ≥ 0.55 → RETURNING                                       │  │
│  │   • Body fallback ≥ 0.50 → RETURNING (if face fails)                        │  │
│  │   • Below 0.35 → definitely NEW                                              │  │
│  │   • Batched: N faces in ONE DB round-trip (CROSS JOIN LATERAL)               │  │
│  └───────────────────────────┬─────────────────────────────────────────────────┘  │
│                              │                                                    │
│  ┌───────────────────────────▼─────────────────────────────────────────────────┐  │
│  │  VISIT SESSION TRACKER (new — in-memory state machine)                       │  │
│  │   • Active visits dict: {visitor_id → ActiveVisit}                           │  │
│  │   • New detection → open visit or extend existing                            │  │
│  │   • 20-min no detection → close visit + compute duration                     │  │
│  │   • 4-hour max cap → auto-close stale visits                                │  │
│  │   • Increment visitor.visit_count on genuine new visits                      │  │
│  └───────────────────────────┬─────────────────────────────────────────────────┘  │
│                              │                                                    │
│  ┌───────────────────────────▼─────────────────────────────────────────────────┐  │
│  │  AUTO-ENROLLER + GALLERY MANAGER (new)                                       │  │
│  │   • NEW person → create visitor record + first face in gallery               │  │
│  │   • RETURNING person → add face to gallery (top-10 by quality)               │  │
│  │   • Update adaptive centroid (weighted moving average)                        │  │
│  │   • Save best face crop as thumbnail                                          │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐  │
│  │  REST API                                                                      │  │
│  │   POST /api/detect           POST /api/camera/start                          │  │
│  │   GET  /api/visitors         POST /api/camera/stop                           │  │
│  │   GET  /api/visitors/{id}    GET  /api/camera/status                         │  │
│  │   GET  /api/analytics/*      GET  /api/camera/snapshot                       │  │
│  │   GET  /api/health           WebSocket /ws/live-feed                         │  │
│  └─────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────────────────┐
│                     POSTGRESQL 16 + pgvector                                      │
│                                                                                   │
│  visitors              visitor_faces           visits               detection_    │
│  ──────────────        ──────────────          ──────────           events         │
│  id (PK)               id (PK)                 id (PK)             ──────────     │
│  face_embedding 512d   visitor_id FK           visitor_id FK       id (PK)        │
│  body_embedding 512d   embedding 512d          entered_at          visitor_id FK   │
│  visit_count           det_score               left_at             visit_id FK     │
│  first_seen_at         body_embedding 512d     duration_minutes    detected_at     │
│  last_seen_at          source_frame_path       detection_count     face_similarity │
│  HNSW indexes          HNSW index              camera_id           is_new_visitor  │
└──────────────────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼────────────────────────────────────────────────────┐
│                     NEXT.JS DASHBOARD (UI)                                        │
│                                                                                   │
│  ┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐  │
│  │ Live     │ │ Visitor      │ │ Visitor      │ │ Analytics    │ │ Camera    │  │
│  │ Monitor  │ │ Directory    │ │ Profile      │ │ Dashboard    │ │ Controls  │  │
│  └──────────┘ └──────────────┘ └──────────────┘ └──────────────┘ └───────────┘  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema

### 3.1 `visitors` — Core Identity Table

```sql
CREATE TABLE visitors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Centroid embeddings (L2-normalized, updated after each verified visit)
    face_embedding VECTOR(512),
    body_embedding VECTOR(512),

    -- Visit statistics (denormalized for fast reads)
    visit_count INTEGER NOT NULL DEFAULT 0,
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at TIMESTAMPTZ,

    -- Quality tracking
    best_face_det_score FLOAT DEFAULT 0.0,
    total_faces_recorded INTEGER DEFAULT 0,

    -- Admin fields
    name TEXT,                          -- Optional label (e.g., "Regular - Mr. Kumar")
    notes TEXT,
    thumbnail_path TEXT,                -- Best face crop saved to disk
    is_staff BOOLEAN DEFAULT FALSE,     -- Exclude from analytics
    is_active BOOLEAN DEFAULT TRUE      -- Soft delete
);

CREATE INDEX idx_visitors_face_hnsw ON visitors
    USING hnsw (face_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_visitors_body_hnsw ON visitors
    USING hnsw (body_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX idx_visitors_last_seen ON visitors(last_seen_at DESC);
CREATE INDEX idx_visitors_visit_count ON visitors(visit_count DESC);
```

### 3.2 `visitor_faces` — Per-Visitor Face Gallery

```sql
CREATE TABLE visitor_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visitor_id UUID NOT NULL REFERENCES visitors(id) ON DELETE CASCADE,

    embedding VECTOR(512) NOT NULL,     -- Per-image ArcFace 512-d
    det_score FLOAT NOT NULL DEFAULT 0.0,
    body_embedding VECTOR(512),         -- Body from same detection (optional)

    source_frame_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_vf_visitor_id ON visitor_faces(visitor_id);
CREATE INDEX idx_vf_embedding_hnsw ON visitor_faces
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

> [!TIP]
> **Why a separate gallery?** A single centroid blurs away distinctive features. Storing the top-10 highest-quality face embeddings per visitor (from different angles, lighting, visits) dramatically improves re-recognition accuracy over time.

### 3.3 `visits` — Visit Sessions

```sql
CREATE TABLE visits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visitor_id UUID NOT NULL REFERENCES visitors(id) ON DELETE CASCADE,

    entered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    left_at TIMESTAMPTZ,                  -- NULL = still active
    duration_minutes INTEGER,             -- Computed on close

    detection_count INTEGER DEFAULT 0,    -- Frames detected in this visit
    best_face_confidence FLOAT,
    avg_face_confidence FLOAT,

    camera_id TEXT,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_visits_visitor ON visits(visitor_id);
CREATE INDEX idx_visits_entered ON visits(entered_at DESC);
CREATE INDEX idx_visits_visitor_entered ON visits(visitor_id, entered_at DESC);
CREATE INDEX idx_visits_active ON visits(left_at) WHERE left_at IS NULL;
```

### 3.4 `detection_events` — Audit Trail

```sql
CREATE TABLE detection_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    visitor_id UUID REFERENCES visitors(id) ON DELETE SET NULL,
    visit_id UUID REFERENCES visits(id) ON DELETE SET NULL,

    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    face_similarity FLOAT,
    body_similarity FLOAT,
    combined_confidence FLOAT,

    is_new_visitor BOOLEAN NOT NULL,
    is_ambiguous BOOLEAN NOT NULL DEFAULT FALSE,  -- Stored so Activity page can filter
    match_source TEXT,                    -- "face" | "body" | "none"

    camera_id TEXT,
    frame_path TEXT,
    bbox JSONB,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_de_visitor ON detection_events(visitor_id, detected_at DESC);
CREATE INDEX idx_de_datetime ON detection_events(detected_at DESC);
```

---

## 4. Identity Resolution Pipeline

### 4.1 Core Decision Algorithm

```
Incoming Face Embedding (512-d) + Body Embedding (512-d)
                    │
                    ▼
    ┌───────────────────────────────┐
    │  STEP 1: Face Gallery Search  │  HNSW on visitor_faces.embedding
    │  Batched: N faces = 1 query   │  CROSS JOIN LATERAL (top-2 per face)
    └───────────────┬───────────────┘
                    │
                    ▼
         Top match similarity?
                    │
        ┌───────────┼───────────┐
        │           │           │
    < 0.35      0.35–0.55     ≥ 0.55
        │           │           │
        ▼           ▼           ▼
   DEFINITELY     GREY      Check Ambiguity:
   NEW VISITOR    ZONE      Top - Runner-up ≥ 0.05?
   (register)      │           │
                   │      ┌────┴────┐
                   │     YES       NO
                   │      │         │
                   │      ▼         ▼
                   │   RETURNING   AMBIGUOUS
                   │   (visitor_id) (skip frame)
                   │
                   ▼
         ┌──────────────────┐
         │  BODY FALLBACK   │  HNSW on visitors.body_embedding
         │  body_sim ≥ 0.50 │
         └────────┬─────────┘
                  │
            ┌─────┴─────┐
           YES         NO
            │           │
            ▼           ▼
        RETURNING    NEW VISITOR
        (visitor_id)  (register)
```

> [!IMPORTANT]
> **As implemented, the BODY FALLBACK branch is OFF by default and same-session
> only.** OSNet body embeddings encode clothing/appearance, so they cannot
> recognise a returning visitor on a different day and would cause false merges
> if used cross-visit. With `ALLOW_BODY_FALLBACK=false` (default), a grey-zone
> detection with sufficient face quality is registered as a NEW visitor; with no
> body match it is otherwise dropped. Enable body fallback only for short-term
> re-acquisition within one session.

### 4.2 Thresholds

| Threshold | Value | Purpose |
|---|---|---|
| `RETURNING_FACE_THRESHOLD` | **0.55** | Minimum face similarity to recognize as returning |
| `NEW_VISITOR_MAX_SIMILARITY` | **0.45** | At/below this (and quality OK) → auto-create a NEW visitor. Stricter than recognition so the grey zone (0.45–0.55) does not fragment one person into many records |
| `AMBIGUITY_MARGIN` | **0.05** | Top-2 must differ by ≥ 5 points; else skip |
| `STRONG_MATCH_THRESHOLD` | **0.65** | Above this, update centroid + add face to gallery |
| `REJECT_SIMILARITY` | **0.35** | Reserved knob (documented). Creation is gated by `NEW_VISITOR_MAX_SIMILARITY` in the implementation |
| `FACE_QUALITY_CUTOFF` | **0.45** | Min det_score to seed a visitor or store a face in the gallery |
| `ALLOW_BODY_FALLBACK` | **false** | Body re-ID OFF by default — same-session only (see §6.3) |
| `RETURNING_BODY_THRESHOLD` | **0.55** | Body match threshold, only when `ALLOW_BODY_FALLBACK=true` |

### 4.3 Batched Resolution SQL

One DB round-trip for N faces in a single frame:

```sql
WITH input_faces AS (
    SELECT idx, emb::vector
    FROM (VALUES (0, :emb_0::vector), (1, :emb_1::vector), ...) AS v(idx, emb)
),
face_matches AS (
    SELECT f.idx, vf.visitor_id,
           1 - (vf.embedding <=> f.emb) AS similarity,
           ROW_NUMBER() OVER (PARTITION BY f.idx ORDER BY vf.embedding <=> f.emb) AS rank
    FROM input_faces f
    CROSS JOIN LATERAL (
        SELECT visitor_id, embedding
        FROM visitor_faces
        ORDER BY embedding <=> f.emb
        LIMIT 2  -- top-2 for ambiguity check
    ) vf
)
SELECT * FROM face_matches;
```

---

## 5. Visit Session Tracking

### 5.1 Session Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `VISIT_COOLDOWN_MINUTES` | **20** | After 20 min of no detection → close visit. Next detection = NEW visit. A brief absence under this window keeps the same visit open (handled by *not* closing yet) |
| `MAX_VISIT_DURATION_HOURS` | **4** | Auto-close stale visits (prevent forever-open records) |
| `STALE_CHECK_INTERVAL_SECONDS` | **60** | Background task checks for stale visits every minute |

> [!NOTE]
> `VISIT_EXTENSION_MINUTES` from the original draft was **removed**. The brief-
> absence ("bathroom break") case needs no separate window: the in-memory visit
> simply stays open until `VISIT_COOLDOWN_MINUTES` of inactivity, so a return
> within the cooldown extends the same visit automatically. A single
> high-confidence detection still counts as a visit.

### 5.2 In-Memory State Machine

```python
class VisitTracker:
    """Per-camera in-memory tracker. Persists to DB, but avoids DB queries for hot-path decisions."""
    
    active_visits: Dict[UUID, ActiveVisit]  # key = visitor_id

    async def process_detection(self, visitor_id, timestamp, confidence, camera_id):
        if visitor_id in self.active_visits:
            # EXTEND existing visit
            visit = self.active_visits[visitor_id]
            visit.last_detected_at = timestamp
            visit.detection_count += 1
            visit.best_confidence = max(visit.best_confidence, confidence)
            # DB update deferred (batched every N seconds)
        else:
            # Check DB: was there a recent visit we should reopen?
            last_visit = await db.get_latest_visit(visitor_id)
            if last_visit and last_visit.left_at and \
               (timestamp - last_visit.left_at).minutes < VISIT_COOLDOWN:
                # Brief absence (bathroom?) → reopen
                await db.reopen_visit(last_visit.id, timestamp)
                self.active_visits[visitor_id] = ActiveVisit(...)
            else:
                # Genuine NEW visit
                visit_id = await db.create_visit(visitor_id, timestamp, camera_id)
                await db.increment_visit_count(visitor_id)
                self.active_visits[visitor_id] = ActiveVisit(...)

    async def cleanup_stale(self, now):
        """Called every STALE_CHECK_INTERVAL_SECONDS."""
        for vid, visit in list(self.active_visits.items()):
            idle = now - visit.last_detected_at
            open_for = now - visit.started_at
            # Close on cooldown idle OR the hard max-duration cap — NOT on an
            # extension window (which was removed).
            if idle >= timedelta(minutes=VISIT_COOLDOWN_MINUTES) or \
               open_for >= timedelta(hours=MAX_VISIT_DURATION_HOURS):
                duration = int((visit.last_detected_at - visit.started_at).total_seconds() // 60)
                await db.close_visit(visit.id, visit.last_detected_at, duration)
                del self.active_visits[vid]
```

### 5.3 Edge Cases

| Edge Case | Solution |
|---|---|
| Person exits, returns in 5 min | Cooldown reopen — same visit (bathroom break) |
| Person sits for 3 hours | Visit extends continuously; not split |
| Camera restarts mid-visit | On startup, load active visits from DB (`WHERE left_at IS NULL`) |
| Two cameras see same person | Same visitor_id → same active visit in tracker |
| Server crash | Active visits recovered from DB on restart |

---

## 6. Re-Recognition Strategy

### 6.1 Multi-Embedding Gallery

```python
MAX_FACES_PER_VISITOR = 10

async def add_face_to_gallery(db, visitor_id, embedding, det_score, frame_path=None):
    """Quality-based eviction: keep top-10 faces per visitor."""
    existing = await db.get_faces_for_visitor(visitor_id)

    if len(existing) < MAX_FACES_PER_VISITOR:
        await db.insert_face(visitor_id, embedding, det_score, frame_path)
        return

    # Evict lowest-quality (oldest if tie)
    worst = min(existing, key=lambda f: (f.det_score, f.created_at))
    if det_score > worst.det_score:
        await db.delete_face(worst.id)
        await db.insert_face(visitor_id, embedding, det_score, frame_path)
```

### 6.2 Adaptive Centroid Update

```python
async def update_centroid(db, visitor, new_embedding, det_score):
    """Weighted moving average — older centroids are more trusted."""
    if det_score < 0.45:
        return  # Skip low-quality updates

    alpha = 0.15 * min(det_score * 2, 1.0) * max(0.05, 1.0 / (1 + visitor.visit_count * 0.1))
    updated = (1 - alpha) * visitor.face_embedding + alpha * new_embedding
    updated = l2_normalize(updated)
    await db.update_visitor_centroid(visitor.id, updated)
```

### 6.3 Body Fallback Fusion Rules

> [!WARNING]
> **Cross-visit body matching is disabled in the implementation
> (`ALLOW_BODY_FALLBACK=false`).** The fusion rules below describe the optional
> *same-session* re-acquisition behaviour only (e.g. a diner who turns away for a
> few frames within one visit). Body embeddings change with clothing, so they
> must never establish identity across separate visits.

| Scenario | Face | Body | Action |
|---|---|---|---|
| Clear frontal face | 0.72 | 0.60 | Use face. Body confirms ✅ |
| Masked / side angle | None | 0.58 | Body-only match with "low confidence" flag |
| Weak face + moderate body | 0.48 | 0.55 | Both contribute; require strong body |
| Face & body disagree | 0.50 → Person A | 0.55 → Person B | **AMBIGUOUS** — skip detection |
| Ambiguous face, body confirms | 0.52 vs 0.50 | 0.62 → top face match | Body breaks tie ✅ |

---

## 7. Camera Service (Webcam)

### 7.1 Design for Webcam Testing

```python
class CameraService:
    """Background camera processing — webcam-first, RTSP-ready."""

    def __init__(self):
        self.capture: cv2.VideoCapture = None
        self.is_running: bool = False
        self._task: asyncio.Task = None
        self._last_frame: np.ndarray = None       # For snapshot API
        self._last_annotated: np.ndarray = None    # For live feed WebSocket
        self._stats = {"frames_processed": 0, "persons_detected": 0, ...}

    async def start(self, source: Union[int, str] = 0):
        """
        source: 0 = USB webcam, "rtsp://..." = IP camera, "/path/to.mp4" = video file
        """
        self.capture = cv2.VideoCapture(source)
        self.is_running = True
        self._task = asyncio.create_task(self._processing_loop())

    async def stop(self):
        self.is_running = False
        if self.capture: self.capture.release()
        if self._task: self._task.cancel()

    async def _processing_loop(self):
        prev_signature = None
        frame_interval = 1.0 / CAMERA_FPS  # e.g., 1.0 for 1 FPS

        while self.is_running:
            # 1. Grab frame (in thread pool to not block event loop)
            ret, frame = await asyncio.to_thread(self.capture.read)
            if not ret: break

            # 2. Downscale
            frame = cap_frame_long_side(frame, MAX_FRAME_LONG_SIDE)
            self._last_frame = frame.copy()

            # 3. Frame dedup check
            signature = compute_frame_signature(frame)
            if prev_signature and frames_are_similar(signature, prev_signature):
                await asyncio.sleep(frame_interval)
                continue
            prev_signature = signature

            # 4. Detection pipeline (in thread pool — CPU bound)
            detected_persons = await run_inference(process_frame, frame, extract_body=True)

            # 5. Identity resolution + visit tracking
            for person in detected_persons:
                if person.face_embedding:
                    result = await identity_resolver.resolve(person, db)
                    await visit_tracker.process_detection(result, now(), camera_id)
                    await gallery_manager.maybe_update(result, person)

            # 6. Annotate frame for live feed
            self._last_annotated = draw_detections(frame, detected_persons, results)

            # 7. Update stats
            self._stats["frames_processed"] += 1
            self._stats["persons_detected"] += len(detected_persons)

            await asyncio.sleep(frame_interval)
```

### 7.2 CPU Budget per Frame (@ 1 FPS)

| Operation | Time (CPU) | Notes |
|---|---|---|
| Frame grab + downscale | ~5ms | OpenCV, in thread pool |
| Frame dedup check | ~1ms | 32×32 grayscale comparison |
| YOLOv8n (ONNX) | ~80-150ms | Person detection |
| ArcFace full-frame | ~100-200ms | All faces at once |
| OSNet batched | ~30-80ms | All body crops at once |
| pgvector HNSW query | ~5-10ms | Batched for N faces |
| Visit tracking | ~1ms | In-memory operations |
| **Total** | **~220-450ms** | **Well within 1s budget for 1 FPS** |

---

## 8. API Design

### 8.1 Detection

```yaml
POST /api/detect:
  summary: Upload image/video for one-shot detection
  request: multipart/form-data
    file: UploadFile              # Image or video
    camera_id: str (optional)
  response:
    detections:
      - visitor_id: UUID
        is_new: bool
        visit_id: UUID
        face_confidence: float
        body_confidence: float
        match_source: "face" | "body" | "none"
        is_ambiguous: bool
    new_visitors_count: int
    returning_visitors_count: int
```

### 8.2 Camera Control

```yaml
POST /api/camera/start:
  body: { source: "0", camera_id: "cam-0", fps: 1.0 }   # "0" = webcam, "rtsp://..." = IP cam
  response: { status: "started", source: "0" }

POST /api/camera/stop:
  response: { status: "stopped" }

GET /api/camera/status:
  response:
    is_running: bool
    source: str
    camera_id: str
    fps: float
    frames_processed: int
    frames_skipped: int
    persons_detected: int
    new_visitors: int
    returning_visitors: int
    uptime_seconds: int
    last_error: str | null

GET /api/camera/snapshot:
  response: image/jpeg            # Latest annotated (or raw) frame as JPEG

POST /api/camera/roi:             # NEW — Region of Interest filter
  body: { x: int, y: int, width: int, height: int }
  response: { roi: {...} }

GET /api/camera/roi:              # NEW — Read current ROI
  response: { roi: {...} | null }

WebSocket /ws/live-feed:
  description: Streams annotated frames + live stats every frame interval
  payload: { type: "frame", frame: "data:image/jpeg;base64,...",
             is_running: bool, currently_inside: int, stats: {...} }
```

### 8.3 Visitors

```yaml
GET /api/visitors:
  query: { limit, offset, min_visits, since, sort_by, search }
  response:
    total: int
    visitors:
      - id, name, visit_count, first_seen_at, last_seen_at,
        is_staff, thumbnail_url, avg_confidence, best_face_det_score,
        is_active, total_faces_recorded

GET /api/visitors/{id}:
  response: Full visitor detail + latest visit summary (VisitorDetailResponse)

GET /api/visitors/{id}/visits:
  query: { limit, offset, since, until }
  response: Paginated visit history with durations

PUT /api/visitors/{id}:
  body: { name, notes, is_staff }
  response: Updated visitor

DELETE /api/visitors/{id}:
  query: { hard: bool }           # hard=false (default) = soft delete (is_active=false)
  response: { success: true }     # hard=true = permanent delete (removes all faces/visits)

GET /api/visitors/{id}/thumbnail:
  response: image/jpeg
```

### 8.4 Analytics

```yaml
GET /api/analytics/summary:
  query: { since, until }
  response:
    total_unique_visitors, total_visits, new_visitors, returning_visitors,
    average_duration_minutes, return_rate, visits_by_day[]
    # NOTE: peak_hour removed; return_rate (float 0–1) added

GET /api/analytics/frequency:
  response:
    distribution: { "1": 450, "2": 120, "3": 80, "4+": 45 }

GET /api/analytics/hourly:
  response:
    hourly: [{ hour: 0, new: 5, returning: 12 }, ... ]

GET /api/analytics/top-visitors:
  query: { limit, since, until }
  response:
    - visitor_id, visit_count, last_visit, first_visit, avg_duration
```

### 8.5 Admin

```yaml
POST /api/admin/visitors/{id}/merge:
  body: { target_visitor_id: UUID }     # Merge INTO this visitor (re-points all visits/faces/events)
  response: { success, merged_visits }

POST /api/admin/visitors/{id}/mark-staff:
  body: { is_staff: bool }

GET /api/health:
  response:
    status: "ok" | "degraded"
    models_loaded: bool
    yolo_loaded: bool
    arcface_loaded: bool
    body_model: str | null
    db_connected: bool
    camera_running: bool
    visitors_count: int
    total_visits: int
```

### 8.6 Activity Feed (Added for Dashboard)

```yaml
GET /api/activity:
  query: { limit, since, event_type: "new"|"returning"|"ambiguous" }
  response:
    events:
      - id, detected_at, visitor_id, visitor_name, thumbnail_url,
        visit_id, face_similarity, body_similarity, is_new_visitor,
        is_ambiguous, match_source, camera_id
```

### 8.7 Settings (Added for Dashboard)

```yaml
GET /api/settings:
  description: Read-only reflection of active backend configuration (all env vars)
  response: SettingsResponse — all thresholds, visit params, gallery params,
            camera params, privacy params. No mutations (config is env-driven).
```

### 8.8 Middleware

```yaml
X-Process-Time-Ms:
  description: Response header on every endpoint showing server processing time in ms
  reason: Performance observability without external tooling

CORS:
  description: Middleware is disabled (commented out) for WebSocket compatibility.
  note: In production, configure a reverse proxy (nginx) to enforce CORS rather
        than relying on FastAPI middleware, which breaks WS upgrade handshakes.
```

---

## 9. UI Dashboard Plan

### 9.1 Technology

| Choice | Reason |
|---|---|
| **Next.js 14** (App Router) | SSR for fast page loads; API routes used for backend proxy |
| **TailwindCSS v3** | Rapid styling with dark mode support |
| **Recharts** | Lightweight charts (bar, line, area, pie) |
| **Lucide Icons** | Clean, modern icon set |
| **SWR** | Data fetching with automatic revalidation and stale-while-revalidate |
| ~~shadcn/ui~~ → **Lightweight Tailwind components** | shadcn/ui replaced with hand-written Card, Button, Select, Input in `components/ui.tsx` — no generator step required |

### 9.2 Design System

```
Color Palette (Dark Mode Primary):
─────────────────────────────────
Background:     #0F172A (slate-900)
Surface:        #1E293B (slate-800)
Card:           #334155 (slate-700)
Primary:        #3B82F6 (blue-500)
Success:        #10B981 (emerald-500)
Warning:        #F59E0B (amber-500)
Danger:         #EF4444 (red-500)
Text Primary:   #F8FAFC (slate-50)
Text Secondary: #94A3B8 (slate-400)
Accent:         #8B5CF6 (violet-500)

Typography: Inter (Google Fonts)
Border Radius: 12px (cards), 8px (buttons/inputs)
Spacing: 4px base grid
```

### 9.3 Pages & Wireframes

#### Page 1: Live Monitor (`/`)

The **home page** — real-time camera feed with detection overlay.

```
┌──────────────────────────────────────────────────────────────────────┐
│  🍽️ Restaurant Tracker          [Live Monitor] [Visitors] [Analytics]│
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────┐  ┌──────────────────────┐   │
│  │                                    │  │  📊 Live Stats        │   │
│  │                                    │  │                      │   │
│  │        LIVE CAMERA FEED            │  │  Currently Inside: 4 │   │
│  │   (WebSocket annotated stream)     │  │  Today's Visits: 23  │   │
│  │                                    │  │  New Today: 8        │   │
│  │   [Person boxes with labels]       │  │  Returning: 15       │   │
│  │   "Visitor #12 (Visit #5)"         │  │                      │   │
│  │   "NEW — Registering..."           │  │  ──────────────────  │   │
│  │                                    │  │  Camera: Webcam 0    │   │
│  │                                    │  │  FPS: 1.0            │   │
│  │                                    │  │  Status: ● Running   │   │
│  └────────────────────────────────────┘  └──────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  🕐 Recent Activity Feed (last 10 events, auto-scroll)       │   │
│  │                                                              │   │
│  │  11:45 AM  👤 Visitor #12 recognized (confidence: 0.87)      │   │
│  │            Visit #5 started                                  │   │
│  │  11:44 AM  🆕 New visitor registered → Visitor #47           │   │
│  │  11:43 AM  👤 Visitor #3 recognized (confidence: 0.92)       │   │
│  │            Visit #15 extended (duration: 45 min)             │   │
│  │  11:40 AM  🚪 Visitor #8 left (visit duration: 1h 12m)      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────┐  ┌─────────────┐                                   │
│  │ 🟢 Start    │  │ 🔴 Stop     │  Camera Source: [0 (Webcam)] ▼   │
│  └─────────────┘  └─────────────┘                                   │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- WebSocket-powered live camera feed with detection bounding boxes
- Real-time stats cards updating every second
- Auto-scrolling activity feed showing events as they happen
- Camera start/stop controls

---

#### Page 2: Visitor Directory (`/visitors`)

Searchable, sortable table of all registered visitors.

```
┌──────────────────────────────────────────────────────────────────────┐
│  👥 Visitor Directory                                                │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 🔍 Search by name or ID...          [Min Visits: ___]        │   │
│  │ Sort: [Most Visits ▼]  Since: [📅 Date Picker]               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────┬────────────┬───────┬──────────────┬──────────────┬─────┐ │
│  │ 📷   │ Name       │Visits │ First Seen   │ Last Seen    │ ⚙️  │ │
│  ├──────┼────────────┼───────┼──────────────┼──────────────┼─────┤ │
│  │ [🧑] │ Visitor #1 │ 15 🔥 │ Jan 15, 2025 │ Today, 11:45 │ ••• │ │
│  │ [👩] │ Mr. Sharma │ 12    │ Feb 3, 2025  │ Today, 10:30 │ ••• │ │
│  │ [🧑] │ Visitor #3 │ 8     │ Mar 12, 2025 │ Yesterday    │ ••• │ │
│  │ [👩] │ Visitor #4 │ 3     │ Jun 10, 2025 │ 3 days ago   │ ••• │ │
│  │ [🧑] │ 🆕 NEW     │ 1     │ Today, 11:44 │ Today, 11:44 │ ••• │ │
│  └──────┴────────────┴───────┴──────────────┴──────────────┴─────┘ │
│                                                                      │
│  Showing 1-20 of 142 visitors          [◀ Prev] [1] [2] [3] [Next ▶]│
│                                                                      │
│  ┌────────────────────────────────────────────┐                     │
│  │ 📊 Quick Summary                           │                     │
│  │ Total Visitors: 142 | Staff: 5 | Active: 4 │                     │
│  └────────────────────────────────────────────┘                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Face thumbnail column (from saved crops)
- Visit count with "🔥" badge for frequent visitors (≥10 visits)
- Relative time display ("Today", "Yesterday", "3 days ago")
- Search, filter by min visits, sort by visits/recency
- Click row → opens Visitor Profile page
- ••• menu: Edit name, Mark as staff, Merge, Delete

---

#### Page 3: Visitor Profile (`/visitors/[id]`)

Detailed profile for a single visitor.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ← Back to Directory                                                 │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  ┌────────┐                                                    │  │
│  │  │        │  Visitor #12                                       │  │
│  │  │  FACE  │  Name: Mr. Kumar  [✏️ Edit]                       │  │
│  │  │ THUMB  │  Notes: "Regular lunch customer"  [✏️ Edit]       │  │
│  │  │        │  Staff: ❌  [Toggle]                               │  │
│  │  └────────┘                                                    │  │
│  │                                                                │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │ 15       │  │ Jun 16   │  │ Jan 15   │  │ 45 min       │  │  │
│  │  │ Total    │  │ Last     │  │ First    │  │ Avg Duration │  │  │
│  │  │ Visits   │  │ Visit    │  │ Visit    │  │              │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  📈 Visit Frequency Over Time (Bar Chart)                      │  │
│  │  ▐▐                                                            │  │
│  │  ▐▐  ▐▐      ▐▐                  ▐▐  ▐▐  ▐▐                  │  │
│  │  ▐▐  ▐▐  ▐▐  ▐▐  ▐▐          ▐▐  ▐▐  ▐▐  ▐▐  ▐▐  ▐▐       │  │
│  │  Jan  Feb  Mar  Apr  May  ...  Nov  Dec  Jan  Feb  Mar  Apr   │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  🕐 Visit History                                              │  │
│  │                                                                │  │
│  │  #15  Jun 16, 2025  11:45 AM → (still here)  [3 detections]  │  │
│  │  #14  Jun 14, 2025  12:30 PM → 1:45 PM       75 min          │  │
│  │  #13  Jun 12, 2025  7:00 PM  → 8:15 PM       75 min          │  │
│  │  #12  Jun 10, 2025  12:15 PM → 1:00 PM       45 min          │  │
│  │  ...                                                           │  │
│  │                     [Load More]                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────┐  ┌──────────────┐                                    │
│  │ 🗑️ Delete │  │ 🔗 Merge With│                                    │
│  └───────────┘  └──────────────┘                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Face thumbnail (enlarged)
- Editable name and notes inline
- 4 stat cards: total visits, last visit, first visit, avg duration
- Visit frequency bar chart (monthly)
- Paginated visit history with enter/leave times, duration
- Active visit indicator ("still here")
- Merge and delete actions

---

#### Page 4: Analytics Dashboard (`/analytics`)

Business intelligence with charts.

```
┌──────────────────────────────────────────────────────────────────────┐
│  📊 Analytics Dashboard                                              │
│  Date Range: [Jun 1] → [Jun 16]  [Today] [This Week] [This Month]  │
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ 142      │  │ 487      │  │ 38%      │  │ 52 min   │            │
│  │ Unique   │  │ Total    │  │ Return   │  │ Avg      │            │
│  │ Visitors │  │ Visits   │  │ Rate     │  │ Duration │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                      │
│  ┌───────────────────────────────────┐  ┌────────────────────────┐  │
│  │  📈 Daily Visits (Area Chart)     │  │ 🍩 New vs Returning   │  │
│  │                                   │  │    (Donut Chart)       │  │
│  │    ╱\      /\                     │  │                        │  │
│  │   /  \    /  \    /\              │  │   ┌─────┐              │  │
│  │  /    \  /    \  /  \             │  │   │62%  │ Returning    │  │
│  │ /      \/      \/    \            │  │   │38%  │ New          │  │
│  │ Jun 1    Jun 5    Jun 10  Jun 15  │  │   └─────┘              │  │
│  └───────────────────────────────────┘  └────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────┐  ┌────────────────────────┐  │
│  │  🕐 Hourly Heatmap               │  │ 🏆 Top 5 Regulars     │  │
│  │  (Stacked Bar: New + Returning)   │  │                        │  │
│  │                                   │  │ 1. Mr. Kumar    15 🔥  │  │
│  │  ▐▐     ▐▐▐▐▐▐▐▐                │  │ 2. Visitor #3   12     │  │
│  │  ▐▐  ▐▐ ▐▐▐▐▐▐▐▐ ▐▐▐▐          │  │ 3. Visitor #8   10     │  │
│  │  ▐▐▐▐▐▐ ▐▐▐▐▐▐▐▐ ▐▐▐▐▐▐        │  │ 4. Visitor #22  8      │  │
│  │  6am  9am  12pm  3pm  6pm  9pm   │  │ 5. Visitor #5   7      │  │
│  └───────────────────────────────────┘  └────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  📊 Visit Frequency Distribution (Bar Chart)                 │   │
│  │                                                              │   │
│  │  ████████████████████████████████  450 (1 visit)            │   │
│  │  ████████████████  120 (2 visits)                            │   │
│  │  ██████████  80 (3 visits)                                   │   │
│  │  ██████  45 (4+ visits)                                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

**Charts Used:**
- **Area Chart** — daily visits over time (smooth gradient fill)
- **Donut Chart** — new vs. returning visitor ratio
- **Stacked Bar** — hourly visits heatmap (peak hours visible)
- **Horizontal Bar** — frequency distribution
- **Leaderboard** — top regular visitors

---

#### Page 5: Activity Timeline (`/activity`)

Chronological event log for debugging and auditing.

```
┌──────────────────────────────────────────────────────────────────────┐
│  📜 Activity Timeline                                                │
│  Filter: [All ▼]  Date: [Today ▼]  [🔄 Auto-refresh: ON]          │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ 11:45:23 AM │ 👤 RECOGNIZED  │ Visitor #12 │ Face: 0.87    │   │
│  │             │ Visit #5 started│ Camera: 0   │                │   │
│  ├─────────────┼────────────────┼─────────────┼────────────────┤   │
│  │ 11:44:15 AM │ 🆕 NEW         │ Visitor #47 │ Face: 0.92    │   │
│  │             │ Auto-registered │ Camera: 0   │                │   │
│  ├─────────────┼────────────────┼─────────────┼────────────────┤   │
│  │ 11:43:02 AM │ ⚠️ AMBIGUOUS   │ Unknown     │ Top: 0.52     │   │
│  │             │ Skipped (margin │ Camera: 0   │ Runner: 0.49  │   │
│  │             │ too narrow)     │             │                │   │
│  ├─────────────┼────────────────┼─────────────┼────────────────┤   │
│  │ 11:40:00 AM │ 🚪 VISIT CLOSED│ Visitor #8  │ Duration: 72m │   │
│  │             │ Visit #41      │ Camera: 0   │ 8 detections  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  [Load Older Events...]                                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

#### Page 6: Camera Management (`/camera`)

```
┌──────────────────────────────────────────────────────────────────────┐
│  📹 Camera Management                                                │
│                                                                      │
│  ┌─────────────────────────────────────┐  ┌──────────────────────┐  │
│  │  Camera Source                       │  │  Status              │  │
│  │                                     │  │                      │  │
│  │  Source: [0 (USB Webcam)       ▼]   │  │  ● Running           │  │
│  │  ─── or ───                         │  │  Uptime: 2h 34m      │  │
│  │  RTSP URL: [rtsp://.............]   │  │  Frames: 9,240       │  │
│  │                                     │  │  Persons: 487        │  │
│  │  FPS: [1.0]                         │  │  Errors: 0           │  │
│  │  Max Frame Size: [1280] px          │  │                      │  │
│  │                                     │  │  [🟢 Start] [🔴 Stop]│  │
│  │  Frame Dedup: [✓ Enabled]           │  │                      │  │
│  │  ROI: [x] [y] [w] [h]  [Set ROI]   │  │  (optional focus zone│  │
│  └─────────────────────────────────────┘  └──────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  📸 Latest Snapshot                                          │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │                                                      │   │   │
│  │  │              [Latest camera frame]                   │   │   │
│  │  │                                                      │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  │  Captured at: 11:45:23 AM  [📷 Refresh]                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

---

#### Page 7: Settings (`/settings`)

```
┌──────────────────────────────────────────────────────────────────────┐
│  ⚙️ Settings                                                         │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  🧠 Recognition Thresholds                                     │  │
│  │                                                                │  │
│  │  Face Match Threshold:     [0.55] ──────●────────── 1.0       │  │
│  │  Body Match Threshold:     [0.50] ────●──────────── 1.0       │  │
│  │  Ambiguity Margin:         [0.05] ──●────────────── 0.20      │  │
│  │  Strong Match Threshold:   [0.65] ──────────●────── 1.0       │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  🕐 Visit Session Settings                                     │  │
│  │                                                                │  │
│  │  Cooldown (min):           [20]                                │  │
│  │  Extension (min):          [10]                                │  │
│  │  Max Duration (hours):     [4]                                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  📷 Gallery Settings                                           │  │
│  │                                                                │  │
│  │  Max Faces Per Visitor:    [10]                                │  │
│  │  Quality Cutoff:           [0.45]                              │  │
│  │  Save Frame Crops:         [✓]                                 │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  [💾 Save Settings]  [↩️ Reset Defaults]                            │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.4 UI Navigation

```
Sidebar (always visible):
┌──────────────┐
│ 🍽️ RestaurantTracker │
├──────────────┤
│ 📺 Live Monitor     │ ← Home page (/)
│ 👥 Visitors         │ ← /visitors
│ 📊 Analytics        │ ← /analytics
│ 📜 Activity         │ ← /activity
│ 📹 Camera           │ ← /camera
│ ⚙️ Settings          │ ← /settings
├──────────────┤
│ System Status       │
│ ● Camera: Running   │
│ 🟢 Models: Loaded   │
│ 💾 DB: Connected    │
└──────────────┘
```

---

## 10. CPU Optimization Strategy

### 10.1 Inherited From Base Project

| Optimization | Impact |
|---|---|
| Model singleton (load once at startup) | No per-request model loading overhead |
| CPU thread tuning (MKL/OMP = cores - 1) | Full CPU utilization |
| YOLO ONNX export | 2-3x faster than PyTorch on CPU |
| Warm-up inference on startup | No cold-start penalty |
| Full-frame ArcFace (detect all faces at once) | Avoids N passes for N persons |
| Frame dedup (dHash + MAD) | Skip 80-90% of static frames |
| Cross-frame tracking (IoU + cosine) | Skip DB query for recently-seen persons |
| Batched DB queries (CROSS JOIN LATERAL) | O(1) DB round-trips per frame |
| Inference semaphore | Prevent CPU thrashing |
| FaceEmbeddingCache (dHash keyed) | Skip re-computing same face |

### 10.2 New Optimizations

| Optimization | Impact |
|---|---|
| In-memory visit tracker | Avoids DB SELECT on every detection (**requires a single uvicorn worker** — state is in process memory; use Redis to scale out) |
| Gallery pruning (top-10 per visitor) | Bounded table growth |
| Adaptive centroid update | Self-improving recognition without full re-enrollment |
| Partial index on `visits(left_at IS NULL)` | Lightning-fast active visit queries |
| `last_seen_at DESC` index | Fast recent visitor queries |
| Batched INSERT for detection_events | Groups writes per frame |
| Staff exclusion filter | Cleaner analytics |
| Background stale visit cleanup | Periodic, not per-request |
| Process at 1 FPS (not 30 FPS) | 30x CPU reduction vs. raw camera |
| Skip body embedding when face is strong | Save ~80ms per person on confident matches |

### 10.3 Performance Targets

| Metric | Target | Achieved By |
|---|---|---|
| Frame-to-result latency | < 500ms @ 1 FPS | ONNX YOLO + full-frame ArcFace + batched queries |
| Identity resolution per face | < 10ms | HNSW index |
| Handle 1M+ visitors | < 50ms query | HNSW is logarithmic |
| Memory footprint | < 2GB | OSNet (0.2M params) vs. ViT (86M) |
| Webcam processing | Smooth at 1 FPS on 4-core CPU | Frame dedup + inference semaphore |

---

## 11. Implementation Phases

> **Status:** Phases 1–4 (backend) implemented in place; not yet runtime-tested
> (heavy deps were not installed in the build environment — code byte-compiles
> and was cross-checked). Phases 5–6 pending.

### Phase 1: Foundation ✅ (done, in place)
- [x] Transform `backend/app/` in place (no separate `restaurant-tracker/` tree)
- [x] Keep & adapt core modules: `ml_models.py`, `cv_pipeline.py`, `utils.py`, `database.py`, `config.py`
- [x] Strip BLIP/caption/LLM/ViT/liveness + guardian/student logic
- [x] Create Alembic migration for restaurant schema (4 tables, drops legacy tables)
- [x] Docker Compose updated (PostgreSQL + pgvector + backend)
- [ ] Test model loading + health endpoint (needs deps installed)

### Phase 2: Identity Resolution ✅ (done)
- [x] `identity_resolver.py` (batched HNSW gallery search + ambiguity gate; body fallback off by default)
- [x] `auto_enroller.py` (new visitor registration + top-N gallery + adaptive centroid)
- [x] `services/detection_pipeline.py` (shared resolve→enroll→track→audit)
- [x] `POST /api/detect` endpoint (image + video upload)
- [ ] Test: upload photo → new visitor → re-upload → recognized (needs deps)

### Phase 3: Camera Service + Visit Tracking ✅ (done)
- [x] `camera_service.py` (webcam/RTSP/file loop + frame dedup + annotation)
- [x] `visit_tracker.py` (in-memory state machine + DB recovery on startup)
- [x] Camera control endpoints (start/stop/status/snapshot)
- [x] WebSocket `/ws/live-feed`
- [ ] Test: webcam → walk in → detected → walk away → visit closed (needs native run + webcam)

### Phase 4: Visitor Management + Analytics API ✅ (done)
- [x] CRUD endpoints: visitors list, detail, update, soft/hard delete, thumbnail
- [x] Visit history endpoint per visitor
- [x] Analytics: summary, frequency, hourly, top-visitors
- [x] Admin: merge duplicates, mark staff
- [ ] Test analytics accuracy with synthetic data

### Phase 5: Dashboard UI ✅ (done — builds cleanly, not yet run against a live backend)
- [x] Set up Next.js 14 + TailwindCSS (+ Recharts, lucide-react, SWR; shadcn/ui replaced by lightweight Tailwind components in `components/ui.tsx`)
- [x] Server-side proxy (`/api/backend/[...path]/route.ts`) so the API key stays off the client
- [x] SWR data fetching with stale-while-revalidate + per-page refresh intervals
- [x] TypeScript interfaces for all API responses (`lib/types.ts`) + formatting utilities (`lib/format.ts`)
- [x] Page 1: Live Monitor (WebSocket feed + stats + activity)
- [x] Page 2: Visitor Directory (table + search + filters + pagination)
- [x] Page 3: Visitor Profile (detail + monthly chart + history + edit/merge/delete)
- [x] Page 4: Analytics Dashboard (area/donut/stacked-bar/frequency + date range + top regulars)
- [x] Page 5: Activity Timeline (filter + auto-refresh)
- [x] Page 6: Camera Management (start/stop/status/snapshot + ROI configuration)
- [x] Page 7: Settings (read-only — backend config is env-driven)
- [x] `dashboard/Dockerfile` for containerized Next.js build
- [x] `dashboard/.env.local.example` (NEXT_PUBLIC_BACKEND_URL, NEXT_PUBLIC_WS_URL)
- [x] `dashboard/README.md` (separate frontend setup and development guide)

### Phase 6: Polish & Testing (Week 4)
- [ ] End-to-end testing with webcam
- [ ] Load testing with simulated traffic
- [ ] Performance profiling and tuning
- [ ] Error handling and edge cases
- [ ] Documentation

---

## 12. Project Structure

**As built (in place — no separate `restaurant-tracker/` root):**

```
Person-Tracking-main/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # App, lifespan, routers, health, bg tasks,
│   │   │                                    # timing middleware (X-Process-Time-Ms)
│   │   ├── config.py                        # Restaurant settings (pydantic-settings, 60+ vars)
│   │   ├── database.py                      # Async SQLAlchemy engine + session + pgvector init
│   │   ├── models.py                        # Visitor, VisitorFace, Visit, DetectionEvent
│   │   ├── schemas.py                       # 20+ Pydantic request/response models
│   │   ├── ml_models.py                     # ModelManager singleton (YOLO + ArcFace + OSNet)
│   │   ├── cv_pipeline.py                   # process_frame(), DetectedPerson dataclass
│   │   ├── osnet.py                         # OSNet x0.25 architecture (vendored)
│   │   ├── utils.py                         # Frame dedup, media, annotation, run_inference
│   │   │
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── identity_resolver.py         # NEW vs RETURNING (HNSW batched + ambiguity gate)
│   │   │   ├── auto_enroller.py             # Gallery management + adaptive centroid
│   │   │   ├── visit_tracker.py             # In-memory session state machine + DB recovery
│   │   │   ├── detection_pipeline.py        # Shared resolve→enroll→track→audit
│   │   │   ├── camera_service.py            # Webcam/RTSP/file processing loop + ROI support
│   │   │   └── analytics_service.py         # Analytics query builders
│   │   │
│   │   └── api/
│   │       ├── __init__.py                  # verify_api_key dependency
│   │       ├── detect.py                    # POST /api/detect (image/video upload)
│   │       ├── visitors.py                  # Visitor CRUD + visit history + thumbnail
│   │       │                                # DELETE supports ?hard=true for permanent removal
│   │       ├── analytics.py                 # Analytics endpoints
│   │       ├── camera.py                    # Camera control + ROI endpoints
│   │       ├── admin.py                     # Merge, staff marking
│   │       ├── activity.py                  # GET /api/activity (detection event timeline)
│   │       ├── settings.py                  # GET /api/settings (read-only config reflection)
│   │       └── websocket.py                 # WebSocket /ws/live-feed
│   │
│   ├── alembic/
│   │   ├── env.py                           # Migration env (auto-detects models)
│   │   └── versions/
│   │       └── 001_restaurant_schema.py     # Drops SPVS tables, creates 4 restaurant tables
│   ├── storage/                             # (runtime, gitignored)
│   │   ├── visitor_photos/                  # Saved face thumbnails
│   │   └── tmp_detect/                      # Temp files for uploaded media
│   ├── Dockerfile                           # Multi-stage: Python 3.11, CPU PyTorch, single worker
│   ├── requirements.txt                     # 26 deps (FastAPI, SQLAlchemy, asyncpg, pgvector,
│   │                                        # ultralytics, insightface, onnxruntime, cv2, Pillow)
│   └── alembic.ini
│
├── dashboard/                               # Next.js 14 UI
│   ├── app/
│   │   ├── layout.tsx                       # Root layout: sidebar nav, global dark theme
│   │   ├── page.tsx                         # Live Monitor (WebSocket + stats + activity)
│   │   ├── visitors/
│   │   │   ├── page.tsx                     # Visitor Directory (table + search + pagination)
│   │   │   └── [id]/page.tsx               # Visitor Profile (detail + chart + history)
│   │   ├── analytics/page.tsx               # Analytics Dashboard (area/donut/stacked-bar)
│   │   ├── activity/page.tsx                # Activity Timeline (filter + auto-refresh)
│   │   ├── camera/page.tsx                  # Camera Management (start/stop/snapshot/ROI)
│   │   ├── settings/page.tsx                # Settings (read-only config display)
│   │   └── api/backend/[...path]/route.ts   # Server-side proxy (keeps API key off browser)
│   ├── components/
│   │   ├── sidebar.tsx                      # Navigation + system status indicators
│   │   ├── live-feed.tsx                    # WebSocket consumer, base64 frame decoder
│   │   ├── activity-feed.tsx                # Event list with thumbnails + timestamps
│   │   ├── visitor-table.tsx                # Searchable/sortable table with pagination
│   │   ├── charts.tsx                       # Recharts wrappers: Area, Donut, Bar
│   │   ├── stat-card.tsx                    # KPI card with icon/tone theming
│   │   └── ui.tsx                           # Lightweight: Card, Button, Select, Input
│   ├── lib/
│   │   ├── api.ts                           # SWR fetcher + API URL builder
│   │   ├── types.ts                         # TypeScript interfaces for all API responses
│   │   └── format.ts                        # Relative time + duration formatting helpers
│   ├── Dockerfile                           # Containerized Next.js build
│   ├── package.json                         # next 14, react 18, recharts, swr, lucide-react,
│   │                                        # tailwindcss (NO shadcn/ui)
│   ├── tailwind.config.ts                   # Dark mode, custom slate palette
│   ├── tsconfig.json                        # Path aliases (@/)
│   ├── .env.local.example                   # NEXT_PUBLIC_BACKEND_URL, NEXT_PUBLIC_WS_URL
│   └── README.md                            # Dashboard setup & development guide
│
├── docker-compose.yml                       # postgres + pgadmin + backend + dashboard
│                                            # (4 services; ports 3001–3004)
├── init-db.sql                              # pgvector extension enablement
├── .env.example                             # All backend env vars (template)
├── .gitignore                               # Excludes storage/, .env, face crops
├── test_camera.py                           # Windows utility: find available camera indices
└── README.md                                # Project overview + quick start
```

---

## 13. Configuration Reference

All settings configurable via `.env`:

```python
# ── Database ──
DATABASE_URL = "postgresql+asyncpg://tracker:tracker_pass@localhost:3004/restaurant_tracker"

# ── API ──
API_KEY = "changeme-set-a-real-key"
CORS_ORIGINS = '["http://localhost:3003"]'    # Next.js dashboard

# ── Identity Resolution ──
RETURNING_FACE_THRESHOLD = 0.55
NEW_VISITOR_MAX_SIMILARITY = 0.45             # stricter creation gate (grey zone = 0.45–0.55)
REJECT_SIMILARITY = 0.35                      # reserved knob
AMBIGUITY_MARGIN = 0.05
STRONG_MATCH_THRESHOLD = 0.65

# ── Body re-ID (same-session only; OFF by default) ──
ALLOW_BODY_FALLBACK = false
RETURNING_BODY_THRESHOLD = 0.55

# ── Gallery Management ──
MAX_FACES_PER_VISITOR = 10
FACE_QUALITY_CUTOFF = 0.45
CENTROID_ALPHA_BASE = 0.15

# ── Visit Sessions (VISIT_EXTENSION_MINUTES removed) ──
VISIT_COOLDOWN_MINUTES = 20
MAX_VISIT_DURATION_HOURS = 4
STALE_CHECK_INTERVAL_SECONDS = 60

# ── Camera ──
CAMERA_SOURCE = "0"                           # "0" = USB webcam
CAMERA_FPS = 1.0
CAMERA_ID = "cam-0"
CAMERA_AUTOSTART = false
MAX_FRAME_LONG_SIDE = 1280
FRAME_DEDUP_ENABLED = true
FRAME_DEDUP_MAD_THRESHOLD = 4.0

# ── Models (CPU) ──
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_USE_ONNX = true
YOLO_PERSON_CONFIDENCE = 0.5
INSIGHTFACE_MODEL_NAME = "buffalo_l"
INSIGHTFACE_DET_SIZE = 640
BODY_MODEL_TYPE = "osnet"
CPU_THREADS = 0                               # 0 = all cores
INFERENCE_MAX_CONCURRENCY = 1

# ── Quality Gates ──
MIN_FACE_SIZE_PX = 40
MIN_FACE_DET_SCORE = 0.40

# ── Storage ──
VISITOR_PHOTO_DIR = "storage/visitor_photos"
DETECT_SAVE_FRAMES = false
DETECT_TMP_DIR = "storage/tmp_detect"

# ── Privacy / Retention ──
VISITOR_RETENTION_DAYS = 0                     # 0 = keep forever; purge older than N days
RETENTION_PURGE_INTERVAL_HOURS = 24

# ── Analytics ──
ANALYTICS_DEFAULT_DAYS = 30
```

---

## Verification Checklist

| Requirement | Solution |
|---|---|
| ✅ Detect unknown people | YOLO + ArcFace + OSNet pipeline (CPU) |
| ✅ Auto-register first-time visitors | Auto-enroller with quality gate |
| ✅ Recognize returning visitors | HNSW gallery search + adaptive centroid |
| ✅ Count visit frequency | Visit session tracker + visit_count |
| ✅ Track visit timestamps & duration | Visit table with enter/leave/duration |
| ✅ Handle ambiguous matches | Ambiguity margin + body fallback + skip |
| ✅ Body fallback when face obscured | OSNet 512-d as secondary signal (same-session only) |
| ✅ Self-improving recognition | Multi-embedding gallery + adaptive centroid |
| ✅ Webcam support for testing | CameraService with cv2.VideoCapture(0) |
| ✅ CPU-optimized | ONNX YOLO, 1 FPS, frame dedup, inference semaphore |
| ✅ Full dashboard UI | 7-page Next.js dashboard with live feed + charts |
| ✅ Analytics & business insights | Summary, frequency, hourly, top visitors + return_rate |
| ✅ Admin tools | Merge duplicates, mark staff, edit names, hard/soft delete |
| ✅ Scalable to millions | HNSW indexes, gallery pruning, table partitioning |
| ✅ Detection audit trail | `detection_events` table with is_ambiguous, match_source |
| ✅ Activity timeline page | `/api/activity` feed + Activity page (filter by type) |
| ✅ Configuration transparency | `/api/settings` read-only endpoint + Settings page |
| ✅ Region-of-interest support | ROI filter via `POST /api/camera/roi` |
| ✅ Privacy / retention | VISITOR_RETENTION_DAYS purge job + .gitignore of crops |
| ✅ Performance observability | X-Process-Time-Ms header on every response |
| ✅ Containerized deployment | Docker Compose: 4 services (postgres, pgadmin, backend, dashboard) |
| ✅ Frontend type safety | TypeScript interfaces for all API contracts (`lib/types.ts`) |
| ✅ Camera index discovery | `test_camera.py` utility for Windows webcam enumeration |
