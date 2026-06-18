# Restaurant Visitor Tracker — Optimization Plan

> Comprehensive optimization roadmap addressing: duplicate identity fragmentation,
> cross-camera correlation, embedding quality, re-identification accuracy, and
> computational efficiency. Derived from deep-dive analysis of the current
> codebase (2026-06-17 snapshot).

**Created:** 2026-06-18
**Status:** Planning phase
**Target:** Reduce false-new rate by 50%+, eliminate cross-camera identity splits, improve matching accuracy across pose/lighting variation.

---

## 1. Problem Catalogue: What's Breaking

### 1.1 Same Person → Multiple Identities (Identity Fragmentation)

**Root cause:** When the same person appears from a different angle, lighting, or
in a different camera, their embedding drifts below `RETURNING_FACE_THRESHOLD`
(0.55) — the system registers them as NEW instead of RETURNING.

| Scenario | Why It Breaks | Impact |
|----------|--------------|--------|
| Person enters from camera-0, later seen on camera-1 | No cross-camera identity correlation exists. Each camera resolves independently against the static gallery with no spatial/temporal handoff | **Duplicate visitor record per camera** |
| Person turns 90° (frontal → profile) | Gallery may lack a profile face for that person; the profile embedding vectors toward a different cluster | **New registration even though they're in the building** |
| Person removes glasses / mask / hat | Embedding centroid drifts; existing gallery faces are from different appearances | **Ambiguous or NEW decision** |
| Restaurant lighting changes (day→night) | CLAHE + gamma helps but embedding still shifts ~0.05-0.10 cosine | **Same person falls into grey zone or NEW threshold** |
| Two people in frame, heavy occlusion | Face assignment may swap persons between frames; gallery gets wrong faces | **Polluted gallery → cascading misidentification** |

### 1.2 Weak Anti-Duplicate Defenses

| Current Defense | Gap |
|-----------------|-----|
| `temporal_consistency.py` | ✓ Works within 30s window and 150px — but person must STAY in same camera. No cross-camera awareness. |
| `auto_merge_duplicates` | ✓ Manually triggered or via scheduled API call. **Not a background daemon** — duplicates accumulate between sweeps. |
| `probable_duplicate` review flags | ✓ Catches `top_similarity` near threshold. But only fires at registration time — doesn't catch when a visitor's centroid *later* drifts toward another. |
| Diversity check (cosine < 0.85) | ✓ Prevents near-duplicate gallery faces. But 0.85 is **very permissive** — faces at 0.84 similarity are distinct poses of the same person, added to gallery as if they were different. |
| Ambiguity gate | ✓ Prevents wrong merge. But also creates **orphan detections** — a real returning person whose embedding sits near another visitor's gets AMBIGUOUS and dropped, creating a gap in their visit. |

### 1.3 Gallery Quality Decay Over Time

- **det_score-only eviction** (`auto_enroller.py:183-196`): The eviction policy
  picks the lowest `det_score` face, not the overall **least useful** face.
  A sharp profile at det_score 0.62 may be evicted in favour of a blurry frontal
  at det_score 0.64 — but the profile is more valuable for diversity.

- **No periodic re-scoring**: Once a face is in the gallery, its utility is
  never reassessed. Faces added on day 1 (good lighting) may be worse than
  faces added on day 30 (better camera angle), but both sit forever.

- **Centroid drift from low-quality faces**: `update_centroid` uses a weighted
  average, but even a det_score=0.48 face (above `FACE_QUALITY_CUTOFF`) nudges
  the centroid. Over 50 visits, the centroid can drift away from the person's
  true embedding center.

### 1.4 Body Re-ID Is Effectively Useless

- **Disabled by default** (`ALLOW_BODY_FALLBACK=False`)
- **Same-session only** — OSNet embeddings are clothing-dependent, so they
  cannot identify a regular customer on a different day.
- **Separate body search** — body fallback is a second DB round-trip per
  grey-zone face (`_search_body`), not batched with the face search.
- **No body+face joint embedding** — the two 512-d vectors are never fused into
  a multimodal embedding; they're used independently.

### 1.5 No Multi-Object Tracking (MOT)

Each frame is independently resolved. There is no:
- Kalman filter predicting where a person will be next frame
- Hungarian algorithm matching detections frame-to-frame
- Track ID persistence across frames within a visit

The `temporal_consistency.py` gate is a lightweight substitute but doesn't
handle: crossing paths, temporary occlusion, or camera handoff.

---

## 2. Optimization Plan — Tiered by Impact & Effort

### TIER 1 — High Impact, Low Effort (Weeks 1-2)

#### 2.1 Embedding Quality-Weighted Gallery Eviction

**Current:** Evicts the face with lowest `det_score` in the target bin.

**Optimization:** Replace single-metric eviction with a composite score:
```
gallery_score = 0.35 * clarity    (Laplacian + frontality)
              + 0.30 * det_score   (InsightFace confidence)
              + 0.20 * uniqueness  (1 - max_cosine_sim_to_other_gallery_faces)
              + 0.15 * recency     (normalized days since added)
```
- Evict the face with the **lowest composite score** instead of lowest det_score.
- The `uniqueness` term ensures we keep faces that cover underrepresented angles.
- The `recency` term prevents keeping an old, low-quality face over a newer one.

**Files to modify:** `auto_enroller.py` — `_find_eviction_candidate`, `add_face_to_gallery`

**Expected gain:** 15-25% fewer false-new registrations from gallery degradation.

---

#### 2.2 Periodic Background Gallery Re-Scoring

**Current:** Gallery faces are scored only when operator triggers "auto-clean faces" manually.

**Optimization:** A background task (runs every 24h) that:
1. Re-scores every gallery face using `compute_clarity`
2. Updates `VisitorFace.clarity_score` in DB
3. Logs faces with degrading clarity for review
4. Optionally auto-evicts faces that dropped below `FACE_CLARITY_CUTOFF` (keeping the best one per pose bin, not just best overall)

**New settings:**
```
GALLERY_RESCORE_INTERVAL_HOURS: int = 24
GALLERY_AUTO_PRUNE: bool = True
GALLERY_MIN_FACES_PER_POSE: dict = {"frontal": 2, "left": 1, "right": 1, "down": 1}
```

**Files to modify:** `auto_enroller.py` (new function), `main.py` (background task startup)

**Expected gain:** Prevents gallery rot; improves return-recognition accuracy 10-15%.

---

#### 2.3 Enhanced Pose-Aware Search with Adjacent Bins

**Current:** Pose-aware search matches `exact_bin → frontal → unknown`.

**Problem:** A `left` profile face at -80° yaw has low cosine similarity to a
`right` profile at +80° — but they're mirror-symmetric and should still match
better than matching against `frontal`.

**Optimization:** Add adjacent-bin matching with a configurable rank multiplier:
```
ORDER BY CASE
    WHEN vf.pose_bin = p.pose_bin          THEN 1   -- exact match
    WHEN vf.pose_bin = adjacent(p.pose_bin) THEN 2   -- left↔frontal, right↔frontal
    WHEN vf.pose_bin = 'frontal'            THEN 3   -- frontal fallback
    WHEN vf.pose_bin = mirror(p.pose_bin)    THEN 4   -- left↔right (mirror)
    ELSE 5                                              -- unknown / fallback
END
```

Adjacency map:
- `left` ↔ `frontal`, `frontal` ↔ `right`
- Mirror: `left` ↔ `right` (faces share the same underlying bone structure)

**Files to modify:** `identity_resolver.py` — `_search_faces_batch`

**Expected gain:** 10-20% better recall for profile faces, fewer grey-zone drops.

---

#### 2.4 Smarter Grey-Zone Handling with Feature Fusion

**Current:** Grey zone (0.45 < sim < 0.55) → body fallback if enabled; else NEW only
if quality passes cutoff; else dropped. Body fallback is a separate query.

**Optimization:** In the grey zone, fuse face + body + spatial signals:
```
grey_score = 0.55 * face_cosine_sim
           + 0.25 * body_cosine_sim       (if body embedding exists)
           + 0.10 * spatial_consistency   (1 - px_dist/max_dist from temporal gate)
           + 0.10 * pose_match_bonus      (1.0 if exact bin match, else 0.5)
```
- If `grey_score >= GREY_FUSION_THRESHOLD` (default 0.52) → RETURNING
- This uses ALL available signals before resorting to NEW or DROP.

**Body embeddings** should be included in the batched gallery search (add a
body_similarity column to the LATERAL join) rather than requiring a separate DB
round-trip per face.

**Files to modify:** `identity_resolver.py`, `detection_pipeline.py`

**Expected gain:** 20-30% fewer wrong decisions in the grey zone.

---

### TIER 2 — Medium Impact, Medium Effort (Weeks 3-5)

#### 2.5 Multi-Object Tracking (DeepSORT / ByteTrack)

**Current:** Frame-by-frame identity resolution. No inter-frame track association.

**Optimization:** Add a lightweight Kalman filter + Hungarian assignment layer:

```
Frame t-1:  tracks = [Track(vid=A, bbox=...), Track(vid=B, bbox=...)]
Frame t:    detections = [Det(bbox=...), Det(bbox=...)]
            → Hungarian match using IoU + embedding cosine distance
            → Unmatched detections → resolve identity normally
            → Unmatched tracks → mark as "temporarily lost" (coast for N frames)
```

Track state machine:
```
ACTIVE → (no detection for K frames) → LOST → (detection reassociates) → ACTIVE
                                          ↘ (no detection for T seconds) → GONE
```

**Why this helps:**
- A person turning away (face disappears) is still tracked by bbox IoU —
  eliminates 60-80% of temporal-gate false-new registrations.
- Crossing paths — two people walk past each other and bboxes swap — are
  resolved by embedding consistency, not just bbox overlap.
- Camera handoff — when a person exits camera A's FOV and enters camera B's,
  the track state can be shared (via Redis or DB) to correlate identities.

**Implementation:**
- Add `backend/app/services/multi_tracker.py`
- Integrate into `camera_service.py` and `detection_pipeline.py`
- Use `filterpy` or `scipy` for Kalman filter (8-d state: cx, cy, a, h, vx, vy, va, vh)
- Hungarian algorithm via `scipy.optimize.linear_sum_assignment`

**Files to create:** `multi_tracker.py`
**Files to modify:** `camera_service.py`, `detection_pipeline.py`, `config.py`

**New settings:**
```
MOT_ENABLED: bool = True
MOT_MAX_COAST_FRAMES: int = 30        # frames before marking track LOST
MOT_MAX_LOST_SECONDS: float = 5.0     # seconds before deleting track
MOT_IOU_MATCH_THRESHOLD: float = 0.3  # min IoU for bbox-only association
MOT_EMBED_MATCH_WEIGHT: float = 0.6   # weight of embedding vs IoU in cost matrix
```

**Expected gain:** 40-60% reduction in within-camera identity fragmentation.

---

#### 2.6 Cross-Camera Identity Correlation

**Current:** Each camera resolves identities independently. Camera B has no
knowledge of who was seen on Camera A.

**Optimization:** A cross-camera handoff system with three layers:

**Layer 1 — Spatial Handoff (immediate):**
When a track exits camera A's FOV, calculate the physical direction vector and
estimated time-to-enter for adjacent cameras. When camera B sees a NEW face
within the expected time window, boost its similarity threshold by 0.10.

**Layer 2 — Temporal Handoff (medium-term):**
Maintain a Redis/DB "recently seen" set keyed by visitor_id → {camera_id,
last_seen, embedding, pose_bin}. Before creating a NEW visitor, check if this
person was recently seen on any camera.

**Layer 3 — Global Gallery Sync (long-term):**
When a high-confidence match updates a visitor's gallery on camera A, all other
cameras' in-memory caches are invalidated (or notified via Redis pub/sub) to
use the updated gallery.

**Implementation:**
- Add `camera_adjacency` configuration (which cameras share physical space)
  ```
  CAMERA_ADJACENCY: dict = {"cam-0": ["cam-1"], "cam-1": ["cam-0", "cam-2"]}
  ```
- `CrossCameraCoordinator` class managing the shared recently-seen pool
- Redis-backed for multi-process; in-memory dict for single-process

**Files to create:** `cross_camera.py`
**Files to modify:** `detection_pipeline.py`, `camera_service.py`, `identity_resolver.py`, `config.py`

**Expected gain:** Eliminates 80-90% of cross-camera duplicate registrations.

---

#### 2.7 Periodic Background Dedup Sweep

**Current:** `auto_merge_duplicates` only runs on explicit API call.

**Optimization:** A background task (runs every 6h) that:
1. Queries all visitor pairs whose centroids have cosine similarity ≥
   `BACKGROUND_DEDUP_SIMILARITY` (default 0.75 — a very confident same-person threshold)
2. Runs the merge ONLY when:
   - Both visitors have ≥ 3 gallery faces (enough data to trust the match)
   - The pooled gallery after merge passes a quality check (no single bin > 60% of faces)
   - Neither visitor is staff
3. Auto-merges high-confidence pairs; flags medium-confidence pairs for review

**This is much more aggressive than the current `auto_merge_duplicates`**
because it's comparing centroids (not individual gallery faces) at a much
higher threshold — safe because a centroid at 0.75+ cosine is almost
certainly the same person.

**Files to modify:** `review_queue.py` (new function), `main.py` (background task)
**New settings:**
```
BACKGROUND_DEDUP_ENABLED: bool = True
BACKGROUND_DEDUP_INTERVAL_HOURS: int = 6
BACKGROUND_DEDUP_SIMILARITY: float = 0.75
BACKGROUND_DEDUP_MIN_GALLERY_SIZE: int = 3
```

**Expected gain:** Proactively merges 70-85% of duplicates without operator
intervention.

---

#### 2.8 Embedding Ensemble — Face + Body Joint Representation

**Current:** Face and body embeddings are 512-d vectors used independently.
The body embedding is clothing-dependent; the face embedding is pose/lighting-dependent.

**Optimization:** Create a **fused embedding** that combines face + body:
- **For same-session:** Concatenate [face_emb (512), body_emb (512)] and reduce
  to 512-d via a learned PCA projection or a small MLP trained offline.
- **For cross-session:** Use face embedding only, but with the body embedding as
  a confidence booster (grey-zone fusion from Tier 1).

The fused embedding helps same-session re-acquisition (body+face jointly
discriminate better than either alone), which is the most common scenario in a
restaurant setting.

**Implementation:**
- Train PCA on a corpus of same-person face+body pairs from existing detection data
- Store `fused_embedding` on `detection_events` for same-session use
- Use in temporal consistency gate and short-term re-acquisition

**Files to modify:** `identity_resolver.py`, `models.py` (add column), migration 010

**Expected gain:** 15-25% better same-session re-identification.

---

### TIER 3 — High Impact, High Effort (Weeks 5-8)

#### 2.9 Appearance-Aware Re-Identification

**Current:** Body re-ID (OSNet) is the only appearance signal, and it's clothing-
dependent — useless across visits.

**Optimization:** Add **clothing-agnostic appearance features**:
- **Gait features** (if walking is captured) — OpenPose keypoints → gait energy image
- **Height estimation** — from bbox dimensions + camera calibration
- **Build estimation** — shoulder width from YOLO pose keypoints (YOLOv8-pose variant)

These features are stable across visits (same person, different clothes, same height).
Store them on the `visitors` row and use as a **pre-filter** before face matching:
```
if abs(height_diff) > 10cm or abs(build_diff) > 0.3:
    skip this gallery entry entirely (can't be the same person)
```

This reduces the HNSW search space and prevents clearly-wrong matches.

**Files to create:** `appearance_features.py`
**Files to modify:** `models.py` (add height/build columns), migration 011, `identity_resolver.py`

**New settings:**
```
APPEARANCE_PRE_FILTER: bool = True
HEIGHT_TOLERANCE_CM: float = 10.0
BUILD_TOLERANCE: float = 0.3
```

**Expected gain:** 10-15% fewer false-positive matches, especially for visitors
with similar facial features.

---

#### 2.10 Adaptive Threshold Per Visitor

**Current:** Global thresholds (`RETURNING_FACE_THRESHOLD=0.55`) apply to all
visitors uniformly.

**Problem:** A visitor with 50 high-quality gallery faces should have a higher
returning threshold (more evidence = more confidence needed to match). A new
visitor with 2 faces should have a lower threshold (less evidence = we should
be more permissive to avoid fragmenting them further).

**Optimization:** Per-visitor adaptive threshold:
```
per_visitor_threshold = base_threshold
                      + 0.03 * min(gallery_size / 10, 1.0)   -- more faces = stricter
                      - 0.05 * new_visitor_penalty             -- (< 5 faces = looser)
                      - 0.02 * max(0, 1 - avg_clarity)         -- low-quality gallery = looser
```
- Clamp to [base - 0.10, base + 0.10]
- Stored on `visitors.adaptive_threshold`, recalculated on every gallery update
- Used in `_decide_from_face` instead of the global threshold

**Files to modify:** `identity_resolver.py`, `auto_enroller.py`, `models.py`
**Migration:** 012

**Expected gain:** 20-30% reduction in false-new for infrequent visitors,
10-15% reduction in false-returning for well-known visitors.

---

#### 2.11 Gallery Pruning by Information Gain

**Current:** Gallery keeps up to 10 faces per visitor. Eviction is per-pose-bin
with det_score as the only metric.

**Optimization:** Maximize **information gain** per gallery slot:
1. Compute pairwise cosine similarity matrix of all gallery faces for a visitor
2. Remove the face that contributes the **least unique information** (highest
   average similarity to all other faces)
3. Prioritize keeping faces that:
   - Cover distinct pose bins (spread across yaw angles)
   - Have high clarity (sharp + frontal)
   - Are from different lighting conditions (day/night)
   - Are from different visits (temporal diversity)

```
info_gain(face_i) = clarity(face_i) * (1 - mean_cos_sim(face_i, all_other_faces))
```

**Files to modify:** `auto_enroller.py` — `add_face_to_gallery`

**Expected gain:** Better recall with same gallery size (more representative
faces per visitor).

---

#### 2.12 Post-Detection Similarity Re-Ranking

**Current:** Top-2 gallery matches come from HNSW approximate search. HNSW
recall is typically 95-99%, but the 1-5% misses can cause false-NEW decisions.

**Optimization:** After the HNSW batch search returns top-2, re-rank those
candidates using a **more expensive but more accurate** comparison:
- Run the query face embedding through the gallery visitor's full set of faces
  (not just the HNSW-retrieved top-2)
- Compute `max_cos_sim(query, all_gallery_faces_of_visitor)` as the true
  similarity
- This catches the case where HNSW missed a high-similarity gallery face

**Trade-off:** Adds ~5-10ms per face (reading the visitor's full gallery from DB).
Acceptable because identity resolution is not the bottleneck (inference is).

**Files to modify:** `identity_resolver.py` — `resolve_batch`

**Expected gain:** Eliminates nearly all HNSW false-negative misses.

---

## 3. Computational Efficiency Optimizations

### 3.1 Incremental Gallery HNSW Index Rebuild

**Current:** The pgvector HNSW index (`visitor_faces_embedding_idx`) is
rebuilt from scratch on every `REINDEX` or vacuum. For 10k+ faces this can
lock the table for seconds.

**Optimization:** Use pgvector's `ivfflat` index for the write-heavy gallery
table (faster inserts) and `hnsw` for the read-heavy centroid table. Or use
partial indexes partitioned by visitor activity.

**Migration:** Update index strategy.

### 3.2 dHash Cache with Approximate Matching

**Current:** `FaceEmbeddingCache` uses exact dHash matching — only pixel-
identical aligned crops hit the cache.

**Optimization:** Allow approximate dHash matching (Hamming distance ≤ 2) to
reuse embeddings for near-identical crops (slight lighting shift, 1-2 pixel
jitter). This requires:
- A locality-sensitive hash (LSH) or a BK-tree for fast approximate lookup
- Configurable Hamming distance threshold

**Settings:**
```
EMBEDDING_CACHE_HAMMING_DISTANCE: int = 2
```

**Expected gain:** 2-3x more cache hits in real-world scenarios with minor
camera jitter.

### 3.3 ROI-Aware Embedding Cache

**Current:** The embedding cache is camera-wide. When ROI changes, cached
embeddings from outside the new ROI are useless.

**Optimization:** Key the cache by `(dHash, roi_hash)`. When ROI changes,
invalidate only the stale entries, not the entire cache.

### 3.4 ONNX for ArcFace & OSNet

**Current:** Only YOLO is exported to ONNX. ArcFace and OSNet run via
PyTorch/InsightFace.

**Optimization:** Export ArcFace backbone and OSNet to ONNX for CPU inference
(2-3x speedup on CPU, same as YOLO). InsightFace's detection model stays in
PyTorch (it's relatively light).

**Expected gain:** 30-40% faster CPU inference for face recognition pass.

---

## 4. Accuracy Improvements — Threshold & Algorithm Tuning

### 4.1 Dual-Threshold System

**Current:** Single `RETURNING_FACE_THRESHOLD` (0.55) gates all returning decisions.

**Optimization:** Two thresholds for different scenarios:

| Scenario | Threshold | Rationale |
|----------|-----------|-----------|
| Same camera, same visit, last seen < 2 min ago | 0.45 | Person hasn't changed — temporal proximity is strong evidence |
| Same camera, different visit (≥ 20 min gap) | 0.55 | Same setting but could be a different person |
| Different camera, same visit (cross-camera) | 0.58 | Higher bar — different angle/lighting |
| Different camera, different visit | 0.62 | Highest bar — maximum uncertainty |
| First detection of the day (no prior session) | 0.50 | Permissive — we'd rather match than fragment |

**Implementation:** `_decide_from_face` accepts a `context: ResolutionContext`
dataclass with camera_id, visit_state, time_since_last_seen. The effective
threshold is selected from a lookup table.

**Files to modify:** `identity_resolver.py`, `detection_pipeline.py`, `config.py`

### 4.2 Post-Merge Gallery Dedup

**Current:** `merge_visitors` pools both visitors' gallery faces but doesn't
de-duplicate. If both visitors had faces of the same pose at similar quality,
the merged gallery has redundant entries.

**Optimization:** After merge, run a dedup pass:
1. Within each pose bin, compute pairwise cosine similarity
2. If two faces have similarity > 0.95, keep the higher-clarity one
3. Recompute centroid from deduped gallery

**Files to modify:** `visitor_merge.py`

### 4.3 Temporal Gate — Multi-Feature Matching

**Current:** Temporal gate uses cosine similarity (0.7 weight) + spatial proximity (0.3).

**Optimization:** Add body embedding and pose consistency:
```
score = 0.50 * face_cosine_sim
      + 0.20 * spatial_proximity   (1 - px_dist / max_dist)
      + 0.15 * body_cosine_sim     (if body embedding available)
      + 0.10 * pose_consistency    (1 if same bin, 0.5 if adjacent, 0 else)
      + 0.05 * size_consistency    (1 - abs(height_ratio - 1))
```

**Files to modify:** `temporal_consistency.py`

---

## 5. Data Quality & Auto-Recovery

### 5.1 Stale Visitor Detection

**Current:** No mechanism to detect visitors whose gallery has degraded over
time (e.g., centroid drifted, best faces were accidentally evicted).

**Optimization:** Weekly job that:
1. For each active visitor with ≥ 5 visits, compute the mean cosine similarity
   of their last 10 detection embeddings to their own centroid
2. If mean < 0.55, flag as `possible_degraded` — the gallery no longer
   represents this visitor well
3. Auto-trigger gallery cleanup or suggest operator review

### 5.2 Detection Event Anomaly Detection

**Current:** Detection events are logged but not analyzed for patterns.

**Optimization:** Analyze detection event stream for anomalies:
- Sudden spike in `ambiguous` rate → lighting change or camera moved
- Sudden drop in face detection rate → camera obstruction
- Visitor with alternating high/low face similarity → centroid drift
- Multiple visitors with same detection timestamp and bbox → YOLO hallucination

Alert via webhook if anomaly score exceeds threshold.

### 5.3 Confidence Calibration

**Current:** ArcFace cosine similarity values are used directly as confidence.
But cosine similarity is NOT a calibrated probability — 0.55 from ArcFace
means something different for masked vs. unmasked, profile vs. frontal.

**Optimization:** Maintain a per-threshold calibration curve:
- For each (pose_bin, mask_state) combination, track: when similarity=X,
  what % of the time was it truly the same person?
- Adjust the effective threshold to maintain a target false-accept rate (e.g., 1%)

This requires ground-truth labels — use operator merge/review actions as
weak labels (if an operator merged two visitors, they WERE the same person).

---

## 6. Implementation Priority Matrix

| # | Optimization | Impact | Effort | Risk | Priority |
|---|-------------|--------|--------|------|----------|
| 2.1 | Quality-weighted eviction | High | Low | Low | **1** |
| 2.3 | Adjacent-bin pose search | High | Low | Low | **2** |
| 2.4 | Grey-zone feature fusion | High | Medium | Low | **3** |
| 2.7 | Background dedup sweep | High | Medium | Medium | **4** |
| 2.2 | Periodic gallery re-scoring | Medium | Low | Low | **5** |
| 4.1 | Dual-threshold system | High | Medium | Medium | **6** |
| 2.8 | Face+body fused embedding | Medium | Medium | Low | **7** |
| 2.5 | Multi-object tracking (MOT) | High | High | High | **8** |
| 2.6 | Cross-camera correlation | High | High | High | **9** |
| 2.10 | Per-visitor adaptive threshold | High | Medium | High | **10** |
| 4.3 | Multi-feature temporal gate | Medium | Low | Low | **11** |
| 2.12 | Post-HNSW re-ranking | Medium | Low | Low | **12** |
| 3.2 | Approximate dHash cache | Medium | Medium | Low | **13** |
| 2.11 | Info-gain gallery pruning | Medium | Medium | Medium | **14** |
| 2.9 | Appearance-aware re-ID | Medium | High | Medium | **15** |
| 4.2 | Post-merge gallery dedup | Low | Low | Low | **16** |
| 5.1 | Stale visitor detection | Medium | Medium | Low | **17** |
| 5.3 | Confidence calibration | Medium | High | High | **18** |
| 3.4 | ONNX for ArcFace/OSNet | Medium | High | Medium | **19** |
| 5.2 | Detection anomaly detection | Low | Medium | Low | **20** |

---

## 7. New Settings Summary (to add to config.py)

```python
# ── Tier 1 Optimizations ────────────────────────────────────
# Quality-weighted gallery eviction composite weights
GALLERY_SCORE_CLARITY_WEIGHT: float = 0.35
GALLERY_SCORE_DETSCORE_WEIGHT: float = 0.30
GALLERY_SCORE_UNIQUENESS_WEIGHT: float = 0.20
GALLERY_SCORE_RECENCY_WEIGHT: float = 0.15

# Periodic gallery re-scoring
GALLERY_RESCORE_INTERVAL_HOURS: int = 24
GALLERY_AUTO_PRUNE: bool = True
GALLERY_MIN_FACES_PER_POSE: dict = {"frontal": 2, "left": 1, "right": 1, "down": 1}

# Adjacent pose-bin matching
POSE_ADJACENT_MATCHING: bool = True

# Grey-zone feature fusion
GREY_ZONE_FUSION_ENABLED: bool = True
GREY_FUSION_THRESHOLD: float = 0.52
GREY_FACE_WEIGHT: float = 0.55
GREY_BODY_WEIGHT: float = 0.25
GREY_SPATIAL_WEIGHT: float = 0.10
GREY_POSE_WEIGHT: float = 0.10

# ── Tier 2 Optimizations ────────────────────────────────────
# Multi-object tracking
MOT_ENABLED: bool = True
MOT_MAX_COAST_FRAMES: int = 30
MOT_MAX_LOST_SECONDS: float = 5.0
MOT_IOU_MATCH_THRESHOLD: float = 0.3
MOT_EMBED_MATCH_WEIGHT: float = 0.6

# Cross-camera correlation
CROSS_CAMERA_ENABLED: bool = True
CAMERA_ADJACENCY: dict = {}
CROSS_CAMERA_TEMPORAL_WINDOW_SECONDS: float = 30.0

# Background dedup sweep
BACKGROUND_DEDUP_ENABLED: bool = True
BACKGROUND_DEDUP_INTERVAL_HOURS: int = 6
BACKGROUND_DEDUP_SIMILARITY: float = 0.75
BACKGROUND_DEDUP_MIN_GALLERY_SIZE: int = 3

# Face+body fused embedding
FUSED_EMBEDDING_ENABLED: bool = False
FUSED_EMBEDDING_DIM: int = 512

# ── Tier 3 Optimizations ────────────────────────────────────
# Appearance-aware pre-filter
APPEARANCE_PRE_FILTER: bool = False
HEIGHT_TOLERANCE_CM: float = 10.0
BUILD_TOLERANCE: float = 0.3

# Per-visitor adaptive threshold
ADAPTIVE_THRESHOLD_ENABLED: bool = False
ADAPTIVE_THRESHOLD_MIN_OFFSET: float = -0.10
ADAPTIVE_THRESHOLD_MAX_OFFSET: float = 0.10

# Post-HNSW re-ranking
HNSW_POST_RERANK: bool = True

# Approximate dHash embedding cache
EMBEDDING_CACHE_HAMMING_DISTANCE: int = 0   # 0 = exact only

# ── Dual-threshold system ───────────────────────────────────
DUAL_THRESHOLD_ENABLED: bool = True
THRESHOLD_SAME_CAMERA_RECENT: float = 0.45
THRESHOLD_SAME_CAMERA_NEW_VISIT: float = 0.55
THRESHOLD_CROSS_CAMERA_SAME_VISIT: float = 0.58
THRESHOLD_CROSS_CAMERA_NEW_VISIT: float = 0.62
```

---

## 8. Migration Plan

| Migration | Description |
|-----------|-------------|
| 010 | Add `clarity_score` index, `gallery_score` column to visitor_faces |
| 011 | Add `height_est`, `build_est` columns to visitors |
| 012 | Add `adaptive_threshold` column to visitors |
| 013 | Add `fused_embedding` column to detection_events |
| 014 | Add cross-camera tracking tables (`cross_camera_handoffs`) |

---

## 9. Success Metrics

| Metric | Current (estimated) | Target | Measured By |
|--------|---------------------|--------|-------------|
| False-new rate (person fragmented into ≥2 records) | ~15-25% | < 8% | Manual audit of 100 visitors, count duplicates |
| Cross-camera duplicate rate | ~40-60% | < 10% | Same person seen on 2 cameras = 2 records vs 1 |
| Ambiguous drop rate (real person gets AMBIGUOUS) | ~5-10% | < 3% | detection_events with is_ambiguous=True, manual review |
| Grey-zone resolution accuracy | ~60% | > 80% | Manual labeling of grey-zone decisions |
| Gallery quality score (avg clarity) | ~0.55 | > 0.65 | `SELECT AVG(clarity_score) FROM visitor_faces` |
| Duplicate merge backlog | ~50-200 | < 20 | `SELECT COUNT(*) FROM review_queue WHERE resolved=FALSE AND flag_type='probable_duplicate'` |
| Per-frame inference time | ~800-1200ms (CPU) | < 900ms | Monitoring p95 frame latency |
| HNSW search recall | ~95% | > 99% | Post-re-rank comparison against exact search |

---

## 10. Testing Strategy

For each optimization:

1. **Offline benchmark** — Run on a labeled validation set of 500+ frames with
   known ground-truth identities. Measure precision/recall/F1 before and after.

2. **A/B shadow mode** — New logic runs in parallel with old logic; only old
   logic's decisions are committed. Compare outputs to measure divergence.

3. **Canary deployment** — Enable on one camera for 48h; compare new-visitor
   creation rate, ambiguous rate, and review queue growth against baseline.

4. **Rollback plan** — Every optimization is behind a feature flag
   (`*_ENABLED`). Disabling the flag instantly reverts to old behavior
   without a code deploy.

---

## Appendix A: Key Files Reference

| File | Current Role | Optimization Impact |
|------|-------------|---------------------|
| `identity_resolver.py` | Face matching + batch search | 2.3, 2.4, 2.12, 4.1, 2.10 |
| `auto_enroller.py` | Gallery management + centroid | 2.1, 2.2, 2.11, 4.2 |
| `temporal_consistency.py` | Same-person re-acquisition | 4.3 |
| `review_queue.py` | Flag + auto-merge | 2.7 |
| `visitor_merge.py` | Visitor merge logic | 4.2 |
| `detection_pipeline.py` | Detection orchestration | 2.5, 2.6, 2.8, 4.1 |
| `cascade_pipeline.py` | Body-skip cascade | 3.4 |
| `cv_pipeline.py` | CV pipeline | 2.9, 3.3 |
| `camera_service.py` | Camera + streaming | 2.5, 2.6 |
| `visit_tracker.py` | Visit state machine | 2.6 |
| `config.py` | All settings | All sections |
| `models.py` | DB schema | All sections |

---

## Appendix B: Known Risks & Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| MOT Kalman filter predictions drift on sudden camera movement | Medium | Reset Kalman covariance on scene-change detection (frame dedup MAD spike) |
| Cross-camera false handoff (two different people look similar) | High | Require high confidence (≥0.65) + appearance check + temporal window validation |
| Background dedup mistakenly merges two different people | High | Very conservative threshold (0.75+) + min gallery size + human review for medium-confidence pairs |
| Per-visitor adaptive threshold loosens too much for sparse visitors | Medium | Clamp to [-0.10, +0.10]; never go below 0.45 absolute |
| ONNX ArcFace export changes embedding values slightly vs PyTorch | Medium | Validate cosine distribution on 10k pairs before switching; keep PyTorch as fallback |
| Fused embedding degrades when body is occluded (seated patron) | Medium | Only fuse when body bbox area > 20% of person bbox (not heavily occluded) |
| Approximate dHash creates false embedding reuse | Low | Hamming 2 on 64-bit hash = 2/64 ≈ 3% bit difference; safe for aligned crops |