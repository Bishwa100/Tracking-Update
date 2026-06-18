# Restaurant Visitor Tracker - Optimization and Accuracy Plan v3

> Updated audit and implementation plan based on `TECHNIQUES_AND_APPROACHES.md`
> and the current backend implementation under `backend/app/`.
>
> This document focuses on the real failure mode you called out: the same person
> being detected as a new person when seen from another camera angle or another
> camera. It also lists duplicated code, dead or underused optimizations, and a
> staged path to improve accuracy without creating unsafe false merges.

**Last updated:** 2026-06-18

---

## 1. Executive Summary

The current system already has strong foundations:

- Full-frame single-pass face detection with InsightFace / ArcFace.
- Multi-pose gallery search through `visitor_faces`.
- pgvector HNSW search.
- Ambiguity handling to avoid merging two different people.
- Temporal consistency gate.
- Frame de-duplication and a parallel camera pipeline.
- Review queue and manual/auto merge workflow.

The biggest remaining problem is identity fragmentation:

```text
Same physical person
  -> different head angle / lighting / camera
  -> lower face similarity
  -> resolver enters grey zone
  -> detection is promoted to NEW
  -> duplicate visitor record is created
```

The highest-impact fix is not a new model first. The first fix is decision
logic: stop creating a new visitor from a single grey-zone frame. Then add
tracklets, continuous pose, per-visitor thresholds, and cross-camera
reconciliation.

### Top Priority Issues

| Priority | Issue | Why it matters |
|---|---|---|
| P0 | Grey-zone detections become `NEW` too easily | Main cause of duplicate visitors from side angles, masks, and bad lighting |
| P0 | Identity decisions are per-frame, not per-tracklet | One bad frame can permanently create a duplicate |
| P0 | Cross-camera context is not used in identity decisions | Same person can be registered separately on each camera |
| P1 | Only pose bin is persisted, not yaw/pitch/roll | A mild side angle and extreme profile are both just `left` or `right` |
| P1 | `FaceEmbeddingCache` has no eviction | Long-running camera streams can grow memory over time |
| P1 | Periocular masked-face embedding is implemented but not wired | Masked people only get a threshold offset, not a better embedding path |
| P1 | Cascade body-skip pipeline exists but is not used | OSNet may run even when body embedding is not needed |
| P2 | Repeated bbox, crop, similarity, and visit-state logic | Makes accuracy changes harder and bug-prone |

---

## 2. Current Implementation Snapshot

### 2.1 Current detection path

```text
camera_service.py / api/detect.py
  -> cv_pipeline.process_frame()
      -> YOLO person detection
      -> InsightFace full-frame face detection
      -> face assignment to person boxes
      -> optional fallback ArcFace on person crop
      -> OSNet body embeddings
  -> detection_pipeline.process_detections()
      -> mask heuristic
      -> identity_resolver.resolve_batch()
      -> auto_enroller register/update
      -> visit_tracker
      -> detection_events audit
```

### 2.2 Current identity decision thresholds

| Setting | Current default | Current use |
|---|---:|---|
| `RETURNING_FACE_THRESHOLD` | `0.55` | Above this means returning if ambiguity check passes |
| `NEW_VISITOR_MAX_SIMILARITY` | `0.45` | Below this means new in `_decide_from_face()` |
| `REJECT_SIMILARITY` | `0.35` | Defined and patchable, but not used by `identity_resolver.py` |
| `AMBIGUITY_MARGIN` | `0.05` | Top match must beat runner-up by this margin |
| `STRONG_MATCH_THRESHOLD` | `0.65` | High-confidence gallery/centroid update |
| `FACE_QUALITY_CUTOFF` | `0.45` | Minimum face quality for enrollment/gallery |

### 2.3 Current database identity data

| Table | Current fields | Missing for this plan |
|---|---|---|
| `visitors` | centroid face/body embeddings, stats, consent, thumbnail | personal threshold stats, quality distribution |
| `visitor_faces` | embedding, det_score, body_embedding, crop_path, clarity_score, `pose_bin` | yaw, pitch, roll, source camera, source event |
| `detection_events` | visitor_id, visit_id, similarity, ambiguity, camera_id, bbox | tracklet_id, candidate/top-match metadata, cross-camera group |
| `visits` | visitor_id, enter/leave, confidence, camera_id | multi-camera path / transition metadata |

---

## 3. Root Cause: Same Person Becomes Multiple Visitors

### 3.1 Grey-zone promotion to `NEW`

`identity_resolver._decide_from_face()` returns `match_source="none"` for a
grey-zone match. Then `resolve_batch()` promotes that grey-zone result to a new
visitor if `det_score >= FACE_QUALITY_CUTOFF`.

Current behavior:

```text
top similarity = 0.50
returning threshold = 0.55
new max similarity = 0.45
face quality = 0.80

Result:
  not returning
  not directly new
  grey zone
  quality is good
  -> promoted to NEW
```

This is exactly the scenario where a returning person at a new angle becomes a
duplicate visitor.

**Fix direction:** Grey zone must mean `hold`, `tracklet`, `review`, or
`candidate returning`, not immediate new registration.

### 3.2 Single-frame enrollment

`register_new_visitor()` creates a visitor from the first eligible face. If that
face is side-angle, low-light, motion-blurred, or partially occluded, the
gallery starts from a weak identity seed.

**Fix direction:** New enrollment should require a short tracklet with multiple
consistent observations, or a very low best-gallery similarity.

### 3.3 Pose is too coarse

`cv_pipeline.estimate_pose()` computes `yaw`, `pitch`, and `roll`, but
`auto_enroller` stores only `pose_bin` in `visitor_faces`. The resolver receives
only the bin.

Problem:

```text
yaw 20 degrees  -> right
yaw 75 degrees  -> right
```

Those two faces can have very different ArcFace embeddings, but the resolver
cannot distinguish them during search or thresholding.

**Fix direction:** Store continuous pose and use angular distance in search and
gallery selection.

### 3.4 Top-2 ambiguity is too shallow

The SQL query returns only top-2 gallery rows. If the top two rows both belong
to the same visitor, the resolver cannot see the closest different visitor.

Correct logic should be:

1. Search top K gallery rows, for example K=5 or K=10.
2. Collapse rows to the best score per visitor.
3. Compare the best visitor against the best different visitor.
4. Treat multiple rows from the same visitor as supporting evidence, not
   ambiguity.

### 3.5 Cross-camera context is not used

`camera_id` is stored on `DetectionEvent` and `Visit`, but identity resolution
does not use it. `visitor_faces` does not store source camera. The temporal gate
is process-local and stores no camera topology.

Result:

```text
Camera A sees visitor at entrance -> visitor_1
Camera B sees same person from side angle -> top similarity too low
Resolver has no camera transition context
Grey zone becomes NEW -> visitor_2
```

### 3.6 Masked-face support is partial

`mask_detector.extract_periocular_region()` exists, but
`detection_pipeline.py` only marks the detection as masked and applies
`MASKED_FACE_THRESHOLD_OFFSET`. It does not re-embed the eye region.

### 3.7 Body re-ID is not a durable identity signal

OSNet body embeddings are clothing-dependent. They are useful for same-session
re-acquisition or cross-camera handoff during the same visit, but they should
not be trusted across days.

**Fix direction:** Use body and clothing color as short-term evidence only,
weighted below face similarity.

---

## 4. Duplicate and Redundant Code Audit

### 4.1 Bounding box and crop logic

Repeated bbox clamping, cropping, offsetting, and center checks appear in:

- `cv_pipeline.py`
- `services/detection_pipeline.py`
- `services/camera_service.py`
- `services/cascade_pipeline.py`
- `utils.py`

**Plan:** Add `backend/app/utils/geometry.py`.

Recommended helpers:

```python
def clamp_bbox(bbox: dict, frame_shape: tuple) -> dict: ...
def bbox_area(bbox: dict) -> float: ...
def bbox_center(bbox: dict) -> tuple[float, float]: ...
def bbox_iou(a: dict, b: dict) -> float: ...
def crop_from_frame(frame, bbox, margin: float = 0.0, min_size: int = 4): ...
def offset_bbox(bbox: dict, dx: int, dy: int) -> dict: ...
```

### 4.2 Similarity and normalization logic

Current repeated logic:

- SQL cosine similarity in `identity_resolver.py`.
- NumPy cosine in `temporal_consistency.py`.
- Dot-product similarity in `auto_enroller.py`.
- Normalization in `utils.py`, `ml_models.py`, and then again in `cv_pipeline.py`.

**Plan:** Add `backend/app/utils/similarity.py`.

Recommended helpers:

```python
def normalize_embedding(embedding) -> list[float]: ...
def cosine_similarity(a, b, assume_normalized: bool = True) -> float: ...
def pairwise_cosine(embeddings) -> np.ndarray: ...
```

Keep SQL vector search in `identity_resolver.py`, but use the shared utilities
for in-memory comparisons.

### 4.3 Visit tracking state machine

`visit_tracker.py` and `redis_visit_tracker.py` duplicate the visit lifecycle:
open, update, close after cooldown, seated cooldown, max-duration cap.

**Plan:** Extract a shared state machine and provide storage backends:

```text
VisitTrackerCore
  -> InMemoryVisitStore
  -> RedisVisitStore
```

### 4.4 Frame de-duplication

Frame de-duplication is used separately in live camera and upload paths.

**Plan:** Add a `FrameDedupBuffer` with:

- previous signature
- threshold
- ROI-aware signature source
- counters for hit rate

### 4.5 Mask detector region extraction

`mask_detector.py` extracts lower and upper face regions multiple times.
`lower_bright` is computed but not used in the final mask decision.

**Plan:** Consolidate region stats into one helper:

```python
def region_stats(face_crop, y0: float, y1: float) -> dict:
    return {"mean": ..., "std": ..., "crop": ...}
```

### 4.6 Face image saving

`_save_thumbnail()` and `_save_face_crop()` in `auto_enroller.py` repeat
directory creation, path construction, and `cv2.imwrite()` handling.

**Plan:** Use one image persistence helper for visitor assets.

### 4.7 Camera parallel/sequential paths

`camera_service.py` has two processing paths with repeated logic for:

- frame read and de-duplication
- inference call
- DB processing
- stats update
- annotation building

**Plan:** Extract a shared `FrameProcessor` used by both modes.

---

## 5. Dead, Partial, or Underused Optimizations

| Feature | Current state | Action |
|---|---|---|
| `cascade_pipeline.process_frame_cascade()` | Implemented but camera/upload paths call `process_frame()` | Add `PIPELINE_CASCADE` setting and wire it |
| `FACE_CONF_SKIP_BODY` | Configured but only used inside inactive cascade path | Activate cascade or skip OSNet when body signal is not needed |
| `preprocess_face_for_recognition()` | Defined in `utils.py`, not wired into recognition path | A/B test on aligned recognition crops |
| `extract_periocular_region()` | Defined but not called after mask detection | Re-embed masked eye region or store as second candidate |
| `_is_group_frame()` | Defined in `cv_pipeline.py`, not used downstream | Suppress new enrollment in crowded ambiguous frames |
| `REJECT_SIMILARITY` | Defined and patchable, not used in resolver | Use as hard-new floor |
| `flag_ambiguous_visitor()` | Defined but not called | Wire to ambiguous-rate tracking |
| `FaceEmbeddingCache` | Works but has no size limit | Add LRU/TTL eviction and metrics |

---

## 6. Best Accuracy Improvements

### 6.1 Replace immediate grey-zone new registration

New decision policy:

| Condition | Decision |
|---|---|
| No gallery match and face quality is high | Candidate new, but prefer tracklet confirmation |
| `top_sim <= REJECT_SIMILARITY` | Strong new candidate |
| `REJECT_SIMILARITY < top_sim < RETURNING_FACE_THRESHOLD` | Grey zone: do not create immediately |
| `top_sim >= RETURNING_FACE_THRESHOLD` and clear different-visitor margin | Returning |
| Top visitor and best different visitor too close | Ambiguous/review |

Recommended result states:

```python
match_source = "face"       # returning
match_source = "new"        # confirmed new
match_source = "grey_zone"  # hold/tracklet/review
match_source = "ambiguous"  # competing visitor identities
match_source = "temporal"   # recovered from recent track
match_source = "cross_camera" # recovered from another camera
```

### 6.2 Top-K visitor-level ambiguity

Change `_search_faces_batch()` from top-2 rows to top-K rows. Then collapse to
best score per visitor.

Pseudo-logic:

```python
rows = search_gallery_top_k(query_embedding, k=10)
best_by_visitor = keep_best_row_per_visitor(rows)
top = best_by_visitor[0]
runner = first different visitor, if any

if top.sim >= threshold and (runner is None or top.sim - runner.sim >= margin):
    return RETURNING
if runner and top.sim - runner.sim < margin:
    return AMBIGUOUS
```

### 6.3 Tracklet-based registration

Add a short-lived tracklet buffer before enrollment.

```text
Frame 1: face seen, weak similarity
Frame 2: same bbox trajectory, same person
Frame 3: face turns, better embedding
  -> aggregate tracklet embedding
  -> resolve once
  -> create/update one visitor
```

Minimum viable implementation:

- 2 to 5 second tracklet window.
- Associate detections by bbox IoU/center distance and embedding similarity.
- Store embeddings, det_score, pose, bbox, timestamp, camera_id.
- Use quality-weighted mean embedding for final resolution.
- Require at least 2 consistent observations before creating a visitor unless
  there is no gallery match at all and similarity is below `REJECT_SIMILARITY`.

### 6.4 Persist continuous pose

Migration:

```sql
ALTER TABLE visitor_faces ADD COLUMN yaw FLOAT;
ALTER TABLE visitor_faces ADD COLUMN pitch FLOAT;
ALTER TABLE visitor_faces ADD COLUMN roll FLOAT;
ALTER TABLE visitor_faces ADD COLUMN source_camera_id TEXT;
```

Resolver improvement:

```sql
ORDER BY
  CASE
    WHEN vf.yaw IS NOT NULL AND ABS(vf.yaw - :yaw) < 15 THEN 1
    WHEN vf.yaw IS NOT NULL AND ABS(vf.yaw - :yaw) < 35 THEN 2
    WHEN vf.pose_bin = :pose_bin THEN 3
    WHEN vf.pose_bin = 'frontal' THEN 4
    ELSE 5
  END,
  vf.embedding <=> :embedding
```

### 6.5 Per-visitor adaptive thresholds

Some visitors have stable embeddings; others vary strongly by pose, glasses,
lighting, or masks. Add threshold stats per visitor.

Migration:

```sql
ALTER TABLE visitors ADD COLUMN expected_match_similarity FLOAT;
ALTER TABLE visitors ADD COLUMN match_similarity_std FLOAT;
ALTER TABLE visitors ADD COLUMN personal_returning_threshold FLOAT;
ALTER TABLE visitors ADD COLUMN personal_new_threshold FLOAT;
```

Computation:

```python
pairwise = pairwise_cosine(visitor_gallery_embeddings)
mean = pairwise.mean()
std = pairwise.std()
personal_returning = clamp(mean - 2 * std, 0.40, 0.70)
personal_new = min(settings.NEW_VISITOR_MAX_SIMILARITY, personal_returning - 0.10)
```

### 6.6 Gallery quality gating

Current gallery growth is pose-aware, but it should also ask whether the new face
helps the gallery.

Add a face only when at least one is true:

- it covers a missing pose/yaw range
- it has better clarity than the worst face in its pose range
- it improves recall against recent tracklet samples
- it is a high-confidence returning match

Reject when:

- it is a near duplicate of an existing face
- it is low clarity
- it comes from a crowded/group frame and is not strongly matched

### 6.7 Masked-face re-embedding

When mask heuristic returns true:

1. Extract periocular crop.
2. Run recognition on the periocular crop.
3. Resolve both full-face and periocular embeddings.
4. Prefer the result with clearer decision margin.

Store `is_masked` and `mask_confidence` on `detection_events` later if useful.

### 6.8 Temporal match should update gallery

When the temporal gate converts a `NEW` into a returning visitor, the current
code treats it as returning for visit tracking, but does not update the gallery
or centroid. That misses the exact hard-angle sample the system needs to learn.

**Plan:** If temporal match confidence is high enough, call
`auto_enroller.update_after_match()` with `match_source="temporal"`.

---

## 7. Cross-Camera Identity Plan

### 7.1 Required principle

Cross-camera matching must be conservative. A duplicate visitor is annoying; a
false merge corrupts the gallery and analytics. Use auto-merge only for very
high confidence. Use review queue for medium confidence.

### 7.2 Phase A: camera-aware new-candidate check

Before creating a new visitor, search recent visitors from all cameras.

Inputs:

- candidate embedding
- candidate pose
- camera_id
- timestamp
- bbox
- body embedding if available

Candidate filters:

- seen within the last N minutes
- not impossible by camera topology
- not opted out or inactive
- similarity above a loose floor

Decision:

```text
face high + topology valid -> returning via cross_camera
face medium + body/color support -> review queue
face weak or impossible transition -> keep as new candidate/tracklet
```

### 7.3 Phase B: camera topology

Add a simple table or config:

```text
camera_a, camera_b, min_travel_seconds, max_expected_seconds
entrance, dining, 5, 120
dining, billing, 3, 90
entrance, kitchen, blocked
```

Use it to reject impossible matches and prioritize plausible transitions.

### 7.4 Phase C: background reconciliation job

Every few minutes:

1. Find visitors created recently.
2. Compare against visitors seen nearby in time.
3. Use face gallery, pose, body, color, and camera topology.
4. Queue `cross_camera_probable_duplicate`.
5. Auto-merge only above a strict threshold.

Recommended thresholds:

| Decision | Starting threshold |
|---|---:|
| Auto cross-camera returning | `>= 0.68` face similarity with clear margin |
| Queue review | `0.52 - 0.68` plus topology support |
| Auto merge duplicate visitors | `>= 0.72` after re-checking current galleries |

These values must be calibrated on local footage.

### 7.5 Phase D: same-session body and color fusion

For cross-camera handoff during the same visit:

```text
score =
  0.65 * face_similarity +
  0.20 * body_similarity +
  0.10 * clothing_color_similarity +
  0.05 * topology_score
```

Only use body/color within the same day/session. Do not use clothing to
recognize returning customers across days.

---

## 8. Best Performance Optimizations

### 8.1 Skip OSNet unless it is useful

Current camera and upload paths call `process_frame(..., extract_body=True)`.
If `ALLOW_BODY_FALLBACK=False` and cross-camera body fusion is disabled, body
embeddings are expensive and mostly unused.

Plan:

- Add `PIPELINE_CASCADE`.
- Use `process_frame_cascade()` when enabled.
- Or set `extract_body=False` unless body fallback/cross-camera body is enabled.

### 8.2 Add LRU to `FaceEmbeddingCache`

Use `OrderedDict` or `functools.lru_cache` style eviction.

Recommended settings:

```python
FACE_CACHE_MAX_ENTRIES = 10_000
FACE_CACHE_TTL_SECONDS = 3_600
```

Expose metrics:

- cache hits
- cache misses
- cache size
- eviction count

### 8.3 Stream video frames instead of loading all frames

`extract_video_frames()` returns a list. For longer uploads, this can hold many
frames in memory.

Plan:

- Convert frame extraction to an async/sync generator.
- Process each sampled frame immediately.
- Keep only dedup signature and tracklet state.

### 8.4 Vectorize YOLO post-processing

`ModelManager.detect_persons()` loops through boxes and transfers each box from
device to CPU individually. Convert all boxes/confidences in one transfer.

### 8.5 Batch small-face rescue

`cv_pipeline.refine_small_face()` can trigger several additional ArcFace passes
in crowded frames. Collect failed small faces and run rescue in one batched or
limited pass where possible.

### 8.6 Dynamic frame sizing

Current `MAX_FRAME_LONG_SIDE=1280` is global.

Plan:

- Near camera: lower long side to 960 or 720.
- Far camera / small faces: keep 1280 or increase detector size.
- Track per-camera face size distributions and recommend settings.

### 8.7 Bulk DB writes

Current path inserts/updates per processed detection.

Plan:

- Bulk insert `DetectionEvent` rows per frame/tracklet.
- Cache recently loaded `Visitor` rows for a short TTL.
- Batch review queue inserts.

---

## 9. Schema Changes

### 9.1 Accuracy schema

```sql
ALTER TABLE visitor_faces
  ADD COLUMN yaw FLOAT,
  ADD COLUMN pitch FLOAT,
  ADD COLUMN roll FLOAT,
  ADD COLUMN source_camera_id TEXT,
  ADD COLUMN source_detection_event_id UUID;

ALTER TABLE visitors
  ADD COLUMN expected_match_similarity FLOAT,
  ADD COLUMN match_similarity_std FLOAT,
  ADD COLUMN personal_returning_threshold FLOAT,
  ADD COLUMN personal_new_threshold FLOAT;
```

### 9.2 Tracklet and cross-camera schema

```sql
ALTER TABLE detection_events
  ADD COLUMN tracklet_id UUID,
  ADD COLUMN top_match_id UUID,
  ADD COLUMN top_match_similarity FLOAT,
  ADD COLUMN candidate_state VARCHAR(32),
  ADD COLUMN correlation_group_id UUID,
  ADD COLUMN is_cross_camera_match BOOLEAN DEFAULT FALSE;
```

Optional tables:

```sql
CREATE TABLE camera_topology (
  id UUID PRIMARY KEY,
  camera_a TEXT NOT NULL,
  camera_b TEXT NOT NULL,
  min_travel_seconds FLOAT,
  max_expected_seconds FLOAT,
  transition_enabled BOOLEAN DEFAULT TRUE
);

CREATE TABLE visitor_merge_audit (
  id UUID PRIMARY KEY,
  source_visitor_id UUID,
  target_visitor_id UUID,
  reason TEXT,
  similarity FLOAT,
  merged_by TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 10. Implementation Roadmap

### Phase 0: Baseline and safety checks

| Task | Files | Result |
|---|---|---|
| Add metrics for grey-zone-to-new rate | `identity_resolver.py`, `detection_events` | Know how often duplicates are being created |
| Add metrics for top match distribution of new visitors | `review_queue.py`, analytics | Tune thresholds from data |
| Build labelled test clips | docs/test data | Measure fragmentation and false merges |
| Add regression tests for resolver decisions | `tests/` | Prevent future threshold regressions |

### Phase 1: Immediate accuracy and stability fixes

| Task | Files | Impact |
|---|---|---|
| Use `REJECT_SIMILARITY` and stop grey-zone auto-new | `identity_resolver.py` | Largest duplicate reduction |
| Search top-K and compare best different visitor | `identity_resolver.py` | Better ambiguity handling |
| Add LRU/TTL to face cache | `ml_models.py`, `config.py` | Prevent memory growth |
| Wire periocular masked path | `detection_pipeline.py`, `mask_detector.py`, `ml_models.py` | Better masked matching |
| Activate cascade or skip body when unused | `camera_service.py`, `api/detect.py`, `cascade_pipeline.py`, `config.py` | Faster inference |
| Wire ambiguous/new duplicate review flags | `detection_pipeline.py`, `review_queue.py` | Better operator feedback |
| Remove duplicate face crop extraction | `detection_pipeline.py` | Small CPU cleanup |

### Phase 2: Code consolidation

| Task | Files | Result |
|---|---|---|
| Add `utils/geometry.py` | new + bbox call sites | One source for bbox math |
| Add `utils/similarity.py` | new + resolver/enroller/temporal | Consistent embedding math |
| Add `FrameDedupBuffer` | `camera_service.py`, `api/detect.py` | Shared de-dup behavior |
| Consolidate visitor image saving | `auto_enroller.py` | Less duplicate IO code |
| Extract visit tracker core | `visit_tracker.py`, `redis_visit_tracker.py` | One visit state machine |
| Share camera frame processing | `camera_service.py` | Less parallel/sequential drift |

### Phase 3: Multi-angle identity

| Task | Files | Result |
|---|---|---|
| Persist yaw/pitch/roll/source camera | `models.py`, Alembic, `auto_enroller.py` | Fine-grained pose history |
| Pose-angle-aware search ordering | `identity_resolver.py` | Better side-angle recall |
| Per-visitor adaptive thresholds | `models.py`, `auto_enroller.py`, `identity_resolver.py` | Less one-size-fits-all matching |
| Tracklet buffer before new enrollment | new `services/tracklet.py`, `detection_pipeline.py` | Prevent one-frame duplicate IDs |
| Gallery quality gating | `auto_enroller.py`, `face_quality.py` | Better gallery, fewer bad seeds |
| Temporal match updates gallery | `detection_pipeline.py` | Learn from hard angle recoveries |

### Phase 4: Cross-camera identity

| Task | Files | Result |
|---|---|---|
| Add camera topology config/table | `models.py`, migration, admin API | Transition constraints |
| Add cross-camera candidate resolver | new `services/cross_camera_resolver.py` | Check recent visitors before creating new |
| Add reconciliation job | new `services/cross_camera_dedup.py` | Merge/flag duplicates after the fact |
| Add source camera to gallery search filters | `identity_resolver.py` | Camera-aware ranking |
| Add body/color same-session fusion | new utility/service | Better handoff when face is weak |
| Add merge audit and gallery trim after merge | `visitor_merge.py`, migration | Safer merge workflow |

### Phase 5: Advanced model and runtime optimization

| Task | Result |
|---|---|
| Model adapter for face recognition backbones | Evaluate AdaFace/MagFace without rewriting pipeline |
| TensorRT/OpenVINO model paths | Faster YOLO/ArcFace/OSNet |
| ByteTrack/DeepSORT integration | Stable track IDs before identity resolution |
| 3D face normalization/frontalization trial | Better extreme-angle matching |
| Self-supervised gallery cleaning | Detect polluted visitor records and stale faces |

---

## 11. Validation Metrics

Track these before and after each phase.

| Metric | How to compute | Target |
|---|---|---|
| Duplicate visitor rate | probable duplicate flags / new visitors | Down |
| Grey-zone-to-new rate | grey-zone detections promoted to new | Near zero after Phase 1 |
| Fragmentation rate | labelled person assigned multiple visitor IDs | Down 50%+ |
| False merge rate | different labelled people assigned same visitor ID | Must stay near zero |
| Cross-camera recall | same person matched across cameras | Up |
| Ambiguous decision rate | ambiguous / total resolved faces | Down only if false merges stay low |
| p95 frame latency | capture to processed result | Stable or down |
| Memory RSS over 24h | process memory | Flat after cache LRU |
| Gallery quality | clarity score distribution and faces per visitor | Improve without exceeding cap |
| Review queue precision | reviewed flags that are true duplicates | Up |

### Minimum labelled test set

Build a small local evaluation set:

- 10 to 20 people.
- Each person across frontal, left, right, down, masked/unmasked where possible.
- At least two cameras or two camera angles.
- Several crowded frames.
- Ground-truth mapping from frame/track to person.

This is enough to tune thresholds safely before changing production behavior.

---

## 12. Recommended Configuration Direction

Add these settings:

```python
PIPELINE_CASCADE: bool = True
FACE_CACHE_MAX_ENTRIES: int = 10_000
FACE_CACHE_TTL_SECONDS: int = 3_600

IDENTITY_TOP_K: int = 10
GREY_ZONE_POLICY: str = "tracklet"  # "drop" | "review" | "tracklet"
TRACKLET_ENABLED: bool = True
TRACKLET_WINDOW_SECONDS: float = 2.0
TRACKLET_MIN_OBSERVATIONS_NEW: int = 2

POSE_CONTINUOUS_SEARCH: bool = True
ADAPTIVE_VISITOR_THRESHOLDS: bool = True

CROSS_CAMERA_ENABLED: bool = False
CROSS_CAMERA_LOOKBACK_SECONDS: float = 180.0
CROSS_CAMERA_REVIEW_THRESHOLD: float = 0.52
CROSS_CAMERA_AUTO_THRESHOLD: float = 0.68
CROSS_CAMERA_AUTO_MERGE_THRESHOLD: float = 0.72
CROSS_CAMERA_BODY_COLOR_ENABLED: bool = False
```

Default `CROSS_CAMERA_ENABLED` to `False` until topology and review workflow are
ready. Accuracy features that reduce new enrollment, like grey-zone holding, can
be enabled earlier.

---

## 13. File Change Map

| File | Expected changes |
|---|---|
| `backend/app/services/identity_resolver.py` | top-K search, visitor-level ambiguity, grey-zone policy, adaptive thresholds, pose-angle ranking |
| `backend/app/services/detection_pipeline.py` | tracklet gate, periocular path, review flags, temporal gallery update |
| `backend/app/services/auto_enroller.py` | persist pose values/camera, gallery quality gating, visitor threshold stats |
| `backend/app/cv_pipeline.py` | pass continuous pose through consistently, batch small-face rescue later |
| `backend/app/ml_models.py` | LRU face cache, vectorized YOLO post-processing, model adapter later |
| `backend/app/services/camera_service.py` | cascade wiring, body-skip policy, shared frame processor |
| `backend/app/api/detect.py` | streaming frame processing, cascade/body-skip parity with camera path |
| `backend/app/models.py` | pose/camera columns, visitor threshold stats, tracklet/correlation fields |
| `backend/app/services/temporal_consistency.py` | camera-aware entries, optional shared Redis state |
| `backend/app/services/visitor_merge.py` | merge audit, gallery trim, temporal gate cleanup |
| `backend/app/services/review_queue.py` | cross-camera duplicate flags, better ambiguity/new-candidate metadata |
| New `backend/app/utils/geometry.py` | bbox, crop, IoU, offset helpers |
| New `backend/app/utils/similarity.py` | normalization and cosine helpers |
| New `backend/app/services/tracklet.py` | short-lived tracklet buffer |
| New `backend/app/services/cross_camera_resolver.py` | real-time cross-camera candidate matching |
| New `backend/app/services/cross_camera_dedup.py` | background duplicate reconciliation |

---

## 14. Execution Order Recommendation

Do this order for the best risk/reward:

1. Stop grey-zone auto-new and add resolver tests.
2. Add top-K visitor-level ambiguity.
3. Add LRU cache and skip unused body embeddings.
4. Wire periocular masked matching.
5. Add tracklet buffer for new enrollment.
6. Persist yaw/pitch/roll and source camera.
7. Add per-visitor adaptive thresholds.
8. Add cross-camera candidate resolver behind a disabled-by-default flag.
9. Add background reconciliation and review queue workflow.
10. Evaluate model upgrades only after the decision pipeline is stable.

This keeps the system conservative: fewer duplicate visitors without increasing
the chance of merging two different people.

---

## 15. Open Questions

1. How many cameras are expected in production, and where are they physically
   located?
2. Is a 1 to 2 second delay acceptable before creating a brand-new visitor?
3. Should cross-camera matches be auto-applied, review-only, or mixed by
   confidence?
4. Is there labelled footage available for threshold calibration?
5. Is production CPU-only or GPU-backed?
6. Should body/color re-ID be enabled only during active visits?
7. What is the acceptable memory budget per camera stream?

---

*End of v3 plan.*
