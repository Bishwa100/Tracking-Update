# Restaurant Visitor Tracker — Optimization & Improvement Plan

> Comprehensive optimization plan addressing duplicate detection, cross-camera identity resolution, accuracy improvements, and performance enhancements.
>
> **Last Updated:** 2026-06-18

---

## 1. Executive Summary

This document outlines a detailed plan to address the critical issues identified in the current implementation:

1. **Cross-camera duplicate detection** — Same person appearing on multiple cameras is incorrectly registered as new visitors
2. **Identity resolution accuracy** — Improve matching precision across different poses, lighting conditions, and time
3. **Performance optimization** — Reduce computational overhead while maintaining accuracy
4. **Code deduplication** — Identify and consolidate repeated logic across the codebase

---

## 2. Issues Analysis

### 2.1 Cross-Camera Identity Problem

**Current State:**
- Each camera operates independently in the detection pipeline
- No mechanism to correlate detections across different cameras
- Same person appearing on Camera A and Camera B creates two separate visitor records

**Root Cause:**
- `detection_pipeline.py` processes each camera stream in isolation
- `identity_resolver.py` searches only within the `visitor_faces` table without cross-camera context
- No camera-aware similarity scoring that accounts for simultaneous detections

**Impact:**
- One physical person → multiple visitor records
- Incorrect analytics (inflated visitor counts)
- Poor customer experience (treated as new visitor each time)

### 2.2 Duplicate Code Patterns

**Identified Duplications:**

| Location | Duplication Type |
|----------|------------------|
| `identity_resolver.py` & `auto_enroller.py` | Face embedding similarity computation (cosine similarity) |
| `cv_pipeline.py` & `cascade_pipeline.py` | Frame preprocessing (CLAHE, auto-gamma) |
| `visit_tracker.py` & `redis_visit_tracker.py` | Visit state machine logic |
| Multiple services | dHash computation for face caching |
| `auto_tuning.py` & `monitoring.py` | Background loop patterns |

### 2.3 Accuracy Issues

- Pose-aware search exists but may not handle extreme angles well
- Mask detection uses simple heuristic (could be improved)
- Small face rescue is limited to single upscaling pass
- No cross-frame temporal tracking beyond 30-second window for same person

---

## 3. Optimization Plan

### 3.1 Cross-Camera Identity Resolution (Priority: HIGH)

#### 3.1.1 Add Camera-Aware Correlation Layer

```
Proposed Architecture Change:

┌─────────────────────────────────────────────────────────────────┐
│  CAMERA CORRELATION SERVICE (new)                              │
│  ─────────────────────────────────────────────────────────── │
│  1. Receive detections from ALL active cameras                 │
│  2. Within a time window (e.g., 500ms), find overlapping       │
│     person detections across cameras                           │
│  3. Compare embeddings to determine if same person            │
│  4. If match found, attribute to existing visitor              │
│  5. If no match, allow individual camera to create new         │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation Details:**
- New service: `cross_camera_resolver.py`
- Time window: configurable (default 500ms)
- Spatial correlation: check if bounding boxes could be the same person based on position in restaurant layout
- Decision logic:
  ```
  IF camera_A.detection AND camera_B.detection WITHIN 500ms:
      IF similarity(emb_A, emb_B) >= CROSS_CAMERA_THRESHOLD (0.70):
          MERGE as same person
      ELSE:
          INDEPENDENT detections
  ```

#### 3.1.2 Database Schema Enhancement

```sql
-- Add camera correlation tracking
ALTER TABLE detection_events ADD COLUMN correlated_group_id UUID;
ALTER TABLE detection_events ADD COLUMN is_cross_camera_match BOOLEAN DEFAULT FALSE;

-- Index for fast correlation lookups
CREATE INDEX idx_correlated_group ON detection_events(correlated_group_id);
```

#### 3.1.3 New Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CROSS_CAMERA_ENABLED` | True | Enable cross-camera deduplication |
| `CROSS_CAMERA_TIME_WINDOW_MS` | 500 | Time window for correlation |
| `CROSS_CAMERA_SIMILARITY_THRESHOLD` | 0.70 | Min similarity to merge |
| `CROSS_CAMERA_SPATIAL_WEIGHT` | 0.3 | Weight for spatial proximity |

### 3.2 Identity Resolution Improvements (Priority: HIGH)

#### 3.2.1 Enhanced Pose-Aware Search

Current: Basic pose bin matching (frontal, left, right, down)
Proposed: Multi-scale pose embedding

```python
# New: Pose-aware embedding augmentation
def augment_embedding_with_pose(embedding, pose_bin, det_score):
    """
    Modulate embedding based on pose to improve cross-pose matching.
    - Frontal faces: use as-is
    - Profile faces: apply learned transformation toward frontal space
    - Detection confidence modulates the transformation strength
    """
    # Implementation: apply pose-specific linear transformation
    # that projects profile embeddings closer to frontal space
    pass
```

#### 3.2.2 Multi-Frame Ensemble Matching

Instead of single-frame matching, aggregate embeddings across multiple frames:

```python
async def resolve_with_temporal_ensemble(
    faces: List[DetectedFace],
    db: AsyncSession,
    window_frames: int = 5,
) -> ResolutionResult:
    """
    Combine embeddings from recent frames for more robust matching.
    1. Collect embeddings from last N frames for same person track
    2. Compute mean embedding (weighted by det_score)
    3. Run HNSW search on ensemble embedding
    4. Fall back to per-frame if ensemble is ambiguous
    """
```

#### 3.2.3 Adaptive Threshold by Camera

Different cameras may have different lighting, angles, or quality:
```python
# Per-camera adaptive thresholds stored in runtime_settings
CAMERA_ADAPTIVE_THRESHOLDS = {
    "camera_1": {"returning": 0.52, "new_max": 0.42},
    "camera_2": {"returning": 0.58, "new_max": 0.48},
}
```

### 3.3 Performance Optimizations (Priority: MEDIUM)

#### 3.3.1 Embedding Cache Improvements

**Current:** dHash based cache (exact match)
**Proposed:** Add approximate cache with LSH (Locality-Sensitive Hashing)

```python
class LSHFaceEmbeddingCache:
    """
    LSH-based approximate embedding cache.
    - Uses random projections for fast approximate lookups
    - Reduces ArcFace inference by ~60% for cached content
    """
    def __init__(self, num_tables=8, projection_dim=16):
        self.tables = [np.random.randn(512, projection_dim) for _ in range(num_tables)]
    
    def _get_bucket(self, embedding):
        buckets = []
        for table in self.tables:
            bucket = tuple((np.dot(embedding, table.T) > 0).astype(int))
            buckets.append(bucket)
        return tuple(buckets)
```

#### 3.3.2 Batch Processing Optimization

Current: Batched DB queries per frame
Proposed: Multi-frame aggregation for even better batch efficiency

```python
# Aggregate embeddings across multiple frames before DB query
async def resolve_batch_optimized(
    all_faces: List[dict],  # Faces from multiple frames
    db: AsyncSession,
    aggregate_window: int = 10,  # Frames to aggregate
) -> List[ResolutionResult]:
    """
    1. Group faces by person track ID
    2. For each track, compute weighted average embedding
    3. Single DB query for all unique tracks
    4. Distribute results back to original detections
    """
```

#### 3.3.3 Model Optimization

- **YOLOv8n → YOLOv8s**: Slightly heavier but more accurate for small faces
- **ArcFace → CanvasFace**: Newer model with better cross-pose performance
- **OSNet → OSNet x0.5**: Better body re-ID with moderate performance cost

### 3.4 Code Deduplication (Priority: MEDIUM)

#### 3.4.1 Create Shared Utility Module

```python
# New: app/utils/similarity.py
def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b))

def batch_cosine_similarity(embeddings: List[List[float]], query: List[float]) -> List[float]:
    """Compute cosine similarity of multiple embeddings against a query."""
    # Vectorized computation for efficiency
    pass

def dhash_embedding(face_crop: np.ndarray, size: int = 8) -> int:
    """Compute difference hash of a face crop."""
    # Move to shared location
    pass
```

#### 3.4.2 Consolidate Background Loops

Create base class for background services:
```python
# New: app/services/base_background_service.py
class BaseBackgroundService:
    """Base class for background services with common patterns."""
    
    async def start(self):
        """Start the background loop."""
        
    async def stop(self):
        """Gracefully stop the service."""
        
    async def _run_loop(self):
        """Override this to implement specific logic."""
        
    async def health_check(self) -> bool:
        """Override for custom health checks."""
```

### 3.5 Accuracy Improvements (Priority: HIGH)

#### 3.5.1 Enhanced Mask Detection

Current: Simple std-dev heuristic
Proposed: Lightweight CNN for mask detection

```python
class MaskDetectorCNN:
    """
    Lightweight mask detector (~100KB).
    - Runs on CPU in <5ms
    - Trained on masked/unmasked face crops
    - Returns: is_masked, mask_confidence
    """
    def __init__(self):
        # Load lightweight MobileNetV2-based model
        pass
```

#### 3.5.2 Small Face Multi-Scale Rescue

Current: Single upscaling pass
Proposed: Multi-scale pyramid for very small faces

```python
def refine_small_face_multi_scale(face_crop, max_scales=3):
    """
    Try multiple scales for small face rescue.
    1. Current scale (already implemented)
    2. 2x upscale
    3. 3x upscale with ensemble
    Returns best embedding across all scales.
    """
```

#### 3.5.3 Quality-Aware Threshold Adjustment

```python
def dynamic_threshold(
    det_score: float,
    pose_bin: str,
    lighting_condition: str,  # bright/normal/dark
    camera_id: str,
) -> tuple[float, float]:
    """
    Adjust thresholds based on detection quality.
    - High det_score + frontal + good lighting: stricter thresholds
    - Low det_score + profile + poor lighting: relaxed thresholds
    """
    base_returning = settings.RETURNING_FACE_THRESHOLD
    base_new = settings.NEW_VISITOR_MAX_SIMILARITY
    
    # Modulation factors
    score_factor = min(det_score * 0.2, 0.1)  # Up to ±0.1 based on score
    pose_factor = 0.05 if pose_bin != "frontal" else 0.0
    lighting_factor = {"dark": 0.08, "normal": 0.0, "bright": -0.03}.get(lighting_condition, 0.0)
    
    returning = base_returning - score_factor + pose_factor + lighting_factor
    new = base_new - score_factor + pose_factor + lighting_factor
    
    return returning, new
```

---

## 4. Implementation Roadmap

### Phase 1: Core Cross-Camera Resolution (Week 1-2)

| Task | Effort | Owner |
|------|--------|-------|
| Create `cross_camera_resolver.py` service | High | TBD |
| Add database schema for correlation | Medium | TBD |
| Implement time-window correlation logic | High | TBD |
| Add configuration parameters | Low | TBD |
| Unit tests for cross-camera matching | Medium | TBD |

### Phase 2: Identity Resolution Enhancements (Week 2-3)

| Task | Effort | Owner |
|------|--------|-------|
| Implement pose-aware embedding augmentation | Medium | TBD |
| Add multi-frame ensemble matching | High | TBD |
| Create per-camera adaptive thresholds | Medium | TBD |
| Integrate with existing resolver | Medium | TBD |

### Phase 3: Performance & Code Quality (Week 3-4)

| Task | Effort | Owner |
|------|--------|-------|
| Implement LSH embedding cache | Medium | TBD |
| Consolidate similarity utilities | Low | TBD |
| Create base background service class | Medium | TBD |
| Optimize batch processing | Medium | TBD |

### Phase 4: Accuracy Improvements (Week 4-5)

| Task | Effort | Owner |
|------|--------|-------|
| Train/p集成 lightweight mask detector | High | TBD |
| Implement multi-scale small face rescue | Medium | TBD |
| Add dynamic threshold adjustment | Medium | TBD |
| A/B testing framework for thresholds | High | TBD |

---

## 5. Success Metrics

### 5.1 Cross-Camera Deduplication
- [ ] < 5% duplicate visitor rate across cameras
- [ ] Verify with synthetic test data (same person on 2+ cameras)
- [ ] Monitor `probable_duplicate` queue for reduction

### 5.2 Identity Resolution Accuracy
- [ ] False new rate < 2% (currently ~5%)
- [ ] False merge rate < 0.5%
- [ ] Auto-merge success rate > 90%

### 5.3 Performance
- [ ] Frame processing latency p95 < 500ms
- [ ] ArcFace inference reduced by 50%+ via caching
- [ ] Memory usage stable under continuous operation

### 5.4 Code Quality
- [ ] Eliminate identified duplicate patterns
- [ ] All shared utilities in `app/utils/`
- [ ] > 80% test coverage on core services

---

## 6. Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Cross-camera false merges | Medium | High | Require high similarity (0.70+) + human review queue |
| Performance regression | Low | Medium | Benchmark before/after, rollback capability |
| Database schema changes | Medium | Medium | Alembic migrations, backward compatible |
| New model integration issues | Medium | Medium | Keep fallback to current models |

---

## 7. Appendix: Configuration Reference

### New Settings to Add

```python
# app/config.py additions
class Settings(BaseSettings):
    # Cross-camera settings
    CROSS_CAMERA_ENABLED: bool = True
    CROSS_CAMERA_TIME_WINDOW_MS: int = 500
    CROSS_CAMERA_SIMILARITY_THRESHOLD: float = 0.70
    CROSS_CAMERA_SPATIAL_WEIGHT: float = 0.3
    
    # Enhanced resolution
    ENSEMBLE_WINDOW_FRAMES: int = 5
    LSH_CACHE_TABLES: int = 8
    LSH_CACHE_PROJECTION_DIM: int = 16
    
    # Accuracy improvements
    MASK_DETECTOR_ENABLED: bool = True
    MULTI_SCALE_FACE_RESCUE: bool = True
    DYNAMIC_THRESHOLDS: bool = True
```

---

*Document Version: 1.0*
*Next Review: 2026-06-25*