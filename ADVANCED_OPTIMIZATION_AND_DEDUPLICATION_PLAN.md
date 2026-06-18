# Advanced Optimization & Deduplication Plan (Multi-Angle Focus)

**Date:** 2026-06-18
**Objective:** Solve the "same person across multiple camera angles detected as new person" issue, optimize the pipeline, and enhance the deduplication logic.

---

## 1. The Core Problem: Multi-Angle Duplication

### Why is this happening?
The current identity resolution relies heavily on ArcFace embeddings and spatial-temporal gates:
1. **Pose Discrepancy:** ArcFace embeddings for a direct frontal face and an extreme profile (side) face of the same person can differ significantly, dropping their cosine similarity below the `RETURNING_FACE_THRESHOLD` (0.55).
2. **Cross-Camera Spatial Failure:** The existing `temporal_consistency.py` gate uses pixel distance (`< 150px`) to catch same-person fragmentation. This fails across multiple cameras because pixel coordinates do not align across different video streams.
3. **Underutilized Body Features:** OSNet body re-ID embeddings are extracted but currently disabled by default or restricted strictly to grey-zone fallback in single-camera sessions.

---

## 2. Plan: Resolving Multi-Angle & Cross-Camera Duplicates

### A. Cross-Camera Body Re-ID Fusion (Session Linking)
Instead of relying purely on faces, we will utilize **Body Re-ID (OSNet)** as the primary cross-camera bridge during a live session.
- **How it works:** If Camera A detects a "Frontal" face and Camera B detects a "Profile" face simultaneously (or within a short time window), they might appear as two different faces. However, their clothing/body embedding will be nearly identical (> 0.85 similarity).
- **Action:** If two active visits have highly similar body embeddings and overlapping time windows, instantly link them as the same visitor and merge their face galleries.

### B. Tracklet-Based Identity Resolution
Currently, resolution runs on a per-frame basis. We will introduce **Tracklets** (short sequences of detections of the same physical body in one camera view).
- **How it works:** Track a person using a fast tracker (like ByteTrack/DeepSORT) for a few seconds. Collect multiple face angles as they walk.
- **Action:** Send the *best* face from the tracklet (or a pool of faces) to the DB for identity resolution all at once. This avoids premature "NEW VISITOR" registration on the first blurry/profile frame.

### C. Dynamic Pose-Adaptive Thresholds
- **How it works:** If the pipeline detects a `profile` face (e.g., looking sharply left), but the top database match only has `frontal` faces in their gallery, we mathematically expect a lower similarity score.
- **Action:** Dynamically lower the `RETURNING_FACE_THRESHOLD` (e.g., from 0.55 to 0.48) *only* when comparing mismatched pose bins, provided the body embedding similarity is very high.

### D. Global (Topology-Aware) Temporal Gate
- **How it works:** Replace the naive `150px` spatial check with a cross-camera logical gate.
- **Action:** If a visitor disappears from Camera 1 and a "new" visitor appears in Camera 2 a few seconds later with a matching body embedding, the system will assume they are the same person transitioning between zones, overriding the face threshold.

---

## 3. Plan: System & Performance Optimizations

### A. Offline / Nightly Global Deduplication
Real-time constraints mean some duplicates will always slip through.
- **Action:** Implement a nightly background job running a clustering algorithm (e.g., DBSCAN) over all visitors created that day. It will use a weighted combination of Face Similarity (0.6) + Body Similarity (0.4) + Time Overlap to propose high-confidence automatic merges.

### B. Cross-Stream Batching
Currently, inference is parallelized per camera stream. 
- **Action:** Introduce a global batching queue. If 4 cameras submit frames at the same time, YOLO and OSNet should batch the crops into a single tensor (e.g., batch size 16) to maximize GPU utilization and throughput.

### C. Upgrade Feature Extractors (Future-Proofing)
- Evaluate upgrading from OSNet to a more modern Re-ID model (e.g., **StrongSORT / BoT-SORT**) which handles occlusion and varied lighting better.
- Evaluate switching from standard ArcFace to a **pose-invariant face recognition model** (e.g., AdaFace) which is specifically trained to handle extreme low-quality and high-angle faces.

---

## 4. Implementation Steps

1. **Phase 1: Body Re-ID Activation & Tuning**
   - Enable `ALLOW_BODY_FALLBACK` by default.
   - Implement the Cross-Camera Session Linker to fuse active visits using body similarity.
2. **Phase 2: Pose-Adaptive Thresholding**
   - Modify `identity_resolver.py` to accept dynamic thresholds based on the requested `pose_bin` vs. the available `pose_bin` in the gallery.
3. **Phase 3: Nightly Auto-Dedup Sweep**
   - Write a standalone script/cron job to cluster and merge duplicate visitors based on combined face/body scores.
4. **Phase 4: Tracklet Tracking**
   - Integrate ByteTrack into `cv_pipeline.py` to buffer detections and only attempt recognition on the optimal frame within a tracklet.
