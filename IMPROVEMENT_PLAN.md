# Accuracy & Performance Improvement Plan

**Date:** 2026-06-23
**Scope:** Targeted fixes for recognition accuracy and pipeline throughput.

This plan complements [ADVANCED_OPTIMIZATION_AND_DEDUPLICATION_PLAN.md](ADVANCED_OPTIMIZATION_AND_DEDUPLICATION_PLAN.md).
Most of that earlier plan (tracklets, pose-adaptive thresholds, parallel pipeline,
frame dedup, embedding cache, cross-camera scaffolding) is **already shipped**, so
the remaining wins are surgical rather than structural.

---

## Part 1 — Implemented in this change (safe, config-gated, defaults preserve behaviour)

### 1.1 HNSW `ef_search` tuning  *(accuracy — highest impact)*
**Problem.** The gallery HNSW indexes are built with `m=16, ef_construction=64`
(`alembic/versions/001_restaurant_schema.py`), but the resolver query never set
`hnsw.ef_search`, leaving pgvector's default of **40**. The pose-aware search does
`LIMIT :top_k` (default 10) *inside* a `CROSS JOIN LATERAL` that also filters on
`is_active`, `consent_status != 'opted_out'`, and `pose_bin`. When those filters
reject candidates, an `ef_search` of 40 can run out of graph candidates before it
fills `top_k` true matches → a returning visitor scores below threshold → they are
**re-registered as a new visitor** (the system's #1 duplication failure mode).

**Fix.** `SET LOCAL hnsw.ef_search = HNSW_EF_SEARCH` (default **100**) at the start
of each resolve transaction in `identity_resolver._search_faces_batch`. `SET LOCAL`
scopes it to the transaction; it does not leak to other queries on the pooled
connection. Cost at this gallery size is negligible; recall gain on filtered
searches is the main win.

- Knob: `HNSW_EF_SEARCH` (int, default 100; `0` = leave server default).
- Clamped to `>= IDENTITY_TOP_K` automatically.

### 1.2 Per-person ArcFace fallback is now gated  *(speed)*
**Problem.** In `cv_pipeline.process_frame`, every person box that the single
full-frame ArcFace pass did **not** assign a face to triggers a *second* full
`extract_face_data` call on that person's crop. In a crowded frame that is `N`
extra ArcFace forward passes — the dominant per-frame cost when many people are
present.

**Fix.** Gated behind `PER_PERSON_FACE_FALLBACK` (default **True**, so behaviour is
unchanged). Deployments with good full-frame face detection (close/medium range)
can set it `False` to cut crowd-frame latency materially. The full-frame
small-face rescue (`refine_small_face`) is unaffected.

### 1.3 Empty-frame early-out  *(speed, minor)*
`process_frame` now returns immediately when YOLO finds **no** persons *and* the
ArcFace pass finds **no** faces, skipping the per-person and body-queue loops.

### 1.4 Documented a non-fix (avoid dead complexity)
A NumPy "exact re-rank" of HNSW results was considered and **rejected**: pgvector's
`1 - (embedding <=> query)` is already the *exact* cosine for our L2-normalized
embeddings, and the outer `ORDER BY similarity DESC` already sorts returned rows
exactly. HNSW is approximate only in *which* rows it returns — addressed by 1.1
(`ef_search`), not by re-scoring rows already returned. Noted in `config.py`.

---

## Part 2 — Recommended next (larger; not in this change)

### 2.1 Consolidate the per-match DB round-trips  *(speed)*
`auto_enroller.update_after_match` issues several sequential `SELECT`s per
confident match: `_is_diverse_embedding` (gallery fetch), `add_face_to_gallery`
(another gallery fetch), and `recompute_adaptive_thresholds` (a third). On a busy
stream that is 3+ round-trips per returning visitor per frame.
- **Action:** fetch the visitor's gallery (`embedding, det_score, pose_bin, body_embedding`)
  **once** per match and pass it down to diversity check + eviction + threshold
  recompute. ~2–3× fewer queries on the hottest write path.
- **Risk:** low; pure refactor with the same SQL semantics.

### 2.2 Move the masked-face periocular pass out of the per-detection loop  *(speed)*
`detection_pipeline.process_detections` runs mask detection and a *second* ArcFace
embed (periocular region) per masked face, then a **second** `resolve_batch`. Batch
the periocular embeds across all masked faces in the frame (already partly done)
and ensure mask detection itself is vectorized/short-circuited when
`MASK_DETECTION_ENABLED` is off.

### 2.3 Persist `ef_search` at the index/role level too  *(robustness)*
Belt-and-suspenders for 1.1: `ALTER DATABASE ... SET hnsw.ef_search = 100` (or set
per-role) so any future query path that forgets the `SET LOCAL` still benefits.
Keep the `SET LOCAL` as the authoritative per-tx value.

### 2.4 Re-rank candidates by a *fused* score, not face-only  *(accuracy)*
When body embeddings are available and the face is grey-zone, compute a fused
`w_f * face_sim + w_b * body_sim` for **same-session** candidates before the
ambiguity gate (not across visits — body is clothing-dependent, already documented).
This is a more principled version of the current binary body fallback.

### 2.5 Detector input-size auto-selection  *(speed/accuracy trade)*
`INSIGHTFACE_DET_SIZE=640` maximises small-face recall but is the slowest setting.
For close-range single-camera footage, 480 or 320 is 1.5–4× faster at almost no
recall loss. Consider auto-picking based on median detected face size over a warmup
window, exposed as a runtime setting.

### 2.6 Model upgrades (evaluate, don't rush)
- **AdaFace** in place of ArcFace `buffalo_l` for pose/low-quality robustness
  (directly attacks the profile-vs-frontal similarity drop).
- **BoT-SORT / StrongSORT** body re-ID for occlusion robustness vs OSNet x0.25.
  Both are drop-in at the embedding interface; gate behind config and A/B on
  `detection_events` before switching defaults.

---

## Part 3 — How to measure (before/after)

See the "Benchmarking" section below / the assistant's message. Key signals to
watch in `detection_events` and `/api/analytics/*`:
- **new vs returning ratio** — 1.1 should *lower* the new-visitor rate (fewer
  duplicates) without raising false merges.
- **`grey_zone` / `ambiguous` / `pose_hold` event counts** — 1.1 should pull some
  grey-zone holds up into confident `face` matches.
- **per-frame `process_frame` timing logs** (DEBUG) — 1.2/1.3 should drop arcface
  time on crowded frames when `PER_PERSON_FACE_FALLBACK=False`.
- **duplicate count** — number of visitors the nightly dedup sweep proposes to
  merge should fall.
