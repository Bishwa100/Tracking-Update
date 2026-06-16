
# Restaurant Visitor Detection System — Real-World Failure Analysis & Solutions

## Methodology
Based on the uploaded implementation plan, I analyzed the entire pipeline from camera capture → detection → identity resolution → visit tracking → dashboard display. Each stage has specific failure modes under real-world conditions.

---

## CATEGORY 1: Face Recognition & Identity Resolution Failures

### 1.1 Same Person, Different Angle → Detected as New Person
**Severity: CRITICAL | Frequency: HIGH**

**Root Causes:**
- ArcFace is trained on frontal faces; profile/extreme angles produce embeddings that fall into the grey zone (0.35-0.55 similarity)
- With `NEW_VISITOR_MAX_SIMILARITY=0.45` and `RETURNING_FACE_THRESHOLD=0.55`, a side-angle face (similarity ~0.40) triggers auto-registration as "new visitor"
- The multi-embedding gallery stores only 10 faces; if those 10 are all frontal, any profile view is effectively unknown
- No 3D face modeling or pose-aware embedding normalization

**Evidence from Plan:**
- "ArcFace full-frame pass → all faces at once" — no mention of pose estimation
- Gallery stores "top-10 by quality" but quality is based on `det_score` (detection confidence), not pose diversity
- Centroid update uses weighted average — if profile embeddings are rare, centroid stays frontal-biased

**Real-World Scenario:**
A regular customer enters facing the camera (frontal, recognized). Sits down, turns to talk to friend (profile view). System sees profile, similarity to centroid = 0.42 → registers as NEW visitor #47. Customer turns back to camera (frontal, recognized as #12). Now one person has two records.

**Solutions (in order of effectiveness):**
1. **Pose-Aware Gallery Management** — Store faces grouped by yaw angle bins (frontal: -15° to +15°, left profile: -90° to -45°, right profile: +45° to +90°). Query only the matching bin. Requires adding pose estimation (e.g., 68-point landmarks or a lightweight pose estimator).
2. **3D Face Reconstruction + Frontalization** — Use a lightweight 3DMM (3D Morphable Model) or PRNet to frontalize profile faces before embedding extraction. Adds ~50-100ms but eliminates angle sensitivity.
3. **Dynamic Threshold by Pose** — Lower `RETURNING_FACE_THRESHOLD` for non-frontal poses (e.g., 0.45 for profile vs 0.55 for frontal). Requires pose estimation.
4. **Temporal Consistency Gate** — If a "new" visitor appears within 30 seconds and 2 meters (pixel distance) of a known visitor who just disappeared, force-merge or flag for review. Uses cross-frame tracking.
5. **Gallery Diversity Enforcement** — When adding to gallery, ensure at least 3 different pose bins are represented. If a new pose is detected and gallery is full, evict the most redundant pose (not just lowest quality).

---

### 1.2 Same Person, Different Lighting → Detected as New Person
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- ArcFace embeddings are somewhat robust to lighting but extreme cases (backlit silhouette, harsh overhead restaurant lighting, sunlight through windows) shift embeddings into grey zone
- No illumination normalization in preprocessing
- Restaurant environments have highly variable lighting (day/night, warm/cool LEDs, window glare)

**Real-World Scenario:**
Daytime visit: customer sits near window, well-lit, recognized. Evening visit: same table, now backlit by window, face in shadow. Embedding similarity = 0.38 → new visitor.

**Solutions:**
1. **Histogram Equalization + Gamma Correction** — Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to face crops before ArcFace. Adds ~2ms, significantly improves low-light recognition.
2. **Illumination-Invariant Embedding Space** — Fine-tune ArcFace (or add a projection layer) on restaurant-specific lighting variations. Requires collecting a small dataset (~500 pairs across lighting conditions).
3. **Multi-Exposure Fusion** — Capture 3 frames at different exposures (if camera supports it) or use temporal averaging over 3-5 frames to reduce lighting variance.
4. **Adaptive Centroid with Lighting Context** — Store lighting metadata (brightness histogram) with each gallery face. During matching, weight gallery faces by lighting similarity to current frame.

---

### 1.3 Same Person, Facial Expression Changes → Detected as New Person
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- Smiling vs neutral vs talking expressions change facial geometry
- ArcFace is somewhat robust but extreme expressions (laughing, frowning) can shift embeddings by 0.05-0.10
- Restaurant context: people eating, talking, laughing — expressions change constantly

**Solutions:**
1. **Expression-Normalized Embeddings** — Use a model trained with expression augmentation (e.g., MagFace or a custom ArcFace with expression-mixup training).
2. **Temporal Aggregation** — For visit recognition, collect 5-10 face embeddings over the first 30 seconds and use the median embedding (more robust than single-frame).
3. **Expression-Aware Gallery** — Store at least 2 expressions per pose bin: neutral and smiling.

---

### 1.4 Same Person, Aging / Appearance Changes → Not Recognized
**Severity: MEDIUM | Frequency: LOW (but cumulative)**

**Root Causes:**
- Over months, hair changes, facial hair growth, weight changes, glasses on/off shift embeddings
- Adaptive centroid (`alpha=0.15`) updates slowly; a significant appearance change requires ~20+ visits to drift the centroid enough
- No explicit "appearance change detection" mechanism

**Solutions:**
1. **Age-Drift Detection** — Track embedding drift over time. If a visitor's recent embeddings consistently differ from centroid by 0.40-0.50 (grey zone) but are internally consistent, trigger a "gradual re-enrollment" with higher alpha (0.50) for 5 visits.
2. **Seasonal Gallery Refresh** — Every 90 days, replace the oldest 3 gallery faces with recent high-quality ones. Prevents gallery from becoming "stale."
3. **Multi-Model Ensemble** — Use both ArcFace and a secondary model (e.g., MobileFaceNet) trained on different data. If both agree on identity despite ArcFace being in grey zone, accept the match.

---

### 1.5 False Merges: Two Similar-Looking People → Same Visitor Record
**Severity: CRITICAL | Frequency: MEDIUM**

**Root Causes:**
- Ambiguity gate (`top - runner_up < 0.05 → REJECT`) catches some cases but:
  - Runner-up might not be in top-2 if database is large (HNSW approximate search can miss true nearest neighbors)
  - Twins, siblings, or people from same ethnic group with similar features
  - Low-quality face (partial occlusion) reduces inter-person variance

**Real-World Scenario:**
Two brothers visit on different days. Brother A is registered. Brother B visits; face similarity to A = 0.58 (above threshold), runner-up is a stranger at 0.30. Difference = 0.28 > 0.05 → Brother B is merged into Brother A's record. Now analytics show one person with impossible visit patterns.

**Solutions:**
1. **Strict Quality Gate for Auto-Recognition** — Only auto-recognize (without human review) if `det_score >= 0.60` AND face is frontal (pose confidence). Lower-quality detections require body fallback or manual confirmation.
2. **Multi-Factor Verification** — Combine face + body + temporal + spatial consistency. If face says "returning" but body embedding differs significantly (indicating different clothing/build), flag as ambiguous even if face margin is wide.
3. **Visitor "Confidence Score" Decay** — Each visitor record has a confidence score. First visit = 0.3 (tentative). After 3 consistent recognitions = 0.7 (confirmed). High-confidence records require stronger evidence to match. Prevents early false merges from poisoning the database.
4. **Periodic Deduplication Job** — Weekly batch job: compute all-pairs similarity matrix for visitors with >5 visits. Flag pairs with similarity >0.75 for admin review (possible false merge).

---

### 1.6 Partial Face Occlusion (Mask, Hand, Menu) → Recognition Failure
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- ArcFace requires full face; partial occlusion drops `det_score` below `MIN_FACE_DET_SCORE=0.40`
- Even if detected, embedding quality is poor (missing nose/mouth region)
- Restaurant context: people hold menus, phones, cups; COVID-era mask wearing; hand-to-face gestures while eating

**Solutions:**
1. **Occlusion-Robust Face Detector** — Replace ArcFace's built-in detector with RetinaFace or YOLO-Face trained on occluded faces. Detects partial faces at lower confidence but higher recall.
2. **Per-Facial-Component Embeddings** — Extract embeddings from visible components only (eyes-only, upper-face). Requires model modification but handles masks/hand occlusions.
3. **Body Fallback Enhancement** — When face is occluded, rely entirely on body re-ID. As noted in the plan, body fallback is OFF by default. For restaurants, enable body fallback WITHIN a visit session (same clothing) but NEVER cross-visit. This requires tracking "session body embeddings" separately from "cross-visit face embeddings."
4. **Temporal Occlusion Handling** — If a person was recognized 5 seconds ago and now face is occluded, maintain identity via cross-frame tracking (IoU + cosine). Don't require re-recognition every frame.

---

### 1.7 Glasses On/Off / Headwear Changes
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- Glasses significantly alter eye region embeddings
- Hats/caps change forehead/hairline geometry
- These are common in restaurants (sunglasses on head, reading glasses, religious headwear)

**Solutions:**
1. **Accessory-Aware Gallery** — Detect accessories (glasses, hat) and store separate embeddings: "with_glasses" and "without_glasses". Query the appropriate variant.
2. **Eye-Region Suppression** — During embedding extraction, reduce weight of eye-region features (where glasses have most impact). Use attention masking.

---

## CATEGORY 2: Body Re-ID Failures

### 2.1 Body Fallback Creates False Cross-Visit Matches
**Severity: CRITICAL | Frequency: HIGH (if enabled)**

**Root Causes:**
- OSNet encodes clothing color, texture, bag style — all change between visits
- Plan correctly notes this: "OSNet embeddings are clothing/appearance dependent — useless and dangerous for recognising a regular on a different day"
- However, if someone enables `ALLOW_BODY_FALLBACK=true` without understanding this, false merge rate will spike

**Real-World Scenario:**
Visitor A wears a red jacket on Monday (registered). Visitor B wears a red jacket on Tuesday. Body similarity = 0.62 → Visitor B merged into Visitor A. Now one record has two different people.

**Solutions:**
1. **Session-Scoped Body Embeddings** — Store body embeddings in `active_visits` (in-memory only), not in `visitors` table. Use body matching ONLY to re-acquire a person who turned away during the SAME meal. Clear body embeddings when visit closes.
2. **Body + Face Consistency Check** — If body match suggests returning but face match is in grey zone, require BOTH to agree within a time window (e.g., face must match within 10 seconds of body match). If face never matches, treat as new visitor.
3. **Clothing Change Detection** — Compute body embedding drift during a visit. If body embedding changes >0.30 within 5 minutes (person took off jacket), ignore body for the rest of the session.

---

### 2.2 Similar Body Types + Same Clothing Colors → False Body Matches
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- OSNet x0.25 is lightweight (0.2M params) — good for speed, poor at fine-grained discrimination
- Two people wearing black shirts, similar builds → body similarity = 0.55+ (above threshold)
- Restaurant uniforms (staff wearing same shirt) → all staff might merge into one record

**Solutions:**
1. **Staff Exclusion from Body Matching** — Mark staff records (`is_staff=true`). Never use body fallback to match against staff (prevents customer-staff merges).
2. **Body + Spatial Context** — Body match is only valid if the person is in a plausible location (not jumping between tables instantly). Use cross-frame tracking continuity.
3. **Upgrade to OSNet-AIN or OSNet-IBN** — These variants have better discriminative power at same speed. Or use a two-stage approach: OSNet for fast filtering, then a heavier model (e.g., BOT) for ambiguous cases.

---

## CATEGORY 3: Visit Session Tracking Failures

### 3.1 Brief Absence (Bathroom Break) → New Visit Created
**Severity: MEDIUM | Frequency: HIGH**

**Root Causes:**
- `VISIT_COOLDOWN_MINUTES=20` — if someone is away for 21 minutes, new visit is created
- Restaurant context: bathroom break + ordering at counter + phone call outside easily exceeds 20 minutes
- Plan says "the brief-absence case is handled by not closing within the cooldown" but this is vague — if no detection for 20 min, visit IS closed

**Real-World Scenario:**
Customer eats for 30 min, goes to bathroom (5 min), returns. If camera doesn't detect them in bathroom (no line of sight), 5 min absence + 16 min remaining = visit closes at 21 min. Returns, detected as NEW visit #2. Analytics show 2 visits instead of 1.

**Solutions:**
1. **Smart Cooldown with Context** — Extend cooldown to 45 minutes for seated customers (detected at a table). Use table/chair detection (YOLO can detect chairs) to infer "person is likely still here even if not visible."
2. **Zone-Based Tracking** — Define zones: "dining area", "entrance", "bathroom corridor". If person was in dining area and disappears, assume they're still in restaurant (not a new visit) until they exit through entrance zone.
3. **Posture/Activity Detection** — If person left a bag/jacket on chair (detectable by YOLO), keep visit open even if person is not visible. "Possession proxy" indicates continued presence.

---

### 3.2 Long Meal → Visit Split by Max Duration Cap
**Severity: LOW | Frequency: LOW**

**Root Causes:**
- `MAX_VISIT_DURATION_HOURS=4` — auto-closes visits after 4 hours
- Some restaurants have all-day customers (coffee shop, co-working space)
- Creates artificial visit splits in analytics

**Solutions:**
1. **Venue-Type Configuration** — Coffee shop mode: `MAX_VISIT_DURATION_HOURS=8`. Fine dining mode: `MAX_VISIT_DURATION_HOURS=4`. Make this a runtime setting per camera.
2. **Activity-Based Extension** — If detections continue consistently (every 1-5 minutes) for 4+ hours, extend max duration by 2 hours. Only cap if detections are sporadic (indicating camera artifact or person left).

---

### 3.3 Person Leaves and Returns Next Day Within Cooldown Window
**Severity: LOW | Frequency: LOW**

**Root Causes:**
- If someone leaves at 11:55 PM and returns at 12:05 AM (next day), 10 minutes < 20 min cooldown
- Visit tracker sees this as "same visit" spanning midnight
- Analytics show a 12-hour visit duration

**Solutions:**
1. **Calendar-Day Boundary** — Force-close all visits at midnight (or 3 AM for late-night restaurants). New day = new visit, regardless of cooldown.
2. **Max Gap Within Visit** — Even within cooldown, if gap > 2 hours, split into new visit. Prevents overnight merges.

---

### 3.4 Camera Restart Mid-Visit → Lost Active Visit State
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- In-memory `active_visits` dict is lost on restart
- Plan says "load active visits from DB on startup" but this is recovery, not seamless continuation
- If restart takes >20 seconds, the DB-loaded visits might be stale (person left during downtime)

**Solutions:**
1. **Persistent Visit State with Heartbeat** — Write `active_visits` to Redis every 5 seconds. On restart, load from Redis (not just DB). Redis survives process restarts.
2. **Graceful Degradation on Restart** — On startup, mark all recovered active visits as "tentative." First detection of that visitor confirms continuation; no detection within 5 minutes closes the visit.

---

### 3.5 Two Cameras See Same Person → Double Counted Visits
**Severity: HIGH | Frequency: MEDIUM**

**Root Causes:**
- Each camera has its own `VisitTracker` instance (if multi-worker) or same instance (if single worker)
- Plan says "Same visitor_id → same active visit in tracker" but this assumes shared state
- If scaled to multiple workers (Redis path), visit state must be globally consistent
- Even with single worker, two cameras = two detection streams; if both detect same person simultaneously, two visits might open

**Real-World Scenario:**
Camera A (entrance) detects person entering. Camera B (dining area) detects same person 10 seconds later. If Camera B processes before Camera A's visit is committed to DB, Camera B sees no active visit → creates new visit. Now 2 visits for 1 entry.

**Solutions:**
1. **Global Visit Lock** — Use Redis distributed lock per `visitor_id` for visit creation. First camera to acquire lock creates the visit; others extend it.
2. **Cross-Camera Identity Propagation** — When a person is recognized on Camera A, immediately broadcast to all cameras: "Visitor #12 is now in venue." Camera B suppresses new-visit creation for 30 seconds.
3. **Post-Hoc Deduplication** — Daily batch job: find visits for same visitor with `entered_at` within 5 minutes on different cameras. Merge if spatially plausible (cameras are adjacent).

---

## CATEGORY 4: Detection Pipeline Failures

### 4.1 YOLO Misses Small / Distant / Partially Visible People
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- YOLOv8n is lightweight (fast) but lower recall on small objects
- Restaurant context: people walking past camera quickly, sitting far from camera, partially behind chairs/plants
- `YOLO_PERSON_CONFIDENCE=0.5` — might miss legitimate detections at 0.45 confidence

**Real-World Scenario:**
Person walks from entrance to table. At far end of room, they're small in frame (50px tall). YOLO confidence = 0.42 → not detected. They sit down, now partially occluded by chair. Never detected during entire visit. Analytics show 0 visits for a real customer.

**Solutions:**
1. **Two-Stage Detection** — Use YOLOv8n at full frame for fast filtering, then run a second pass on zoomed regions where people are likely (entrance, tables). Or use YOLOv8s (medium) for 5% better recall at 20% more cost.
2. **Dynamic Confidence Thresholding** — Lower confidence to 0.35 for regions near entrances (people are more likely to be real). Raise to 0.60 for background regions (reduce false positives from wall decorations).
3. **Temporal Integration** — If a person is detected in frame N at confidence 0.55, search for them in frame N+1 even at confidence 0.30 (tracking continuity). Use Kalman filter to predict location.

---

### 4.2 YOLO False Positives (Posters, Mannequins, Reflections)
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- Restaurant decor: framed photos of people, mannequins in display windows, mirrors reflecting patrons
- YOLO can't distinguish real person from photo of person
- These get registered as "new visitors" with low-quality faces (or no face if it's a poster)

**Solutions:**
1. **Liveness Detection (Re-add to pipeline)** — The plan removed liveness/anti-spoofing. Re-add a lightweight liveness model (e.g., Silent-Face-Anti-Spoofing or simple blink detection) for detections with no face embedding (posters have no face) or static face (photos don't blink).
2. **Static Object Filtering** — Track detection locations over time. If a "person" is detected at exactly the same pixel coordinates for >50 frames, it's a poster/mannequin. Add to ignore list.
3. **Depth Estimation** — Use a monocular depth estimator (e.g., MiDaS) to reject detections that are clearly on walls (depth > room size) or too flat (poster depth profile).

---

### 4.3 Frame Dedup Skips Actual Movement
**Severity: MEDIUM | Frequency: LOW**

**Root Causes:**
- dHash + MAD threshold = 4.0 — if a person moves slowly (elderly customer, someone reading), frame signature change might be below threshold
- System skips frame → misses detection opportunity
- Over 1 FPS, slow movement across 10 frames = significant real movement but low per-frame change

**Solutions:**
1. **Adaptive Dedup Threshold** — Lower MAD threshold to 2.0 for regions with recent person detections. Keep 4.0 for empty scenes.
2. **Motion History Image** — Instead of single-frame dHash, maintain a motion history buffer (last 5 frames). Trigger processing if cumulative motion exceeds threshold.

---

### 4.4 Group of People → Bounding Box Overlap / ID Switching
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- YOLO returns overlapping boxes for groups
- ArcFace processes full frame — if two people are close, face landmarks might be confused (landmarks of person A assigned to person B's box)
- Cross-frame tracking (IoU) switches IDs when people cross paths

**Real-World Scenario:**
Two friends walk in together. YOLO boxes overlap. ArcFace detects 2 faces but assigns both to the larger box. One person is "lost" (no face assigned to their box). They get registered as a new visitor later when separated. The other person gets two face embeddings in one frame, potentially corrupting their gallery.

**Solutions:**
1. **Non-Maximum Suppression (NMS) with Face Association** — After YOLO detection, use face landmark locations to disambiguate overlapping boxes. Each face must be within its assigned body box; if not, reassign.
2. **Group Handling Mode** — When >3 people are detected with high box overlap, switch to "group mode": track the group as a single entity until individuals separate (IoU < 0.3 for 3+ seconds), then resolve identities.
3. **Re-ID After Separation** — When group splits, use last-known embeddings + predicted motion to re-assign correct IDs. Maintain "identity hypotheses" for ambiguous cases.

---

## CATEGORY 5: Database & Scale Failures

### 5.1 HNSW Index Degradation with Many Low-Quality Embeddings
**Severity: MEDIUM | Frequency: MEDIUM (over time)**

**Root Causes:**
- Gallery stores 10 faces per visitor. If many are low-quality (side angles, blurry, occluded), the HNSW index contains "noise" embeddings
- Similarity search becomes less reliable as noise increases
- Plan has `FACE_QUALITY_CUTOFF=0.45` but this is for initial enrollment; gallery faces can be lower quality if they pass the weaker "strong match" gate

**Solutions:**
1. **Gallery Quality Audit** — Monthly job: compute intra-visitor embedding variance. If variance > 0.20 (faces are too dissimilar), flag for admin review or auto-prune lowest-quality faces.
2. **HNSW Index Rebuild** — Weekly `REINDEX` of `idx_vf_embedding_hnsw` to maintain search quality as data distribution shifts.
3. **Embedding Whitening** — Apply PCA whitening to gallery embeddings before HNSW indexing. Reduces impact of low-quality outliers.

---

### 5.2 pgvector Query Timeout Under Load
**Severity: HIGH | Frequency: MEDIUM (at scale)**

**Root Causes:**
- Batched query with `CROSS JOIN LATERAL` + `ORDER BY embedding <=> f.emb LIMIT 2` — for N faces, this is N× subqueries
- HNSW is fast but not instant; with 100K+ gallery faces, each subquery takes ~5-10ms
- 10 faces in frame = 50-100ms DB time. At 1 FPS this is fine, but if camera processes burst frames (e.g., group enters), DB can queue up

**Solutions:**
1. **Connection Pooling** — Ensure `asyncpg` pool size = 10-20 connections. The plan doesn't specify pool configuration.
2. **Query Result Caching** — Cache HNSW results for 5 seconds per face signature (dHash of face crop). If same person is in consecutive frames, reuse last result.
3. **Approximate Batch Search** — Use pgvector's `hnsw.ef_search` parameter. Lower to 32 for speed, 64 for accuracy. Make it configurable per deployment.
4. **Read Replica for Analytics** — Route analytics queries (dashboard) to a read replica. Keep primary for hot-path identity resolution.

---

### 5.3 Detection Events Table Bloat
**Severity: MEDIUM | Frequency: HIGH**

**Root Causes:**
- 1 FPS processing, 10 hours/day operation, 50 detections/day average = 1.8M rows/year
- `detection_events` has no partitioning or retention policy beyond `VISITOR_RETENTION_DAYS`
- Even with retention, table bloats between purge runs

**Solutions:**
1. **Time-Range Partitioning** — Partition `detection_events` by month. Drop old partitions instead of DELETE (instant, no bloat).
2. **Aggressive Aggregation** — For analytics, only need hourly aggregates after 30 days. Move raw events to cold storage (S3/MinIO) after 30 days; keep aggregated stats in DB.
3. **Columnar Storage** — Use TimescaleDB or ClickHouse for detection_events (time-series optimized). Keep relational tables in PostgreSQL.



## CATEGORY 7: Dashboard & UX Failures

### 7.1 WebSocket Live Feed Lag / Disconnect
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- WebSocket streams base64 JPEG every 1 second
- Each frame ~50-100KB at 1280px. 100KB/s = 800Kbps per client
- 10 admin clients = 8Mbps upstream from backend
- Plan says "CORS middleware disabled for WebSocket compatibility" — this suggests known WS issues

**Solutions:**
1. **MJPEG Stream Instead of WebSocket** — Use HTTP MJPEG streaming (multipart/x-mixed-replace). More compatible with proxies/load balancers than WebSocket.
2. **Frame Rate Throttling** — Reduce live feed to 0.5 FPS (frame every 2 seconds) for remote clients. Local dashboard gets 1 FPS.
3. **Delta Encoding** — Only send changed regions of frame (where detections moved). Reduces bandwidth by 70% for static backgrounds.

---

### 7.2 Analytics Inaccurate Due to False New/Returning Classification
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- All Category 1 failures (angle, lighting, occlusion) create false "new visitors"
- Dashboard shows inflated "new visitor" count and deflated "return rate"
- Business decisions (marketing, staffing) based on incorrect data

**Real-World Impact:**
Dashboard shows 30% return rate. Actual return rate is 50%. Restaurant invests in customer acquisition (unnecessary) instead of loyalty programs (needed).

**Solutions:**
1. **Confidence-Weighted Metrics** — Don't count detections with `is_ambiguous=true` or `face_similarity < 0.45` in new/returning stats. Report them as "unclassified" with separate KPI.
2. **Human Review Queue** — Flag visitors with <3 visits and high similarity to another visitor for admin review. "Visitor #47 looks similar to Visitor #12. Merge?"
3. **Ground Truth Calibration** — Monthly: admin manually reviews 50 random detections. Compute precision/recall of new/returning classification. Adjust thresholds if precision < 90%.
4. **Analytics Uncertainty Bands** — Show return rate as "45-55%" (range based on ambiguity rate) rather than single number.

---

### 7.3 Settings Page is Read-Only → No Runtime Tuning
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- Plan says "Settings page is read-only (settings are env-driven)"
- Admin must restart backend to change threshold
- In production, restarting drops active visits and interrupts service

**Solutions:**
1. **Runtime Config API** — Allow `POST /api/admin/settings` to update in-memory config. Changes persist to DB and are reloaded on restart. No restart needed.
2. **A/B Threshold Testing** — Allow running two cameras with different thresholds simultaneously. Compare accuracy metrics to find optimal values for the specific venue.
3. **Auto-Tuning** — Track false new/returning rate (via admin feedback or temporal consistency checks). Automatically adjust `RETURNING_FACE_THRESHOLD` ±0.02 weekly to optimize.

---

## CATEGORY 8: Deployment & Operational Failures

### 8.1 Single Worker Requirement Prevents Scaling
**Severity: HIGH | Frequency: MEDIUM (as restaurant grows)**

**Root Causes:**
- "App pinned to a single worker (documented); Redis noted as the scale-out path"
- In-memory `VisitTracker` can't be shared across workers
- If restaurant adds 3 cameras, single worker can't handle 3× load

**Solutions:**
1. **Redis-Backed Visit Tracker** — Implement now, not later. Use Redis Hash for `active_visits` with TTL = `VISIT_COOLDOWN_MINUTES`. All workers read/write same state.
2. **Camera Sharding** — Assign each camera to a specific worker (consistent hashing). Worker A handles cameras 1-2, Worker B handles cameras 3-4. No shared state needed within a worker's camera set.

---

### 8.2 Model Loading Failure on Startup → System Unusable
**Severity: CRITICAL | Frequency: LOW**

**Root Causes:**
- YOLO + ArcFace + OSNet must all load before first request
- ArcFace "buffalo_l" model is ~300MB download on first run
- If download fails (no internet, firewall), system never starts
- Plan says "not yet runtime-tested (heavy deps weren't installed)"

**Solutions:**
1. **Graceful Degradation** — If ArcFace fails to load, fall back to body-only mode (OSNet). If YOLO fails, reject all requests with 503. Don't crash.
2. **Pre-baked Docker Image** — Include model weights in Docker image (not downloaded at runtime). Image is larger but startup is reliable.
3. **Health Endpoint Detail** — `/api/health` should report WHICH model failed and WHY (disk space, network, corrupt file). Not just `arcface_loaded: false`.

---

### 8.3 Docker Compose Port Collision (3001-3010)
**Severity: LOW | Frequency: LOW**

**Root Causes:**
- Ports 3001-3010 are hardcoded in docker-compose
- If host already uses these ports (common dev tools), services fail to start

**Solutions:**
1. **Environment-Driven Ports** — Use `.env` for all ports: `BACKEND_PORT`, `DASHBOARD_PORT`, etc. Default to 3001-3010 but allow override.

---

## CATEGORY 9: Restaurant-Specific Edge Cases

### 9.1 Children / Height Variation
**Severity: MEDIUM | Frequency: MEDIUM**

**Root Causes:**
- Camera mounted at adult eye level → children's faces are at extreme upward angle
- ArcFace performs poorly on extreme vertical angles
- Children also have faster appearance changes (growth, hair changes)

**Solutions:**
1. **Multi-Height Camera Setup** — Use two cameras at different heights OR a wide-angle camera with perspective correction.
2. **Child Detection** — If detected person is <120cm (estimated from bounding box height + camera geometry), apply lower face threshold (0.45) and higher ambiguity margin (0.08) due to less discriminative child faces.

---

### 9.2 Pets / Service Animals Detected as People
**Severity: LOW | Frequency: LOW**

**Root Causes:**
- YOLOv8n sometimes detects dogs/cats as "person" (class confusion)
- ArcFace fails to find face (no harm) but body embedding is computed
- Could create "ghost visitors" with no face, only body

**Solutions:**
1. **Face-Required Enrollment** — Don't auto-enroll if no face is detected, regardless of body detection. Plan already does this partially via `FACE_QUALITY_CUTOFF`.
2. **Animal Classifier** — Add a lightweight animal detector (MobileNet) to reject detections before identity resolution.

---

### 9.3 Customer Wearing Mask (Post-COVID Norm)
**Severity: HIGH | Frequency: HIGH**

**Root Causes:**
- Mask covers 50% of face → ArcFace embedding is unreliable
- Plan has no mask-specific handling
- In some regions, mask-wearing is still common in restaurants

**Solutions:**
1. **Mask Detection + Periocular Recognition** — Detect mask presence. If masked, use only eye/forehead region for embedding (periocular recognition). Requires retraining or using a mask-aware model like MaskedFace-Net.
2. **Body-Primary Mode for Masked** — When mask detected, rely primarily on body re-ID (within session) and don't attempt cross-visit recognition. Mark visit as "masked — identity uncertain."

---

### 9.4 Customer Changes Seat Multiple Times
**Severity: LOW | Frequency: MEDIUM**

**Root Causes:**
- Customer moves from bar to table, or table to table
- Cross-frame tracking might lose them if out of frame for extended period
- Each "re-detection" after loss might be treated as new visit if >20 min

**Solutions:**
1. **Venue Graph Tracking** — Model the restaurant as a graph (entrance → bar → dining area → bathroom → exit). If person was last seen in bar and now appears in dining area (connected node), maintain same visit. If they appear at entrance (exit required), new visit.

---

### 9.5 Drive-Through / Pickup Window (if applicable)
**Severity: LOW | Frequency: LOW**

**Root Causes:**
- Car windows, sunglasses, partial face visible
- Brief interaction (<1 minute) — not a "visit" in traditional sense

**Solutions:**
1. **Separate Camera Profile** — Drive-through camera uses different rules: no enrollment, only counting. Or use license plate recognition instead of face for drive-through analytics.

---

## CATEGORY 10: CPU Optimization Gaps

### 10.1 ArcFace Full-Frame Pass is Still Expensive
**Severity: MEDIUM | Frequency: CONSTANT**

**Root Causes:**
- "Full-frame ArcFace (detect all faces at once)" — this is better than N crops, but still processes entire frame through ResNet100 backbone
- At 1280px long side, frame is large; ArcFace resizes to 640×640 internally
- 100-200ms per frame is 10-20% of 1-second budget

**Solutions:**
1. **Face Region Pre-Cropping** — Use a lightweight face detector (e.g., YuNet or MediaPipe Face Detection) at 5 FPS to get face ROIs. Only run ArcFace on ROIs, not full frame. YuNet is 5ms vs ArcFace's 100ms.
2. **MobileFaceNet Instead of ArcFace** — MobileFaceNet is 4× faster with <2% accuracy drop. Use for initial filtering; ArcFace only for ambiguous cases.
3. **INT8 Quantization** — Convert ArcFace to ONNX with INT8 quantization. 2× speedup with minimal accuracy loss.
4. **Batch Multiple Frames** — If camera is at 1 FPS but system can handle 2 FPS, batch 2 frames and process together. Better GPU/CPU utilization through batching.

---

### 10.2 OSNet Body Re-ID is Redundant for Confident Face Matches
**Severity: LOW | Frequency: CONSTANT**

**Root Causes:**
- Plan says "Skip body embedding when face is strong" — good, but the check happens AFTER computing both
- YOLO still runs on full frame; body crops are extracted even if face is strong

**Solutions:**
1. **Cascade Architecture** — Run YOLO first. For each detection, run face detector ONLY. If face confidence > 0.60, skip body extraction entirely. Only extract body for face-confidence < 0.60 detections. Saves 30-80ms per confident detection.
2. **YOLO Person Confidence Thresholding** — If YOLO person confidence < 0.70 AND no face detected, don't bother with body extraction (likely false positive).

---

### 10.3 No NPU / Edge TPU Utilization
**Severity: LOW | Frequency: N/A (future-proofing)**

**Root Causes:**
- Plan is CPU-only
- Modern edge devices (Coral TPU, Intel NCS2, Apple Neural Engine) can accelerate inference 10-100×

**Solutions:**
1. **ONNX Runtime with Execution Providers** — Configure ONNX Runtime to use OpenVINO (Intel), CoreML (Apple), or CUDA (NVIDIA) if available. Falls back to CPU if not.
2. **Model Export Formats** — Maintain models in multiple formats: ONNX (general), TFLite (mobile/edge), OpenVINO IR (Intel). Load best available at runtime.

---

## SUMMARY: Priority Matrix

| Priority | Issue | Category | Solution Complexity | Impact if Fixed |
|----------|-------|----------|-------------------|-----------------|
| P0 | Same person, different angle → new person | 1.1 | High | Very High |
| P0 | GDPR/BIPA auto-enrollment violation | 6.1 | Medium | Critical (legal) |
| P0 | Group overlap / ID switching | 4.4 | High | Very High |
| P1 | Lighting variation → new person | 1.2 | Medium | High |
| P1 | Partial occlusion (menu, hand, mask) | 1.6 | Medium | High |
| P1 | Brief absence → new visit | 3.1 | Medium | High |
| P1 | YOLO misses small/distant people | 4.1 | Medium | High |
| P1 | Analytics inaccurate due to false classification | 7.2 | Low | High |
| P2 | Single worker scaling limit | 8.1 | High | Medium |
| P2 | Body fallback false merges | 2.1 | Low | Medium |
| P2 | HNSW index degradation | 5.1 | Low | Medium |
| P2 | CPU optimization (cascade architecture) | 10.2 | Low | Medium |
| P3 | Expression changes | 1.3 | High | Low |
| P3 | Aging/appearance drift | 1.4 | High | Low |
| P3 | Glasses on/off | 1.7 | Medium | Low |
| P3 | Pets detected as people | 9.2 | Low | Low |

---

## RECOMMENDED IMPLEMENTATION ORDER

### Week 1: Critical Fixes
1. **Pose-aware gallery management** (bins: frontal, left, right) — fixes #1 failure mode
2. **Consent workflow + anonymized mode** — legal compliance
3. **Group handling + face-body association** — fixes ID switching
4. **CLAHE preprocessing** — fixes lighting issues

### Week 2: Accuracy Improvements
5. **Temporal consistency gate** — prevents same-person fragmentation
6. **Smart cooldown with table detection** — fixes bathroom break issue
7. **Occlusion-robust detection** — mask/menu handling
8. **Confidence-weighted analytics** — honest metrics

### Week 3: Scale & Performance
9. **Redis-backed visit tracker** — enables multi-worker
10. **Cascade architecture** (face-first, body-only-if-needed) — CPU optimization
11. **Query result caching** — DB performance
12. **Partitioning for detection_events** — long-term storage

### Week 4: Polish
13. **Auto-tuning thresholds** — self-improving system
14. **Staff pre-registration** — operational workflow
15. **Human review queue** — ongoing quality assurance
