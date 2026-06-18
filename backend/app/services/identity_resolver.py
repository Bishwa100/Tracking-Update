"""
Identity resolver — decides NEW vs RETURNING for each detected face.

Strategy:
  1. Batched HNSW search over visitor_faces.embedding (top-K per input face in a
     single DB round-trip via VALUES + CROSS JOIN LATERAL), then collapsed to the
     best score PER VISITOR so the ambiguity runner-up is a different person.
  2. Threshold + ambiguity gate:
       top_sim >= RETURNING_FACE_THRESHOLD and clears runner-up by AMBIGUITY_MARGIN
         → RETURNING
       runner-up within AMBIGUITY_MARGIN
         → AMBIGUOUS (skip — don't risk a false merge)
       top_sim <= REJECT_SIMILARITY
         → NEW (confident stranger)
       REJECT_SIMILARITY < top_sim < RETURNING_FACE_THRESHOLD (grey zone)
         → optional body fallback, else HELD ("grey_zone") per GREY_ZONE_POLICY —
           NOT registered as a new visitor (that fragments one person at a new
           angle into many records).
  3. Body fallback is OFF by default and same-session only — OSNet embeddings are
     clothing dependent and must NOT be used to recognise visitors across visits.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ResolutionResult:
    visitor_id: Optional[UUID] = None
    is_new: bool = False
    is_ambiguous: bool = False
    face_similarity: float = 0.0
    body_similarity: float = 0.0
    match_source: str = "none"  # "face" | "body" | "new" | "grey_zone" | "none"
    # Best-scoring gallery visitor for this face, even when we DIDN'T match them
    # (i.e. a new/grey-zone decision). Lets the review queue show "similar to whom".
    top_match_id: Optional[UUID] = None


async def _search_faces_batch(
    embeddings: List[List[float]], db: AsyncSession,
    pose_bins: Optional[List[str]] = None,
) -> List[List[Tuple[UUID, float]]]:
    """
    Return the top-K (visitor_id, similarity) gallery rows per input face
    embedding, in input order (K = settings.IDENTITY_TOP_K). The caller
    collapses these to the best score per visitor, so K > 2 ensures the closest
    DIFFERENT visitor is visible to the ambiguity gate even when several of the
    nearest rows belong to the same person. When pose_bins is provided each
    embedding is compared against its matching pose bin first, then frontal as a
    fallback, then unknown — giving pose-aware gallery search in one round-trip.
    """
    if not embeddings:
        return []

    top_k = max(2, int(settings.IDENTITY_TOP_K))
    params: dict = {"top_k": top_k}
    for i, emb in enumerate(embeddings):
        params[f"emb_{i}"] = str(emb)
        if pose_bins:
            params[f"pose_{i}"] = pose_bins[i]

    values_sql = ", ".join(
        rf"({i}, (:emb_{i})\:\:vector)" for i in range(len(embeddings))
    )

    if pose_bins:
        # Pose-binned search: exact bin + frontal fallback + unknown fallback
        pose_values = ", ".join(
            rf"({i}, :pose_{i})" for i in range(len(embeddings))
        )
        query = text(rf"""
            WITH input_faces AS (
                SELECT idx, emb FROM (VALUES {values_sql}) AS v(idx, emb)
            ),
            input_poses AS (
                SELECT idx, pose_bin FROM (VALUES {pose_values}) AS p(idx, pose_bin)
            )
            SELECT f.idx, m.visitor_id, m.similarity
            FROM input_faces f
            JOIN input_poses p ON p.idx = f.idx
            CROSS JOIN LATERAL (
                SELECT vf.visitor_id,
                       1 - (vf.embedding <=> f.emb) AS similarity
                FROM visitor_faces vf
                JOIN visitors vis ON vis.id = vf.visitor_id
                WHERE vis.is_active = TRUE
                  AND vis.consent_status != 'opted_out'
                  AND (
                      vf.pose_bin = p.pose_bin
                      OR vf.pose_bin = 'frontal'
                      OR vf.pose_bin = 'unknown'
                  )
                ORDER BY
                    CASE
                        WHEN vf.pose_bin = p.pose_bin THEN 1
                        WHEN vf.pose_bin = 'frontal' THEN 2
                        ELSE 3
                    END,
                    vf.embedding <=> f.emb
                LIMIT :top_k
            ) m
            ORDER BY f.idx, m.similarity DESC
        """)
    else:
        query = text(rf"""
            SELECT v.idx, m.visitor_id, m.similarity
            FROM (VALUES {values_sql}) AS v(idx, emb)
            CROSS JOIN LATERAL (
                SELECT vf.visitor_id,
                       1 - (vf.embedding <=> v.emb) AS similarity
                FROM visitor_faces vf
                JOIN visitors vis ON vis.id = vf.visitor_id
                WHERE vis.is_active = TRUE
                  AND vis.consent_status != 'opted_out'
                ORDER BY vf.embedding <=> v.emb
                LIMIT :top_k
            ) m
            ORDER BY v.idx, m.similarity DESC
        """)

    result = await db.execute(query, params)
    grouped: List[List[Tuple[UUID, float]]] = [[] for _ in embeddings]
    for row in result.all():
        grouped[int(row.idx)].append((row.visitor_id, float(row.similarity)))
    return grouped


async def _search_body(embedding: List[float], db: AsyncSession) -> Optional[Tuple[UUID, float]]:
    """Closest visitor by body centroid (used only when body fallback is enabled)."""
    query = text(r"""
        SELECT id, 1 - (body_embedding <=> :emb\:\:vector) AS similarity
        FROM visitors
        WHERE body_embedding IS NOT NULL AND is_active = TRUE
        ORDER BY body_embedding <=> :emb\:\:vector
        LIMIT 1
    """)
    row = (await db.execute(query, {"emb": str(embedding)})).first()
    if row is None:
        return None
    return row.id, float(row.similarity)


def _best_per_visitor(
    matches: List[Tuple[UUID, float]]
) -> List[Tuple[UUID, float]]:
    """
    Collapse raw top-K gallery rows to the best similarity per visitor, sorted
    by similarity descending. Several rows of one visitor (different pose faces)
    count as supporting evidence for that visitor, NOT as competing candidates —
    so the runner-up used by the ambiguity gate is always a DIFFERENT visitor.
    """
    best: dict = {}
    for vid, sim in matches:
        if vid not in best or sim > best[vid]:
            best[vid] = sim
    return sorted(best.items(), key=lambda kv: kv[1], reverse=True)


def _decide_from_face(
    matches: List[Tuple[UUID, float]],
    threshold_offset: float = 0.0,
) -> ResolutionResult:
    """Apply thresholds + (visitor-level) ambiguity gate to one face's matches."""
    if not matches:
        return ResolutionResult(is_new=True, match_source="new")

    collapsed = _best_per_visitor(matches)
    top_id, top_sim = collapsed[0]
    runner_up = collapsed[1] if len(collapsed) > 1 else None  # a different visitor

    # Masked faces get a loosened returning threshold (offset is negative).
    effective_threshold = settings.RETURNING_FACE_THRESHOLD + threshold_offset

    if top_sim >= effective_threshold:
        if (
            runner_up is not None
            and (top_sim - runner_up[1]) < settings.AMBIGUITY_MARGIN
        ):
            return ResolutionResult(
                is_ambiguous=True, face_similarity=top_sim, match_source="none",
                top_match_id=top_id,
            )
        return ResolutionResult(
            visitor_id=top_id,
            face_similarity=top_sim,
            match_source="face",
            top_match_id=top_id,
        )

    # Confident stranger: clearly below any known visitor → definitely NEW.
    if top_sim <= (settings.REJECT_SIMILARITY + threshold_offset):
        return ResolutionResult(
            is_new=True, face_similarity=top_sim, match_source="new",
            top_match_id=top_id,
        )

    # Grey zone (REJECT_SIMILARITY < top_sim < RETURNING_FACE_THRESHOLD): not a
    # confident match and not a confident stranger. Hold it — registering here is
    # the main source of duplicate records for the same person at a new angle.
    return ResolutionResult(
        is_new=False, face_similarity=top_sim, match_source="grey_zone",
        top_match_id=top_id,
    )


async def resolve_batch(
    faces: List[dict],
    db: AsyncSession,
) -> List[ResolutionResult]:
    """
    Resolve a list of detected faces in one DB round-trip.

    faces: [{"face_embedding": [...], "body_embedding": [...] | None,
             "det_score": float, "pose_bin": str | None}].
    Returns one ResolutionResult per input face, in order.
    """
    if not faces:
        return []

    face_embeddings = [f["face_embedding"] for f in faces]
    pose_bins = [f.get("pose_bin") or "unknown" for f in faces]
    grouped = await _search_faces_batch(face_embeddings, db, pose_bins=pose_bins)

    results: List[ResolutionResult] = []
    for face, matches in zip(faces, grouped):
        res = _decide_from_face(matches, threshold_offset=face.get("threshold_offset", 0.0))

        # Grey-zone body fallback (same-session re-acquisition only, opt-in).
        if (
            res.match_source == "grey_zone"
            and settings.ALLOW_BODY_FALLBACK
            and face.get("body_embedding")
        ):
            body = await _search_body(face["body_embedding"], db)
            if body is not None and body[1] >= settings.RETURNING_BODY_THRESHOLD:
                res = ResolutionResult(
                    visitor_id=body[0],
                    face_similarity=res.face_similarity,
                    body_similarity=body[1],
                    match_source="body",
                    top_match_id=res.top_match_id,
                )

        # Remaining grey-zone faces are HELD by default (match_source stays
        # "grey_zone" → detection_pipeline records an audit event and does not
        # register a visitor). Only the explicit "register" escape hatch falls
        # back to the legacy behaviour of seeding a new visitor.
        if (
            res.match_source == "grey_zone"
            and settings.GREY_ZONE_POLICY == "register"
            and face.get("det_score", 0.0) >= settings.FACE_QUALITY_CUTOFF
        ):
            res = ResolutionResult(
                is_new=True,
                face_similarity=res.face_similarity,
                match_source="new",
                top_match_id=res.top_match_id,
            )

        results.append(res)
    return results
