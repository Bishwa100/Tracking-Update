"""Visitor CRUD + visit history endpoints."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.responses import FileResponse
from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import verify_api_key
from app.database import get_db
from app.models import Visit, Visitor
from app.schemas import (
    ConsentUpdateRequest,
    VisitListResponse,
    VisitorDetailResponse,
    VisitorListResponse,
    VisitorSummary,
    VisitorUpdateRequest,
    VisitSummary,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/visitors", tags=["visitors"])


def _thumbnail_url(visitor: Visitor) -> Optional[str]:
    return f"/api/visitors/{visitor.id}/thumbnail" if visitor.thumbnail_path else None


def _visit_summary(v: Visit) -> VisitSummary:
    return VisitSummary(
        id=v.id,
        entered_at=v.entered_at,
        left_at=v.left_at,
        duration_minutes=v.duration_minutes,
        detection_count=v.detection_count or 0,
        best_face_confidence=v.best_face_confidence,
        camera_id=v.camera_id,
        is_active=v.left_at is None,
    )


@router.get("", response_model=VisitorListResponse)
async def list_visitors(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    min_visits: Optional[int] = Query(None, ge=0),
    since: Optional[datetime] = Query(None),
    search: Optional[str] = Query(None, description="Match visitor name or id prefix"),
    sort_by: str = Query("last_seen", pattern="^(last_seen|visit_count|first_seen)$"),
    include_staff: bool = Query(False),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """List visitors with filtering, sorting, and pagination."""
    query = select(Visitor).where(Visitor.is_active.is_(True))
    if not include_staff:
        query = query.where(Visitor.is_staff.is_(False))
    if min_visits is not None:
        query = query.where(Visitor.visit_count >= min_visits)
    if since is not None:
        query = query.where(Visitor.last_seen_at >= since)
    if search:
        term = search.strip()
        query = query.where(
            Visitor.name.ilike(f"%{term}%")
            | func.cast(Visitor.id, String).ilike(f"{term}%")
        )

    total = await db.scalar(select(func.count()).select_from(query.subquery()))

    sort_col = {
        "last_seen": Visitor.last_seen_at,
        "visit_count": Visitor.visit_count,
        "first_seen": Visitor.first_seen_at,
    }[sort_by]
    query = query.order_by(sort_col.desc().nullslast()).limit(limit).offset(offset)

    visitors = (await db.execute(query)).scalars().all()
    return VisitorListResponse(
        total=total or 0,
        visitors=[
            VisitorSummary(
                id=v.id,
                name=v.name,
                visit_count=v.visit_count,
                first_seen_at=v.first_seen_at,
                last_seen_at=v.last_seen_at,
                is_staff=v.is_staff,
                is_active=v.is_active,
                best_face_det_score=v.best_face_det_score,
                thumbnail_url=_thumbnail_url(v),
            )
            for v in visitors
        ],
    )


@router.get("/{visitor_id}", response_model=VisitorDetailResponse)
async def get_visitor(
    visitor_id: UUID,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    visitor = await db.get(Visitor, visitor_id)
    if visitor is None or not visitor.is_active:
        raise HTTPException(status_code=404, detail="Visitor not found.")

    latest = (
        await db.execute(
            select(Visit)
            .where(Visit.visitor_id == visitor_id)
            .order_by(Visit.entered_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    return VisitorDetailResponse(
        id=visitor.id,
        name=visitor.name,
        notes=visitor.notes,
        visit_count=visitor.visit_count,
        first_seen_at=visitor.first_seen_at,
        last_seen_at=visitor.last_seen_at,
        is_staff=visitor.is_staff,
        is_active=visitor.is_active,
        best_face_det_score=visitor.best_face_det_score,
        total_faces_recorded=visitor.total_faces_recorded or 0,
        thumbnail_url=_thumbnail_url(visitor),
        latest_visit=_visit_summary(latest) if latest else None,
        consent_status=getattr(visitor, "consent_status", None),
        consent_at=getattr(visitor, "consent_at", None),
        consent_method=getattr(visitor, "consent_method", None),
        opted_out_at=getattr(visitor, "opted_out_at", None),
    )


@router.get("/{visitor_id}/gallery-insights")
async def get_gallery_insights(
    visitor_id: UUID,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """
    Phase 3/4 visibility for one visitor: gallery pose coverage, which cameras
    captured their faces, the computed per-visitor adaptive thresholds, and any
    merges folded into this record.
    """
    from app.models import VisitorFace, VisitorMergeAudit

    visitor = await db.get(Visitor, visitor_id)
    if visitor is None or not visitor.is_active:
        raise HTTPException(status_code=404, detail="Visitor not found.")

    faces = (
        await db.execute(
            select(VisitorFace).where(VisitorFace.visitor_id == visitor_id)
        )
    ).scalars().all()

    pose_coverage: dict[str, int] = {}
    camera_coverage: dict[str, int] = {}
    for f in faces:
        pose_coverage[f.pose_bin or "unknown"] = pose_coverage.get(f.pose_bin or "unknown", 0) + 1
        if f.source_camera_id:
            camera_coverage[f.source_camera_id] = camera_coverage.get(f.source_camera_id, 0) + 1

    merges = (
        await db.execute(
            select(VisitorMergeAudit)
            .where(VisitorMergeAudit.target_visitor_id == visitor_id)
            .order_by(VisitorMergeAudit.created_at.desc())
            .limit(20)
        )
    ).scalars().all()

    return {
        "gallery_size": len(faces),
        "pose_coverage": pose_coverage,
        "camera_coverage": camera_coverage,
        "adaptive_thresholds": {
            "expected_match_similarity": visitor.expected_match_similarity,
            "match_similarity_std": visitor.match_similarity_std,
            "personal_returning_threshold": visitor.personal_returning_threshold,
            "personal_new_threshold": visitor.personal_new_threshold,
        },
        "merges": [
            {
                "id": str(m.id),
                "source_visitor_id": str(m.source_visitor_id) if m.source_visitor_id else None,
                "reason": m.reason,
                "similarity": m.similarity,
                "merged_by": m.merged_by,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in merges
        ],
    }


@router.get("/{visitor_id}/visits", response_model=VisitListResponse)
async def get_visitor_visits(
    visitor_id: UUID,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    exists = await db.get(Visitor, visitor_id)
    if exists is None:
        raise HTTPException(status_code=404, detail="Visitor not found.")

    base = select(Visit).where(Visit.visitor_id == visitor_id)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    visits = (
        await db.execute(
            base.order_by(Visit.entered_at.desc()).limit(limit).offset(offset)
        )
    ).scalars().all()
    return VisitListResponse(
        total=total or 0, visits=[_visit_summary(v) for v in visits]
    )


@router.put("/{visitor_id}", response_model=VisitorDetailResponse)
async def update_visitor(
    visitor_id: UUID,
    request: VisitorUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    visitor = await db.get(Visitor, visitor_id)
    if visitor is None or not visitor.is_active:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    if request.name is not None:
        visitor.name = request.name
    if request.notes is not None:
        visitor.notes = request.notes
    if request.is_staff is not None:
        visitor.is_staff = request.is_staff
    await db.commit()
    return await get_visitor(visitor_id, db, _key)


@router.post("/{visitor_id}/consent", response_model=VisitorDetailResponse)
async def update_consent(
    visitor_id: UUID,
    request: ConsentUpdateRequest,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """
    Set a visitor's consent status. Opting out stops them being matched (the
    identity resolver excludes consent_status='opted_out') and clears their
    entries from the temporal-consistency gate immediately. Their embeddings are
    purged by the retention job after OPTED_OUT_EMBEDDING_TTL_DAYS.
    """
    valid = {"implicit", "explicit", "opted_out"}
    if request.consent_status not in valid:
        raise HTTPException(status_code=400, detail=f"consent_status must be one of {valid}.")

    visitor = await db.get(Visitor, visitor_id)
    if visitor is None or not visitor.is_active:
        raise HTTPException(status_code=404, detail="Visitor not found.")

    now = datetime.now(timezone.utc)
    visitor.consent_status = request.consent_status
    visitor.consent_method = request.method
    if request.consent_status == "opted_out":
        visitor.opted_out_at = now
    else:
        visitor.consent_at = now
        visitor.opted_out_at = None
    await db.commit()

    if request.consent_status == "opted_out":
        try:
            from app.services.temporal_consistency import temporal_gate
            temporal_gate.clear_visitor(visitor_id)
        except Exception:
            logger.debug("Could not clear temporal gate for %s", visitor_id)

    return await get_visitor(visitor_id, db, _key)


@router.delete("/{visitor_id}")
async def delete_visitor(
    visitor_id: UUID,
    hard: bool = Query(False, description="Permanently delete instead of soft delete"),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    visitor = await db.get(Visitor, visitor_id)
    if visitor is None:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    if hard:
        await db.delete(visitor)
    else:
        visitor.is_active = False
    await db.commit()
    return {"success": True, "visitor_id": str(visitor_id), "hard": hard}


@router.get("/{visitor_id}/thumbnail")
async def get_visitor_thumbnail(
    visitor_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    visitor = await db.get(Visitor, visitor_id)
    if visitor is None or not visitor.thumbnail_path:
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    path = Path(visitor.thumbnail_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail file missing.")
    return FileResponse(str(path), media_type="image/jpeg")
