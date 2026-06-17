"""Admin endpoints — merge duplicate visitors, mark staff."""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Security
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import verify_api_key
from app.database import get_db
from app.schemas import MarkStaffRequest, MergeRequest
from app.models import Visitor
from app.services.visitor_merge import MergeError, merge_visitors

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/visitors/{visitor_id}/merge")
async def merge_visitor(
    visitor_id: UUID,
    request: MergeRequest,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """Merge `visitor_id` INTO target_visitor_id, then delete the source."""
    try:
        result = await merge_visitors(db, visitor_id, request.target_visitor_id)
    except MergeError as exc:
        status = 400 if "itself" in str(exc) else 404
        raise HTTPException(status_code=status, detail=str(exc))
    return {"success": True, **result}


@router.post("/visitors/{visitor_id}/mark-staff")
async def mark_staff(
    visitor_id: UUID,
    request: MarkStaffRequest,
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    visitor = await db.get(Visitor, visitor_id)
    if visitor is None:
        raise HTTPException(status_code=404, detail="Visitor not found.")
    visitor.is_staff = request.is_staff
    await db.commit()
    return {"success": True, "visitor_id": str(visitor_id), "is_staff": request.is_staff}
