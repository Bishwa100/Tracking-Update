"""Analytics endpoints."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, Security
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import verify_api_key
from app.database import get_db
from app.schemas import (
    AnalyticsSummary,
    FrequencyDistribution,
    HourlyBreakdown,
    TopVisitor,
)
from app.services import analytics_service

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    return AnalyticsSummary(**await analytics_service.summary(db, since, until))


@router.get("/frequency", response_model=FrequencyDistribution)
async def analytics_frequency(
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    return FrequencyDistribution(**await analytics_service.frequency(db))


@router.get("/hourly", response_model=HourlyBreakdown)
async def analytics_hourly(
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    return HourlyBreakdown(**await analytics_service.hourly(db, since, until))


@router.get("/top-visitors", response_model=list[TopVisitor])
async def analytics_top_visitors(
    limit: int = Query(10, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    rows = await analytics_service.top_visitors(db, limit)
    return [TopVisitor(**r) for r in rows]


@router.get("/confidence-weighted")
async def analytics_confidence_weighted(
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    min_confidence: float = Query(0.40, ge=0.0, le=1.0),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """Summary with confidence-weighted unique visitor count."""
    return await analytics_service.confidence_weighted_summary(db, since, until, min_confidence)


@router.get("/detection-quality")
async def analytics_detection_quality(
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """Detection quality band breakdown (high / medium / low confidence)."""
    return await analytics_service.detection_quality_report(db, since, until)


@router.get("/pipeline-quality")
async def analytics_pipeline_quality(
    since: Optional[datetime] = Query(None),
    until: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
    _key: str = Security(verify_api_key),
):
    """Decision-pipeline health: grey-zone / ambiguous rates + recovery counts."""
    return await analytics_service.pipeline_quality(db, since, until)
