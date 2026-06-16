"""
SQLAlchemy ORM models with pgvector embedding columns.

Restaurant visitor tracking schema:
    visitors          — core identity (centroid embeddings + visit stats)
    visitor_faces     — per-visitor multi-pose face gallery (top-N by quality)
    visits            — visit sessions (enter/leave/duration)
    detection_events  — per-detection audit trail
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, ForeignKey, Text,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Visitor(Base):
    """A unique person seen by the system (auto-registered on first detection)."""

    __tablename__ = "visitors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Centroid embeddings (L2-normalized, adaptively updated on confident matches).
    face_embedding = Column(Vector(512), nullable=True)
    body_embedding = Column(Vector(512), nullable=True)

    # Denormalized visit statistics for fast reads.
    visit_count = Column(Integer, nullable=False, default=0)
    first_seen_at = Column(DateTime(timezone=True), default=utcnow)
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    # Quality tracking.
    best_face_det_score = Column(Float, default=0.0)
    total_faces_recorded = Column(Integer, default=0)

    # Admin / label fields.
    name = Column(Text, nullable=True)            # optional human label
    notes = Column(Text, nullable=True)
    thumbnail_path = Column(Text, nullable=True)  # best face crop on disk
    is_staff = Column(Boolean, default=False)     # exclude from analytics
    is_active = Column(Boolean, default=True)     # soft delete

    # Consent / privacy (GDPR / BIPA compliance)
    consent_status = Column(String(20), nullable=True, default="implicit")
    consent_at = Column(DateTime(timezone=True), nullable=True)
    consent_method = Column(String(50), nullable=True)
    opted_out_at = Column(DateTime(timezone=True), nullable=True)

    # Recognition confidence (0.3 = tentative, 1.0 = confirmed staff/explicit)
    visit_confidence = Column(Float, nullable=True, default=0.3)

    faces = relationship(
        "VisitorFace", back_populates="visitor", cascade="all, delete-orphan"
    )
    visits = relationship(
        "Visit", back_populates="visitor", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "idx_visitors_face_hnsw",
            face_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"face_embedding": "vector_cosine_ops"},
        ),
        Index(
            "idx_visitors_body_hnsw",
            body_embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"body_embedding": "vector_cosine_ops"},
        ),
        Index("idx_visitors_last_seen", last_seen_at.desc()),
        Index("idx_visitors_visit_count", visit_count.desc()),
    )


class VisitorFace(Base):
    """One face embedding in a visitor's multi-pose gallery."""

    __tablename__ = "visitor_faces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    visitor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("visitors.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    embedding = Column(Vector(512), nullable=False)
    det_score = Column(Float, nullable=False, default=0.0)
    body_embedding = Column(Vector(512), nullable=True)
    source_frame_path = Column(Text, nullable=True)
    pose_bin = Column(String(20), nullable=True, default="unknown")
    created_at = Column(DateTime(timezone=True), default=utcnow)

    visitor = relationship("Visitor", back_populates="faces")

    __table_args__ = (
        Index(
            "idx_visitor_faces_hnsw",
            embedding,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class Visit(Base):
    """A single visit session for a visitor."""

    __tablename__ = "visits"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    visitor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("visitors.id", ondelete="CASCADE"),
        nullable=False,
    )

    entered_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    left_at = Column(DateTime(timezone=True), nullable=True)  # NULL = still active
    duration_minutes = Column(Integer, nullable=True)

    detection_count = Column(Integer, default=0)
    best_face_confidence = Column(Float, nullable=True)
    avg_face_confidence = Column(Float, nullable=True)

    camera_id = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    visitor = relationship("Visitor", back_populates="visits")

    __table_args__ = (
        Index("idx_visits_visitor", visitor_id),
        Index("idx_visits_entered", entered_at.desc()),
        Index("idx_visits_visitor_entered", visitor_id, entered_at.desc()),
        # Partial index: lightning-fast lookup of still-open visits.
        Index("idx_visits_active", left_at, postgresql_where=(left_at.is_(None))),
    )


class DetectionEvent(Base):
    """Per-detection audit trail (one row per recognised/created detection)."""

    __tablename__ = "detection_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    visitor_id = Column(
        UUID(as_uuid=True),
        ForeignKey("visitors.id", ondelete="SET NULL"),
        nullable=True,
    )
    visit_id = Column(
        UUID(as_uuid=True),
        ForeignKey("visits.id", ondelete="SET NULL"),
        nullable=True,
    )

    detected_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)
    face_similarity = Column(Float, nullable=True)
    body_similarity = Column(Float, nullable=True)

    is_new_visitor = Column(Boolean, nullable=False, default=False)
    is_ambiguous = Column(Boolean, nullable=False, default=False)
    match_source = Column(String(16), nullable=True)  # "face" | "body" | "new" | "none"

    camera_id = Column(Text, nullable=True)
    frame_path = Column(Text, nullable=True)
    bbox = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow)

    __table_args__ = (
        Index("idx_de_visitor", visitor_id, detected_at.desc()),
        Index("idx_de_datetime", detected_at.desc()),
    )
