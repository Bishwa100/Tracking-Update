"""Add pose_bin to visitor_faces and consent fields to visitors.

Revision ID: 002
Revises: 001
Create Date: 2026-06-16
"""

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("visitor_faces", sa.Column("pose_bin", sa.String(20), nullable=True, server_default="unknown"))
    op.add_column("visitors", sa.Column("consent_status", sa.String(20), nullable=True, server_default="implicit"))
    op.add_column("visitors", sa.Column("consent_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("visitors", sa.Column("consent_method", sa.String(50), nullable=True))
    op.add_column("visitors", sa.Column("opted_out_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("visitors", sa.Column("visit_confidence", sa.Float(), nullable=True, server_default="0.3"))
    # Index for fast opted-out exclusion in HNSW queries
    op.create_index("ix_visitors_consent_status", "visitors", ["consent_status"])


def downgrade() -> None:
    op.drop_index("ix_visitors_consent_status", table_name="visitors")
    op.drop_column("visitors", "visit_confidence")
    op.drop_column("visitors", "opted_out_at")
    op.drop_column("visitors", "consent_method")
    op.drop_column("visitors", "consent_at")
    op.drop_column("visitors", "consent_status")
    op.drop_column("visitor_faces", "pose_bin")
