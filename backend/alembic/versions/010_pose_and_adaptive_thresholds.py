"""Continuous pose + source camera on gallery faces, and per-visitor thresholds.

Phase 3 (multi-angle identity):
  • visitor_faces gains yaw/pitch/roll (continuous head pose, previously computed
    then discarded) + source_camera_id (which camera captured the gallery face).
  • visitors gains per-visitor adaptive threshold stats so visitors with high
    within-person embedding variance can match at a lower bar without loosening
    the global threshold.

Revision ID: 010
Revises: 009
Create Date: 2026-06-18
"""

from alembic import op
import sqlalchemy as sa


revision = "010_pose_and_adaptive_thresholds"
down_revision = "009_face_crop_clarity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("visitor_faces", sa.Column("yaw", sa.Float(), nullable=True))
    op.add_column("visitor_faces", sa.Column("pitch", sa.Float(), nullable=True))
    op.add_column("visitor_faces", sa.Column("roll", sa.Float(), nullable=True))
    op.add_column(
        "visitor_faces", sa.Column("source_camera_id", sa.Text(), nullable=True)
    )

    op.add_column(
        "visitors", sa.Column("expected_match_similarity", sa.Float(), nullable=True)
    )
    op.add_column(
        "visitors", sa.Column("match_similarity_std", sa.Float(), nullable=True)
    )
    op.add_column(
        "visitors",
        sa.Column("personal_returning_threshold", sa.Float(), nullable=True),
    )
    op.add_column(
        "visitors", sa.Column("personal_new_threshold", sa.Float(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("visitors", "personal_new_threshold")
    op.drop_column("visitors", "personal_returning_threshold")
    op.drop_column("visitors", "match_similarity_std")
    op.drop_column("visitors", "expected_match_similarity")

    op.drop_column("visitor_faces", "source_camera_id")
    op.drop_column("visitor_faces", "roll")
    op.drop_column("visitor_faces", "pitch")
    op.drop_column("visitor_faces", "yaw")
