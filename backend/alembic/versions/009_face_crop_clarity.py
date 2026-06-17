"""Add per-face crop path + cached clarity score to visitor_faces.

Enables the operator-triggered "auto-clean faces" action: each gallery face now
keeps its own tight crop on disk so blur/sharpness can be measured, and a cached
clarity score (landmark frontality + blur + det_score) drives pruning.

Revision ID: 009
Revises: 008
Create Date: 2026-06-17
"""

from alembic import op
import sqlalchemy as sa


revision = "009_face_crop_clarity"
down_revision = "008_review_queue_match"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "visitor_faces",
        sa.Column("crop_path", sa.Text(), nullable=True),
    )
    op.add_column(
        "visitor_faces",
        sa.Column("clarity_score", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("visitor_faces", "clarity_score")
    op.drop_column("visitor_faces", "crop_path")
