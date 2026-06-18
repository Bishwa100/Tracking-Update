"""Cross-camera identity: camera topology + visitor merge audit.

Phase 4:
  • camera_topology — pairwise transition constraints between cameras
    (min/max travel seconds) so impossible cross-camera matches are rejected and
    plausible ones prioritised.
  • visitor_merge_audit — append-only record of every merge (manual, auto-dedup,
    cross-camera) for traceability and rollback investigation.

Revision ID: 011
Revises: 010
Create Date: 2026-06-18
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


revision = "011_cross_camera"
down_revision = "010_pose_and_adaptive_thresholds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "camera_topology",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_a", sa.Text(), nullable=False),
        sa.Column("camera_b", sa.Text(), nullable=False),
        sa.Column("min_travel_seconds", sa.Float(), nullable=True),
        sa.Column("max_expected_seconds", sa.Float(), nullable=True),
        sa.Column("transition_enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_camera_topology_pair", "camera_topology", ["camera_a", "camera_b"], unique=True
    )

    op.create_table(
        "visitor_merge_audit",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("source_visitor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("target_visitor_id", UUID(as_uuid=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("similarity", sa.Float(), nullable=True),
        sa.Column("merged_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "idx_merge_audit_target", "visitor_merge_audit", ["target_visitor_id"]
    )


def downgrade() -> None:
    op.drop_index("idx_merge_audit_target", table_name="visitor_merge_audit")
    op.drop_table("visitor_merge_audit")
    op.drop_index("idx_camera_topology_pair", table_name="camera_topology")
    op.drop_table("camera_topology")
