"""Runtime settings table for persisted threshold changes.

Revision ID: 005
Revises: 004
Create Date: 2026-06-16
"""

from alembic import op
import sqlalchemy as sa

revision = "005_runtime_settings"
down_revision = "004_partition_detection_events"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "runtime_settings",
        sa.Column("key", sa.String(100), primary_key=True),
        sa.Column("value", sa.Text(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("updated_by", sa.String(100), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("runtime_settings")
