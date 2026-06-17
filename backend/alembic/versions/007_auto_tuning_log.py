"""Auto-tuning log table.

Revision ID: 007
Revises: 006
Create Date: 2026-06-16
"""

from alembic import op
import sqlalchemy as sa

revision = "007_auto_tuning_log"
down_revision = "006_review_queue"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "auto_tuning_log",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("tuned_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("setting_key", sa.String(100), nullable=False),
        sa.Column("old_value", sa.Float(), nullable=False),
        sa.Column("new_value", sa.Float(), nullable=False),
        sa.Column("false_new_rate", sa.Float(), nullable=True),
        sa.Column("detail", sa.Text(), nullable=True),  # JSON summary
    )
    op.create_index("ix_auto_tuning_log_tuned_at", "auto_tuning_log", ["tuned_at"])


def downgrade() -> None:
    op.drop_table("auto_tuning_log")
