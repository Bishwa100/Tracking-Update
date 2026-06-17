"""Human review queue table.

Revision ID: 006
Revises: 005
Create Date: 2026-06-16
"""

from alembic import op
import sqlalchemy as sa

revision = "006_review_queue"
down_revision = "005_runtime_settings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "review_queue",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("visitor_id", sa.UUID(), sa.ForeignKey("visitors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("flag_type", sa.String(50), nullable=False),
        sa.Column("detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
        sa.Column("resolved", sa.Boolean(), server_default="FALSE", nullable=False),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(100), nullable=True),
        sa.UniqueConstraint("visitor_id", "flag_type", name="uq_review_queue_visitor_flag"),
    )
    op.create_index("ix_review_queue_resolved", "review_queue", ["resolved"])
    op.create_index("ix_review_queue_created_at", "review_queue", ["created_at"])


def downgrade() -> None:
    op.drop_table("review_queue")
