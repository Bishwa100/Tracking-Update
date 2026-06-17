"""Consent audit log table.

Revision ID: 003
Revises: 002
Create Date: 2026-06-16
"""

from alembic import op
import sqlalchemy as sa

revision = "003_consent_system"
down_revision = "002_pose_bin_consent"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "consent_log",
        sa.Column("id", sa.UUID(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("visitor_id", sa.UUID(), sa.ForeignKey("visitors.id", ondelete="CASCADE"), nullable=False),
        sa.Column("action", sa.String(30), nullable=False),   # granted / revoked / opted_out
        sa.Column("method", sa.String(50), nullable=True),    # implicit / qr_code / staff / api
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("user_agent", sa.String(255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()"), nullable=False),
    )
    op.create_index("ix_consent_log_visitor_id", "consent_log", ["visitor_id"])
    op.create_index("ix_consent_log_created_at", "consent_log", ["created_at"])


def downgrade() -> None:
    op.drop_table("consent_log")
