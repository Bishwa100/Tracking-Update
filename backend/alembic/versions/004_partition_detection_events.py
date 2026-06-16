"""Monthly partitioning of detection_events (range on detected_at).

This migration:
1. Renames the existing monolithic table to detection_events_legacy.
2. Creates a new partitioned parent table detection_events.
3. Copies rows from legacy into the partitioned table.
4. Drops the legacy table.
5. Creates partitions for the current and next 3 months.

Revision ID: 004
Revises: 003
Create Date: 2026-06-16
"""

from datetime import date, timedelta

from alembic import op
import sqlalchemy as sa

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def _months(start: date, n: int):
    """Yield (year, month) for n consecutive months starting at start."""
    y, m = start.year, start.month
    for _ in range(n):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _partition_name(year: int, month: int) -> str:
    return f"detection_events_{year}_{month:02d}"


def _bounds(year: int, month: int) -> tuple[str, str]:
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def upgrade() -> None:
    conn = op.get_bind()

    # Step 1 — rename existing table
    conn.execute(sa.text("ALTER TABLE detection_events RENAME TO detection_events_legacy"))

    # Step 2 — create partitioned parent (no data, no indexes)
    conn.execute(sa.text("""
        CREATE TABLE detection_events (
            id UUID NOT NULL DEFAULT gen_random_uuid(),
            visitor_id UUID REFERENCES visitors(id) ON DELETE SET NULL,
            visit_id UUID REFERENCES visits(id) ON DELETE SET NULL,
            detected_at TIMESTAMPTZ NOT NULL,
            face_similarity FLOAT,
            body_similarity FLOAT,
            is_new_visitor BOOLEAN DEFAULT FALSE,
            is_ambiguous BOOLEAN DEFAULT FALSE,
            match_source VARCHAR(20),
            camera_id VARCHAR(50),
            frame_path VARCHAR(500),
            bbox JSONB
        ) PARTITION BY RANGE (detected_at)
    """))

    # Step 3 — create partitions for current + next 3 months
    today = date.today()
    for year, month in _months(today, 4):
        lo, hi = _bounds(year, month)
        pname = _partition_name(year, month)
        conn.execute(sa.text(f"""
            CREATE TABLE {pname}
            PARTITION OF detection_events
            FOR VALUES FROM ('{lo}') TO ('{hi}')
        """))

    # Step 4 — create a default partition for rows outside defined ranges
    conn.execute(sa.text("""
        CREATE TABLE detection_events_default
        PARTITION OF detection_events DEFAULT
    """))

    # Step 5 — migrate rows
    conn.execute(sa.text("""
        INSERT INTO detection_events
        SELECT * FROM detection_events_legacy
    """))

    # Step 6 — recreate indexes on the parent (inherited by child partitions)
    conn.execute(sa.text("CREATE INDEX ix_de_detected_at ON detection_events (detected_at)"))
    conn.execute(sa.text("CREATE INDEX ix_de_visitor_id ON detection_events (visitor_id)"))

    # Step 7 — drop legacy
    conn.execute(sa.text("DROP TABLE detection_events_legacy"))


def downgrade() -> None:
    conn = op.get_bind()
    # Collect all partitions
    result = conn.execute(sa.text("""
        SELECT inhrelid::regclass AS name
        FROM pg_inherits
        WHERE inhparent = 'detection_events'::regclass
    """))
    partitions = [r.name for r in result]

    # Detach and drop each partition, then recreate monolithic table
    conn.execute(sa.text("""
        CREATE TABLE detection_events_legacy AS
        SELECT * FROM detection_events
    """))
    conn.execute(sa.text("DROP TABLE detection_events CASCADE"))
    conn.execute(sa.text("ALTER TABLE detection_events_legacy RENAME TO detection_events"))
