"""baseline: cases, events, llm_calls

The three tables the service has used since Phase 7 step 1, written out in full. This file is a snapshot and must never import from
app.database.models: the models change over time, and a baseline that follows them would stop describing the database it created.
A database made before Alembic existed (by Base.metadata.create_all) already has exactly these tables; app/database/migrate.py
checks that and stamps it with this revision instead of running it.

Revision ID: 0001
Revises:
Create Date: 2026-10-29
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

# JSONB on Postgres (indexable, compact), plain JSON on SQLite. The same variant the models use.
JSON = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "cases",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("indication", sa.Text(), nullable=True),
        sa.Column("labs", JSON, nullable=True),
        sa.Column("image_path", sa.String(length=512), nullable=True),
        sa.Column("image_sha256", sa.String(length=64), nullable=True),
        sa.Column("prediction", JSON, nullable=True),
        sa.Column("explanations", JSON, nullable=True),
        sa.Column("evidence", JSON, nullable=True),
        sa.Column("timings", JSON, nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_cases_created_at", "cases", ["created_at"])
    op.create_index("ix_cases_status", "cases", ["status"])

    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=True),
        sa.Column("detail", JSON, nullable=True),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_events_case_id_id", "events", ["case_id", "id"])

    op.create_table(
        "llm_calls",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("case_id", sa.String(length=32), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=False),
        sa.Column("purpose", sa.String(length=64), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("seconds", sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(["case_id"], ["cases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_calls_case_id", "llm_calls", ["case_id"])


def downgrade() -> None:
    op.drop_index("ix_llm_calls_case_id", table_name="llm_calls")
    op.drop_table("llm_calls")
    op.drop_index("ix_events_case_id_id", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_cases_status", table_name="cases")
    op.drop_index("ix_cases_created_at", table_name="cases")
    op.drop_table("cases")
