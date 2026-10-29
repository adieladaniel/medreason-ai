"""
Database tables. Three of them, each answering one question:

  cases       what was submitted, where the job is, and the result of each stage so far
  events      what happened and how long it took (drives the live progress view and is the
              audit trail of a run)
  llm_calls   exactly what the local language model was asked and what it said, with timing

The result of each pipeline stage is stored as soon as it exists, in its own column, so a client
can read the prediction while the slow explanations are still being computed.

Types are portable: JSON is JSONB on Postgres and plain JSON on SQLite, so the same models run in
both. Timestamps are timezone-aware UTC.

Changing a table here also needs a migration: `alembic revision --autogenerate -m "what changed"`, then read the file it
writes (migrations/versions/). tests/test_migrations.py fails while the models and the migrations disagree.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

PortableJSON = JSON().with_variant(JSONB(), "postgresql")

# Order matters: the progress view shows these as steps.
STAGES = ["queued", "predicting", "explaining", "retrieving", "writing", "verifying", "done"]
# "partial" means the prediction exists but a later stage (explanations, evidence, the written note) failed,
# so a client can show what there is and say what is missing. "failed" means there is no prediction.
STATUSES = ["queued", "running", "done", "partial", "failed"]


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(32), default="queued")
    error: Mapped[str | None] = mapped_column(Text, default=None)

    # inputs, each optional, at least one required (checked by the service, not the schema)
    indication: Mapped[str | None] = mapped_column(Text, default=None)
    labs: Mapped[dict | None] = mapped_column(PortableJSON, default=None)
    image_path: Mapped[str | None] = mapped_column(String(512), default=None)
    image_sha256: Mapped[str | None] = mapped_column(String(64), default=None)

    # stage results
    prediction: Mapped[dict | None] = mapped_column(PortableJSON, default=None)    # belief, consistency, plan
    explanations: Mapped[dict | None] = mapped_column(PortableJSON, default=None)  # per-modality attributions
    evidence: Mapped[dict | None] = mapped_column(PortableJSON, default=None)      # queries, passages, text
    timings: Mapped[dict] = mapped_column(PortableJSON, default=dict)               # stage -> seconds

    events: Mapped[list["Event"]] = relationship(back_populates="case", cascade="all, delete-orphan",
                                                 order_by="Event.id", passive_deletes=True)
    llm_calls: Mapped[list["LlmCall"]] = relationship(back_populates="case", cascade="all, delete-orphan",
                                                      order_by="LlmCall.id", passive_deletes=True)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (Index("ix_events_case_id_id", "case_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))          # started, finished, failed
    seconds: Mapped[float | None] = mapped_column(Float, default=None)
    detail: Mapped[dict | None] = mapped_column(PortableJSON, default=None)

    case: Mapped[Case] = relationship(back_populates="events")


class LlmCall(Base):
    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    model: Mapped[str] = mapped_column(String(64))
    purpose: Mapped[str] = mapped_column(String(64))
    prompt: Mapped[str] = mapped_column(Text)
    response: Mapped[str] = mapped_column(Text)
    seconds: Mapped[float] = mapped_column(Float)

    case: Mapped[Case] = relationship(back_populates="llm_calls")
