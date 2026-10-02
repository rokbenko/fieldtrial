"""Database tables (docs/PLAN.md section 9), as SQLAlchemy 2 typed ORM models."""

from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.engine import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from fieldtrial.store.ids import uuid7


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC datetimes. SQLite stores them naive; this restores the zone."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Store as naive UTC."""
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        """Load as aware UTC."""
        return None if value is None else value.replace(tzinfo=UTC)


def utcnow() -> datetime:
    """The current time, timezone-aware in UTC."""
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Base class of all tables."""

    type_annotation_map: ClassVar[dict[Any, Any]] = {
        dict[str, Any]: JSON,
        list[Any]: JSON,
        datetime: UTCDateTime,
    }


def _id() -> Mapped[str]:
    return mapped_column(String(36), primary_key=True, default=uuid7)


class Study(Base):
    """One pre-registered study."""

    __tablename__ = "study"

    id: Mapped[str] = _id()
    name: Mapped[str] = mapped_column(String(64))
    title: Mapped[str | None] = mapped_column(Text)
    design_yaml: Mapped[str] = mapped_column(Text)
    design_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))  # draft | locked | running | closed
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    locked_at: Mapped[datetime | None]
    unblinded_at: Mapped[datetime | None]
    closed_at: Mapped[datetime | None]
    fieldtrial_version: Mapped[str] = mapped_column(String(64))


class Arm(Base):
    """An arm of the study, with its blind code."""

    __tablename__ = "arm"
    __table_args__ = (
        UniqueConstraint("study_id", "key"),
        UniqueConstraint("study_id", "blind_code"),
    )

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    key: Mapped[str] = mapped_column(String(64))
    blind_code: Mapped[str] = mapped_column(String(8))
    label: Mapped[str | None] = mapped_column(Text)
    runner: Mapped[str] = mapped_column(String(32))
    policy_ref: Mapped[dict[str, Any]]
    serving: Mapped[dict[str, Any]]
    policy_fingerprint: Mapped[str | None] = mapped_column(String(128))


class Condition(Base):
    """An initial setup the robot is tested in."""

    __tablename__ = "condition"
    __table_args__ = (UniqueConstraint("study_id", "key"),)

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    key: Mapped[str] = mapped_column(String(512))
    factors: Mapped[dict[str, Any]]
    position: Mapped[int]


class ScheduleSlot(Base):
    """One planned (or rescheduled) trial in the run order."""

    __tablename__ = "schedule_slot"
    __table_args__ = (Index("ix_slot_study_seq", "study_id", "seq"),)

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    seq: Mapped[int]
    block: Mapped[int]
    replicate: Mapped[int]
    position: Mapped[int]
    condition_id: Mapped[str] = mapped_column(ForeignKey("condition.id"))
    arm_id: Mapped[str] = mapped_column(ForeignKey("arm.id"))
    origin: Mapped[str] = mapped_column(String(16))  # planned | rescheduled
    status: Mapped[str] = mapped_column(String(16))  # pending | done | void


class Session(Base):
    """One continuous working period by one operator on one rig."""

    __tablename__ = "session"

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    operator: Mapped[str] = mapped_column(String(128))
    rig: Mapped[str] = mapped_column(String(128))
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    ended_at: Mapped[datetime | None]
    rig_check: Mapped[dict[str, Any]] = mapped_column(default=dict)
    environment: Mapped[dict[str, Any]] = mapped_column(default=dict)
    software: Mapped[dict[str, Any]] = mapped_column(default=dict)
    notes: Mapped[str | None] = mapped_column(Text)


class Trial(Base):
    """One rollout attempt for a schedule slot."""

    __tablename__ = "trial"
    __table_args__ = (Index("ix_trial_study", "study_id"),)

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    slot_id: Mapped[str] = mapped_column(ForeignKey("schedule_slot.id"))
    session_id: Mapped[str] = mapped_column(ForeignKey("session.id"))
    attempt: Mapped[int]
    status: Mapped[str] = mapped_column(String(16))  # running | completed | invalid
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    ended_at: Mapped[datetime | None]
    duration_s: Mapped[float | None]
    stage_index: Mapped[int | None]  # -1 = no stage reached
    success: Mapped[bool | None]
    termination: Mapped[str | None] = mapped_column(String(32))
    failure_tags: Mapped[list[Any]] = mapped_column(default=list)
    notes: Mapped[str | None] = mapped_column(Text)
    invalid_reason: Mapped[str | None] = mapped_column(Text)
    media: Mapped[list[Any]] = mapped_column(default=list)
    auto_labels: Mapped[dict[str, Any]] = mapped_column(default=dict)
    version: Mapped[int] = mapped_column(default=1)


class Event(Base):
    """Append-only audit log entry."""

    __tablename__ = "event"
    __table_args__ = (Index("ix_event_study_ts", "study_id", "ts"),)

    id: Mapped[str] = _id()
    study_id: Mapped[str] = mapped_column(ForeignKey("study.id"))
    ts: Mapped[datetime] = mapped_column(default=utcnow)
    kind: Mapped[str] = mapped_column(String(64))
    actor: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(default=dict)
