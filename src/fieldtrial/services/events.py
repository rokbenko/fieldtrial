"""The append-only event log. Every service write adds exactly one event in its transaction."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from fieldtrial.store import models as m


def append_event(
    db: Session, study_id: str, kind: str, actor: str, payload: dict[str, Any] | None = None
) -> m.Event:
    """Add an event to the current transaction."""
    event = m.Event(study_id=study_id, kind=kind, actor=actor, payload=payload or {})
    db.add(event)
    return event


def list_events(db: Session, study_id: str, kind: str | None = None) -> list[m.Event]:
    """Events of a study in time order, optionally of one kind."""
    query = select(m.Event).where(m.Event.study_id == study_id)
    if kind is not None:
        query = query.where(m.Event.kind == kind)
    return list(db.scalars(query.order_by(m.Event.ts, m.Event.id)))
