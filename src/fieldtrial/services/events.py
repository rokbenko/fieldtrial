"""The append-only event log. Every service write adds exactly one event in its transaction.

Writes that a client may retry (a double tap, a lost response) take an ``idempotency_key``.
The key is stored on the write's event together with the result; a repeat with the same key
returns that stored result instead of writing again.
"""

from collections.abc import Callable
from typing import Any, TypeVar

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from fieldtrial.services._context import ServiceError, StudyContext
from fieldtrial.store import models as m

R = TypeVar("R")
MAX_KEY_LENGTH = 64


def append_event(
    db: Session,
    study_id: str,
    kind: str,
    actor: str,
    payload: dict[str, Any] | None = None,
    *,
    idempotency_key: str | None = None,
) -> m.Event:
    """Add an event to the current transaction."""
    event = m.Event(
        study_id=study_id,
        kind=kind,
        actor=actor,
        payload=payload or {},
        idempotency_key=idempotency_key,
    )
    db.add(event)
    return event


def list_events(db: Session, study_id: str, kind: str | None = None) -> list[m.Event]:
    """Events of a study in time order, optionally of one kind."""
    query = select(m.Event).where(m.Event.study_id == study_id)
    if kind is not None:
        query = query.where(m.Event.kind == kind)
    return list(db.scalars(query.order_by(m.Event.ts, m.Event.id)))


def events_after(ctx: StudyContext, last_id: str | None, *, limit: int = 100) -> list[m.Event]:
    """Events newer than ``last_id`` (ids are UUIDv7, so they sort by time)."""
    with ctx.db() as db:
        query = select(m.Event).where(m.Event.study_id == ctx.study_id)
        if last_id:
            query = query.where(m.Event.id > last_id)
        return list(db.scalars(query.order_by(m.Event.id).limit(limit)))


def latest_event_id(ctx: StudyContext) -> str | None:
    """Id of the newest event, or None."""
    with ctx.db() as db:
        return db.scalar(
            select(m.Event.id)
            .where(m.Event.study_id == ctx.study_id)
            .order_by(m.Event.id.desc())
            .limit(1)
        )


def _stored(ctx: StudyContext, key: str) -> m.Event | None:
    with ctx.db() as db:
        return db.scalar(
            select(m.Event).where(m.Event.study_id == ctx.study_id, m.Event.idempotency_key == key)
        )


def _replay(event: m.Event, kind: str, decode: Callable[[dict[str, Any]], R]) -> R:
    if event.kind != kind:
        raise ServiceError(f"idempotency key already used for a different request ({event.kind})")
    return decode(event.payload.get("result", {}))


def idempotent(
    ctx: StudyContext,
    key: str | None,
    kind: str,
    decode: Callable[[dict[str, Any]], R],
    write: Callable[[], R],
) -> R:
    """Run ``write`` once per ``key``; a repeat returns the stored result via ``decode``.

    ``write`` must store its result under ``payload["result"]`` of the ``kind`` event it
    appends with this key. Without a key, ``write`` simply runs.
    """
    if key is None:
        return write()
    if not 1 <= len(key) <= MAX_KEY_LENGTH:
        raise ServiceError(f"an idempotency key must have 1 to {MAX_KEY_LENGTH} characters")
    previous = _stored(ctx, key)
    if previous is not None:
        return _replay(previous, kind, decode)
    try:
        return write()
    except IntegrityError:
        # A concurrent request with the same key committed first.
        previous = _stored(ctx, key)
        if previous is None:
            raise
        return _replay(previous, kind, decode)
