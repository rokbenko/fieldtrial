"""Opening a study folder for service calls."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session, sessionmaker

from fieldtrial.design import StudySpec, parse_study
from fieldtrial.store import models as m
from fieldtrial.store.db import DB_FILENAME, open_database, session_factory


class ServiceError(RuntimeError):
    """A request that cannot be carried out in the study's current state."""


class ConcurrencyError(ServiceError):
    """The record changed since it was read (optimistic concurrency check failed)."""


@dataclass
class StudyContext:
    """An open, locked study: its folder, locked design and database."""

    folder: Path
    study_id: str
    spec: StudySpec
    engine: Engine
    sessions: sessionmaker[Session]

    def db(self) -> Session:
        """A new database session (use with ``with ... begin()``)."""
        return self.sessions()


def resolve_folder(folder: str | Path) -> Path:
    """The study folder for a path to a folder or to its ``study.yaml``."""
    p = Path(folder)
    return p.parent if p.name == "study.yaml" else p


@contextmanager
def open_study(folder: str | Path) -> Iterator[StudyContext]:
    """Open a locked study. The design comes from the database, not the (editable) YAML."""
    root = resolve_folder(folder)
    db_path = root / DB_FILENAME
    if not db_path.exists():
        raise ServiceError(f"{root} is not locked yet; run `fieldtrial lock {root}` first")
    engine = open_database(db_path)
    try:
        sessions = session_factory(engine)
        with sessions() as db:
            study = db.scalars(select(m.Study)).one()
            spec = parse_study(study.design_yaml).spec
            study_id = study.id
        yield StudyContext(root, study_id, spec, engine, sessions)
    finally:
        engine.dispose()
