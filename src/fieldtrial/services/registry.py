"""Long-lived access to the studies under a served folder (for the web console and API).

Opening a study runs the database migrations, which is too slow to repeat on every request.
The registry keeps one engine per study and reloads the design when its hash changes, for
example after ``fieldtrial amend`` in another terminal.
"""

import threading
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select

from fieldtrial.design import parse_study
from fieldtrial.services._context import ServiceError, StudyContext, resolve_folder
from fieldtrial.services.study import StudyEntry, list_studies
from fieldtrial.store import models as m
from fieldtrial.store.db import DB_FILENAME, open_database, session_factory


class UnknownStudyError(ServiceError):
    """No study with this slug under the served folder."""


@dataclass
class _Open:
    context: StudyContext
    design_hash: str


class StudyRegistry:
    """The studies at ``root``: one study folder, or a folder of study folders."""

    def __init__(self, root: str | Path) -> None:
        self.root = resolve_folder(root)
        if not self.root.is_dir():
            raise ServiceError(f"{self.root} is not a folder")
        self.single = (self.root / "study.yaml").exists()
        self._open: dict[str, _Open] = {}
        self._lock = threading.Lock()

    def entries(self) -> list[StudyEntry]:
        """All studies, in folder order."""
        return list_studies(self.root)

    def folder(self, slug: str) -> Path:
        """The folder of the study with this slug (its folder name)."""
        if self.single:
            if slug != self.root.name:
                raise UnknownStudyError(f"no study {slug!r}")
            return self.root
        candidate = self.root / slug
        if (
            "/" in slug
            or "\\" in slug
            or slug in ("", ".", "..")
            or not (candidate / "study.yaml").exists()
        ):
            raise UnknownStudyError(f"no study {slug!r}")
        return candidate

    def get(self, slug: str) -> StudyContext:
        """The open, locked study with this slug."""
        folder = self.folder(slug)
        if not (folder / DB_FILENAME).exists():
            raise ServiceError(f"study {slug!r} is not locked yet; run `fieldtrial lock {folder}`")
        with self._lock:
            entry = self._open.get(slug)
            if entry is None:
                engine = open_database(folder / DB_FILENAME)
                entry = self._load(folder, engine)
                self._open[slug] = entry
                return entry.context
            with entry.context.db() as db:
                current = db.scalars(select(m.Study.design_hash)).one()
            if current != entry.design_hash:
                entry = self._load(folder, entry.context.engine)
                self._open[slug] = entry
            return entry.context

    @staticmethod
    def _load(folder: Path, engine: Engine) -> _Open:
        sessions = session_factory(engine)
        with sessions() as db:
            study = db.scalars(select(m.Study)).one()
            spec = parse_study(study.design_yaml).spec
            context = StudyContext(folder, study.id, spec, engine, sessions)
            return _Open(context, study.design_hash)

    def close(self) -> None:
        """Release every database connection."""
        with self._lock:
            for entry in self._open.values():
                entry.context.engine.dispose()
            self._open.clear()
