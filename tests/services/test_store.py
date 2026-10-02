"""Storage layer: ids, migrations and SQLite settings."""

import re
from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import text

from fieldtrial.store.db import open_database
from fieldtrial.store.ids import uuid7
from fieldtrial.store.models import Base

_UUID7 = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def test_uuid7_format_and_order() -> None:
    ids = [uuid7() for _ in range(200)]
    assert all(_UUID7.match(i) for i in ids)
    assert len(set(ids)) == len(ids)
    # The 48-bit millisecond timestamp prefix never goes backwards.
    stamps = [i.replace("-", "")[:12] for i in ids]
    assert stamps == sorted(stamps)


def test_migrations_match_models(tmp_path: Path) -> None:
    engine = open_database(tmp_path / "x.db", create=True)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert diff == []


def test_sqlite_settings(tmp_path: Path) -> None:
    engine = open_database(tmp_path / "x.db", create=True)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar() == 1
    engine.dispose()


def test_reopen_is_idempotent(tmp_path: Path) -> None:
    open_database(tmp_path / "x.db", create=True).dispose()
    engine = open_database(tmp_path / "x.db")
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version_num FROM alembic_version")).scalar()
    engine.dispose()
    assert version == "0001_baseline"
