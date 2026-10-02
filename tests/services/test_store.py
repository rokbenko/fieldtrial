"""Storage layer: ids, migrations and SQLite settings."""

import re
from pathlib import Path

from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text

from fieldtrial.store.db import alembic_config, open_database
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
    assert version == ScriptDirectory.from_config(alembic_config()).get_current_head()


def test_upgrade_from_baseline_keeps_data(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    engine = create_engine(f"sqlite:///{path}")
    config = alembic_config()
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.upgrade(config, "0001_baseline")
        conn.execute(
            text(
                "INSERT INTO study (id, name, design_yaml, design_hash, status, created_at, "
                "fieldtrial_version) VALUES ('s', 'n', '', 'h', 'locked', '2026-01-01', '0')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO event (id, study_id, ts, kind, actor, payload) "
                "VALUES ('e', 's', '2026-01-01', 'design_locked', 'a', '{}')"
            )
        )
    engine.dispose()
    engine = open_database(path)
    with engine.connect() as conn:
        row = conn.execute(text("SELECT kind, idempotency_key FROM event")).one()
    assert tuple(row) == ("design_locked", None)
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.downgrade(config, "0001_baseline")
    engine.dispose()
