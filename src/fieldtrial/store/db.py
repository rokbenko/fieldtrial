"""Opening a study database: SQLite in WAL mode with foreign keys, migrated to the latest schema."""

from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

DB_FILENAME = "fieldtrial.db"
_MIGRATIONS = Path(__file__).parent / "migrations"


def _configure_sqlite(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def alembic_config() -> Config:
    """Alembic configuration pointing at the packaged migrations."""
    config = Config()
    config.set_main_option("script_location", str(_MIGRATIONS))
    return config


def open_database(path: str | Path, *, create: bool = False) -> Engine:
    """Open (or with ``create=True``, create) a study database, migrated to the latest schema."""
    db_path = Path(path)
    if not create and not db_path.exists():
        raise FileNotFoundError(f"{db_path} does not exist; lock the study first")
    engine = create_engine(f"sqlite:///{db_path}")
    event.listen(engine, "connect", _configure_sqlite)
    config = alembic_config()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    return engine


def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Sessions that keep loaded objects usable after commit."""
    return sessionmaker(engine, expire_on_commit=False)
