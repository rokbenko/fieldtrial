"""Alembic environment. Migrations run programmatically from :func:`fieldtrial.store.db.open_database`."""

from alembic import context

from fieldtrial.store.models import Base

connection = context.config.attributes["connection"]
context.configure(
    connection=connection,
    target_metadata=Base.metadata,
    render_as_batch=True,  # SQLite cannot ALTER most things; batch mode recreates tables
)
with context.begin_transaction():
    context.run_migrations()
