"""Idempotency keys on events, so retried requests are applied once.

Revision ID: 0002_idempotency
"""

import sqlalchemy as sa
from alembic import op

revision = "0002_idempotency"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.add_column(sa.Column("idempotency_key", sa.String(length=64), nullable=True))
        batch_op.create_index("ix_event_idempotency", ["study_id", "idempotency_key"], unique=True)


def downgrade() -> None:
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.drop_index("ix_event_idempotency")
        batch_op.drop_column("idempotency_key")
