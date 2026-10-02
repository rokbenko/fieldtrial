"""Baseline schema (docs/PLAN.md section 9).

Revision ID: 0001_baseline
"""

import sqlalchemy as sa
from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "study",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("design_yaml", sa.Text(), nullable=False),
        sa.Column("design_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("locked_at", sa.DateTime(), nullable=True),
        sa.Column("unblinded_at", sa.DateTime(), nullable=True),
        sa.Column("closed_at", sa.DateTime(), nullable=True),
        sa.Column("fieldtrial_version", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "arm",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("blind_code", sa.String(length=8), nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("runner", sa.String(length=32), nullable=False),
        sa.Column("policy_ref", sa.JSON(), nullable=False),
        sa.Column("serving", sa.JSON(), nullable=False),
        sa.Column("policy_fingerprint", sa.String(length=128), nullable=True),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("study_id", "blind_code"),
        sa.UniqueConstraint("study_id", "key"),
    )
    op.create_table(
        "condition",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=512), nullable=False),
        sa.Column("factors", sa.JSON(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("study_id", "key"),
    )
    op.create_table(
        "event",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("ts", sa.DateTime(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("actor", sa.String(length=128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.create_index("ix_event_study_ts", ["study_id", "ts"], unique=False)

    op.create_table(
        "session",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("operator", sa.String(length=128), nullable=False),
        sa.Column("rig", sa.String(length=128), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("rig_check", sa.JSON(), nullable=False),
        sa.Column("environment", sa.JSON(), nullable=False),
        sa.Column("software", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "schedule_slot",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("block", sa.Integer(), nullable=False),
        sa.Column("replicate", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("condition_id", sa.String(length=36), nullable=False),
        sa.Column("arm_id", sa.String(length=36), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(
            ["arm_id"],
            ["arm.id"],
        ),
        sa.ForeignKeyConstraint(
            ["condition_id"],
            ["condition.id"],
        ),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("schedule_slot", schema=None) as batch_op:
        batch_op.create_index("ix_slot_study_seq", ["study_id", "seq"], unique=False)

    op.create_table(
        "trial",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("slot_id", sa.String(length=36), nullable=False),
        sa.Column("session_id", sa.String(length=36), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("ended_at", sa.DateTime(), nullable=True),
        sa.Column("duration_s", sa.Double(), nullable=True),
        sa.Column("stage_index", sa.Integer(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=True),
        sa.Column("termination", sa.String(length=32), nullable=True),
        sa.Column("failure_tags", sa.JSON(), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("invalid_reason", sa.Text(), nullable=True),
        sa.Column("media", sa.JSON(), nullable=False),
        sa.Column("auto_labels", sa.JSON(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["session_id"],
            ["session.id"],
        ),
        sa.ForeignKeyConstraint(
            ["slot_id"],
            ["schedule_slot.id"],
        ),
        sa.ForeignKeyConstraint(
            ["study_id"],
            ["study.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("trial", schema=None) as batch_op:
        batch_op.create_index("ix_trial_study", ["study_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("trial", schema=None) as batch_op:
        batch_op.drop_index("ix_trial_study")

    op.drop_table("trial")
    with op.batch_alter_table("schedule_slot", schema=None) as batch_op:
        batch_op.drop_index("ix_slot_study_seq")

    op.drop_table("schedule_slot")
    op.drop_table("session")
    with op.batch_alter_table("event", schema=None) as batch_op:
        batch_op.drop_index("ix_event_study_ts")

    op.drop_table("event")
    op.drop_table("condition")
    op.drop_table("arm")
    op.drop_table("study")
