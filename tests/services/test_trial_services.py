"""Running trials: sessions, outcomes, invalidation and rescheduling, concurrency, events."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from fieldtrial.services import ConcurrencyError, ServiceError, open_study
from fieldtrial.services.session import end_session, start_session
from fieldtrial.services.trial import (
    collect_records,
    complete_trial,
    invalidate_trial,
    next_slot,
    pending_slots,
    start_trial,
)
from fieldtrial.store import models as m


def _events(folder: Path) -> list[str]:
    with open_study(folder) as ctx, ctx.db() as db:
        return list(db.scalars(select(m.Event.kind).order_by(m.Event.ts, m.Event.id)))


def test_run_one_trial(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        slot = next_slot(ctx)
        assert slot is not None
        assert slot.seq == 1
        assert slot.total == 8
        trial = start_trial(ctx, slot.slot_id, session)
        assert trial.attempt == 1
        assert trial.version == 1
        # A running slot is skipped by next_slot.
        following = next_slot(ctx)
        assert following is not None
        assert following.seq == 2
        with pytest.raises(ServiceError, match="already has a running trial"):
            start_trial(ctx, slot.slot_id, session)
        done = complete_trial(
            ctx, trial.trial_id, stage_index=3, termination="success", duration_s=11.5
        )
        assert done.status == "completed"
        assert done.version == 2
        records, info = collect_records(ctx)
        assert info.status == "running"
        assert info.pending_slots == 7
        end_session(ctx, session)
        with pytest.raises(ServiceError, match="already ended"):
            end_session(ctx, session)
        with pytest.raises(ServiceError, match="has ended"):
            start_trial(ctx, following.slot_id, session)
    assert len(records) == 1
    assert records[0].success is True
    assert records[0].duration_s == 11.5
    assert records[0].operator == "ana"
    assert _events(locked_study) == [
        "design_locked",
        "session_started",
        "trial_started",
        "trial_completed",
        "session_ended",
    ]


def test_success_is_reaching_success_stage(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        complete_trial(
            ctx, trial.trial_id, stage_index=2, termination="stuck", failure_tags=["dropped"]
        )
        records, _ = collect_records(ctx)
    assert records[0].success is False
    assert records[0].failure_tags == ("dropped",)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stage_index": 4, "termination": "success"}, "stage_index"),
        ({"stage_index": -2, "termination": "stuck"}, "stage_index"),
        ({"stage_index": 0, "termination": "bored"}, "termination"),
        ({"stage_index": 0, "termination": "stuck", "failure_tags": ["x"]}, "failure tags"),
    ],
)
def test_bad_outcomes_rejected(locked_study: Path, kwargs: dict[str, object], message: str) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        with pytest.raises(ServiceError, match=message):
            complete_trial(ctx, trial.trial_id, **kwargs)  # type: ignore[arg-type]


def test_session_needs_operator(locked_study: Path) -> None:
    with open_study(locked_study) as ctx, pytest.raises(ServiceError):
        start_session(ctx, operator=" ", rig="rig-1")


def test_invalidate_reschedules_at_end_of_block(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        before = pending_slots(ctx)
        first = before[0]
        block_size = sum(1 for s in before if s.block == first.block)
        session = start_session(ctx, operator="ana", rig="rig-1")
        trial = start_trial(ctx, first.slot_id, session)
        with pytest.raises(ServiceError, match="needs a reason"):
            invalidate_trial(ctx, trial.trial_id, "")
        invalid = invalidate_trial(ctx, trial.trial_id, "gripper fault")
        assert invalid.status == "invalid"
        after = pending_slots(ctx)
        # Same number of pending slots; the replacement sits right after the block.
        assert len(after) == len(before)
        replacement = next(s for s in after if s.seq == block_size + 1)
        assert (replacement.block, replacement.condition, replacement.arm) == (
            first.block,
            first.condition,
            first.arm,
        )
        # Every later slot moved back by one, keeping seq unique.
        seqs = [s.seq for s in after]
        assert len(set(seqs)) == len(seqs)
        assert max(seqs) == len(before) + 1
        # The retry is attempt 2 of a fresh slot; invalid trials are kept and reported.
        retry = start_trial(ctx, replacement.slot_id, session)
        assert retry.attempt == 1
        records, info = collect_records(ctx)
        assert [r.status for r in records] == ["invalid"]
        assert records[0].invalid_reason == "gripper fault"
        assert info.planned_slots == 8
        with pytest.raises(ServiceError, match="already invalid"):
            invalidate_trial(ctx, trial.trial_id, "again")


def test_invalidate_completed_trial_reschedules_at_end(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        complete_trial(ctx, trial.trial_id, stage_index=3, termination="success")
        invalidate_trial(ctx, trial.trial_id, "wrong part loaded", reschedule="end", actor="bo")
        last = pending_slots(ctx)[-1]
        assert last.seq == 9
        assert (last.condition, last.arm) == (slot.condition, slot.arm)


def test_optimistic_concurrency(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        complete_trial(
            ctx,
            trial.trial_id,
            stage_index=3,
            termination="success",
            expected_version=trial.version,
        )
        # A second client still holding version 1 must not overwrite the result.
        with pytest.raises(ConcurrencyError):
            invalidate_trial(ctx, trial.trial_id, "stale", expected_version=trial.version)
        with pytest.raises(ServiceError, match="not running"):
            complete_trial(ctx, trial.trial_id, stage_index=0, termination="stuck")


def test_every_write_logs_one_event(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        writes = 1
        for _ in range(3):
            slot = next_slot(ctx)
            assert slot is not None
            trial = start_trial(ctx, slot.slot_id, session)
            complete_trial(ctx, trial.trial_id, stage_index=0, termination="stuck")
            writes += 2
        with ctx.db() as db:
            events = db.scalar(select(func.count()).select_from(m.Event))
    assert events == writes + 1  # plus design_locked


def test_failed_write_logs_nothing(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="rig-1")
        with pytest.raises(ServiceError):
            start_trial(ctx, "no-such-slot", session)
    assert _events(locked_study) == ["design_locked", "session_started"]
