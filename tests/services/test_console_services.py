"""Services the console and API need: idempotency, undo, edits, media, sessions, listings."""

from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select
from tests.services.conftest import write_small_study

from fieldtrial.services import ConcurrencyError, ServiceError, open_study
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.events import events_after, latest_event_id
from fieldtrial.services.session import (
    console_state,
    end_session,
    get_session,
    list_sessions,
    start_session,
)
from fieldtrial.services.study import is_blinded, list_studies, lock_study, unblind_study
from fieldtrial.services.trial import (
    attach_media,
    complete_trial,
    edit_trial,
    get_trial,
    invalidate_trial,
    list_trials,
    next_slot,
    pending_slots,
    reopen_trial,
    start_trial,
)
from fieldtrial.store import models as m
from fieldtrial.store.models import utcnow


def _event_count(folder: Path, kind: str | None = None) -> int:
    with open_study(folder) as ctx, ctx.db() as db:
        query = select(func.count()).select_from(m.Event)
        if kind:
            query = query.where(m.Event.kind == kind)
        return int(db.scalar(query) or 0)


def _run_one(ctx, session: str, stage: int = 3) -> str:  # type: ignore[no-untyped-def]
    slot = next_slot(ctx)
    assert slot is not None
    trial = start_trial(ctx, slot.slot_id, session)
    complete_trial(
        ctx, trial.trial_id, stage_index=stage, termination="success" if stage == 3 else "stuck"
    )
    return trial.trial_id


def test_idempotent_writes(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        s1 = start_session(ctx, operator="ana", rig="r", idempotency_key="k-session")
        s2 = start_session(ctx, operator="ana", rig="r", idempotency_key="k-session")
        assert s1 == s2
        slot = next_slot(ctx)
        assert slot is not None
        t1 = start_trial(ctx, slot.slot_id, s1, idempotency_key="k-start")
        t2 = start_trial(ctx, slot.slot_id, s1, idempotency_key="k-start")
        assert t1 == t2
        done = [
            complete_trial(
                ctx, t1.trial_id, stage_index=3, termination="success", idempotency_key="k-done"
            )
            for _ in range(3)
        ]
        assert done[0] == done[1] == done[2]
        assert done[0].version == 2
        with pytest.raises(ServiceError, match="different request"):
            end_session(ctx, s1, idempotency_key="k-done")
        with pytest.raises(ServiceError, match="1 to 64"):
            start_session(ctx, operator="a", rig="r", idempotency_key="x" * 65)
        end_session(ctx, s1, idempotency_key="k-end")
        end_session(ctx, s1, idempotency_key="k-end")
    assert _event_count(locked_study, "session_started") == 1
    assert _event_count(locked_study, "trial_started") == 1
    assert _event_count(locked_study, "trial_completed") == 1
    assert _event_count(locked_study, "session_ended") == 1


def test_undo_completed_trial(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="r")
        first = _run_one(ctx, session)
        before = console_state(ctx, session)
        assert before.undoable is not None
        assert before.undoable.trial_id == first
        assert before.done == 1
        view = reopen_trial(ctx, first, session_id=session)
        assert view.status == "running"
        state = console_state(ctx, session)
        assert state.running is not None
        assert state.running.trial_id == first
        assert state.running.stage_index is None
        assert state.up_next is None
        assert state.done == 0
        complete_trial(ctx, first, stage_index=1, termination="stuck")
        assert get_trial(ctx, first).success is False
        with pytest.raises(ServiceError, match="only the session"):
            reopen_trial(ctx, first, session_id="other")
        late = utcnow() + timedelta(seconds=11)
        with pytest.raises(ServiceError, match="10 s"):
            reopen_trial(ctx, first, now=late)
        assert console_state(ctx, session, now=late).undoable is None
        second = _run_one(ctx, session)
        with pytest.raises(ServiceError, match="latest trial"):
            reopen_trial(ctx, first)
        reopen_trial(ctx, second, window_s=None)
        with pytest.raises(ServiceError, match="nothing to undo"):
            reopen_trial(ctx, second)
    assert _event_count(locked_study, "trial_reopened") == 2


def test_undo_invalid_voids_replacement(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="r")
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        invalidate_trial(ctx, trial.trial_id, "gripper fault")
        assert len(pending_slots(ctx)) == 8
        reopen_trial(ctx, trial.trial_id)
        pending = pending_slots(ctx)
        assert len(pending) == 8
        assert pending[0].slot_id == slot.slot_id
        # Once the replacement has been run, the invalid trial can no longer be undone.
        invalidate_trial(ctx, trial.trial_id, "gripper fault again")
        replacement = next(
            s for s in pending_slots(ctx) if s.condition == slot.condition and s.arm == slot.arm
        )
        start_trial(ctx, replacement.slot_id, session)
        with pytest.raises(ServiceError):
            reopen_trial(ctx, trial.trial_id, window_s=None)


def test_edit_trial_logs_old_and_new(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="r")
        trial_id = _run_one(ctx, session)
        with pytest.raises(ServiceError, match="needs a reason"):
            edit_trial(ctx, trial_id, actor="bo", reason=" ", stage_index=1)
        with pytest.raises(ServiceError, match="nothing to change"):
            edit_trial(ctx, trial_id, actor="bo", reason="x", stage_index=3)
        view = edit_trial(
            ctx,
            trial_id,
            actor="bo",
            reason="video shows a tilted insertion",
            stage_index=2,
            termination="stuck",
            failure_tags=["dropped"],
            notes="re-labelled",
            expected_version=2,
        )
        assert view.version == 3
        detail = get_trial(ctx, trial_id)
        assert (detail.stage_index, detail.success, detail.failure_tags) == (2, False, ("dropped",))
        with pytest.raises(ConcurrencyError):
            edit_trial(ctx, trial_id, actor="bo", reason="x", notes="y", expected_version=2)
        with pytest.raises(ServiceError, match="unknown failure tags"):
            edit_trial(ctx, trial_id, actor="bo", reason="x", failure_tags=["nope"])
        with ctx.db() as db:
            event = db.scalars(select(m.Event).where(m.Event.kind == "trial_edited")).one()
        assert event.payload["old"]["stage_index"] == 3
        assert event.payload["new"]["stage_index"] == 2
        assert event.actor == "bo"


def test_edits_after_unblinding_are_deviations(tmp_path: Path) -> None:
    folder = write_small_study(tmp_path / "s")
    path = folder / "study.yaml"
    path.write_text(path.read_text().replace("blinding: operator", "blinding: none"))
    lock_study(folder)
    with open_study(folder) as ctx:
        assert not is_blinded(ctx)
        session = start_session(ctx, operator="ana", rig="r")
        trial_id = _run_one(ctx, session)
    unblind_study(folder)
    with open_study(folder) as ctx:
        edit_trial(ctx, trial_id, actor="bo", reason="relabel", stage_index=0)
    kinds = [d.kind for d in analyze_study(folder).deviations]
    assert "late_edit" in kinds


def test_attach_media(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        session = start_session(ctx, operator="ana", rig="r")
        trial_id = _run_one(ctx, session)
        first = attach_media(ctx, trial_id, "../../etc/clip 1.mp4", b"abc", actor="ana")
        second = attach_media(ctx, trial_id, "clip 1.mp4", b"def", actor="ana")
        again = attach_media(ctx, trial_id, "x.jpg", b"1", actor="ana", idempotency_key="m")
        assert attach_media(ctx, trial_id, "x.jpg", b"1", actor="ana", idempotency_key="m") == again
        assert first == f"media/{trial_id}/clip_1.mp4"
        assert second == f"media/{trial_id}/clip_1-1.mp4"
        assert (locked_study / first).read_bytes() == b"abc"
        assert get_trial(ctx, trial_id).media == (first, second, again)
        with pytest.raises(ServiceError, match="needs a name"):
            attach_media(ctx, trial_id, "...", b"x", actor="ana")
        with pytest.raises(ServiceError, match="no trial"):
            attach_media(ctx, "nope", "a.jpg", b"x", actor="ana")


def test_sessions_and_listings(locked_study: Path) -> None:
    with open_study(locked_study) as ctx:
        assert is_blinded(ctx)
        first = latest_event_id(ctx)
        session = start_session(ctx, operator=" ana ", rig="rig-1", rig_check={"Lamp on": True})
        assert get_session(ctx, session).operator == "ana"
        assert [s.session_id for s in list_sessions(ctx, open_only=True)] == [session]
        slot = next_slot(ctx)
        assert slot is not None
        trial = start_trial(ctx, slot.slot_id, session)
        with pytest.raises(ServiceError, match="running trial"):
            end_session(ctx, session)
        invalidate_trial(ctx, trial.trial_id, "fault")
        end_session(ctx, session)
        assert list_sessions(ctx, open_only=True) == []
        assert [t.status for t in list_trials(ctx)] == ["invalid"]
        new = events_after(ctx, first)
        assert [e.kind for e in new] == [
            "session_started",
            "trial_started",
            "trial_invalidated",
            "session_ended",
        ]
        assert events_after(ctx, new[-1].id) == []
        with pytest.raises(ServiceError):
            get_session(ctx, "nope")
        with pytest.raises(ServiceError):
            get_trial(ctx, "nope")


def test_list_studies(tmp_path: Path) -> None:
    write_small_study(tmp_path / "b")
    lock_study(tmp_path / "b")
    write_small_study(tmp_path / "a")
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "study.yaml").write_text("not: [valid")
    (tmp_path / "notes").mkdir()
    entries = list_studies(tmp_path)
    assert [(e.slug, e.status, e.locked) for e in entries] == [
        ("a", "draft", False),
        ("b", "locked", True),
        ("c", "invalid", False),
    ]
    assert [e.slug for e in list_studies(tmp_path / "b")] == ["b"]
    with pytest.raises(ServiceError):
        list_studies(tmp_path / "missing")


def test_idempotency_race(locked_study: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two requests with one key both miss the pre-check; the unique index decides."""
    from fieldtrial.services import events

    with open_study(locked_study) as ctx:
        first = start_session(ctx, operator="ana", rig="r", idempotency_key="race")
        real = events._stored
        calls = {"n": 0}

        def miss_once(c, key):  # type: ignore[no-untyped-def]
            calls["n"] += 1
            return None if calls["n"] == 1 else real(c, key)

        monkeypatch.setattr(events, "_stored", miss_once)
        again = start_session(ctx, operator="ana", rig="r", idempotency_key="race")
    assert again == first
    assert calls["n"] == 2
    assert _event_count(locked_study, "session_started") == 1
