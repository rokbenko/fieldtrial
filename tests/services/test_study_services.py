"""Study lifecycle: init, lock, amend, unblind, status, and the event log."""

from pathlib import Path

import pytest
from sqlalchemy import func, select

from fieldtrial.design import StudyValidationError
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import (
    amend_study,
    init_study,
    lock_study,
    plan_study,
    study_status,
    unblind_study,
    validate_study,
)
from fieldtrial.store import models as m
from fieldtrial.store.db import DB_FILENAME


def _count(folder: Path, model: type[m.Base], **where: object) -> int:
    with open_study(folder) as ctx, ctx.db() as db:
        query = select(func.count()).select_from(model)
        for column, value in where.items():
            query = query.where(getattr(model, column) == value)
        return int(db.scalar(query) or 0)


def test_init_writes_folder(tmp_path: Path) -> None:
    folder = tmp_path / "s"
    assert init_study(folder, template="serving-sweep", name="sweep") == folder / "study.yaml"
    assert (folder / "media").is_dir()
    assert (folder / "reports").is_dir()
    assert validate_study(folder).spec.name == "sweep"
    with pytest.raises(ServiceError):
        init_study(folder)


def test_init_rejects_unknown_template(tmp_path: Path) -> None:
    with pytest.raises((ServiceError, ValueError)):
        init_study(tmp_path / "s", template="nope")


def test_validate_reports_line(small_study: Path) -> None:
    path = small_study / "study.yaml"
    path.write_text(path.read_text().replace("alpha: 0.05", "alpha: 0.9"))
    with pytest.raises(StudyValidationError) as info:
        validate_study(small_study)
    assert any(issue.line is not None for issue in info.value.issues)


def test_plan(small_study: Path) -> None:
    plan = plan_study(small_study, baseline=0.7)
    assert len(plan.schedule) == 8
    assert plan.trials_per_arm == {"baseline": 4, "q50": 4}
    assert plan.n_conditions == 4
    # With 4 trials per arm nothing small is detectable.
    assert plan.detectable_increase is None or plan.detectable_increase > 0.2


def test_lock_creates_schedule_and_event(locked_study: Path) -> None:
    assert (locked_study / DB_FILENAME).exists()
    assert _count(locked_study, m.ScheduleSlot, status="pending") == 8
    assert _count(locked_study, m.Event, kind="design_locked") == 1
    with pytest.raises(ServiceError, match="already locked"):
        lock_study(locked_study)


def test_lock_failure_leaves_no_database(small_study: Path) -> None:
    path = small_study / "study.yaml"
    path.write_text(path.read_text().replace("seed: 20261001", "seed: -1"))
    with pytest.raises(StudyValidationError):
        lock_study(small_study)
    assert not (small_study / DB_FILENAME).exists()


def test_open_unlocked_study_fails(small_study: Path) -> None:
    with pytest.raises(ServiceError, match="not locked"), open_study(small_study):
        pass


def test_design_comes_from_database_not_yaml(locked_study: Path) -> None:
    path = locked_study / "study.yaml"
    path.write_text(path.read_text().replace("range: [1, 4]", "range: [1, 6]"))
    with open_study(locked_study) as ctx:
        assert ctx.spec.conditions.count() == 4


def test_amend_adds_and_voids(locked_study: Path) -> None:
    path = locked_study / "study.yaml"
    original = path.read_text()
    with pytest.raises(ServiceError, match="has not changed"):
        amend_study(locked_study, "nothing")
    with pytest.raises(ServiceError, match="needs a reason"):
        amend_study(locked_study, " ")

    path.write_text(original.replace("range: [1, 4]", "range: [1, 6]"))
    grown = amend_study(locked_study, "more slots")
    assert grown.added_slots == 4
    assert grown.voided_slots == 0
    assert grown.old_hash != grown.new_hash
    assert _count(locked_study, m.ScheduleSlot, status="pending") == 12

    path.write_text(original.replace("range: [1, 4]", "range: [1, 3]"))
    shrunk = amend_study(locked_study, "fewer slots")
    assert shrunk.voided_slots == 6
    assert _count(locked_study, m.ScheduleSlot, status="pending") == 6
    assert _count(locked_study, m.Event, kind="amendment") == 2
    assert study_status(locked_study).amendments == 2


def test_amend_new_arm_gets_fresh_code(locked_study: Path) -> None:
    path = locked_study / "study.yaml"
    text = path.read_text().replace(
        "conditions:",
        "  - id: q70\n    runner: manual\n    serving: {inference.queue_threshold: 70}\n\n"
        "conditions:",
    )
    path.write_text(text)
    result = amend_study(locked_study, "third arm")
    assert result.added_slots == 4
    codes = unblind_study(locked_study)
    assert sorted(codes.values()) == ["baseline", "q50", "q70"]
    assert len(set(codes)) == 3


def test_amend_cannot_change_seed(locked_study: Path) -> None:
    path = locked_study / "study.yaml"
    path.write_text(path.read_text().replace("seed: 20261001", "seed: 7"))
    with pytest.raises(ServiceError, match="seed"):
        amend_study(locked_study, "reseed")


def test_status_is_blinded_until_unblinded(locked_study: Path) -> None:
    simulate_study(locked_study, {"baseline": 0.5, "q50": 0.9}, seed=3)
    status = study_status(locked_study)
    assert status.blinded
    assert status.per_arm is None
    assert status.done == 8
    assert status.pending == 0

    codes = unblind_study(locked_study)
    assert set(codes.values()) == {"baseline", "q50"}
    status = study_status(locked_study)
    assert not status.blinded
    assert status.per_arm is not None
    assert sum(n for _, n in status.per_arm.values()) == 8
    unblind_study(locked_study)
    assert _count(locked_study, m.Event, kind="unblinded") == 2


def test_unblinded_design_shows_arms(tmp_path: Path) -> None:
    from tests.services.conftest import write_small_study

    folder = write_small_study(tmp_path / "open")
    path = folder / "study.yaml"
    path.write_text(path.read_text().replace("blinding: operator", "blinding: none"))
    lock_study(folder)
    status = study_status(folder)
    assert not status.blinded
    assert status.per_arm == {"baseline": (0, 0), "q50": (0, 0)}
