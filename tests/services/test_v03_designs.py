"""Anytime-valid stopping and best-arm selection, through the services (v0.3)."""

from pathlib import Path

import pytest

from fieldtrial.design.hashing import design_hash
from fieldtrial.design.loader import load_study
from fieldtrial.services import open_study
from fieldtrial.services.adaptive import adaptive_status, check_after_block
from fieldtrial.services.events import list_events
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import init_study, lock_study


def _anytime_study(folder: Path, *, slots: int = 40, min_blocks: int | None = None) -> Path:
    init_study(folder, template="basic", name="any")
    path = folder / "study.yaml"
    text = path.read_text(encoding="utf-8").replace("range: [1, 40]", f"range: [1, {slots}]")
    extra = f", min_blocks: {min_blocks}" if min_blocks else ""
    text = text.replace("  stopping: {rule: fixed}", f"  stopping: {{rule: anytime{extra}}}")
    path.write_text(text, encoding="utf-8")
    lock_study(folder)
    return folder


def _selection_study(folder: Path, *, delta: float = 0.05) -> Path:
    init_study(folder, template="serving-sweep", name="sweep")
    path = folder / "study.yaml"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "  primary:\n    comparison: {treatment: q50, control: default}\n",
        f"  primary: {{}}\n  selection: {{rule: elimination, delta: {delta}}}\n",
    )
    text = text.replace("range: [1, 10]", "range: [1, 30]")  # 3 objects x 30 = 90 blocks
    path.write_text(text, encoding="utf-8")
    lock_study(folder)
    return folder


def _events(folder: Path, kind: str) -> list[dict]:  # type: ignore[type-arg]
    with open_study(folder) as ctx, ctx.db() as db:
        return [dict(e.payload) for e in list_events(db, ctx.study_id, kind)]


def _slot_counts(folder: Path) -> dict[str, int]:
    from sqlalchemy import func, select

    from fieldtrial.store import models as m

    with open_study(folder) as ctx, ctx.db() as db:
        rows = db.execute(
            select(m.ScheduleSlot.status, func.count())
            .where(m.ScheduleSlot.study_id == ctx.study_id)
            .group_by(m.ScheduleSlot.status)
        )
        return {str(status): int(n) for status, n in rows.all()}


def test_anytime_stops_early_on_a_large_difference(tmp_path: Path) -> None:
    folder = _anytime_study(tmp_path / "a")
    result = simulate_study(folder, {"baseline": 0.1, "q50": 0.95}, seed=3)
    assert result.stopped_at == 1
    [look] = _events(folder, "interim_look")
    # Only the rule, the blocks used and the decision: nothing per arm.
    assert set(look) == {"rule", "look", "blocks", "decision", "voided_slots"}
    assert look["rule"] == "anytime"
    assert look["decision"] == "stop"
    assert look["look"] == len(look["blocks"]) < 40
    counts = _slot_counts(folder)
    assert counts["void"] == look["voided_slots"] > 0
    assert "pending" not in counts
    with open_study(folder) as ctx:
        status = adaptive_status(ctx)
        assert status is not None
        assert (status.rule, status.stopped, status.remaining) == ("anytime", True, 0)
        assert check_after_block(ctx) is None  # nothing more after a stop


def test_anytime_keeps_going_on_equal_arms_and_respects_min_blocks(tmp_path: Path) -> None:
    folder = _anytime_study(tmp_path / "b", slots=12)
    result = simulate_study(folder, {"baseline": 0.6, "q50": 0.6}, seed=1)
    assert result.stopped_at is None
    assert _events(folder, "interim_look") == []
    assert result.completed == 24
    # With min_blocks above the study size, even a huge difference never stops it.
    late = _anytime_study(tmp_path / "c", slots=12, min_blocks=20)
    assert simulate_study(late, {"baseline": 0.0, "q50": 1.0}).stopped_at is None


def test_selection_drops_worse_arms_and_cancels_their_trials(tmp_path: Path) -> None:
    folder = _selection_study(tmp_path / "s")
    result = simulate_study(folder, {"default": 0.1, "q50": 0.15, "sync": 0.97}, seed=2)
    looks = _events(folder, "selection_look")
    assert looks, "expected at least one elimination"
    dropped = [e["arm"] for look in looks for e in look["eliminated"]]
    assert set(dropped) == {"default", "q50"}
    assert all(e["by"] == "sync" for look in looks for e in look["eliminated"])
    assert looks[-1]["decision"] == "stop"
    with open_study(folder) as ctx:
        status = adaptive_status(ctx)
        assert status is not None
        assert status.rule == "elimination"
        assert status.stopped
        assert len(status.dropped) == 2
    # Dropped arms are reported by blind code, never by arm id.
    assert set(result.dropped) == set(status.dropped)
    assert not set(status.dropped) & {"default", "q50", "sync"}
    counts = _slot_counts(folder)
    assert "pending" not in counts
    assert counts["void"] == sum(look["voided_slots"] for look in looks)


def test_selection_keeps_equal_arms(tmp_path: Path) -> None:
    folder = _selection_study(tmp_path / "e")
    result = simulate_study(folder, {"default": 0.7, "q50": 0.7, "sync": 0.7}, seed=4)
    assert _events(folder, "selection_look") == []
    assert result.completed == 270


def test_new_settings_validate_and_hash(tmp_path: Path) -> None:
    folder = _anytime_study(tmp_path / "h")
    spec = load_study(folder).spec
    assert spec.analysis.stopping.rule == "anytime"
    h1 = design_hash(spec)
    text = (
        (folder / "study.yaml")
        .read_text()
        .replace("{rule: anytime}", "{rule: anytime, min_blocks: 1}")
    )
    (folder / "study.yaml").write_text(text)
    assert design_hash(load_study(folder).spec) == h1  # the default, written out


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("{rule: fixed}", "{rule: anytime, looks: 3}", "only for rule: group_sequential"),
        ("{rule: fixed}", "{rule: fixed, min_blocks: 3}", "only for rule: anytime"),
    ],
)
def test_stopping_validation(tmp_path: Path, old: str, new: str, message: str) -> None:
    init_study(tmp_path / "v", template="basic", name="v")
    path = tmp_path / "v" / "study.yaml"
    path.write_text(path.read_text().replace(f"  stopping: {old}", f"  stopping: {new}"))
    with pytest.raises(Exception, match=message):
        load_study(path)


def test_selection_validation(tmp_path: Path) -> None:
    init_study(tmp_path / "w", template="serving-sweep", name="w")
    path = tmp_path / "w" / "study.yaml"
    base = path.read_text()
    with_comparison = base.replace("  multiplicity: holm", "  selection: {}\n  multiplicity: holm")
    path.write_text(with_comparison)
    with pytest.raises(Exception, match="selection is the primary analysis"):
        load_study(path)


def test_completing_a_trial_runs_the_look(tmp_path: Path) -> None:
    """The console and the API complete trials through complete_trial, which checks."""
    from fieldtrial.services.session import start_session
    from fieldtrial.services.trial import complete_trial, next_slot, start_trial

    folder = _anytime_study(tmp_path / "live", slots=20)
    # 39 trials without looks (as if recorded elsewhere), then the last one live.
    simulate_study(folder, {"baseline": 0.0, "q50": 1.0}, max_trials=39, interim=False)
    assert _events(folder, "interim_look") == []
    with open_study(folder) as ctx:
        slot = next_slot(ctx)
        assert slot is not None
        session = start_session(ctx, operator="op", rig="rig")
        trial = start_trial(ctx, slot.slot_id, session)
        stage = ctx.spec.rubric.success_index if slot.arm == "q50" else -1
        complete_trial(ctx, trial.trial_id, stage_index=stage, termination="other")
    [look] = _events(folder, "interim_look")
    assert look["decision"] == "stop"
    assert look["rule"] == "anytime"


def test_anytime_analysis_and_report(tmp_path: Path) -> None:
    from fieldtrial.report import render_html, render_markdown
    from fieldtrial.services.analysis import analyze_study
    from fieldtrial.services.study import unblind_study

    folder = _anytime_study(tmp_path / "r")
    simulate_study(folder, {"baseline": 0.1, "q50": 0.95}, seed=3)
    unblind_study(folder)
    res = analyze_study(folder)
    p = res.primary
    assert (p.method, p.test, p.rejected) == ("anytime", "betting_paired", True)
    assert p.pvalue is not None
    assert p.pvalue <= 0.05
    assert p.ci is not None
    assert p.ci.low > 0
    assert res.anytime is not None
    assert res.anytime.stopped_at == res.anytime.rejected_at == res.anytime.blocks
    assert len(res.anytime.sequence) == res.anytime.blocks
    assert any("stopped after" in s for s in res.summary)
    assert any("Anytime-valid design" in s for s in res.summary)
    render_markdown(res)
    render_html(res)


def test_selection_analysis_and_report(tmp_path: Path) -> None:
    from fieldtrial.report import render_html, render_markdown
    from fieldtrial.services.analysis import analyze_study
    from fieldtrial.services.study import unblind_study

    folder = _selection_study(tmp_path / "t")
    simulate_study(folder, {"default": 0.1, "q50": 0.15, "sync": 0.97}, seed=2)
    unblind_study(folder)
    res = analyze_study(folder)
    assert res.primary.method == "selection"
    assert res.primary.treatment == "sync"
    assert res.primary.rejected
    assert res.selection is not None
    assert res.selection.best == "sync"
    assert res.selection.stopped
    assert {e.arm for e in res.selection.eliminations} == {"default", "q50"}
    assert len(res.selection.pairs) == 3
    assert res.summary[0].startswith("Selected arm: sync.")
    render_markdown(res)
    render_html(res)
