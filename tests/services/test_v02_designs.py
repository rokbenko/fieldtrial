"""Crossover rounds, checkpoint ladders and group-sequential stopping, through the services."""

import json
from pathlib import Path

import pytest

from fieldtrial.analysis.sequential import complete_pairs, score_z
from fieldtrial.report import render_html, render_markdown
from fieldtrial.services import ServiceError, open_study
from fieldtrial.services.analysis import analyze_study
from fieldtrial.services.events import list_events
from fieldtrial.services.interim import interim_status, run_interim
from fieldtrial.services.simulate import simulate_study
from fieldtrial.services.study import amend_study, init_study, lock_study, unblind_study
from fieldtrial.services.trial import collect_records
from fieldtrial.stats import spending_boundaries


def _sequential_study(folder: Path, *, slots: int = 20, looks: int = 4) -> Path:
    init_study(folder, template="basic", name="seq")
    path = folder / "study.yaml"
    text = path.read_text(encoding="utf-8").replace("range: [1, 40]", f"range: [1, {slots}]")
    text = text.replace(
        "  stopping: {rule: fixed}",
        f"  stopping: {{rule: group_sequential, looks: {looks}}}",
    )
    path.write_text(text, encoding="utf-8")
    lock_study(folder)
    return folder


def _interim_payloads(folder: Path) -> list[dict]:  # type: ignore[type-arg]
    with open_study(folder) as ctx, ctx.db() as db:
        return [dict(e.payload) for e in list_events(db, ctx.study_id, "interim_look")]


def test_interim_looks_run_when_due_and_reveal_nothing(tmp_path: Path) -> None:
    folder = _sequential_study(tmp_path / "s")
    with open_study(folder) as ctx:
        status = interim_status(ctx)
    assert status is not None
    assert (status.next_look, status.blocks_needed, status.due) == (1, 5, False)
    with pytest.raises(ServiceError, match="due after 5 complete blocks; 0 so far"):
        run_interim(folder)

    simulate_study(folder, {"baseline": 0.7, "q50": 0.7}, max_trials=10, interim=False)
    result = run_interim(folder)
    assert (result.look, result.planned_looks, result.complete_blocks) == (1, 4, 5)
    [payload] = _interim_payloads(folder)
    # The event holds the blocks used, the fraction, the boundary and the decision only.
    assert set(payload) == {
        "look",
        "planned_looks",
        "fraction",
        "blocks",
        "boundary",
        "decision",
        "voided_slots",
    }
    assert payload["fraction"] == pytest.approx(0.25)
    assert payload["boundary"] == pytest.approx(spending_boundaries([0.25]).boundaries[0])
    with pytest.raises(ServiceError, match="due after 10 complete blocks"):
        run_interim(folder)


def test_a_stop_voids_the_rest_and_the_report_explains_it(tmp_path: Path) -> None:
    folder = _sequential_study(tmp_path / "s", slots=40)
    result = simulate_study(folder, {"baseline": 0.3, "q50": 0.98}, seed=3)
    assert result.stopped_at is not None
    assert result.completed < 80
    with open_study(folder) as ctx:
        status = interim_status(ctx)
        assert status is not None
        assert status.stopped_at == result.stopped_at
        assert status.next_look is None
    with pytest.raises(ServiceError, match="already stopped"):
        run_interim(folder)

    unblind_study(folder)
    res = analyze_study(folder)
    assert res.primary.method == "group_sequential"
    assert res.primary.rejected
    assert res.sequential is not None
    assert res.sequential.stopped_at == result.stopped_at
    assert res.primary.ci is not None
    assert res.primary.ci.method == "repeated_ci"
    assert not [d for d in res.deviations if d.kind == "incomplete"]
    assert any("stopped at interim look" in s for s in res.summary)
    # The recomputed statistic at the stopping look matches the recorded blocks.
    payloads = _interim_payloads(folder)
    with open_study(folder) as ctx:
        records, _ = collect_records(ctx)
    pairs = complete_pairs(records, "q50", "baseline")
    stop = payloads[-1]
    z, _, _ = score_z([pairs[b] for b in stop["blocks"]])
    assert res.sequential.looks[-1].z == pytest.approx(z)
    assert "Group-sequential looks" in render_markdown(res)
    assert "Group-sequential looks" in render_html(res, charts=False)


def test_without_a_stop_the_final_look_spends_the_rest(tmp_path: Path) -> None:
    folder = _sequential_study(tmp_path / "s", slots=20, looks=2)
    simulate_study(folder, {"baseline": 0.7, "q50": 0.7}, seed=1)
    unblind_study(folder)
    res = analyze_study(folder)
    assert res.sequential is not None
    kinds = [k.kind for k in res.sequential.looks]
    assert kinds[-1] == "final"
    assert res.sequential.looks[-1].fraction == 1.0
    design = spending_boundaries([k.fraction for k in res.sequential.looks], spend_all=True)
    assert res.sequential.looks[-1].boundary == pytest.approx(design.boundaries[-1])


def test_skipped_interim_looks_are_reported(tmp_path: Path) -> None:
    folder = _sequential_study(tmp_path / "s", slots=20, looks=3)
    simulate_study(folder, {"baseline": 0.7, "q50": 0.7}, interim=False)
    unblind_study(folder)
    res = analyze_study(folder)
    notes = [d.message for d in res.deviations if d.kind == "interim"]
    assert any("2 planned interim looks were not run" in n for n in notes)


def test_crossover_study_end_to_end(tmp_path: Path) -> None:
    folder = tmp_path / "tray"
    init_study(folder, template="crossover-rounds", name="tray")
    path = folder / "study.yaml"
    path.write_text(
        path.read_text()
        .replace("range: [1, 12]", "range: [1, 4]")
        .replace("rounds: 16", "rounds: 8")
    )
    lock_study(folder)
    simulate_study(folder, {"baseline": 0.5, "candidate": 0.9}, seed=2)
    unblind_study(folder)
    res = analyze_study(folder)
    assert res.primary.method == "crossover"
    assert res.crossover is not None
    assert res.crossover.cycles == 4
    assert res.crossover.ab == 2
    assert res.crossover.ba == 2
    assert len(res.crossover.rounds) == 8
    assert {r.arm for r in res.crossover.rounds[:2]} == {"baseline", "candidate"}
    assert res.primary.test == "crossover_randomization"
    # With 4 cycles the smallest two-sided p-value is 1/3: the summary says so honestly.
    assert res.primary.pvalue is not None
    assert res.primary.pvalue >= 1 / 3 - 1e-12
    assert not res.primary.rejected
    html = render_html(res)
    assert "Crossover rounds" in html
    assert "Success rate per crossover round" in html
    data = json.loads(res.model_dump_json())
    assert data["crossover"]["cycles"] == 4


def test_crossover_amendments_cannot_change_rounds(tmp_path: Path) -> None:
    folder = tmp_path / "tray"
    init_study(folder, template="crossover-rounds", name="tray")
    lock_study(folder)
    path = folder / "study.yaml"
    path.write_text(path.read_text().replace("rounds: 16", "rounds: 20"))
    with pytest.raises(ServiceError, match="rounds cannot be changed"):
        amend_study(folder, "more rounds")
    path.write_text(
        path.read_text()
        .replace("rounds: 20", "rounds: 16")
        .replace("secondary: [stage_reached]", "secondary: [stage_reached, time_to_success]")
    )
    assert amend_study(folder, "add a secondary").added_slots == 0


def test_ladder_as_secondary_and_as_primary(tmp_path: Path) -> None:
    folder = tmp_path / "lad"
    init_study(folder, template="checkpoint-ladder", name="lad")
    lock_study(folder)
    rates = {"step-010k": 0.3, "step-020k": 0.7, "step-030k": 0.85, "step-040k": 0.9}
    simulate_study(folder, rates, seed=4)
    unblind_study(folder)
    res = analyze_study(folder)
    assert res.primary.method == "cochran_q"
    assert res.ladder is not None
    assert not res.ladder.association.primary
    assert res.ladder.association.rejected
    assert res.ladder.plateau_method == "tango"
    assert [row.arm for row in res.ladder.plateau] == ["step-010k", "step-020k", "step-030k"]
    assert any("rises with training step" in s for s in res.summary)

    # The same ladder without a comparison: the association test is the primary analysis.
    folder2 = tmp_path / "lad2"
    init_study(folder2, template="checkpoint-ladder", name="lad2")
    path = folder2 / "study.yaml"
    path.write_text(
        path.read_text().replace("    comparison: {treatment: step-040k, control: step-010k}\n", "")
    )
    lock_study(folder2)
    simulate_study(folder2, rates, seed=4)
    unblind_study(folder2)
    res2 = analyze_study(folder2)
    assert res2.primary.method == "ladder"
    assert res2.primary.test == "mantel"
    assert res2.ladder is not None
    assert res2.ladder.association.primary
    assert res2.primary.pvalue == res2.ladder.association.pvalue
    assert res2.sensitivity == []
    assert "Success rate at each checkpoint in training-step order" in render_html(res2)
    md = render_markdown(res2)
    assert "## Checkpoint ladder" in md
    assert "trend" not in md.lower()


def test_status_of_studies_without_interim_rules(tmp_path: Path) -> None:
    folder = tmp_path / "b"
    init_study(folder, template="basic", name="b")
    lock_study(folder)
    with open_study(folder) as ctx:
        assert interim_status(ctx) is None
    with pytest.raises(ServiceError, match="no group-sequential stopping rule"):
        run_interim(folder)


def test_ladder_with_replicates_uses_pooled_counts(tmp_path: Path) -> None:
    folder = tmp_path / "lad"
    init_study(folder, template="checkpoint-ladder", name="lad")
    path = folder / "study.yaml"
    path.write_text(
        path.read_text()
        .replace("range: [1, 20]", "range: [1, 10]")
        .replace("replicates: 1", "replicates: 3")
    )
    lock_study(folder)
    rates = {"step-010k": 0.5, "step-020k": 0.85, "step-030k": 0.9, "step-040k": 0.9}
    simulate_study(folder, rates, seed=5)
    unblind_study(folder)
    res = analyze_study(folder)
    assert res.primary.method == "cochran_q"
    assert res.ladder is not None
    assert res.ladder.plateau_method == "newcombe"
    assert len(res.ladder.plateau) == 3
