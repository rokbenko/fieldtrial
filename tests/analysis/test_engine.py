"""The analysis engine on hand-built records: one test per primary method, plus checks."""

from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.analysis.engine import analyze
from fieldtrial.analysis.records import StudyContextInfo, TrialRecord
from fieldtrial.design import StudySpec, parse_study
from fieldtrial.stats import cmh_test, cochran_q, compare_independent, compare_paired

T0 = datetime(2026, 10, 1, 9, tzinfo=UTC)
NOW = datetime(2026, 10, 2, tzinfo=UTC)


def spec(
    arms: tuple[str, ...] = ("baseline", "q50"),
    *,
    slots: int = 10,
    replicates: int = 1,
    design: str = "randomized_block",
    threshold: float | None = None,
    alternative: str = "two-sided",
    blinding: str = "none",
) -> StudySpec:
    arm_yaml = "\n".join(f"  - {{id: {a}}}" for a in arms)
    if design == "single_arm":
        primary = f"threshold: {threshold}" if threshold is not None else "alpha: 0.05"
    else:
        primary = f"comparison: {{treatment: {arms[-1]}, control: {arms[0]}}}"
    text = f"""
fieldtrial: 1
name: t
rubric:
  stages: [{{id: a, label: A}}, {{id: b, label: B}}]
  success: b
arms:
{arm_yaml}
conditions:
  factors: {{slot: {{range: [1, {slots}]}}}}
  replicates: {replicates}
design: {{type: {design}, seed: 7, blinding: {blinding}}}
analysis:
  primary:
    {primary}
    alternative: {alternative}
  secondary: [stage_reached]
"""
    return parse_study(text).spec


def info(**kwargs: object) -> StudyContextInfo:
    base: dict[str, object] = {
        "design_hash": "h" * 64,
        "status": "running",
        "locked_at": T0,
        "unblinded_at": None,
        "fieldtrial_version": "0.0",
    }
    base.update(kwargs)
    return StudyContextInfo(**base)  # type: ignore[arg-type]


class Recorder:
    """Builds TrialRecords in run order."""

    def __init__(self) -> None:
        self.records: list[TrialRecord] = []

    def add(
        self,
        arm: str,
        block: int,
        success: bool,
        *,
        condition: str | None = None,
        replicate: int = 1,
        session: str = "s1",
        operator: str = "ana",
        status: str = "completed",
        seq: int | None = None,
    ) -> None:
        n = len(self.records)
        self.records.append(
            TrialRecord(
                trial_id=f"t{n}",
                seq=n + 1 if seq is None else seq,
                block=block,
                replicate=replicate,
                condition=condition or f"slot={block}",
                arm=arm,
                blind_code=arm[:2].upper(),
                session_id=session,
                operator=operator,
                rig="r",
                status=status,
                attempt=1,
                origin="planned",
                started_at=T0 + timedelta(minutes=n),
                stage_index=1 if success else 0,
                success=success if status == "completed" else None,
                termination="success" if success else "stuck",
                duration_s=10.0 + n % 3 if success else 45.0,
                invalid_reason="fault" if status == "invalid" else None,
            )
        )


def test_two_arm_paired_primary_with_excluded_blocks() -> None:
    rec = Recorder()
    control = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1]
    treat = [1, 1, 1, 1, 1, 0, 1, 1, 1]  # block 10 has no treatment trial
    for i, ok in enumerate(control, 1):
        rec.add("baseline", i, bool(ok))
    for i, ok in enumerate(treat, 1):
        rec.add("q50", i, bool(ok))
    res = analyze(spec(), rec.records, info(planned_slots=20, pending_slots=1), now=NOW)
    p = res.primary
    expected = compare_paired(3, 1, 9)
    assert p.method == "mcnemar_tango"
    assert (p.n_used, p.blocks_excluded) == (9, 1)
    assert p.pvalue == expected.test.pvalue
    assert p.estimate == pytest.approx(expected.difference)
    assert p.ci is not None
    assert expected.interval is not None
    assert (p.ci.low, p.ci.high) == pytest.approx((expected.interval.low, expected.interval.high))
    assert not p.rejected
    assert p.mde_pp is None  # 9 pairs: no difference reaches 80% power
    indep = compare_independent(8, 9, 7, 10)
    assert res.sensitivity[0].boschloo_p == pytest.approx(indep.primary.pvalue)
    kinds = [d.kind for d in res.deviations]
    assert kinds == ["incomplete", "excluded_blocks"]
    assert res.summary[0].startswith("1 block without a completed trial")
    assert res.summary[1].startswith("No significant difference detected")
    assert res.summary[1].endswith("did not have 80% power for any difference.")
    assert res.arms[1].rate == pytest.approx(8 / 9)
    assert res.provenance.generated_at == NOW
    assert Results_roundtrip(res)


def Results_roundtrip(res: object) -> bool:  # noqa: N802
    from fieldtrial.analysis.results import Results

    assert isinstance(res, Results)
    return Results.model_validate_json(res.model_dump_json()) == res


def test_paired_rejects_and_claims_difference() -> None:
    rec = Recorder()
    for i in range(1, 31):
        rec.add("baseline", i, i % 2 == 0)
        rec.add("q50", i, True)
    res = analyze(spec(slots=30), rec.records, info(planned_slots=60), now=NOW)
    assert res.primary.rejected
    assert res.primary.mde_pp is None
    assert res.summary[0].startswith("q50 succeeded in 100.0% of trials (30/30;")
    assert "McNemar p" in res.summary[0]


def test_three_arms_cochran_q_with_holm() -> None:
    rec = Recorder()
    arms = ("baseline", "q40", "q50")
    for i in range(1, 21):
        rec.add("baseline", i, i % 3 == 0)
        rec.add("q40", i, i % 2 == 0)
        rec.add("q50", i, i != 5)
    res = analyze(spec(arms, slots=20), rec.records, info(planned_slots=60), now=NOW)
    p = res.primary
    assert p.method == "cochran_q"
    matrix = [[int(i % 3 == 0), int(i % 2 == 0), int(i != 5)] for i in range(1, 21)]
    q = cochran_q(matrix, pairs="vs-first", adjustment="holm")
    assert p.statistic == pytest.approx(q.statistic)
    assert [w.treatment for w in p.pairwise] == ["q40", "q50"]
    assert [w.adjusted_pvalue for w in p.pairwise] == pytest.approx(
        [c.adjusted_pvalue for c in q.comparisons]
    )
    assert p.pvalue == p.pairwise[1].adjusted_pvalue
    assert p.rejected
    assert res.summary[0].startswith("Cochran's Q across 3 arms on 20 complete blocks")
    assert "holm-adjusted" in res.summary[2]
    assert len(res.sensitivity) == 2


def test_replicates_use_cmh() -> None:
    rec = Recorder()
    strata = []
    for cond in range(1, 6):
        kt = kc = 0
        for rep in (1, 2, 3):
            block = (cond - 1) * 3 + rep
            t_ok, c_ok = (rep != 3 or cond == 2), (rep == 1)
            rec.add("q50", block, t_ok, condition=f"slot={cond}", replicate=rep)
            rec.add("baseline", block, c_ok, condition=f"slot={cond}", replicate=rep)
            kt, kc = kt + t_ok, kc + c_ok
        strata.append((kt, 3, kc, 3))
    res = analyze(spec(slots=5, replicates=3), rec.records, info(planned_slots=30), now=NOW)
    expected = cmh_test(strata)
    assert res.primary.method == "cmh"
    assert res.primary.pvalue == pytest.approx(expected.test.pvalue)
    assert res.primary.estimate == pytest.approx(expected.odds_ratio)
    assert res.primary.n_used == 5


def test_cmh_one_sided_halves_p() -> None:
    rec = Recorder()
    for cond in range(1, 4):
        for rep in (1, 2):
            block = cond * 10 + rep
            rec.add("q50", block, True, condition=f"slot={cond}", replicate=rep)
            rec.add("baseline", block, rep == 1, condition=f"slot={cond}", replicate=rep)
    two = analyze(spec(slots=3, replicates=2), rec.records, info(), now=NOW).primary
    one = analyze(
        spec(slots=3, replicates=2, alternative="greater"), rec.records, info(), now=NOW
    ).primary
    assert two.pvalue is not None
    assert one.pvalue == pytest.approx(two.pvalue / 2)


def test_single_arm_threshold_and_descriptive() -> None:
    rec = Recorder()
    for i in range(1, 41):
        rec.add("a", i, i > 2)
    res = analyze(
        spec(("a",), slots=40, design="single_arm", threshold=0.8, alternative="greater"),
        rec.records,
        info(),
        now=NOW,
    )
    assert res.primary.method == "threshold"
    assert res.primary.rejected
    assert "above the 80.0% threshold" in res.summary[0]
    assert res.sensitivity == []
    res = analyze(spec(("a",), slots=40, design="single_arm"), rec.records, info(), now=NOW)
    assert res.primary.method == "descriptive"
    assert res.summary[0].endswith("No test was pre-registered.")


def test_no_data_yet() -> None:
    for s in (
        spec(),
        spec(slots=3, replicates=2),
        spec(("a",), design="single_arm", threshold=0.5),
        spec(("a",), design="single_arm"),
    ):
        res = analyze(s, [], info(planned_slots=10, pending_slots=10), now=NOW)
        assert res.primary.pvalue is None
        assert not res.primary.rejected
        assert all(a.rate is None for a in res.arms)
        assert [d.kind for d in res.deviations] == ["incomplete"]


def test_drift_invalid_and_deviation_flags() -> None:
    rec = Recorder()
    for i in range(1, 21):
        early = i <= 10
        rec.add(
            "baseline",
            i,
            early,
            session="s1" if early else "s2",
            operator="ana" if early else "bo",
            seq=2 * i,
        )
        rec.add(
            "q50",
            i,
            True,
            session="s1" if early else "s2",
            operator="ana" if early else "bo",
            seq=2 * i - 1,
        )
    for i in range(1, 13):
        rec.add("q50", 100 + i, False, status="invalid", seq=100 + i)
    amendment = {
        "ts": "2026-10-01T10:00:00+00:00",
        "actor": "ana",
        "reason": "more slots",
        "old_hash": "a" * 64,
        "new_hash": "b" * 64,
        "added_slots": 4,
        "voided_slots": 0,
    }
    res = analyze(
        spec(slots=20, blinding="operator"),
        rec.records,
        info(amendments=(amendment,), unblinded_at=T0, pending_at_unblinding=5),
        now=NOW,
    )
    flagged = {(d.arm, d.grouping) for d in res.drift if d.flagged}
    assert flagged == {("baseline", "session"), ("baseline", "operator")}
    assert res.invalid.flagged
    assert res.invalid.per_arm == {"baseline": 0, "q50": 12}
    assert any(s.startswith("Results changed across sessions for baseline") for s in res.summary)
    assert any(s.startswith("Invalid trials are unevenly spread") for s in res.summary)
    kinds = [d.kind for d in res.deviations]
    assert kinds == ["amendment", "early_unblinding", "out_of_order"]
    assert "more slots" in res.deviations[0].message
    assert len(res.sessions) == 2
    assert res.provenance.operators == ["ana", "bo"]


def test_stages_and_timing() -> None:
    rec = Recorder()
    for i in range(1, 11):
        rec.add("baseline", i, i <= 5)
        rec.add("q50", i, i <= 8)
    res = analyze(spec(), rec.records, info(), now=NOW)
    base = res.stages[0]
    assert [d.count for d in base.distribution] == [0, 5, 5]
    assert base.funnel[1].reached == 5
    assert res.stage_comparisons[0].preregistered
    assert res.timing[1].n_successes == 8
    assert res.timing[1].median_s is not None
    assert not res.timing[1].preregistered
    assert res.conditions[0].condition == "slot=1"


@settings(max_examples=60, deadline=None)
@given(
    n_arms=st.integers(1, 3),
    replicates=st.integers(1, 2),
    trials=st.lists(
        st.tuples(
            st.integers(0, 2),  # arm index
            st.integers(1, 6),  # block
            st.booleans(),  # success
            st.sampled_from(["completed", "completed", "invalid"]),
            st.sampled_from(["s1", "s2"]),
        ),
        max_size=30,
    ),
)
def test_engine_never_crashes_on_partial_data(
    n_arms: int, replicates: int, trials: list[tuple[int, int, bool, str, str]]
) -> None:
    """Property: any partially filled study can be analyzed and rendered."""
    from fieldtrial.report import render_markdown

    arms = ("baseline", "q40", "q50")[:n_arms] if n_arms > 1 else ("a",)
    s = (
        spec(("a",), slots=6, design="single_arm", threshold=0.5)
        if n_arms == 1
        else spec(arms, slots=6, replicates=replicates)
    )
    rec = Recorder()
    for arm_index, block, ok, status, session in trials:
        arm = arms[arm_index % len(arms)]
        rec.add(
            arm,
            block,
            ok,
            status=status,
            session=session,
            replicate=1 + block % replicates,
            condition=f"slot={1 + block % 6}",
        )
    res = analyze(s, rec.records, info(planned_slots=12, pending_slots=3), now=NOW)
    assert res.primary.rejected == (res.primary.pvalue is not None and res.primary.rejected)
    assert render_markdown(res).startswith("# ")
