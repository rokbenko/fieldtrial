"""Turn trial records into ``Results`` (docs/PLAN.md section 15).

The primary analysis follows from the locked design, never from the data:

- ``randomized_block`` with 2 arms and 1 replicate: exact McNemar test with a Tango score
  interval, on the blocks where both arms have a completed trial.
- ``randomized_block`` with 2 arms and several replicates: Cochran–Mantel–Haenszel test
  stratified by condition.
- ``randomized_block`` with more than 2 arms: Cochran's Q on complete blocks, then each arm
  against the control with exact McNemar tests and the pre-registered multiplicity
  adjustment. The primary claim for the treatment arm uses its adjusted p-value.
- ``randomized_block`` with 2 arms and ``stopping: group_sequential``: the McNemar score
  statistic at the planned looks, against Lan–DeMets boundaries; stage-wise p-value and a
  repeated confidence interval.
- ``randomized_block`` with ``analysis.ladder`` and no comparison: Mantel's test of a
  linear association between training step and success, stratified by condition.
- ``crossover_rounds``: period-adjusted difference over crossover cycles with an exact
  randomization test.
- ``single_arm``: exact binomial test against ``threshold`` (descriptive without one).

A pre-registered ladder is always analyzed (association with step and plateau), as the
primary analysis or alongside a primary comparison.

The independent-samples analysis (Boschloo test, Newcombe interval) is always reported as a
sensitivity check. Drift and invalid-trial checks screen at a fixed 0.05 level; they are
flags for a human to look at, not tests of the study hypothesis.
"""

import hashlib
import json
import math
import platform
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import scipy
from scipy import stats as sp_stats

from fieldtrial import __version__
from fieldtrial.analysis import wording
from fieldtrial.analysis.adaptive import anytime_final, selection_final
from fieldtrial.analysis.common import LEVEL, ci_model, num
from fieldtrial.analysis.crossover import crossover_analysis
from fieldtrial.analysis.ladder import ladder_analysis
from fieldtrial.analysis.proxy import agreement, proxy_estimates
from fieldtrial.analysis.records import StudyContextInfo, TrialRecord
from fieldtrial.analysis.results import (
    CI,
    AnytimeSummary,
    ArmSummary,
    ConditionCell,
    ConditionRow,
    CrossoverSummary,
    Deviation,
    DriftCheck,
    EpisodeSummary,
    FunnelStep,
    IndependentComparison,
    InvalidCheck,
    LadderResult,
    Pairwise,
    PrimaryResult,
    Provenance,
    Results,
    RigCheckRow,
    RunnerSummary,
    SelectionSummary,
    SequentialSummary,
    SessionRow,
    StageComparison,
    StageShare,
    StageSummary,
    StudyInfo,
    TimingSummary,
)
from fieldtrial.analysis.sequential import complete_pairs, final_analysis
from fieldtrial.design import StudySpec
from fieldtrial.design import conditions as design_conditions
from fieldtrial.stats import (
    adjust_pvalues,
    cmh_test,
    cochran_q,
    compare_independent,
    compare_paired,
    compare_stages,
    homogeneity_test,
    mde,
    median_time_to_success,
    proportion_ci,
    stage_distribution,
    stage_funnel,
    success_curve,
    test_vs_threshold,
)
from fieldtrial.stats._types import Alternative

FLAG_ALPHA = 0.05
_TIMING_BOOT = 2000
_ci = ci_model
_num = num


@dataclass
class _Extras:
    """Design-specific results collected while the primary analysis runs."""

    ladder: LadderResult | None = None
    crossover: CrossoverSummary | None = None
    sequential: SequentialSummary | None = None
    anytime: AnytimeSummary | None = None
    selection: SelectionSummary | None = None
    notes: list[str] = field(default_factory=list)


def _fingerprint(data: object) -> str:
    text = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class _Data:
    """Records indexed the ways the analyses need them."""

    def __init__(self, spec: StudySpec, records: Sequence[TrialRecord]) -> None:
        self.spec = spec
        self.arm_ids = [a.id for a in spec.arms]
        self.records = list(records)
        self.completed = [r for r in self.records if r.status == "completed"]
        self.invalid = [r for r in self.records if r.status == "invalid"]
        self.by_arm: dict[str, list[TrialRecord]] = {a: [] for a in self.arm_ids}
        for r in self.completed:
            self.by_arm.setdefault(r.arm, []).append(r)
        # Paired outcomes: (block) -> arm -> success. A block is one condition x replicate.
        self.blocks: dict[int, dict[str, bool]] = defaultdict(dict)
        for r in self.completed:
            self.blocks[r.block][r.arm] = bool(r.success)

    def counts(self, arm: str) -> tuple[int, int]:
        rows = self.by_arm.get(arm, [])
        return sum(bool(r.success) for r in rows), len(rows)

    def complete_blocks(self, arms: Sequence[str]) -> list[dict[str, bool]]:
        return [
            outcome for _, outcome in sorted(self.blocks.items()) if all(a in outcome for a in arms)
        ]


def _arm_summaries(data: _Data, codes: dict[str, str], interval: str) -> list[ArmSummary]:
    out = []
    for arm in data.spec.arms:
        k, n = data.counts(arm.id)
        est = proportion_ci(k, n, level=LEVEL, method=interval) if n else None
        out.append(
            ArmSummary(
                arm=arm.id,
                label=arm.label,
                blind_code=codes.get(arm.id, ""),
                successes=k,
                completed=n,
                invalid=sum(r.arm == arm.id for r in data.invalid),
                rate=est.estimate if est else None,
                ci=_ci(est.interval) if est else None,
            )
        )
    return out


def _mde_pp(
    data: _Data, treatment: str, control: str, observed: float, alpha: float
) -> float | None:
    _, n_t = data.counts(treatment)
    k_c, n_c = data.counts(control)
    if not n_t or not n_c:
        return None
    baseline = min(max(k_c / n_c, 0.01), 0.99)
    result = mde(
        baseline, n_c, n_t, alpha=alpha, direction="decrease" if observed < 0 else "increase"
    )
    return result.effect


def _primary(
    data: _Data, info: StudyContextInfo, summary: list[str], extras: _Extras
) -> PrimaryResult:
    spec = data.spec
    primary = spec.analysis.primary
    alternative: Alternative = primary.alternative
    alpha = primary.alpha
    base: dict[str, object] = {"alternative": alternative, "alpha": alpha}

    if spec.design.type == "single_arm":
        arm = data.arm_ids[0]
        k, n = data.counts(arm)
        est = proportion_ci(k, n, level=LEVEL, method=spec.analysis.interval) if n else None
        ci = (est.interval.low, est.interval.high) if est else (math.nan, math.nan)
        if primary.threshold is None:
            if n:
                summary.append(wording.descriptive(arm, k, n, ci, LEVEL))
            return PrimaryResult(
                method="descriptive",
                test=None,
                description="Single arm, no pre-registered threshold: success rate only.",
                treatment=arm,
                control=None,
                n_used=n,
                blocks_excluded=0,
                estimate=est.estimate if est else None,
                estimate_kind="rate",
                ci=_ci(est.interval) if est else None,
                statistic=None,
                pvalue=None,
                rejected=False,
                **base,  # type: ignore[arg-type]
            )
        if not n:
            summary.append(wording.no_data("the threshold test"))
            pvalue, rejected = None, False
        else:
            test = test_vs_threshold(k, n, primary.threshold, alternative=alternative)
            pvalue, rejected = test.pvalue, test.rejects(alpha)
            summary.append(
                wording.threshold(
                    arm=arm,
                    k=k,
                    n=n,
                    ci=ci,
                    p0=primary.threshold,
                    alternative=alternative,
                    pvalue=test.pvalue,
                    rejected=rejected,
                    level=LEVEL,
                )
            )
        return PrimaryResult(
            method="threshold",
            test="binomial",
            description=f"Exact binomial test of {arm} against a threshold of {primary.threshold}.",
            treatment=arm,
            control=None,
            n_used=n,
            blocks_excluded=0,
            estimate=est.estimate if est else None,
            estimate_kind="rate",
            ci=_ci(est.interval) if est else None,
            statistic=None,
            pvalue=pvalue,
            rejected=rejected,
            **base,  # type: ignore[arg-type]
        )

    if spec.design.type == "crossover_rounds":
        return _primary_crossover(data, summary, extras)
    if spec.analysis.selection is not None:
        return _primary_selection(data, info, summary, extras)
    if primary.comparison is None:
        return _primary_ladder(data, summary, extras)
    treatment, control = primary.comparison.treatment, primary.comparison.control

    def rate_ci(k: int, n: int) -> tuple[float, float]:
        iv = proportion_ci(k, n, level=LEVEL, method=spec.analysis.interval).interval
        return iv.low, iv.high

    if len(data.arm_ids) == 2 and spec.conditions.replicates > 1:
        strata: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for r in data.completed:
            cell = strata[r.condition]
            if r.arm == treatment:
                cell[0] += int(bool(r.success))
                cell[1] += 1
            elif r.arm == control:
                cell[2] += int(bool(r.success))
                cell[3] += 1
        usable = [(a, b, c, d) for a, b, c, d in strata.values() if b and d]
        description = (
            f"Cochran–Mantel–Haenszel test of {treatment} vs {control}, stratified by "
            "condition (several replicates per condition)."
        )
        if not usable:
            summary.append(wording.no_data("the stratified comparison"))
            return PrimaryResult(
                method="cmh",
                test="cmh",
                description=description,
                treatment=treatment,
                control=control,
                n_used=0,
                blocks_excluded=0,
                estimate=None,
                estimate_kind="odds_ratio",
                ci=None,
                statistic=None,
                pvalue=None,
                rejected=False,
                **base,  # type: ignore[arg-type]
            )
        res = cmh_test(usable, level=LEVEL)
        # CMH is two-sided by construction; a one-sided alternative halves p in its direction.
        pvalue = res.test.pvalue
        if alternative != "two-sided" and not math.isnan(pvalue):
            agrees = (res.odds_ratio > 1) == (alternative == "greater")
            pvalue = pvalue / 2 if agrees else 1 - pvalue / 2
        rejected = not math.isnan(pvalue) and pvalue <= alpha
        sub = [(a, b, c, d) for a, b, c, d in usable]
        kt, nt = sum(s[0] for s in sub), sum(s[1] for s in sub)
        kc, nc = sum(s[2] for s in sub), sum(s[3] for s in sub)
        observed = kt / nt - kc / nc
        mde_pp = None if rejected else _mde_pp(data, treatment, control, observed, alpha)
        if math.isfinite(res.odds_ratio) and math.isfinite(res.interval.low):
            summary.append(
                wording.odds_ratio(
                    treatment=treatment,
                    control=control,
                    k1=kt,
                    n1=nt,
                    ci1=rate_ci(kt, nt),
                    k2=kc,
                    n2=nc,
                    estimate=res.odds_ratio,
                    ci=(res.interval.low, res.interval.high),
                    pvalue=pvalue,
                    rejected=rejected,
                    mde_pp=mde_pp,
                    level=LEVEL,
                )
            )
        return PrimaryResult(
            method="cmh",
            test="cmh",
            description=description,
            treatment=treatment,
            control=control,
            n_used=res.strata_used,
            blocks_excluded=0,
            estimate=_num(res.odds_ratio),
            estimate_kind="odds_ratio",
            ci=_ci(res.interval),
            statistic=_num(res.test.statistic),
            pvalue=_num(pvalue),
            rejected=rejected,
            mde_pp=mde_pp,
            **base,  # type: ignore[arg-type]
        )

    # Paired designs: the control first, then the other arms in design order.
    arms = [control] + [a for a in data.arm_ids if a != control]
    blocks = data.complete_blocks(arms)
    excluded = sum(1 for outcome in data.blocks.values() if outcome) - len(blocks)
    if excluded:
        summary.append(wording.excluded_blocks(excluded, len(blocks)))

    def pair(arm: str) -> tuple[int, int, int]:
        b = sum(o[arm] and not o[control] for o in blocks)
        c = sum(o[control] and not o[arm] for o in blocks)
        return b, c, len(blocks)

    if len(arms) == 2:
        description = (
            f"Exact McNemar test of {treatment} vs {control} on complete blocks, with a Tango "
            "score interval for the paired difference."
        )
        method = "mcnemar_tango"
    else:
        description = (
            f"Cochran's Q across {len(arms)} arms on complete blocks, then each arm vs "
            f"{control} with exact McNemar tests, {spec.analysis.multiplicity}-adjusted."
        )
        method = "cochran_q"
    if spec.analysis.stopping.rule == "group_sequential":
        return _primary_sequential(data, info, summary, extras, blocks, excluded, rate_ci)
    if spec.analysis.stopping.rule == "anytime":
        return _primary_anytime(data, info, summary, extras, excluded, rate_ci)
    if not blocks:
        summary.append(wording.no_data("the paired comparison"))
        return PrimaryResult(
            method=method,  # type: ignore[arg-type]
            test="mcnemar" if len(arms) == 2 else "cochran_q",
            description=description,
            treatment=treatment,
            control=control,
            n_used=0,
            blocks_excluded=excluded,
            estimate=None,
            estimate_kind="difference",
            ci=None,
            statistic=None,
            pvalue=None,
            rejected=False,
            **base,  # type: ignore[arg-type]
        )

    pairwise: list[Pairwise] = []
    statistic: float | None = None
    if len(arms) == 2:
        b, c, n = pair(treatment)
        mcnemar = compare_paired(b, c, n, alternative=alternative, level=LEVEL).test
        pvalue = mcnemar.pvalue
        adjusted = {treatment: pvalue}
        statistic = _num(mcnemar.statistic)
    else:
        matrix = np.array([[int(o[a]) for a in arms] for o in blocks])
        q = cochran_q(matrix, pairs="vs-first", adjustment=spec.analysis.multiplicity, alpha=alpha)
        statistic = q.statistic
        summary.append(wording.omnibus(len(arms), len(blocks), q.statistic, q.pvalue))
        # The pairwise tests use the pre-registered alternative (cochran_q's are two-sided).
        others = arms[1:]
        raw = {}
        for arm in others:
            b, c, n = pair(arm)
            raw[arm] = compare_paired(b, c, n, alternative=alternative).test.pvalue
        adj = adjust_pvalues(
            [raw[a] for a in others], method=spec.analysis.multiplicity, alpha=alpha
        )
        adjusted = dict(zip(others, adj.adjusted, strict=True))
        pvalue = adjusted[treatment]
    for arm in arms[1:]:
        b, c, n = pair(arm)
        res_arm = compare_paired(b, c, n, alternative=alternative, level=LEVEL)
        pairwise.append(
            Pairwise(
                treatment=arm,
                control=control,
                n_pairs=n,
                b=b,
                c=c,
                difference=res_arm.difference,
                ci=_ci(res_arm.interval),
                pvalue=res_arm.test.pvalue,
                adjusted_pvalue=adjusted[arm],
                rejected=adjusted[arm] <= alpha,
            )
        )
    main = next(p for p in pairwise if p.treatment == treatment)
    rejected = main.rejected
    note = "" if len(arms) == 2 else f", {spec.analysis.multiplicity}-adjusted"
    mde_pp = None
    for p in pairwise:
        kt = sum(o[p.treatment] for o in blocks)
        kc = sum(o[control] for o in blocks)
        n = len(blocks)
        diff = p.difference if p.difference is not None else kt / n - kc / n
        mde_arm = None if p.rejected else _mde_pp(data, p.treatment, control, diff, alpha)
        if p.treatment == treatment:
            mde_pp = mde_arm
        assert p.ci is not None
        summary.append(
            wording.difference(
                treatment=p.treatment,
                control=control,
                k1=kt,
                n1=n,
                ci1=rate_ci(kt, n),
                k2=kc,
                n2=n,
                diff=diff,
                ci=(p.ci.low, p.ci.high),
                test="mcnemar",
                pvalue=p.adjusted_pvalue,
                rejected=p.rejected,
                mde_pp=mde_arm,
                level=LEVEL,
                note=note,
            )
        )
    return PrimaryResult(
        method=method,  # type: ignore[arg-type]
        test="mcnemar" if len(arms) == 2 else "cochran_q",
        description=description,
        treatment=treatment,
        control=control,
        n_used=len(blocks),
        blocks_excluded=excluded,
        estimate=main.difference,
        estimate_kind="difference",
        ci=main.ci,
        statistic=statistic,
        pvalue=pvalue,
        rejected=rejected,
        pairwise=pairwise,
        mde_pp=mde_pp,
        **base,  # type: ignore[arg-type]
    )


def _primary_sequential(
    data: _Data,
    info: StudyContextInfo,
    summary: list[str],
    extras: _Extras,
    blocks: list[dict[str, bool]],
    excluded: int,
    rate_ci: Callable[[int, int], tuple[float, float]],
) -> PrimaryResult:
    spec = data.spec
    primary = spec.analysis.primary
    assert primary.comparison is not None
    treatment, control = primary.comparison.treatment, primary.comparison.control
    stopping = spec.analysis.stopping
    label = wording.SPENDING_LABELS.get(stopping.spending, stopping.spending)
    description = (
        f"Group-sequential comparison of {treatment} vs {control} on complete blocks: McNemar "
        f"score statistic at {stopping.looks} planned looks against {label} error-spending "
        "boundaries, with a stage-wise p-value and a repeated confidence interval."
    )
    base: dict[str, object] = {"alternative": primary.alternative, "alpha": primary.alpha}
    outcome = final_analysis(spec, data.records, info.interim_looks)
    if outcome is None:
        summary.append(wording.no_data("the group-sequential comparison"))
        return PrimaryResult(
            method="group_sequential",
            test="group_sequential",
            description=description,
            treatment=treatment,
            control=control,
            n_used=0,
            blocks_excluded=excluded,
            estimate=None,
            estimate_kind="difference",
            ci=None,
            statistic=None,
            pvalue=None,
            rejected=False,
            **base,  # type: ignore[arg-type]
        )
    extras.sequential = outcome.summary
    extras.notes.extend(outcome.notes)
    seq = outcome.summary
    deciding = seq.looks[(seq.stopped_at or len(seq.looks)) - 1]
    if seq.stopped_at is not None:
        summary.append(
            wording.sequential_stop(
                look=seq.stopped_at,
                looks=seq.planned_looks,
                blocks=deciding.blocks,
                planned=seq.planned_blocks,
                spending=seq.spending,
            )
        )
    n, b, c = outcome.pairs, outcome.b, outcome.c
    pairs_used = blocks if seq.stopped_at is None else None
    kt = kc = 0
    if pairs_used is not None:
        kt = sum(o[treatment] for o in pairs_used)
        kc = sum(o[control] for o in pairs_used)
    else:
        # Counts at the stopping look: concordant successes plus the discordant pairs.
        both = _both_succeeded(data, info, seq.stopped_at or 0, treatment, control)
        kt, kc = both + b, both + c
    diff = (b - c) / n
    ci = _ci(outcome.interval)
    pvalue = outcome.pvalue
    rejected = outcome.rejected
    mde_pp = None if rejected else _mde_pp(data, treatment, control, diff, primary.alpha)
    if pvalue is not None and ci is not None:
        summary.append(
            wording.difference(
                treatment=treatment,
                control=control,
                k1=kt,
                n1=n,
                ci1=rate_ci(kt, n),
                k2=kc,
                n2=n,
                diff=diff,
                ci=(ci.low, ci.high),
                test="group_sequential",
                pvalue=pvalue,
                rejected=rejected,
                mde_pp=mde_pp,
                level=ci.level,
            )
        )
    summary.append(
        wording.sequential_note(
            looks_run=len(seq.looks), looks=seq.planned_looks, spending=seq.spending
        )
    )
    pair = Pairwise(
        treatment=treatment,
        control=control,
        n_pairs=n,
        b=b,
        c=c,
        difference=diff,
        ci=ci,
        pvalue=pvalue if pvalue is not None else math.nan,
        adjusted_pvalue=pvalue if pvalue is not None else math.nan,
        rejected=rejected,
    )
    return PrimaryResult(
        method="group_sequential",
        test="group_sequential",
        description=description,
        treatment=treatment,
        control=control,
        n_used=n,
        blocks_excluded=excluded,
        estimate=diff,
        estimate_kind="difference",
        ci=ci,
        statistic=deciding.z,
        pvalue=pvalue,
        rejected=rejected,
        pairwise=[pair],
        mde_pp=mde_pp,
        **base,  # type: ignore[arg-type]
    )


def _primary_anytime(
    data: _Data,
    info: StudyContextInfo,
    summary: list[str],
    extras: _Extras,
    excluded: int,
    rate_ci: Callable[[int, int], tuple[float, float]],
) -> PrimaryResult:
    spec = data.spec
    primary = spec.analysis.primary
    assert primary.comparison is not None
    treatment, control = primary.comparison.treatment, primary.comparison.control
    description = (
        f"Anytime-valid comparison of {treatment} vs {control}: a betting test of the paired "
        "block differences, checked after every complete block, with an anytime-valid "
        "p-value and a confidence sequence for the paired difference."
    )
    base: dict[str, object] = {"alternative": primary.alternative, "alpha": primary.alpha}
    outcome = anytime_final(spec, data.records, info.interim_looks)
    if outcome is None:
        summary.append(wording.no_data("the anytime-valid comparison"))
        return PrimaryResult(
            method="anytime",
            test="betting_paired",
            description=description,
            treatment=treatment,
            control=control,
            n_used=0,
            blocks_excluded=excluded,
            estimate=None,
            estimate_kind="difference",
            ci=None,
            statistic=None,
            pvalue=None,
            rejected=False,
            **base,  # type: ignore[arg-type]
        )
    extras.anytime = outcome.summary
    extras.notes.extend(outcome.notes)
    res = outcome.result
    seq = outcome.summary
    if seq.stopped_at is not None:
        summary.append(wording.anytime_stop(blocks=seq.stopped_at, planned=seq.planned_blocks))
    used = set(range(1, seq.blocks + 1))
    pairs = {b: o for b, o in complete_pairs(data.records, treatment, control).items() if b in used}
    n = len(pairs)
    kt = sum(t for t, _ in pairs.values())
    kc = sum(c for _, c in pairs.values())
    b = sum(t and not c for t, c in pairs.values())
    c = sum(c and not t for t, c in pairs.values())
    interval = res.sequence.interval()
    ci = CI(low=interval.low, high=interval.high, level=interval.level, method=interval.method)
    rejected = res.rejected_at is not None
    pvalue = res.test.pvalue
    summary.append(
        wording.difference(
            treatment=treatment,
            control=control,
            k1=kt,
            n1=n,
            ci1=rate_ci(kt, n),
            k2=kc,
            n2=n,
            diff=res.estimate,
            ci=(ci.low, ci.high),
            test="betting_paired",
            pvalue=pvalue,
            rejected=rejected,
            mde_pp=None,
            level=ci.level,
            power_note=wording.ANYTIME_POWER,
        )
    )
    summary.append(wording.anytime_note(blocks=n))
    pair = Pairwise(
        treatment=treatment,
        control=control,
        n_pairs=n,
        b=b,
        c=c,
        difference=res.estimate,
        ci=ci,
        pvalue=pvalue,
        adjusted_pvalue=pvalue,
        rejected=rejected,
    )
    return PrimaryResult(
        method="anytime",
        test="betting_paired",
        description=description,
        treatment=treatment,
        control=control,
        n_used=n,
        blocks_excluded=excluded,
        estimate=res.estimate,
        estimate_kind="difference",
        ci=ci,
        statistic=res.test.statistic,
        pvalue=pvalue,
        rejected=rejected,
        pairwise=[pair],
        **base,  # type: ignore[arg-type]
    )


def _primary_selection(
    data: _Data, info: StudyContextInfo, summary: list[str], extras: _Extras
) -> PrimaryResult:
    spec = data.spec
    selection = spec.analysis.selection
    assert selection is not None
    description = (
        f"Best-arm selection over {len(data.arm_ids)} arms by successive elimination on "
        "paired block differences (betting confidence sequences, union bound over all "
        f"pairs, δ = {selection.delta:g}), checked after every complete block."
    )
    base: dict[str, object] = {
        "alternative": "two-sided",
        "alpha": selection.delta,
    }
    outcome = selection_final(spec, data.records, info.selection_looks)
    if outcome is None:
        summary.append(wording.no_data("best-arm selection"))
        return PrimaryResult(
            method="selection",
            test="elimination",
            description=description,
            treatment=None,
            control=None,
            n_used=0,
            blocks_excluded=0,
            estimate=None,
            estimate_kind="difference",
            ci=None,
            statistic=None,
            pvalue=None,
            rejected=False,
            **base,  # type: ignore[arg-type]
        )
    extras.selection = outcome.summary
    extras.notes.extend(outcome.notes)
    sel = outcome.summary
    summary.append(
        wording.selection(
            survivors=sel.survivors,
            eliminations=[(e.arm, e.block) for e in sel.eliminations],
            blocks=sel.blocks,
            planned=sel.planned_blocks,
            delta=sel.delta,
        )
    )
    return PrimaryResult(
        method="selection",
        test="elimination",
        description=description,
        treatment=sel.best,
        control=None,
        n_used=sel.blocks,
        blocks_excluded=0,
        estimate=None,
        estimate_kind="difference",
        ci=None,
        statistic=None,
        pvalue=None,
        rejected=sel.best is not None,
        **base,  # type: ignore[arg-type]
    )


def _both_succeeded(
    data: _Data, info: StudyContextInfo, look: int, treatment: str, control: str
) -> int:
    recorded = next(x for x in info.interim_looks if int(x["look"]) == look)
    wanted = set(recorded.get("blocks", []))
    return sum(
        1
        for block, outcome in data.blocks.items()
        if block in wanted and outcome.get(treatment) and outcome.get(control)
    )


def _primary_crossover(data: _Data, summary: list[str], extras: _Extras) -> PrimaryResult:
    spec = data.spec
    primary = spec.analysis.primary
    assert primary.comparison is not None
    treatment, control = primary.comparison.treatment, primary.comparison.control
    outcome = crossover_analysis(spec, data.records)
    extras.crossover = outcome.summary
    base: dict[str, object] = {"alternative": primary.alternative, "alpha": primary.alpha}
    description = (
        f"Crossover rounds: {treatment} vs {control}, each arm running whole rounds in "
        "cycles of two with randomized order. Period-adjusted difference of round success "
        "rates (Hills–Armitage) with an exact randomization test over the cycle orders."
    )
    res = outcome.result
    if res is None:
        summary.append(wording.no_data("the crossover comparison (both orders are needed)"))
        return PrimaryResult(
            method="crossover",
            test=None,
            description=description,
            treatment=treatment,
            control=control,
            n_used=outcome.summary.cycles_used,
            blocks_excluded=0,
            estimate=None,
            estimate_kind="difference",
            ci=None,
            statistic=None,
            pvalue=None,
            rejected=False,
            **base,  # type: ignore[arg-type]
        )
    rejected = res.test.rejects(primary.alpha)
    ci = _ci(res.interval)
    mde_pp = None
    if not rejected and res.standard_error and res.cycles > 2:
        # Smallest difference with 80% power, given the observed round-to-round variation.
        df = res.cycles - 2
        tail = primary.alpha / 2 if primary.alternative == "two-sided" else primary.alpha
        crit = sp_stats.t.isf(tail, df) + sp_stats.t.isf(0.2, df)
        mde_pp = float(crit * res.standard_error)
    rate = {
        arm: outcome.successes[arm] / outcome.completed[arm] if outcome.completed[arm] else 0.0
        for arm in (treatment, control)
    }
    summary.append(
        wording.crossover(
            treatment=treatment,
            control=control,
            cycles=res.cycles,
            rate_t=rate[treatment],
            rate_c=rate[control],
            diff=res.difference,
            ci=(ci.low, ci.high) if ci else None,
            test=res.test.test,
            pvalue=res.test.pvalue,
            rejected=rejected,
            mde_pp=mde_pp,
            level=LEVEL,
        )
    )
    return PrimaryResult(
        method="crossover",
        test=res.test.test,
        description=description,
        treatment=treatment,
        control=control,
        n_used=res.cycles,
        blocks_excluded=0,
        estimate=res.difference,
        estimate_kind="difference",
        ci=ci,
        statistic=_num(res.test.statistic),
        pvalue=_num(res.test.pvalue),
        rejected=rejected,
        mde_pp=mde_pp,
        **base,  # type: ignore[arg-type]
    )


def _ladder_sentences(data: _Data, ladder: LadderResult, summary: list[str]) -> None:
    a = ladder.association
    summary.append(
        wording.step_association(
            test=a.test,
            statistic=a.statistic,
            pvalue=a.pvalue,
            rejected=a.rejected,
            checkpoints=len(ladder.arms),
        )
    )
    first, last = ladder.arms[0], ladder.arms[-1]
    if a.primary and a.pvalue is not None and not a.rejected:
        k1, n1 = data.counts(last)
        k2, n2 = data.counts(first)
        if n1 and n2:
            observed = k1 / n1 - k2 / n2
            mde_pp = _mde_pp(data, last, first, observed, data.spec.analysis.primary.alpha)
            summary.append(
                wording.check(
                    "Between the first and last checkpoint: "
                    + wording.no_difference_power(n1, n2, mde_pp)
                )
            )
    if ladder.margin is not None and ladder.plateau:
        summary.append(
            wording.plateau(
                plateau_arm=ladder.plateau_arm,
                final_arm=last,
                margin=ladder.margin,
                alpha=data.spec.analysis.primary.alpha,
            )
        )


def _primary_ladder(data: _Data, summary: list[str], extras: _Extras) -> PrimaryResult:
    spec = data.spec
    primary = spec.analysis.primary
    ladder = ladder_analysis(spec, data.records, primary=True)
    assert ladder is not None
    extras.ladder = ladder
    _ladder_sentences(data, ladder, summary)
    first, last = ladder.arms[0], ladder.arms[-1]
    k1, n1 = data.counts(last)
    k2, n2 = data.counts(first)
    a = ladder.association
    return PrimaryResult(
        method="ladder",
        test=a.test,
        description=(
            f"Mantel's test of a linear association between training step and success "
            f"across {len(ladder.arms)} checkpoints, stratified by condition, with the "
            "pre-registered steps as scores."
        ),
        treatment=last,
        control=first,
        n_used=len({r.condition for r in data.completed if r.arm in ladder.arms}),
        blocks_excluded=0,
        estimate=k1 / n1 - k2 / n2 if n1 and n2 else None,
        estimate_kind="difference",
        ci=None,
        statistic=a.statistic,
        pvalue=a.pvalue,
        rejected=a.rejected,
        alternative=a.alternative,
        alpha=primary.alpha,
    )


def _episodes(data: _Data) -> list[EpisodeSummary]:
    out = []
    for arm in data.arm_ids:
        links = [r.episode for r in data.records if r.arm == arm and r.episode]
        if not links:
            continue
        counts = [link.get("intervention_frames") for link in links]
        known = [int(c) for c in counts if c is not None]
        out.append(
            EpisodeSummary(
                arm=arm,
                linked_trials=len(links),
                frames=sum(int(link.get("length", 0)) for link in links),
                trials_with_intervention=sum(1 for c in known if c > 0) if known else None,
                intervention_frames=sum(known) if known else None,
            )
        )
    return out


def _rig_checks(info: StudyContextInfo) -> list[RigCheckRow]:
    return [
        RigCheckRow(
            checked_at=c["ts"],
            session_id=c.get("session_id"),
            source=str(c.get("source", "")),
            shift_px=float(c.get("shift_px", 0.0)),
            brightness_change=float(c.get("brightness_change", 0.0)),
            similarity=float(c.get("similarity", 0.0)),
            flagged=bool(c.get("flagged")),
            reasons=[str(r) for r in c.get("reasons", [])],
        )
        for c in info.rig_checks
    ]


def _runner(data: _Data) -> list[RunnerSummary]:
    """Per-arm runner measurements, for trials whose runner reported any."""
    out = []
    for arm in data.arm_ids:
        rows = [r.runner_metrics for r in data.records if r.arm == arm and r.runner_metrics]
        if not rows:
            continue
        medians = [m["latency_ms_median"] for m in rows if "latency_ms_median" in m]
        p95s = [m["latency_ms_p95"] for m in rows if "latency_ms_p95" in m]
        has_requests = any("requests" in m for m in rows)
        has_exit = any("exit_code" in m for m in rows)
        out.append(
            RunnerSummary(
                arm=arm,
                trials=len(rows),
                requests=int(sum(m.get("requests", 0) for m in rows)) if has_requests else None,
                errors=int(sum(m.get("errors", 0) for m in rows)) if has_requests else None,
                latency_ms_median=float(np.median(medians)) if medians else None,
                latency_ms_p95=max(p95s) if p95s else None,
                abnormal_exits=(
                    sum(
                        1
                        for m in rows
                        if m.get("exited_before_stop") and m.get("exit_code", 0) != 0
                    )
                    if has_exit
                    else None
                ),
            )
        )
    return out


def _sensitivity(data: _Data, summary: list[str]) -> list[IndependentComparison]:
    primary = data.spec.analysis.primary
    if primary.comparison is None:
        return []
    control = primary.comparison.control
    k2, n2 = data.counts(control)
    out = []
    for arm in data.arm_ids:
        if arm == control:
            continue
        k1, n1 = data.counts(arm)
        if not n1 or not n2:
            continue
        res = compare_independent(
            k1, n1, k2, n2, test="boschloo", alternative=primary.alternative, level=LEVEL
        )
        fisher = next(t for t in res.secondary if t.test == "fisher")
        out.append(
            IndependentComparison(
                treatment=arm,
                control=control,
                k1=k1,
                n1=n1,
                k2=k2,
                n2=n2,
                difference=res.difference,
                ci=CI(
                    low=res.interval.low,
                    high=res.interval.high,
                    level=res.interval.level,
                    method=res.interval.method,
                ),
                boschloo_p=res.primary.pvalue,
                fisher_p=fisher.pvalue,
                alternative=primary.alternative,
            )
        )
        summary.append(
            wording.sensitivity(
                treatment=arm,
                control=control,
                diff=res.difference,
                ci=(res.interval.low, res.interval.high),
                pvalue=res.primary.pvalue,
            )
        )
    return out


def _stages(data: _Data) -> tuple[list[StageSummary], list[StageComparison]]:
    spec = data.spec
    names = [s.id for s in spec.rubric.stages]
    n_stages = len(names)
    summaries = []
    per_arm: dict[str, list[int]] = {}
    for arm in data.arm_ids:
        stages = [r.stage_index for r in data.by_arm.get(arm, []) if r.stage_index is not None]
        per_arm[arm] = stages
        if not stages:
            summaries.append(StageSummary(arm=arm, n=0, distribution=[], funnel=[]))
            continue
        dist = stage_distribution(stages, n_stages)
        funnel = stage_funnel(stages, n_stages, level=LEVEL)
        summaries.append(
            StageSummary(
                arm=arm,
                n=len(stages),
                distribution=[
                    StageShare(
                        stage=None if d.stage < 0 else names[d.stage], count=d.count, share=d.share
                    )
                    for d in dist
                ],
                funnel=[
                    FunnelStep(
                        stage=names[f.stage],
                        entered=f.entered,
                        reached=f.reached,
                        conversion=_num(f.conversion),
                        ci=None if math.isnan(f.conversion) else _ci(f.interval),
                        cumulative=f.cumulative,
                    )
                    for f in funnel
                ],
            )
        )
    comparisons = []
    primary = spec.analysis.primary
    if primary.comparison is not None:
        control = primary.comparison.control
        for arm in data.arm_ids:
            if arm == control or len(per_arm[arm]) < 2 or len(per_arm[control]) < 2:
                continue
            test = compare_stages(per_arm[arm], per_arm[control], alternative=primary.alternative)
            comparisons.append(
                StageComparison(
                    treatment=arm,
                    control=control,
                    statistic=_num(test.statistic),
                    pvalue=_num(test.pvalue),
                    preregistered="stage_reached" in spec.analysis.secondary,
                )
            )
    return summaries, comparisons


def _timing(data: _Data) -> list[TimingSummary]:
    out = []
    pre = "time_to_success" in data.spec.analysis.secondary
    for i, arm in enumerate(data.arm_ids):
        rows = [r for r in data.by_arm.get(arm, []) if r.duration_s is not None]
        wins = [float(r.duration_s) for r in rows if r.success and r.duration_s is not None]
        med = median_time_to_success(
            wins, level=LEVEL, n_boot=_TIMING_BOOT, seed=data.spec.design.seed + i
        )
        curve = (
            success_curve(
                [float(r.duration_s or 0.0) for r in rows], [bool(r.success) for r in rows]
            )
            if rows
            else None
        )
        out.append(
            TimingSummary(
                arm=arm,
                n_successes=med.n_successes,
                median_s=_num(med.median),
                ci=_ci(med.interval),
                curve_times=list(curve.times) if curve else [],
                curve_fraction=list(curve.fraction) if curve else [],
                preregistered=pre,
            )
        )
    return out


def _cells(rows: Sequence[TrialRecord], arms: Sequence[str]) -> list[ConditionCell]:
    return [
        ConditionCell(
            arm=arm,
            successes=sum(bool(r.success) for r in rows if r.arm == arm),
            completed=sum(r.arm == arm for r in rows),
        )
        for arm in arms
    ]


def _conditions(data: _Data) -> list[ConditionRow]:
    by_condition: dict[str, list[TrialRecord]] = defaultdict(list)
    for r in data.completed:
        by_condition[r.condition].append(r)
    order = [c.key for c in design_conditions(data.spec)]
    order += sorted(set(by_condition) - set(order))
    return [
        ConditionRow(condition=key, cells=_cells(by_condition.get(key, []), data.arm_ids))
        for key in order
    ]


def _sessions(data: _Data) -> list[SessionRow]:
    groups: dict[str, list[TrialRecord]] = defaultdict(list)
    for r in data.completed:
        groups[r.session_id].append(r)
    rows = []
    for session_id, trials in groups.items():
        first = min(trials, key=lambda r: r.started_at)
        rows.append(
            SessionRow(
                session_id=session_id,
                operator=first.operator,
                rig=first.rig,
                started_at=first.started_at,
                cells=_cells(trials, data.arm_ids),
            )
        )
    return sorted(rows, key=lambda s: s.started_at)


def _drift(data: _Data, summary: list[str]) -> list[DriftCheck]:
    checks = []
    for grouping in ("session", "operator"):
        for arm in data.arm_ids:
            groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for r in data.by_arm.get(arm, []):
                key = r.session_id if grouping == "session" else r.operator
                groups[key][0] += int(bool(r.success))
                groups[key][1] += 1
            usable = [(k, n) for k, n in groups.values() if n]
            if len(usable) < 2:
                continue
            res = homogeneity_test(usable)
            pvalue = _num(res.pvalue)
            flagged = pvalue is not None and pvalue < FLAG_ALPHA
            if flagged and pvalue is not None:
                summary.append(wording.drift_flag(arm, grouping, len(usable), pvalue))
            checks.append(
                DriftCheck(
                    arm=arm,
                    grouping=grouping,
                    groups=len(usable),
                    method=res.method,
                    pvalue=pvalue,
                    flagged=flagged,
                    small_expected=res.small_expected,
                )
            )
    return checks


def _invalid(data: _Data, summary: list[str]) -> InvalidCheck:
    per_arm = {a: sum(r.arm == a for r in data.invalid) for a in data.arm_ids}
    attempts = {a: per_arm[a] + len(data.by_arm.get(a, [])) for a in data.arm_ids}
    groups = [(per_arm[a], attempts[a]) for a in data.arm_ids if attempts[a]]
    if len(data.arm_ids) < 2 or len(groups) < 2 or not sum(per_arm.values()):
        return InvalidCheck(
            per_arm=per_arm, attempts=attempts, method=None, pvalue=None, flagged=False
        )
    res = homogeneity_test(groups)
    pvalue = _num(res.pvalue)
    flagged = pvalue is not None and pvalue < FLAG_ALPHA
    if flagged and pvalue is not None:
        summary.append(wording.invalid_flag(per_arm, attempts, pvalue))
    return InvalidCheck(
        per_arm=per_arm, attempts=attempts, method=res.method, pvalue=pvalue, flagged=flagged
    )


def _deviations(data: _Data, info: StudyContextInfo, excluded: int, used: int) -> list[Deviation]:
    out = [
        Deviation(
            kind="amendment",
            message=(
                f"Amended on {str(a.get('ts', ''))[:10]} by {a.get('actor', '?')}: "
                f"{a.get('reason', '')} (design {str(a.get('old_hash', ''))[:12]} → "
                f"{str(a.get('new_hash', ''))[:12]}; {a.get('added_slots', 0)} trials "
                f"added, {a.get('voided_slots', 0)} removed)."
            ),
        )
        for a in info.amendments
    ]
    if (
        data.spec.design.blinding == "operator"
        and info.unblinded_at is not None
        and (info.pending_at_unblinding or 0) > 0
    ):
        out.append(
            Deviation(
                kind="early_unblinding",
                message=(
                    f"Unblinded on {info.unblinded_at.isoformat(timespec='minutes')} before "
                    f"the study was complete ({info.pending_at_unblinding} trials were still "
                    "pending)."
                ),
            )
        )
    if info.edits_after_unblinding:
        n = info.edits_after_unblinding
        out.append(
            Deviation(
                kind="late_edit",
                message=(
                    f"{n} trial label{'s were' if n != 1 else ' was'} edited after unblinding; "
                    "the event log has the old and new values."
                ),
            )
        )
    if info.pending_slots:
        out.append(
            Deviation(
                kind="incomplete",
                message=(
                    f"The study is not complete: {info.pending_slots} of {info.planned_slots} "
                    "planned trials have not been run."
                ),
            )
        )
    ordered = sorted(data.records, key=lambda r: (r.started_at, r.seq))
    highest, out_of_order = -1, 0
    for r in ordered:
        if r.seq < highest:
            out_of_order += 1
        highest = max(highest, r.seq)
    if out_of_order:
        out.append(
            Deviation(
                kind="out_of_order",
                message=f"{out_of_order} trials were run out of the scheduled order.",
            )
        )
    if excluded:
        out.append(
            Deviation(kind="excluded_blocks", message=wording.excluded_blocks(excluded, used))
        )
    return out


def analyze(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    info: StudyContextInfo,
    *,
    blind_codes: dict[str, str] | None = None,
    now: datetime | None = None,
) -> Results:
    """Analyze a study's trials according to its locked design."""
    data = _Data(spec, records)
    codes = blind_codes or {r.arm: r.blind_code for r in records}
    summary: list[str] = []
    extras = _Extras()
    primary = _primary(data, info, summary, extras)
    if spec.analysis.ladder is not None and extras.ladder is None:
        extras.ladder = ladder_analysis(spec, records, primary=False)
        assert extras.ladder is not None
        _ladder_sentences(data, extras.ladder, summary)
    sensitivity = _sensitivity(data, summary)
    stages, stage_comparisons = _stages(data)
    drift = _drift(data, summary)
    invalid = _invalid(data, summary)
    deviations = _deviations(data, info, primary.blocks_excluded, primary.n_used)
    deviations += [Deviation(kind="interim", message=note) for note in extras.notes]
    rig_rows = _rig_checks(info)
    agreement_rows = agreement(info.proxy_items)
    summary.extend(
        wording.agreement(
            model=row.model,
            n=row.n,
            agreement=row.agreement,
            kappa=row.kappa,
            ci=None if row.ci is None else (row.ci.low, row.ci.high),
        )
        for row in agreement_rows
    )
    arm_of = {code: arm for arm, code in codes.items()}
    proxy_rows, proxy_diffs, proxy_notes = proxy_estimates(spec, info.proxy_items, arm_of)
    summary.extend(
        wording.proxy_estimate(
            arm=prow.arm,
            source=prow.source,
            estimate=prow.estimate,
            ci=(prow.ci.low, prow.ci.high),
            classical=(prow.classical_ci.low, prow.classical_ci.high),
            labeled=prow.labeled,
            unlabeled=prow.unlabeled,
        )
        for prow in proxy_rows
    )
    summary.extend(proxy_notes)
    deviations += [
        Deviation(
            kind="rig_drift",
            message=wording.rig_drift(row.checked_at, row.reasons),
        )
        for row in rig_rows
        if row.flagged
    ]
    comparison = spec.analysis.primary.comparison
    sessions = _sessions(data)
    study = StudyInfo(
        name=spec.name,
        title=spec.title,
        design_hash=info.design_hash,
        design_type=spec.design.type,
        status=info.status,
        seed=spec.design.seed,
        blinding=spec.design.blinding,
        arms=data.arm_ids,
        treatment=comparison.treatment if comparison else None,
        control=comparison.control if comparison else None,
        threshold=spec.analysis.primary.threshold,
        alternative=spec.analysis.primary.alternative,
        alpha=spec.analysis.primary.alpha,
        multiplicity=spec.analysis.multiplicity,
        n_conditions=spec.conditions.count(),
        replicates=spec.conditions.replicates,
        planned_trials=info.planned_slots,
        pending_trials=info.pending_slots,
        stages=[s.id for s in spec.rubric.stages],
        success_stage=spec.rubric.success,
    )
    provenance = Provenance(
        generated_at=now or datetime.now(UTC),
        fieldtrial_version=__version__,
        locked_with_version=info.fieldtrial_version,
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        design_hash=info.design_hash,
        seed=spec.design.seed,
        policy_fingerprints={
            a.id: _fingerprint({"policy": a.policy, "serving": a.serving, "runner": a.runner})
            for a in spec.arms
        },
        sessions=len({r.session_id for r in records}),
        operators=sorted({r.operator for r in records}),
        rigs=sorted({r.rig for r in records}),
    )
    return Results(
        study=study,
        arms=_arm_summaries(data, codes, spec.analysis.interval),
        primary=primary,
        sensitivity=sensitivity,
        stages=stages,
        stage_comparisons=stage_comparisons,
        timing=_timing(data),
        conditions=_conditions(data),
        sessions=sessions,
        drift=drift,
        invalid=invalid,
        deviations=deviations,
        provenance=provenance,
        summary=summary,
        ladder=extras.ladder,
        crossover=extras.crossover,
        sequential=extras.sequential,
        anytime=extras.anytime,
        selection=extras.selection,
        agreement=agreement_rows,
        proxy=proxy_rows,
        proxy_differences=proxy_diffs,
        runner=_runner(data),
        rig_checks=rig_rows,
        episodes=_episodes(data),
    )
