"""Checkpoint ladders in a study: association with training step, and plateau detection.

The association test is Mantel's stratified test, with conditions as strata and the
pre-registered steps as scores (docs/stats/ladders.md). The plateau uses Tango intervals on
complete blocks when every condition runs once per checkpoint, and Newcombe intervals on
the pooled counts otherwise.
"""

from collections.abc import Sequence

import numpy as np

from fieldtrial.analysis.common import ci_model, num
from fieldtrial.analysis.records import TrialRecord
from fieldtrial.analysis.results import LadderResult, PlateauRow, StepAssociation
from fieldtrial.design import StudySpec
from fieldtrial.stats import plateau, plateau_paired, stratified_trend_test


def ladder_analysis(
    spec: StudySpec, records: Sequence[TrialRecord], *, primary: bool
) -> LadderResult | None:
    """Analyze the pre-registered ladder, or return None when the study has none."""
    ladder = spec.analysis.ladder
    if ladder is None:
        return None
    arms = list(ladder.arms)
    alpha = spec.analysis.primary.alpha
    alternative = spec.analysis.primary.alternative if primary else "two-sided"
    completed = [r for r in records if r.status == "completed" and r.arm in arms]

    strata: dict[str, dict[str, list[int]]] = {}
    for r in completed:
        cell = strata.setdefault(r.condition, {a: [0, 0] for a in arms})[r.arm]
        cell[0] += int(bool(r.success))
        cell[1] += 1
    test_name = "mantel"
    statistic = pvalue = None
    if strata:
        res = stratified_trend_test(
            [[(c[a][0], c[a][1]) for a in arms] for c in strata.values()],
            scores=ladder.scores(),
            alternative=alternative,
        )
        statistic, pvalue = num(res.test.statistic), num(res.test.pvalue)
        if res.strata_used == 0:
            statistic = pvalue = None
    association = StepAssociation(
        test=test_name,
        statistic=statistic,
        pvalue=pvalue,
        alternative=alternative,
        rejected=pvalue is not None and pvalue <= alpha,
        primary=primary,
    )

    rows: list[PlateauRow] = []
    plateau_arm: str | None = None
    method: str | None = None
    if ladder.margin is not None:
        result = None
        if spec.conditions.replicates == 1:
            blocks: dict[int, dict[str, bool]] = {}
            for r in completed:
                blocks.setdefault(r.block, {})[r.arm] = bool(r.success)
            matrix = [[int(b[a]) for a in arms] for b in blocks.values() if len(b) == len(arms)]
            if matrix:
                result = plateau_paired(np.array(matrix), margin=ladder.margin, alpha=alpha)
        else:
            counts = [
                (
                    sum(int(bool(r.success)) for r in completed if r.arm == a),
                    sum(1 for r in completed if r.arm == a),
                )
                for a in arms
            ]
            if all(n for _, n in counts):
                result = plateau(counts, margin=ladder.margin, alpha=alpha)
        if result is not None:
            method = result.method
            for step in result.steps:
                interval = ci_model(step.interval)
                assert interval is not None
                rows.append(
                    PlateauRow(
                        arm=arms[step.index],
                        difference=step.difference,
                        ci=interval,
                        tested=step.tested,
                        noninferior=step.noninferior,
                    )
                )
            if result.plateau_index is not None:
                plateau_arm = arms[result.plateau_index]
    return LadderResult(
        arms=arms,
        steps=ladder.scores(),
        association=association,
        margin=ladder.margin,
        plateau_method=method,
        plateau=rows,
        plateau_arm=plateau_arm,
    )
