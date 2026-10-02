"""Statistics for comparing robot policies.

This package is pure: it depends only on numpy and scipy, does no I/O, and imports no
other fieldtrial package. CI enforces this with import-linter contracts.
"""

from fieldtrial.stats._types import (
    AdjustedPValues,
    Alternative,
    ComparisonResult,
    Interval,
    OddsRatio,
    ProportionEstimate,
    TestResult,
)
from fieldtrial.stats.bayes import credible_interval, prob_superiority
from fieldtrial.stats.compare import INDEPENDENT_TESTS, compare_independent
from fieldtrial.stats.drift import HomogeneityResult, homogeneity_test
from fieldtrial.stats.multiplicity import ADJUST_METHODS, adjust_pvalues
from fieldtrial.stats.ordinal import compare_stages, stage_distribution, stage_funnel
from fieldtrial.stats.paired import cochran_q, compare_paired, mcnemar_exact, tango_interval
from fieldtrial.stats.power import (
    POWER_METHODS,
    boschloo_power,
    mcnemar_power,
    mde,
    power,
    sample_size,
    simulate_power,
)
from fieldtrial.stats.proportions import INTERVAL_METHODS, proportion_ci, test_vs_threshold
from fieldtrial.stats.stratified import StratifiedResult, cmh_test
from fieldtrial.stats.timing import median_time_to_success, success_curve

__all__ = [
    "ADJUST_METHODS",
    "INDEPENDENT_TESTS",
    "INTERVAL_METHODS",
    "POWER_METHODS",
    "AdjustedPValues",
    "Alternative",
    "ComparisonResult",
    "HomogeneityResult",
    "Interval",
    "OddsRatio",
    "ProportionEstimate",
    "StratifiedResult",
    "TestResult",
    "adjust_pvalues",
    "boschloo_power",
    "cmh_test",
    "cochran_q",
    "compare_independent",
    "compare_paired",
    "compare_stages",
    "credible_interval",
    "homogeneity_test",
    "mcnemar_exact",
    "mcnemar_power",
    "mde",
    "median_time_to_success",
    "power",
    "prob_superiority",
    "proportion_ci",
    "sample_size",
    "simulate_power",
    "stage_distribution",
    "stage_funnel",
    "success_curve",
    "tango_interval",
    "test_vs_threshold",
]
