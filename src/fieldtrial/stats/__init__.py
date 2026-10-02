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
from fieldtrial.stats.crossover import CrossoverResult, crossover_test
from fieldtrial.stats.drift import HomogeneityResult, homogeneity_test
from fieldtrial.stats.ladder import (
    PlateauResult,
    TrendResult,
    plateau,
    plateau_paired,
    stratified_trend_test,
    trend_test,
)
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
from fieldtrial.stats.sequential import (
    SPENDING_FUNCTIONS,
    SequentialDesign,
    SequentialResult,
    alpha_spending,
    constant_boundaries,
    crossing_probabilities,
    repeated_interval,
    sequential_test,
    spending_boundaries,
)
from fieldtrial.stats.stratified import StratifiedResult, cmh_test
from fieldtrial.stats.timing import median_time_to_success, success_curve

__all__ = [
    "ADJUST_METHODS",
    "INDEPENDENT_TESTS",
    "INTERVAL_METHODS",
    "POWER_METHODS",
    "SPENDING_FUNCTIONS",
    "AdjustedPValues",
    "Alternative",
    "ComparisonResult",
    "CrossoverResult",
    "HomogeneityResult",
    "Interval",
    "OddsRatio",
    "PlateauResult",
    "ProportionEstimate",
    "SequentialDesign",
    "SequentialResult",
    "StratifiedResult",
    "TestResult",
    "TrendResult",
    "adjust_pvalues",
    "alpha_spending",
    "boschloo_power",
    "cmh_test",
    "cochran_q",
    "compare_independent",
    "compare_paired",
    "compare_stages",
    "constant_boundaries",
    "credible_interval",
    "crossing_probabilities",
    "crossover_test",
    "homogeneity_test",
    "mcnemar_exact",
    "mcnemar_power",
    "mde",
    "median_time_to_success",
    "plateau",
    "plateau_paired",
    "power",
    "prob_superiority",
    "proportion_ci",
    "repeated_interval",
    "sample_size",
    "sequential_test",
    "simulate_power",
    "spending_boundaries",
    "stage_distribution",
    "stage_funnel",
    "stratified_trend_test",
    "success_curve",
    "tango_interval",
    "test_vs_threshold",
    "trend_test",
]
