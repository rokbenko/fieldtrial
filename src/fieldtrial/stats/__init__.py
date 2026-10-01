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
from fieldtrial.stats.compare import INDEPENDENT_TESTS, compare_independent
from fieldtrial.stats.multiplicity import ADJUST_METHODS, adjust_pvalues
from fieldtrial.stats.paired import cochran_q, compare_paired, mcnemar_exact, tango_interval
from fieldtrial.stats.proportions import INTERVAL_METHODS, proportion_ci, test_vs_threshold

__all__ = [
    "ADJUST_METHODS",
    "INDEPENDENT_TESTS",
    "INTERVAL_METHODS",
    "AdjustedPValues",
    "Alternative",
    "ComparisonResult",
    "Interval",
    "OddsRatio",
    "ProportionEstimate",
    "TestResult",
    "adjust_pvalues",
    "cochran_q",
    "compare_independent",
    "compare_paired",
    "mcnemar_exact",
    "proportion_ci",
    "tango_interval",
    "test_vs_threshold",
]
