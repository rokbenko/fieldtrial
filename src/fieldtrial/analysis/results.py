"""The versioned ``Results`` model: everything a report shows, as plain JSON-ready data.

``SCHEMA_VERSION`` changes whenever a field is renamed or removed; adding optional fields
does not change it. Rates and differences are proportions (0.925, not 92.5); intervals are
``[low, high]``. Values that cannot be computed are ``None``.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1

PrimaryMethod = Literal[
    "mcnemar_tango",  # 2 arms, complete blocks
    "cochran_q",  # more than 2 arms, complete blocks, pairwise McNemar with adjustment
    "cmh",  # 2 arms with replicates, stratified by condition
    "threshold",  # single arm against a fixed threshold
    "descriptive",  # single arm without a threshold: no test
]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CI(_Model):
    """A confidence interval."""

    low: float
    high: float
    level: float
    method: str


class StudyInfo(_Model):
    """The locked design, as analyzed."""

    name: str
    title: str | None
    design_hash: str
    design_type: str
    status: str
    seed: int
    blinding: str
    arms: list[str]
    treatment: str | None
    control: str | None
    threshold: float | None
    alternative: str
    alpha: float
    multiplicity: str
    n_conditions: int
    replicates: int
    planned_trials: int
    pending_trials: int
    stages: list[str]
    success_stage: str


class ArmSummary(_Model):
    """Per-arm counts. ``rate`` and ``ci`` are ``None`` when the arm has no completed trial."""

    arm: str
    label: str | None
    blind_code: str
    successes: int
    completed: int
    invalid: int
    rate: float | None
    ci: CI | None


class Pairwise(_Model):
    """One arm compared with another inside the primary analysis."""

    treatment: str
    control: str
    n_pairs: int
    b: int  # blocks where only the treatment succeeded
    c: int  # blocks where only the control succeeded
    difference: float | None
    ci: CI | None
    pvalue: float
    adjusted_pvalue: float
    rejected: bool


class PrimaryResult(_Model):
    """The pre-registered primary analysis, chosen from the locked design."""

    method: PrimaryMethod
    test: str | None
    description: str
    treatment: str | None
    control: str | None
    n_used: int  # blocks (paired), strata (CMH) or trials (single arm)
    blocks_excluded: int
    estimate: float | None
    estimate_kind: Literal["difference", "odds_ratio", "rate"]
    ci: CI | None
    statistic: float | None
    pvalue: float | None
    alternative: str
    alpha: float
    rejected: bool
    pairwise: list[Pairwise] = Field(default_factory=list)
    mde_pp: float | None = None  # the smallest difference with 80% power, for the wording


class IndependentComparison(_Model):
    """Arms treated as independent samples (always reported, as a sensitivity analysis)."""

    treatment: str
    control: str
    k1: int
    n1: int
    k2: int
    n2: int
    difference: float
    ci: CI
    boschloo_p: float
    fisher_p: float
    alternative: str


class StageShare(_Model):
    """Share of trials whose furthest stage was ``stage`` (``None`` = no stage reached)."""

    stage: str | None
    count: int
    share: float


class FunnelStep(_Model):
    """Of the trials that reached the previous stage, how many reached this one."""

    stage: str
    entered: int
    reached: int
    conversion: float | None
    ci: CI | None
    cumulative: float


class StageSummary(_Model):
    """Furthest stage reached, per arm."""

    arm: str
    n: int
    distribution: list[StageShare]
    funnel: list[FunnelStep]


class StageComparison(_Model):
    """Brunner–Munzel test on the stage index (secondary)."""

    treatment: str
    control: str
    statistic: float | None
    pvalue: float | None
    preregistered: bool


class TimingSummary(_Model):
    """Time to success, per arm."""

    arm: str
    n_successes: int
    median_s: float | None
    ci: CI | None
    curve_times: list[float]
    curve_fraction: list[float]
    preregistered: bool


class ConditionCell(_Model):
    """Outcomes of one arm in one condition."""

    arm: str
    successes: int
    completed: int


class ConditionRow(_Model):
    """Outcomes per arm for one condition."""

    condition: str
    cells: list[ConditionCell]


class SessionRow(_Model):
    """Outcomes per arm in one session."""

    session_id: str
    operator: str
    rig: str
    started_at: datetime
    cells: list[ConditionCell]


class DriftCheck(_Model):
    """Homogeneity of one arm's success rate across sessions or operators."""

    arm: str
    grouping: Literal["session", "operator"]
    groups: int
    method: str
    pvalue: float | None
    flagged: bool
    small_expected: bool


class InvalidCheck(_Model):
    """Whether invalid trials are spread evenly across arms."""

    per_arm: dict[str, int]
    attempts: dict[str, int]
    method: str | None
    pvalue: float | None
    flagged: bool


class Deviation(_Model):
    """Something that departs from the locked plan."""

    kind: Literal[
        "amendment",
        "early_unblinding",
        "late_edit",
        "incomplete",
        "out_of_order",
        "excluded_blocks",
    ]
    message: str


class Provenance(_Model):
    """Where the numbers came from."""

    generated_at: datetime
    fieldtrial_version: str
    locked_with_version: str
    python: str
    numpy: str
    scipy: str
    design_hash: str
    seed: int
    policy_fingerprints: dict[str, str]
    sessions: int
    operators: list[str]
    rigs: list[str]


class Results(_Model):
    """A complete analysis of one study."""

    schema_version: int = SCHEMA_VERSION
    study: StudyInfo
    arms: list[ArmSummary]
    primary: PrimaryResult
    sensitivity: list[IndependentComparison]
    stages: list[StageSummary]
    stage_comparisons: list[StageComparison]
    timing: list[TimingSummary]
    conditions: list[ConditionRow]
    sessions: list[SessionRow]
    drift: list[DriftCheck]
    invalid: InvalidCheck
    deviations: list[Deviation]
    provenance: Provenance
    summary: list[str]
