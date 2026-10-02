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
    "crossover",  # crossover rounds: period-adjusted difference, randomization test
    "ladder",  # checkpoint ladder without a comparison: association with training step
    "group_sequential",  # 2 arms, complete blocks, McNemar score at planned looks
    "anytime",  # 2 arms, complete blocks, betting test checked after every block
    "selection",  # several arms, successive elimination after every block
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
        "interim",
        "rig_drift",
    ]
    message: str


class StepAssociation(_Model):
    """Test of a linear association between training step and success (ladders)."""

    test: str  # "mantel" (stratified by condition) or "cochran_armitage"
    statistic: float | None
    pvalue: float | None
    alternative: str
    rejected: bool
    primary: bool  # True when this is the pre-registered primary analysis


class PlateauRow(_Model):
    """One checkpoint tested for non-inferiority against the final checkpoint."""

    arm: str
    difference: float  # final minus this checkpoint
    ci: CI
    tested: bool
    noninferior: bool


class LadderResult(_Model):
    """A checkpoint ladder: association with step, and where success levels off."""

    arms: list[str]
    steps: list[float]
    association: StepAssociation
    margin: float | None
    plateau_method: str | None
    plateau: list[PlateauRow] = Field(default_factory=list)
    plateau_arm: str | None = None


class RoundRow(_Model):
    """One round of a crossover design."""

    round: int
    cycle: int
    period: int
    arm: str
    successes: int
    completed: int
    rate: float | None


class CrossoverSummary(_Model):
    """Crossover rounds: per-round results and the period effect."""

    treatment: str
    control: str
    cycles: int
    cycles_used: int
    ab: int  # cycles where the treatment ran first
    ba: int
    rounds: list[RoundRow]
    period_effect: float | None


class LookRow(_Model):
    """One look of a group-sequential design."""

    look: int
    kind: Literal["interim", "final"]
    fraction: float
    blocks: int
    z: float | None
    boundary: float
    crossed: bool
    recorded_decision: Literal["continue", "stop"] | None = None


class SequentialSummary(_Model):
    """Group-sequential stopping: planned and performed looks."""

    spending: str
    planned_looks: int
    planned_fractions: list[float]
    planned_blocks: int
    looks: list[LookRow]
    stopped_at: int | None


class BoundRow(_Model):
    """A confidence-sequence interval after ``blocks`` complete blocks."""

    blocks: int
    low: float
    high: float


class AnytimeSummary(_Model):
    """Anytime-valid stopping: the test after every complete block.

    ``rejected_at`` is the first block count at which the data reject the null;
    ``stopped_at`` the block count at which the study was stopped (from the recorded look).
    """

    min_blocks: int
    planned_blocks: int
    blocks: int
    rejected_at: int | None
    stopped_at: int | None
    capital: float
    sequence: list[BoundRow]


class EliminationRow(_Model):
    """An arm dropped after ``block`` because ``by`` beat it with confidence."""

    arm: str
    block: int
    by: str


class PairRow(_Model):
    """Confidence sequence of ``first - second`` on the blocks that ran both arms."""

    first: str
    second: str
    blocks: int
    estimate: float | None
    low: float
    high: float


class SelectionSummary(_Model):
    """Best-arm selection by successive elimination."""

    delta: float
    min_blocks: int
    planned_blocks: int
    blocks: int
    survivors: list[str]
    best: str | None
    eliminations: list[EliminationRow]
    pairs: list[PairRow]
    stopped: bool


class RunnerSummary(_Model):
    """What the runner measured for one arm (descriptive; not a test).

    ``latency_ms_median`` is the median of the per-trial median request latencies and
    ``latency_ms_p95`` the largest per-trial 95th percentile. ``abnormal_exits`` counts
    command-runner trials that ended with a non-zero exit code before the stop.
    """

    arm: str
    trials: int
    requests: int | None = None
    errors: int | None = None
    latency_ms_median: float | None = None
    latency_ms_p95: float | None = None
    abnormal_exits: int | None = None


class RigCheckRow(_Model):
    """A check of the rig against its reference photo."""

    checked_at: datetime
    session_id: str | None
    source: str
    shift_px: float
    brightness_change: float
    similarity: float
    flagged: bool
    reasons: list[str] = Field(default_factory=list)


class EpisodeSummary(_Model):
    """Linked LeRobot episodes of one arm (descriptive).

    Intervention counts are None when the dataset has no ``intervention`` flags.
    """

    arm: str
    linked_trials: int
    frames: int
    trials_with_intervention: int | None = None
    intervention_frames: int | None = None


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
    ladder: LadderResult | None = None
    crossover: CrossoverSummary | None = None
    sequential: SequentialSummary | None = None
    anytime: AnytimeSummary | None = None
    selection: SelectionSummary | None = None
    runner: list[RunnerSummary] = Field(default_factory=list)
    rig_checks: list[RigCheckRow] = Field(default_factory=list)
    episodes: list[EpisodeSummary] = Field(default_factory=list)
