"""The ``study.yaml`` schema (docs/PLAN.md section 10).

Unknown keys are errors everywhere except inside an arm's ``policy`` and ``serving``, which
are passed through to the runner unchanged.
"""

import itertools
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[
    str,
    Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", min_length=1, max_length=64),
]
FactorValue = int | float | str | bool
MAX_CONDITIONS_WARNING = 500


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Task(_Strict):
    """What the robot is asked to do."""

    instruction: str | None = None
    description: str | None = None


class Stage(_Strict):
    """One progress stage. Reaching a stage implies reaching every earlier stage."""

    id: Identifier
    label: str


class Rubric(_Strict):
    """How a trial is scored: ordered progress stages, the success stage and failure tags."""

    stages: list[Stage] = Field(min_length=1)
    success: Identifier
    failure_tags: list[Identifier] = Field(default_factory=list)
    calibration_media: str | None = None
    rig_checklist: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Rubric":
        ids = [s.id for s in self.stages]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"stage ids must be unique; repeated: {', '.join(duplicates)}")
        if self.success not in ids:
            raise ValueError(f"success stage {self.success!r} is not one of the stages {ids}")
        tags = self.failure_tags
        if len(set(tags)) != len(tags):
            raise ValueError("failure_tags must be unique")
        return self

    @property
    def success_index(self) -> int:
        """Index of the success stage in ``stages``."""
        return [s.id for s in self.stages].index(self.success)


class Limits(_Strict):
    """Per-trial limits."""

    timeout_s: float | None = Field(default=None, gt=0)
    no_progress_s: float | None = Field(default=None, gt=0)
    reset: Literal["independent", "carry_over"] = "independent"


class Arm(_Strict):
    """One thing being evaluated: a policy, its serving configuration and a runner."""

    id: Identifier
    label: str | None = None
    runner: Literal["manual", "sim"] = "manual"
    policy: dict[str, Any] = Field(default_factory=dict)
    serving: dict[str, Any] = Field(default_factory=dict)


class Factor(_Strict):
    """One factor of the conditions: an inclusive integer ``range`` or explicit ``values``."""

    range: tuple[int, int] | None = None
    values: list[FactorValue] | None = None

    @model_validator(mode="after")
    def _check(self) -> "Factor":
        if (self.range is None) == (self.values is None):
            raise ValueError("give exactly one of 'range: [first, last]' or 'values: [...]'")
        if self.range is not None and self.range[0] > self.range[1]:
            raise ValueError(f"range {list(self.range)} is empty: first must be <= last")
        if self.values is not None:
            if not self.values:
                raise ValueError("values must not be empty")
            if len({repr(v) for v in self.values}) != len(self.values):
                raise ValueError("values must be unique")
        return self

    def levels(self) -> list[FactorValue]:
        """The factor's levels in order."""
        if self.range is not None:
            return list(range(self.range[0], self.range[1] + 1))
        assert self.values is not None
        return list(self.values)


class Conditions(_Strict):
    """Initial setups. The cartesian product of all factors is the condition list."""

    factors: dict[Identifier, Factor] = Field(min_length=1)
    replicates: int = Field(default=1, ge=1, le=100)

    def count(self) -> int:
        """Number of distinct conditions (before replication)."""
        return math.prod(len(f.levels()) for f in self.factors.values())

    def expand(self) -> list[dict[str, FactorValue]]:
        """All conditions, in factor order, as ``{factor: level}`` dicts."""
        names = list(self.factors)
        product = itertools.product(*(self.factors[n].levels() for n in names))
        return [dict(zip(names, combo, strict=True)) for combo in product]


class Design(_Strict):
    """How trials are scheduled and blinded."""

    type: Literal["randomized_block", "single_arm"]
    order: Literal["random", "fixed"] = "random"
    blinding: Literal["none", "operator"] = "none"
    seed: int = Field(ge=0, le=2**63 - 1)


class Comparison(_Strict):
    """Primary comparison: ``treatment`` against ``control`` (arm ids)."""

    treatment: Identifier
    control: Identifier


class Primary(_Strict):
    """The pre-registered primary analysis."""

    metric: Literal["success"] = "success"
    comparison: Comparison | None = None
    threshold: float | None = Field(default=None, gt=0, lt=1)
    alternative: Literal["two-sided", "greater", "less"] = "two-sided"
    alpha: float = Field(default=0.05, gt=0, le=0.2)


class Stopping(_Strict):
    """Stopping rule. Only a fixed sample size is supported in v0.1."""

    rule: Literal["fixed"] = "fixed"


class Analysis(_Strict):
    """Pre-registered analysis settings."""

    primary: Primary
    interval: Literal["wilson", "clopper-pearson", "jeffreys", "agresti-coull"] = "wilson"
    multiplicity: Literal["holm", "bonferroni", "bh"] = "holm"
    secondary: list[Literal["stage_reached", "time_to_success"]] = Field(default_factory=list)
    stopping: Stopping = Field(default_factory=Stopping)


class StudySpec(_Strict):
    """A complete ``study.yaml``."""

    fieldtrial: Literal[1]
    name: Identifier
    title: str | None = None
    task: Task = Field(default_factory=Task)
    rubric: Rubric
    limits: Limits = Field(default_factory=Limits)
    arms: list[Arm] = Field(min_length=1)
    conditions: Conditions
    design: Design
    analysis: Analysis

    @field_validator("arms")
    @classmethod
    def _unique_arms(cls, arms: list[Arm]) -> list[Arm]:
        ids = [a.id for a in arms]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"arm ids must be unique; repeated: {', '.join(duplicates)}")
        return arms

    @model_validator(mode="after")
    def _check_design(self) -> "StudySpec":
        arm_ids = [a.id for a in self.arms]
        primary = self.analysis.primary
        if self.limits.reset == "carry_over":
            raise ValueError(
                "limits.reset: carry_over needs design.type: crossover_rounds (arrives in v0.2)"
            )
        if self.design.type == "randomized_block":
            if len(self.arms) < 2:
                raise ValueError("design.type randomized_block needs at least 2 arms")
            if primary.comparison is None:
                raise ValueError("analysis.primary.comparison is required for a comparative design")
            if primary.threshold is not None:
                raise ValueError("analysis.primary.threshold is only for design.type single_arm")
        else:
            if len(self.arms) != 1:
                raise ValueError("design.type single_arm takes exactly 1 arm")
            if primary.comparison is not None:
                raise ValueError("design.type single_arm has no comparison; use threshold")
        if primary.comparison is not None:
            for role in ("treatment", "control"):
                arm = getattr(primary.comparison, role)
                if arm not in arm_ids:
                    raise ValueError(
                        f"analysis.primary.comparison.{role} {arm!r} is not an arm; "
                        f"arms are {arm_ids}"
                    )
            if primary.comparison.treatment == primary.comparison.control:
                raise ValueError("treatment and control must be different arms")
        return self

    def arm(self, arm_id: str) -> Arm:
        """Return the arm with this id."""
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise KeyError(arm_id)
