"""The ``study.yaml`` schema (docs/PLAN.md section 10).

Unknown keys are errors everywhere except inside an arm's ``policy`` and ``serving``, which
are passed through to the runner unchanged.
"""

import itertools
import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fieldtrial.design.capture_config import CaptureConfig
from fieldtrial.design.runner_config import (
    RunnersConfig,
    TemplateError,
    lerobot_arm_error,
    render_command,
    sample_values,
)

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
    runner: Literal["manual", "sim", "command", "openpi_router", "lerobot"] = "manual"
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
    """How trials are scheduled and blinded.

    ``rounds`` is the number of rounds of a ``crossover_rounds`` design: an even number,
    since rounds come in cycles of two (one per arm).
    """

    type: Literal["randomized_block", "single_arm", "crossover_rounds"]
    order: Literal["random", "fixed"] = "random"
    blinding: Literal["none", "operator"] = "none"
    seed: int = Field(ge=0, le=2**63 - 1)
    rounds: int | None = Field(default=None, ge=4, le=200)

    @property
    def cycles(self) -> int:
        """Number of crossover cycles (0 for other designs)."""
        return (self.rounds or 0) // 2


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
    """Stopping rule: a fixed sample size, group-sequential looks, or anytime-valid looks.

    ``group_sequential``: ``looks`` interim and final analyses happen after equal shares of
    the planned blocks, unless ``at`` lists the information fractions (increasing, ending
    at 1). ``anytime``: the study is checked after every complete block, from
    ``min_blocks`` on, with an anytime-valid test (docs/stats/anytime.md).
    """

    rule: Literal["fixed", "group_sequential", "anytime"] = "fixed"
    looks: int | None = Field(default=None, ge=2, le=10)
    spending: Literal["obrien_fleming", "pocock"] = "obrien_fleming"
    at: list[float] | None = None
    min_blocks: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> "Stopping":
        if self.rule != "group_sequential" and (self.looks is not None or self.at is not None):
            raise ValueError("looks and at are only for rule: group_sequential")
        if self.rule != "anytime" and self.min_blocks is not None:
            raise ValueError("min_blocks is only for rule: anytime")
        if self.rule != "group_sequential":
            return self
        if self.looks is None:
            raise ValueError("rule: group_sequential needs looks (2 to 10)")
        if self.at is not None:
            if len(self.at) != self.looks:
                raise ValueError(f"at lists {len(self.at)} fractions but looks is {self.looks}")
            prev = 0.0
            for t in self.at:
                if not prev < t <= 1.0:
                    raise ValueError("at must be strictly increasing fractions in (0, 1]")
                prev = t
            if self.at[-1] != 1.0:
                raise ValueError("the last look in at must be 1 (the full study)")
        return self

    def fractions(self) -> list[float]:
        """Information fraction of each planned look."""
        if self.at is not None:
            return list(self.at)
        assert self.looks is not None
        return [k / self.looks for k in range(1, self.looks + 1)]


class Ladder(_Strict):
    """A checkpoint ladder: arms in training order, their steps, and a plateau margin.

    The trend test uses ``steps`` as scores (0, 1, 2, ... when omitted). ``margin`` is the
    non-inferiority margin for plateau detection, on the success-rate scale.
    """

    arms: list[Identifier] = Field(min_length=3)
    steps: list[float] | None = None
    margin: float | None = Field(default=None, gt=0, lt=1)

    @model_validator(mode="after")
    def _check(self) -> "Ladder":
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("ladder arms must be unique")
        if self.steps is not None:
            if len(self.steps) != len(self.arms):
                raise ValueError(
                    f"ladder steps lists {len(self.steps)} values for {len(self.arms)} arms"
                )
            if any(b <= a for a, b in itertools.pairwise(self.steps)):
                raise ValueError("ladder steps must be strictly increasing")
        return self

    def scores(self) -> list[float]:
        """Trend-test scores, one per ladder arm."""
        if self.steps is not None:
            return list(self.steps)
        return [float(i) for i in range(len(self.arms))]


class Selection(_Strict):
    """Best-arm selection by successive elimination (docs/stats/selection.md).

    After every complete block (from ``min_blocks`` on), an arm is dropped when another arm
    beats it with confidence; its remaining trials are cancelled. With probability at least
    ``1 - delta`` a best arm is never dropped.
    """

    rule: Literal["elimination"] = "elimination"
    delta: float = Field(default=0.05, gt=0, le=0.5)
    min_blocks: int = Field(default=1, ge=1)


class Proxy(_Strict):
    """Proxy-assisted estimates from reward-model scores (docs/stats/ppi.md).

    A score at or above ``threshold`` is shown as a suggested success. With this section,
    reports add PPI++ estimates per arm as a pre-registered secondary analysis.
    """

    threshold: float = Field(default=0.5, gt=0, lt=1)


class Analysis(_Strict):
    """Pre-registered analysis settings."""

    primary: Primary
    interval: Literal["wilson", "clopper-pearson", "jeffreys", "agresti-coull"] = "wilson"
    multiplicity: Literal["holm", "bonferroni", "bh"] = "holm"
    secondary: list[Literal["stage_reached", "time_to_success"]] = Field(default_factory=list)
    stopping: Stopping = Field(default_factory=Stopping)
    ladder: Ladder | None = None
    selection: Selection | None = None
    proxy: Proxy | None = None


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
    runners: RunnersConfig | None = None
    capture: CaptureConfig | None = None

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
        design = self.design
        ladder = self.analysis.ladder
        stopping = self.analysis.stopping
        selection = self.analysis.selection
        if self.limits.reset == "carry_over" and design.type != "crossover_rounds":
            raise ValueError(
                "limits.reset: carry_over needs design.type: crossover_rounds, where an arm "
                "runs a whole round before the scene is reset"
            )
        if design.rounds is not None and design.type != "crossover_rounds":
            raise ValueError("design.rounds is only for design.type crossover_rounds")
        if design.type != "single_arm" and primary.threshold is not None:
            raise ValueError("analysis.primary.threshold is only for design.type single_arm")
        if design.type == "randomized_block":
            if len(self.arms) < 2:
                raise ValueError("design.type randomized_block needs at least 2 arms")
            if primary.comparison is None and ladder is None and selection is None:
                raise ValueError(
                    "analysis.primary.comparison is required for a comparative design "
                    "(or analysis.ladder, whose trend test is then the primary analysis, "
                    "or analysis.selection, to select the best arm)"
                )
        elif design.type == "crossover_rounds":
            if len(self.arms) != 2:
                raise ValueError("design.type crossover_rounds compares exactly 2 arms")
            if design.rounds is None:
                raise ValueError("design.type crossover_rounds needs design.rounds (4 or more)")
            if design.rounds % 2:
                raise ValueError(
                    f"design.rounds must be even (cycles of two rounds), got {design.rounds}"
                )
            if self.conditions.replicates != 1:
                raise ValueError(
                    "design.type crossover_rounds repeats conditions through rounds; "
                    "set conditions.replicates to 1"
                )
            if primary.comparison is None:
                raise ValueError("analysis.primary.comparison is required for a comparative design")
        else:
            if len(self.arms) != 1:
                raise ValueError("design.type single_arm takes exactly 1 arm")
            if primary.comparison is not None:
                raise ValueError("design.type single_arm has no comparison; use threshold")
        if ladder is not None:
            if design.type != "randomized_block":
                raise ValueError("analysis.ladder needs design.type randomized_block")
            for arm in ladder.arms:
                if arm not in arm_ids:
                    raise ValueError(
                        f"analysis.ladder arm {arm!r} is not an arm; arms are {arm_ids}"
                    )
        if stopping.rule != "fixed":
            kind = stopping.rule.replace("_", "-")
            if design.type != "randomized_block" or primary.comparison is None:
                raise ValueError(
                    f"{kind} stopping needs design.type randomized_block with a primary comparison"
                )
            if len(self.arms) != 2:
                raise ValueError(f"{kind} stopping supports exactly 2 arms in this version")
            if self.conditions.replicates != 1:
                raise ValueError(f"{kind} stopping needs conditions.replicates: 1 in this version")
        if selection is not None:
            if design.type != "randomized_block":
                raise ValueError("analysis.selection needs design.type randomized_block")
            if primary.comparison is not None or ladder is not None:
                raise ValueError(
                    "analysis.selection is the primary analysis; remove "
                    "analysis.primary.comparison and analysis.ladder"
                )
            if stopping.rule != "fixed":
                raise ValueError(
                    "analysis.selection stops by itself; leave analysis.stopping at fixed"
                )
            if self.conditions.replicates != 1:
                raise ValueError("analysis.selection needs conditions.replicates: 1")
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

    @model_validator(mode="after")
    def _check_runners(self) -> "StudySpec":
        kinds = {a.runner for a in self.arms}
        switching = kinds & {"command", "openpi_router", "lerobot"}
        if switching and len(kinds) > 1:
            raise ValueError(
                f"arms use different runners ({', '.join(sorted(kinds))}); a switching "
                "runner (command, openpi_router or lerobot) must run every arm"
            )
        config = self.runners or RunnersConfig()
        if "command" in kinds:
            if config.command is None:
                raise ValueError("arms use runner: command; add runners.command.template")
            factors = self.conditions.expand()[0]
            for arm in self.arms:
                try:
                    render_command(
                        config.command.template,
                        sample_values(policy=arm.policy, serving=arm.serving, factors=factors),
                    )
                except TemplateError as exc:
                    raise ValueError(f"runners.command.template for arm {arm.id!r}: {exc}") from exc
        if "openpi_router" in kinds:
            for arm in self.arms:
                url = arm.policy.get("url")
                if not isinstance(url, str) or not url.startswith(("ws://", "wss://")):
                    raise ValueError(
                        f"arm {arm.id!r} uses runner: openpi_router and needs "
                        "policy.url: ws://host:port (its policy server)"
                    )
        if "lerobot" in kinds:
            if config.lerobot is None:
                raise ValueError("arms use runner: lerobot; add runners.lerobot.robot")
            for arm in self.arms:
                error = lerobot_arm_error(arm.policy, arm.serving)
                if error:
                    raise ValueError(f"arm {arm.id!r} uses runner: lerobot and {error}")
        return self

    def arm(self, arm_id: str) -> Arm:
        """Return the arm with this id."""
        for a in self.arms:
            if a.id == arm_id:
                return a
        raise KeyError(arm_id)
