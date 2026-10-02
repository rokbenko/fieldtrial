"""Designs checked after every block: anytime-valid stopping and best-arm selection.

Both use the blocks in schedule order, up to the first block that is not complete yet, so
the order of the data never depends on the outcomes (docs/stats/anytime.md and
docs/stats/selection.md).

- ``analysis.stopping.rule: anytime`` (2 arms): after each complete block, a betting test
  of the paired block differences. When it rejects, the study stops: an ``interim_look``
  event with ``rule: anytime`` records the blocks used, and the remaining trials are
  cancelled. Only stops are recorded; the final analysis recomputes the test from the data.
- ``analysis.selection`` (several arms): after each complete block, successive
  elimination. A ``selection_look`` event records each newly dropped arm; its remaining
  trials are cancelled, and the study stops when one arm remains.

A look records arms and blocks, never success rates, so it can run while the operator is
blinded; the console names dropped arms by blind code only.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from fieldtrial.analysis.records import TrialRecord
from fieldtrial.analysis.results import (
    AnytimeSummary,
    BoundRow,
    EliminationRow,
    PairRow,
    SelectionSummary,
)
from fieldtrial.analysis.sequential import complete_pairs, planned_blocks
from fieldtrial.design import StudySpec
from fieldtrial.stats import AnytimeTestResult, SelectionResult, eliminate, paired_anytime_test

Decision = Literal["continue", "stop"]


class LookError(ValueError):
    """The study has no adaptive rule of this kind."""


# Anytime-valid stopping (2 arms) ------------------------------------------------------------


def _anytime_settings(spec: StudySpec) -> tuple[str, str, int]:
    comparison = spec.analysis.primary.comparison
    stopping = spec.analysis.stopping
    if stopping.rule != "anytime" or comparison is None:
        raise LookError("this study has no anytime stopping rule")
    return comparison.treatment, comparison.control, stopping.min_blocks or 1


def anytime_blocks(
    records: Sequence[TrialRecord], treatment: str, control: str
) -> list[tuple[int, float]]:
    """``(block, treatment - control)`` for blocks 1, 2, ... up to the first incomplete one."""
    pairs = complete_pairs(records, treatment, control)
    out: list[tuple[int, float]] = []
    block = 1
    while block in pairs:
        t, c = pairs[block]
        out.append((block, float(t) - float(c)))
        block += 1
    return out


def _anytime_test(spec: StudySpec, diffs: Sequence[float]) -> AnytimeTestResult:
    primary = spec.analysis.primary
    return paired_anytime_test(list(diffs), primary.alpha, primary.alternative)


@dataclass(frozen=True, slots=True)
class AnytimeLook:
    """A decision to stop an anytime study. Holds nothing that reveals per-arm results."""

    blocks: tuple[int, ...]
    decision: Decision

    def payload(self) -> dict[str, Any]:
        """The ``interim_look`` event payload."""
        return {
            "rule": "anytime",
            "look": len(self.blocks),
            "blocks": list(self.blocks),
            "decision": self.decision,
        }


def anytime_check(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> AnytimeLook | None:
    """The stop decision after the latest complete block, or None when the study goes on."""
    treatment, control, min_blocks = _anytime_settings(spec)
    if any(x.get("decision") == "stop" for x in looks_done):
        return None
    blocks = anytime_blocks(records, treatment, control)
    if len(blocks) < min_blocks:
        return None
    result = _anytime_test(spec, [d for _, d in blocks])
    if result.rejected_at is None:
        return None
    return AnytimeLook(blocks=tuple(b for b, _ in blocks), decision="stop")


@dataclass(frozen=True, slots=True)
class AnytimeOutcome:
    """The final anytime analysis, for the engine."""

    summary: AnytimeSummary
    result: AnytimeTestResult
    notes: tuple[str, ...]


def anytime_final(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> AnytimeOutcome | None:
    """Recompute the test on every complete block. None when no block is complete."""
    treatment, control, min_blocks = _anytime_settings(spec)
    blocks = anytime_blocks(records, treatment, control)
    if not blocks:
        return None
    result = _anytime_test(spec, [d for _, d in blocks])
    stop = next(
        (x for x in looks_done if x.get("rule") == "anytime" and x.get("decision") == "stop"),
        None,
    )
    notes: list[str] = []
    if stop is not None and result.rejected_at is None:
        notes.append(
            f"The study stopped after {int(stop['look'])} blocks, but the current labels no "
            "longer reject the null (labels were edited after the stop)."
        )
    seq = result.sequence
    summary = AnytimeSummary(
        min_blocks=min_blocks,
        planned_blocks=planned_blocks(spec),
        blocks=len(blocks),
        rejected_at=result.rejected_at,
        stopped_at=int(stop["look"]) if stop is not None else None,
        capital=result.test.statistic,
        sequence=[
            BoundRow(blocks=i + 1, low=lo, high=hi)
            for i, (lo, hi) in enumerate(zip(seq.lower, seq.upper, strict=True))
        ],
    )
    return AnytimeOutcome(summary=summary, result=result, notes=tuple(notes))


# Best-arm selection (several arms) ----------------------------------------------------------


def _selection_settings(spec: StudySpec) -> tuple[list[str], float, int]:
    selection = spec.analysis.selection
    if selection is None:
        raise LookError("this study has no analysis.selection")
    return [a.id for a in spec.arms], selection.delta, selection.min_blocks


def recorded_eliminations(looks_done: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Arm -> block after which it was dropped, from ``selection_look`` payloads."""
    out: dict[str, int] = {}
    for look in looks_done:
        for e in look.get("eliminated", []):
            out.setdefault(str(e["arm"]), int(e["block"]))
    return out


def selection_rows(
    records: Sequence[TrialRecord], arms: Sequence[str], dropped: Mapping[str, int]
) -> list[list[float | None]]:
    """Outcomes of blocks 1, 2, ... up to the first incomplete block.

    A block is complete when every arm not yet dropped before it has a completed trial in
    it. A dropped arm counts as missing (None) in later blocks.
    """
    done: dict[tuple[int, str], bool] = {}
    for r in records:
        if r.status == "completed" and r.success is not None:
            done[r.block, r.arm] = bool(r.success)
    rows: list[list[float | None]] = []
    block = 1
    while True:
        row: list[float | None] = []
        for arm in arms:
            if arm in dropped and dropped[arm] < block:
                row.append(None)
            elif (block, arm) in done:
                row.append(float(done[block, arm]))
            else:
                return rows
        if all(v is None for v in row):
            return rows
        rows.append(row)
        block += 1


@dataclass(frozen=True, slots=True)
class SelectionLook:
    """Arms newly dropped after the latest complete block."""

    look: int
    blocks: int
    eliminated: tuple[tuple[str, int, str], ...]  # (arm, block, by)
    decision: Decision

    def payload(self) -> dict[str, Any]:
        """The ``selection_look`` event payload."""
        return {
            "rule": "elimination",
            "look": self.look,
            "blocks": self.blocks,
            "eliminated": [{"arm": a, "block": b, "by": by} for a, b, by in self.eliminated],
            "decision": self.decision,
        }


def _select(
    spec: StudySpec, records: Sequence[TrialRecord], dropped: Mapping[str, int]
) -> tuple[list[list[float | None]], SelectionResult | None]:
    arms, delta, min_blocks = _selection_settings(spec)
    rows = selection_rows(records, arms, dropped)
    if not rows:
        return rows, None
    return rows, eliminate(rows, arms, delta, min_blocks=min_blocks)


def selection_check(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> SelectionLook | None:
    """Arms to drop after the latest complete block, or None when nothing changes."""
    arms, _, _ = _selection_settings(spec)
    if any(x.get("decision") == "stop" for x in looks_done):
        return None
    dropped = recorded_eliminations(looks_done)
    rows, result = _select(spec, records, dropped)
    if result is None:
        return None
    new = tuple((e.arm, e.block, e.by) for e in result.eliminations if e.arm not in dropped)
    if not new:
        return None
    left = len(arms) - len(dropped) - len(new)
    return SelectionLook(
        look=len(looks_done) + 1,
        blocks=len(rows),
        eliminated=new,
        decision="stop" if left <= 1 else "continue",
    )


@dataclass(frozen=True, slots=True)
class SelectionOutcome:
    """The final selection analysis, for the engine."""

    summary: SelectionSummary
    notes: tuple[str, ...]


def selection_final(
    spec: StudySpec,
    records: Sequence[TrialRecord],
    looks_done: Sequence[Mapping[str, Any]],
) -> SelectionOutcome | None:
    """Recompute the elimination on every complete block. None when no block is complete."""
    _, delta, min_blocks = _selection_settings(spec)
    dropped = recorded_eliminations(looks_done)
    rows, result = _select(spec, records, dropped)
    if result is None:
        return None
    recomputed = {e.arm for e in result.eliminations}
    notes = [
        f"Arm {arm} was dropped after block {dropped[arm]}, but the current labels no "
        "longer separate it from the others (labels were edited after the look)."
        for arm in sorted(set(dropped) - recomputed)
    ]
    survivors = [a for a in result.survivors if a not in dropped]
    summary = SelectionSummary(
        delta=delta,
        min_blocks=min_blocks,
        planned_blocks=planned_blocks(spec),
        blocks=len(rows),
        survivors=survivors,
        best=survivors[0] if len(survivors) == 1 else None,
        eliminations=[
            EliminationRow(arm=e.arm, block=e.block, by=e.by) for e in result.eliminations
        ],
        pairs=[
            PairRow(
                first=p.first,
                second=p.second,
                blocks=p.blocks,
                estimate=None if math.isnan(p.estimate) else p.estimate,
                low=p.low,
                high=p.high,
            )
            for p in result.pairs
        ],
        stopped=any(x.get("decision") == "stop" for x in looks_done),
    )
    return SelectionOutcome(summary=summary, notes=tuple(notes))
