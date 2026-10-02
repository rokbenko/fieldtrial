r"""Best-arm selection by successive elimination with anytime-valid confidence sequences.

Arms run in blocks: each block runs every surviving arm once, so the outcomes of two arms
in a block form a paired difference. For every pair of arms :math:`(i, j)` a two-sided
betting confidence sequence (:func:`fieldtrial.stats.confseq.betting_cs`) bounds the mean
difference :math:`\Delta_{ij} = p_i - p_j` from the blocks that ran both. Each sequence uses
level :math:`\delta / P`, where :math:`P = K(K-1)/2` is the number of pairs, so with
probability at least :math:`1 - \delta` every sequence covers its difference at every block.

After each block (from ``min_blocks`` on), an arm :math:`i` is eliminated when some arm
:math:`j` is better with confidence: the lower bound of :math:`\Delta_{ji}` is above 0.
On the event that all sequences cover, a best arm is never eliminated, whenever the
analysis stops. The arms that survive are the selected set: a single survivor is the best
arm with probability at least :math:`1 - \delta`; several survivors could not be told
apart.

This is the successive-elimination scheme of Even-Dar, Mannor and Mansour (2006) with
time-uniform confidence sequences (Howard et al., 2021; Waudby-Smith and Ramdas, 2024)
on within-block differences, and a union bound over all pairs, since the leader is chosen
from the data.

References
----------
Even-Dar, E., Mannor, S. and Mansour, Y. (2006). Action elimination and stopping
conditions for the multi-armed bandit and reinforcement learning problems. *JMLR* 7,
1079–1105.

Waudby-Smith, I. and Ramdas, A. (2024). Estimating means of bounded random variables by
betting. *JRSS B* 86(1), 1–27.
"""

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from fieldtrial.stats._validation import check_open_unit
from fieldtrial.stats.confseq import BREAKS, ConfidenceSequence, betting_cs


@dataclass(frozen=True, slots=True)
class PairBound:
    """Confidence sequence of ``first - second`` at the latest block both arms ran."""

    first: str
    second: str
    blocks: int
    estimate: float
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Elimination:
    """Arm ``arm`` was eliminated after block ``block`` (1-based) because ``by`` beat it."""

    arm: str
    block: int
    by: str


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Outcome of successive elimination over the blocks seen so far.

    Attributes
    ----------
    arms
        All arms, in the order given.
    survivors
        Arms not eliminated; the selected set.
    eliminations
        Eliminations in the order they happened.
    pairs
        Latest bound for every pair of arms (``first`` is listed first in ``arms``).
    blocks
        Number of blocks analysed.
    delta
        Error probability of the whole procedure.
    """

    arms: tuple[str, ...]
    survivors: tuple[str, ...]
    eliminations: tuple[Elimination, ...]
    pairs: tuple[PairBound, ...]
    blocks: int
    delta: float

    @property
    def best(self) -> str | None:
        """The selected arm when exactly one survives, else None."""
        return self.survivors[0] if len(self.survivors) == 1 else None


def _check_outcomes(outcomes: Sequence[Sequence[float | None]], k: int) -> list[list[float | None]]:
    rows: list[list[float | None]] = []
    for b, row in enumerate(outcomes, start=1):
        if len(row) != k:
            raise ValueError(f"block {b} has {len(row)} outcomes for {k} arms")
        clean: list[float | None] = []
        for v in row:
            if v is None or math.isnan(float(v)):
                clean.append(None)
                continue
            f = float(v)
            if not 0.0 <= f <= 1.0:
                raise ValueError(f"outcomes must lie between 0 and 1, got {v!r} in block {b}")
            clean.append(f)
        rows.append(clean)
    return rows


def eliminate(
    outcomes: Sequence[Sequence[float | None]],
    arms: Sequence[str],
    delta: float = 0.05,
    *,
    min_blocks: int = 1,
    breaks: int = BREAKS,
) -> SelectionResult:
    """Successive elimination over blocks of outcomes.

    ``outcomes[b][i]`` is arm ``i``'s outcome (success 1, failure 0, or a score in
    [0, 1]) in block ``b``, or None when the arm did not run in that block (for example
    because it was already eliminated). Blocks are taken in order.
    """
    names = tuple(arms)
    k = len(names)
    if k < 2:
        raise ValueError("selection needs at least 2 arms")
    if len(set(names)) != k:
        raise ValueError("arm names must be unique")
    d = check_open_unit("delta", delta)
    if min_blocks < 1:
        raise ValueError("min_blocks must be at least 1")
    rows = _check_outcomes(outcomes, k)
    pairs = list(itertools.combinations(range(k), 2))
    level = d / len(pairs)

    # Each pair's sequence over the blocks that ran both arms. Bets are predictable, so the
    # sequence on a prefix of the data equals the prefix of the sequence on all the data.
    shared: dict[tuple[int, int], list[int]] = {}
    sequences: dict[tuple[int, int], ConfidenceSequence] = {}
    diffs: dict[tuple[int, int], list[float]] = {}
    for i, j in pairs:
        idx = [b for b, row in enumerate(rows) if row[i] is not None and row[j] is not None]
        shared[i, j] = idx
        dd = [float(rows[b][i]) - float(rows[b][j]) for b in idx]  # type: ignore[arg-type]
        diffs[i, j] = dd
        if dd:
            z = (np.asarray(dd) + 1.0) / 2.0
            sequences[i, j] = betting_cs(z, level, "two-sided", breaks=breaks)

    def bound(i: int, j: int, upto: int) -> tuple[int, float, float]:
        """Shared blocks and (low, high) of p_i - p_j after block ``upto`` (0-based)."""
        key = (i, j) if i < j else (j, i)
        n = sum(1 for b in shared[key] if b <= upto)
        if n == 0:
            return 0, -1.0, 1.0
        cs = sequences[key]
        lo, hi = 2 * cs.lower[n - 1] - 1, 2 * cs.upper[n - 1] - 1
        return (n, lo, hi) if i < j else (n, -hi, -lo)

    alive = set(range(k))
    eliminations: list[Elimination] = []
    for b in range(len(rows)):
        if b + 1 < min_blocks:
            continue
        out: list[tuple[int, int]] = []
        for i in sorted(alive):
            beaters = [(bound(j, i, b)[1], j) for j in range(k) if j != i]
            best_low, by = max(beaters)
            if best_low > 0:
                out.append((i, by))
        if len(out) == len(alive):  # only possible outside the coverage event
            continue
        for i, by in out:
            alive.discard(i)
            eliminations.append(Elimination(arm=names[i], block=b + 1, by=names[by]))

    last = len(rows) - 1
    bounds = []
    for i, j in pairs:
        n, lo, hi = bound(i, j, last)
        dd = diffs[i, j]
        bounds.append(
            PairBound(
                first=names[i],
                second=names[j],
                blocks=n,
                estimate=float(np.mean(dd)) if dd else math.nan,
                low=lo,
                high=hi,
            )
        )
    return SelectionResult(
        arms=names,
        survivors=tuple(names[i] for i in sorted(alive)),
        eliminations=tuple(eliminations),
        pairs=tuple(bounds),
        blocks=len(rows),
        delta=d,
    )
