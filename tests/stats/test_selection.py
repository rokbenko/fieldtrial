"""Best-arm selection by successive elimination (v0.3).

The golden case in ``data/confseq_golden.json`` was produced by an independent
implementation of the same procedure on top of ``confseq.betting.hedged_cs`` (confseq
0.0.11): 3 arms with success rates 0.35, 0.55 and 0.9, 60 blocks, delta = 0.05, pairwise
two-sided sequences at delta / 3, an arm eliminated after the block in which some other
arm's lower bound over it exceeds 0, and no data for an arm after its elimination.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.stats import eliminate

GOLDEN = json.loads((Path(__file__).parent / "data" / "confseq_golden.json").read_text())
CASE = GOLDEN["elimination"]
ARMS = ["a", "b", "c"]


def test_golden_elimination() -> None:
    res = eliminate(CASE["rows"], ARMS, CASE["delta"])
    assert list(res.survivors) == [ARMS[i] for i in CASE["survivors"]]
    assert [(e.arm, e.block, e.by) for e in res.eliminations] == [
        (ARMS[i], b, ARMS[j]) for i, b, j in CASE["eliminations"]
    ]
    assert res.best == "c"
    assert res.blocks == 60
    for pb in res.pairs:
        key = f"{ARMS.index(pb.first)}{ARMS.index(pb.second)}"
        assert pb.low == pytest.approx(CASE["bounds"][key][0], abs=1e-12)
        assert pb.high == pytest.approx(CASE["bounds"][key][1], abs=1e-12)


def test_no_elimination_before_min_blocks() -> None:
    rows = [[1.0, 0.0]] * 30
    early = eliminate(rows, ["x", "y"], 0.05)
    assert early.survivors == ("x",)
    late = eliminate(rows, ["x", "y"], 0.05, min_blocks=31)
    assert late.survivors == ("x", "y")
    assert late.best is None


def test_missing_outcomes_and_validation() -> None:
    rows = [[1.0, None, 0.0], [1.0, float("nan"), 0.0]]
    res = eliminate(rows, ["x", "y", "z"], 0.1)
    yz = next(p for p in res.pairs if (p.first, p.second) == ("y", "z"))
    assert yz.blocks == 0
    assert math.isnan(yz.estimate)
    assert (yz.low, yz.high) == (-1.0, 1.0)
    with pytest.raises(ValueError, match="at least 2 arms"):
        eliminate([[1.0]], ["x"])
    with pytest.raises(ValueError, match="unique"):
        eliminate([[1.0, 0.0]], ["x", "x"])
    with pytest.raises(ValueError, match="2 outcomes for 3 arms"):
        eliminate([[1.0, 0.0]], ["x", "y", "z"])
    with pytest.raises(ValueError, match="between 0 and 1"):
        eliminate([[1.0, 2.0]], ["x", "y"])
    with pytest.raises(ValueError, match="min_blocks"):
        eliminate([[1.0, 0.0]], ["x", "y"], min_blocks=0)


def test_a_cycle_eliminates_nobody() -> None:
    # Pairs run in disjoint blocks: a beats b, b beats c and c beats a. Checked only at the
    # end, every arm is beaten, which cannot happen when all sequences cover; nothing is
    # eliminated then.
    n = None
    rows = [[1.0, 0.0, n]] * 25 + [[n, 1.0, 0.0]] * 25 + [[0.0, n, 1.0]] * 25
    res = eliminate(rows, ["a", "b", "c"], 0.1, min_blocks=75)
    assert res.survivors == ("a", "b", "c")
    assert res.eliminations == ()


@settings(max_examples=25, deadline=None)
@given(
    st.integers(min_value=2, max_value=4).flatmap(
        lambda k: st.lists(
            st.lists(st.sampled_from([0.0, 1.0]), min_size=k, max_size=k),
            min_size=1,
            max_size=25,
        )
    )
)
def test_survivors_are_never_empty_and_eliminated_arms_were_beaten(rows: list) -> None:
    k = len(rows[0])
    arms = [f"arm{i}" for i in range(k)]
    res = eliminate(rows, arms, 0.2, breaks=100)
    assert res.survivors
    assert set(res.survivors) | {e.arm for e in res.eliminations} == set(arms)
    for e in res.eliminations:
        assert e.by != e.arm
    assert len(res.pairs) == len(list(itertools.combinations(arms, 2)))
    for pb in res.pairs:
        assert -1 <= pb.low <= pb.high <= 1


@pytest.mark.slow
def test_no_elimination_when_arms_are_equal() -> None:
    """With equal arms, any elimination is an error; it happens with probability <= delta."""
    rng = np.random.default_rng(20261003)
    delta, k, blocks, reps = 0.1, 4, 60, 300
    errors = 0
    for _ in range(reps):
        rows = (rng.random((blocks, k)) < 0.6).astype(float).tolist()
        errors += bool(eliminate(rows, [str(i) for i in range(k)], delta, breaks=200).eliminations)
    assert errors / reps <= delta + 3 * math.sqrt(delta * (1 - delta) / reps)


@pytest.mark.slow
def test_a_clearly_best_arm_is_selected() -> None:
    rng = np.random.default_rng(3)
    p = [0.3, 0.4, 0.95]
    picked = 0
    for _ in range(40):
        rows = (rng.random((80, 3)) < p).astype(float).tolist()
        picked += eliminate(rows, ["a", "b", "c"], 0.05, breaks=200).best == "c"
    assert picked >= 36
