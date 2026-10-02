"""Schedules, blinding, hashing and the stable RNG (docs/PLAN.md section 11)."""

import json
from collections import Counter
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fieldtrial.design import blind_codes, build_schedule, design_hash, parse_study
from fieldtrial.design.schedule import williams_sequences
from fieldtrial.stats._rng import StableRng
from fieldtrial.templates import TEMPLATES, template_text

SNAPSHOT = json.loads((Path(__file__).parent / "snapshots" / "schedules.json").read_text())


@pytest.mark.parametrize("template", TEMPLATES)
def test_schedule_and_codes_match_snapshot(template: str) -> None:
    # Same seed and design give the identical schedule on every platform and numpy version.
    spec = parse_study(template_text(template, "snapshot")).spec
    schedule = [
        [s.seq, s.block, s.replicate, s.position, s.condition, s.arm] for s in build_schedule(spec)
    ]
    assert schedule == SNAPSHOT[template]["schedule"]
    assert (
        blind_codes([a.id for a in spec.arms], spec.design.seed)
        == SNAPSHOT[template]["blind_codes"]
    )


def test_raw_bit_stream_is_stable() -> None:
    rng = StableRng(20261001, 1)
    assert [str(rng.raw()) for _ in range(5)] == SNAPSHOT["raw_pcg64_seed20261001_stream1"]


def _spec(
    n_arms: int, n_conditions: int, replicates: int = 1, order: str = "random", seed: int = 7
) -> object:
    arms = ", ".join(f"{{id: a{i}}}" for i in range(n_arms))
    return parse_study(
        f"""
fieldtrial: 1
name: s
rubric: {{stages: [{{id: done, label: Done}}], success: done}}
arms: [{arms}]
conditions: {{factors: {{c: {{range: [1, {n_conditions}]}}}}, replicates: {replicates}}}
design: {{type: randomized_block, order: {order}, seed: {seed}}}
analysis: {{primary: {{comparison: {{treatment: a1, control: a0}}}}}}
"""
    ).spec


@settings(max_examples=60, deadline=None)
@given(st.integers(2, 6), st.integers(1, 30), st.integers(1, 3), st.integers(0, 10**9))
def test_complete_blocks_and_counts(n_arms: int, n_cond: int, reps: int, seed: int) -> None:
    spec = _spec(n_arms, n_cond, reps, seed=seed)
    slots = build_schedule(spec)  # type: ignore[arg-type]
    assert [s.seq for s in slots] == list(range(1, len(slots) + 1))
    assert len(slots) == n_arms * n_cond * reps
    blocks: dict[int, list[str]] = {}
    for s in slots:
        blocks.setdefault(s.block, []).append(s.arm)
    for arms in blocks.values():
        assert sorted(arms) == sorted(f"a{i}" for i in range(n_arms))  # every arm once per block
    per_condition = Counter((s.condition, s.arm) for s in slots)
    assert set(per_condition.values()) == {reps}
    # Replicates run as rounds: all of replicate 1 before any of replicate 2.
    assert [s.replicate for s in slots] == sorted(s.replicate for s in slots)


@settings(max_examples=40, deadline=None)
@given(st.integers(2, 6), st.integers(0, 10**9))
def test_positions_are_balanced_over_full_cycles(n_arms: int, seed: int) -> None:
    rows = len(williams_sequences(n_arms))
    spec = _spec(n_arms, rows * 3, seed=seed)
    counts = Counter((s.arm, s.position) for s in build_schedule(spec))  # type: ignore[arg-type]
    # Each Williams row puts every arm in one position, so per full cycle each arm occupies
    # each position rows / n_arms times (1 for even n_arms, 2 for odd).
    assert set(counts.values()) == {3 * rows // n_arms}


@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_williams_design_balances_position_and_carryover(k: int) -> None:
    rows = williams_sequences(k)
    assert len(rows) == (k if k % 2 == 0 else 2 * k)
    for column in zip(*rows, strict=True):
        assert Counter(column) == {i: len(rows) // k for i in range(k)}
    successors = Counter((r[i], r[i + 1]) for r in rows for i in range(k - 1))
    assert len(set(successors.values())) == 1  # every ordered pair equally often


def test_fixed_order_keeps_condition_order() -> None:
    spec = _spec(2, 5, order="fixed")
    blocks = [s.condition for s in build_schedule(spec) if s.position == 1]  # type: ignore[arg-type]
    assert blocks == [f"c={i}" for i in range(1, 6)]


def test_different_seeds_give_different_schedules() -> None:
    a = build_schedule(_spec(3, 10, seed=1))  # type: ignore[arg-type]
    b = build_schedule(_spec(3, 10, seed=2))  # type: ignore[arg-type]
    assert a != b


def test_blind_codes() -> None:
    codes = blind_codes(["a", "b", "c"], 5)
    assert len(set(codes.values())) == 3
    assert all(len(c) == 2 for c in codes.values())
    assert blind_codes(["a", "b", "c"], 5) == codes
    with pytest.raises(ValueError, match="at most"):
        blind_codes([str(i) for i in range(100)], 1)


def test_rng_basics() -> None:
    rng = StableRng(1, 1)
    draws = [rng.below(6) for _ in range(600)]
    assert set(draws) == set(range(6))
    assert 0.0 <= rng.uniform() < 1.0
    with pytest.raises(ValueError, match="positive"):
        rng.below(0)
    with pytest.raises(ValueError, match="non-negative"):
        StableRng(-1, 0)
    with pytest.raises(ValueError, match="at least 1"):
        williams_sequences(0)


BASIC = template_text("basic", "demo")


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('title: "RTC queue threshold on the 21h pi0.5 checkpoint"', 'title: "Another title"'),
        ('label: "queue 50"', 'label: "renamed"'),
        ('{id: lift,     label: "Lifted the correct part"}', '{id: lift, label: "Picked up"}'),
        ("Bimanual pick, handover, insertion.", "Changed description."),
        ("  replicates: 1\n", "  replicates: 1\n"),  # unchanged text: same hash
        ("    metric: success\n", ""),  # dropping an explicit default
    ],
)
def test_cosmetic_edits_keep_the_hash(old: str, new: str) -> None:
    base = design_hash(parse_study(BASIC).spec)
    assert design_hash(parse_study(BASIC.replace(old, new, 1)).spec) == base


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seed: 20261001", "seed: 20261002"),
        ("alpha: 0.05", "alpha: 0.01"),
        ("slot: {range: [1, 40]}", "slot: {range: [1, 41]}"),
        ("inference.queue_threshold: 50", "inference.queue_threshold: 60"),
        ("success: clean", "success: inserted"),
        ("timeout_s: 45", "timeout_s: 60"),
    ],
)
def test_substantive_edits_change_the_hash(old: str, new: str) -> None:
    base = design_hash(parse_study(BASIC).spec)
    assert design_hash(parse_study(BASIC.replace(old, new, 1)).spec) != base
