"""Randomized complete block schedules (docs/PLAN.md section 11).

Each block is one condition and replicate; within a block every arm runs once. The order of
arms inside blocks follows a Williams design, so every arm appears in every position equally
often over each full cycle of sequences, and (for an even number of arms) every arm follows
every other arm equally often. Time-of-day and carryover effects then average out.
"""

from dataclasses import dataclass

from fieldtrial.design._rng import STREAM_SCHEDULE, StableRng
from fieldtrial.design.models import FactorValue, StudySpec


@dataclass(frozen=True, slots=True)
class Condition:
    """One condition: its position in the condition list, a key such as ``slot=17``, and factors."""

    index: int
    key: str
    factors: dict[str, FactorValue]


@dataclass(frozen=True, slots=True)
class Slot:
    """One scheduled trial.

    ``seq`` (1-based) is the global run order, ``block`` (1-based) the block, ``position``
    (1-based) the place inside the block, and ``replicate`` (1-based) the replicate of the
    condition.
    """

    seq: int
    block: int
    replicate: int
    position: int
    condition: str
    arm: str


def condition_key(factors: dict[str, FactorValue]) -> str:
    """Stable, readable key for a condition, for example ``object=cup,slot=3``."""
    return ",".join(f"{name}={value}" for name, value in factors.items())


def conditions(spec: StudySpec) -> list[Condition]:
    """The study's conditions in factor order."""
    return [
        Condition(index=i, key=condition_key(f), factors=f)
        for i, f in enumerate(spec.conditions.expand())
    ]


def williams_sequences(k: int) -> list[list[int]]:
    """Williams design for ``k`` treatments: rows balanced for position and first-order carryover.

    For even ``k`` the design has ``k`` rows; for odd ``k``, ``2k`` rows (each row and its
    reverse).
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    first = [0]
    low, high = 1, k - 1
    take_low = True
    while len(first) < k:
        if take_low:
            first.append(low)
            low += 1
        else:
            first.append(high)
            high -= 1
        take_low = not take_low
    rows = [[(x + shift) % k for x in first] for shift in range(k)]
    if k % 2 == 1 and k > 1:
        rows += [list(reversed(r)) for r in rows]
    return rows


def build_schedule(spec: StudySpec) -> list[Slot]:
    """The full schedule for a study, determined only by the design and its seed.

    - ``randomized_block``: replicates run as rounds; within each round the condition order is
      shuffled (or kept with ``order: fixed``). Arm orders cycle through a shuffled Williams
      design, reshuffled for every full cycle, with arms randomly mapped to its symbols.
    - ``single_arm``: one trial per condition and replicate, in shuffled or fixed order.
    """
    rng = StableRng(spec.design.seed, STREAM_SCHEDULE)
    conds = conditions(spec)
    arm_ids = [a.id for a in spec.arms]

    blocks: list[tuple[int, Condition]] = []
    for replicate in range(1, spec.conditions.replicates + 1):
        round_conditions = list(conds)
        if spec.design.order == "random":
            rng.shuffle(round_conditions)
        blocks.extend((replicate, c) for c in round_conditions)

    symbols = list(arm_ids)
    rng.shuffle(symbols)  # random mapping of arms to Williams symbols
    rows = williams_sequences(len(arm_ids))
    cycle: list[list[int]] = []

    slots: list[Slot] = []
    seq = 0
    for block_number, (replicate, cond) in enumerate(blocks, start=1):
        if not cycle:
            cycle = [list(r) for r in rows]
            rng.shuffle(cycle)
        order = cycle.pop()
        for position, symbol in enumerate(order, start=1):
            seq += 1
            slots.append(
                Slot(
                    seq=seq,
                    block=block_number,
                    replicate=replicate,
                    position=position,
                    condition=cond.key,
                    arm=symbols[symbol],
                )
            )
    return slots
