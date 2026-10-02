"""Randomness that is identical on every platform and numpy version.

NumPy guarantees the raw bit stream of ``PCG64`` seeded from a ``SeedSequence``, but not the
output of ``Generator`` methods such as ``shuffle`` across versions. Schedules must be
reproducible from the study seed forever, so shuffling and sampling here use only the raw
64-bit outputs.
"""

from collections.abc import MutableSequence
from typing import TypeVar

from numpy.random import PCG64, SeedSequence

T = TypeVar("T")

# Independent streams derived from the study seed, one per purpose.
STREAM_SCHEDULE = 1
STREAM_BLINDING = 2
STREAM_SIMULATION = 3

_TWO_64 = 2**64


class StableRng:
    """Uniform integers, floats and shuffles from ``PCG64(SeedSequence([seed, stream]))``."""

    def __init__(self, seed: int, stream: int) -> None:
        if seed < 0 or stream < 0:
            raise ValueError("seed and stream must be non-negative")
        self._bits = PCG64(SeedSequence([seed, stream]))

    def raw(self) -> int:
        """Next raw 64-bit output."""
        return int(self._bits.random_raw())

    def below(self, n: int) -> int:
        """Uniform integer in ``[0, n)``, by rejection sampling (no modulo bias)."""
        if n <= 0:
            raise ValueError("n must be positive")
        limit = (_TWO_64 // n) * n
        while True:
            draw = self.raw()
            if draw < limit:
                return draw % n

    def uniform(self) -> float:
        """Uniform float in ``[0, 1)`` with 53 random bits."""
        return (self.raw() >> 11) * (1.0 / (1 << 53))

    def shuffle(self, items: MutableSequence[T]) -> None:
        """Shuffle in place (Fisher–Yates)."""
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
